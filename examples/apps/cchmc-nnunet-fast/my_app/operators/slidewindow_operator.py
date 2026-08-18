# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SlideWindowOperator: the nnUNet sliding-window inference core, GPU-resident.

Contract (Phase 1 plan 02):

* The model architecture and every fold's weights are loaded **once in
  ``setup()``** (graph-build time); ``compute()`` is inference-only, so a
  second study pays no cold-start cost (INF-008).
* Sliding-window inference replicates the reference nnUNet predictor
  1:1 in its numerically-relevant choices:
    - ``pad_nd_image`` to the patch size (constant 0, centered);
    - ``compute_steps_for_sliding_window(image, patch, tile_step_size=0.5)``
      for the window positions;
    - ``compute_gaussian(patch, sigma_scale=1/8, value_scaling_factor=10)``
      as the blending kernel, with a matching per-voxel visit-count map;
    - ``logits /= visit_counts`` at the end.
  Two deliberate, documented differences from ``nnUNetPredictor`` 2.8.1:
    - the accumulation runs in **FP32** (nnUNet 2.8.1 accumulates the sliding
      window and the TTA sum in FP16); FP16 ``+=`` is non-associative and the
      plan pins the FP32 accumulator (INF-004);
    - the results device is **always CUDA** — nnUNet's OOM handler silently
      re-runs inference with CPU results; this operator never catches
      ``RuntimeError``/OOM and never returns a CPU tensor (INF-001/INF-005).
* TTA mirror flips run in the **exact nnUNet order** (normal pass first, then
  all ``itertools.combinations`` of the ``+2``-shifted allowed mirroring
  axes, sizes 1..N) with **sequential FP32 ``+=``** (INF-003/INF-004).
* Autocast mirrors the reference's boundary exactly: each fold's
  sliding-window inference runs under ``torch.autocast("cuda")`` (FP16
  convs, eager mode) and the per-fold ``load_state_dict`` happens **outside**
  the active autocast — the reference's
  ``predict_logits_from_preprocessed_data`` loads each fold between separate
  autocast scopes, and replicating that is what keeps the numerics
  reproducible (verified: a single autocast around the whole fold loop, with
  mid-loop weight loads, shifts fold outputs by up to ~13 on the airway
  study). TTA/SW/fold accumulation all happens in FP32. Autocast is never
  split across operator boundaries (INF-011).
* The network is built from the bundle ``plans.json`` architecture and the
  weights come from the bundle checkpoint path — no hard-coded trainer class
  (INF-007).

Note on "MONAI sliding window": MONAI 1.3.0's ``sliding_window_inference``
uses a different step generator (fixed ``int(roi*(1-overlap))`` interval via
``_get_scan_interval``) and a different analytic Gaussian kernel than the
reference (verified: normalized-kernel max abs diff 0.034 on the 128^3 patch,
and different step sets on non-dev shapes), so calling it as-is would not
satisfy "same overlap and Gaussian weighting as the reference nnUNet
predictor". This operator therefore runs the same MONAI-style sliding-window
loop (extract patch -> TTA predictor -> gaussian-weighted accumulate ->
divide) using nnUNet's own pure step/Gaussian utilities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from monai.deploy.core import Operator, OperatorSpec

try:  # package-style import (my_app.*)
    from my_app.config import InferenceParams, detect_available_folds, load_inference_params, resolve_checkpoint_name
    from my_app.operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        assert_on_gpu,
        get_study_id,
        nvtx_range,
    )
    from my_app.operators.preprocess_operator import to_holoscan_gpu_tensor
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from config import InferenceParams, detect_available_folds, load_inference_params, resolve_checkpoint_name
    from gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        assert_on_gpu,
        get_study_id,
        nvtx_range,
    )
    from preprocess_operator import to_holoscan_gpu_tensor

__all__ = [
    "SlideWindowOperator",
    "ModelBundle",
    "build_mirror_axis_combinations",
    "build_network_from_params",
    "detect_available_folds",
    "load_model_bundle",
    "mirror_and_predict",
    "predict_logits",
    "resolve_checkpoint_name",
    "sliding_window_predict",
]

# Reference predictor hyperparameters: get_nnunet_monai_predictor(
#   tile_step_size=0.5, use_gaussian=True, use_mirroring=True). These are
# reference predictor constants (not bundle fields); everything bundle-specific
# (patch size, mirror axes, checkpoint, folds, architecture) is config-driven
# via my_app.config.load_inference_params.
DEFAULT_TILE_STEP_SIZE = 0.5
# Reference gaussian kernel: compute_gaussian(patch, sigma_scale=1/8,
# value_scaling_factor=10, device=results_device).
GAUSSIAN_SIGMA_SCALE = 1.0 / 8.0
GAUSSIAN_VALUE_SCALING_FACTOR = 10.0


# ---------------------------------------------------------------------------
# Model resolution / loading (setup-time, once)
# ---------------------------------------------------------------------------


@dataclass
class ModelBundle:
    """Everything the inference core needs, loaded once at setup time."""

    config_name: str
    trainer_name: str  # recorded from checkpoint metadata (not hard-coded)
    network: torch.nn.Module  # CUDA-resident, eval()
    fold_state_dicts: List[Dict[str, torch.Tensor]]  # one per fold, CUDA-resident
    mirror_axes: Optional[Tuple[int, ...]]
    patch_size: Tuple[int, ...]
    num_segmentation_heads: int
    use_mirroring: bool
    use_gaussian: bool
    tile_step_size: float
    device: torch.device


def build_network_from_params(params: InferenceParams, device: Union[str, torch.device] = "cuda"):
    """Build the network from the bundle ``plans.json`` architecture entry and
    load every fold's weights from the resolved checkpoint paths (INF-007).

    No hard-coded trainer class: a custom trainer variant loads through its
    checkpoint path, with the architecture coming from its plans.

    Returns:
        ``(network, fold_state_dicts)`` — the network on ``device`` in eval
        mode (initialized with the first fold's weights) and the CUDA-
        resident per-fold state dicts.
    """
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    device = torch.device(device)
    network = get_network_from_plans(
        params.network_class_name,
        dict(params.network_init_kwargs),
        tuple(params.network_init_kwargs_req_import),
        params.num_input_channels,
        params.num_segmentation_heads,
        allow_init=True,
        deep_supervision=False,
    )

    fold_state_dicts: List[Dict[str, torch.Tensor]] = []
    for path in params.fold_paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        fold_state_dicts.append(ckpt["network_weights"] if "network_weights" in ckpt else ckpt)

    # Initialize the network with the first fold's weights (reference behavior).
    network.load_state_dict(fold_state_dicts[0])
    network = network.to(device)
    network.eval()
    return network, fold_state_dicts


def load_model_bundle(
    model_path: Union[str, Path],
    config_name: str = "3d_fullres",
    checkpoint_name: Optional[str] = None,
    use_mirroring: bool = True,
    use_gaussian: bool = True,
    tile_step_size: float = DEFAULT_TILE_STEP_SIZE,
    device: Union[str, torch.device] = "cuda",
) -> ModelBundle:
    """Load architecture + every fold's weights once, fully on ``device``.

    All bundle-specific values (patch size, mirror axes, checkpoint path,
    folds, architecture) come from ``my_app.config.load_inference_params``
    (INF-006); the checkpoint follows the reference auto-order
    ``final_model.pt > best_model.pt > model.pt`` unless given explicitly,
    and the network is built from the checkpoint's plans (no hard-coded
    trainer class, INF-007).
    """
    device = torch.device(device)
    params = load_inference_params(model_path, config_name, checkpoint_name)
    network, fold_state_dicts = build_network_from_params(params, device)
    # Parity with nnUNetPredictor: it enables cudnn benchmark on cuda devices.
    torch.backends.cudnn.benchmark = True

    return ModelBundle(
        config_name=params.config_name,
        trainer_name=params.trainer_name,
        network=network,
        fold_state_dicts=fold_state_dicts,
        mirror_axes=params.mirror_axes,
        patch_size=params.patch_size,
        num_segmentation_heads=params.num_segmentation_heads,
        use_mirroring=bool(use_mirroring),
        use_gaussian=bool(use_gaussian),
        tile_step_size=float(tile_step_size),
        device=device,
    )


# ---------------------------------------------------------------------------
# Inference core (compute-time; model is never (re-)loaded here)
# ---------------------------------------------------------------------------


def build_mirror_axis_combinations(
    mirror_axes: Sequence[int], input_ndim: int
) -> List[Tuple[int, ...]]:
    """TTA mirror permutations in the exact reference order.

    Replica of the combination builder in
    ``nnUNetPredictor._internal_maybe_mirror_and_predict``: axes are shifted
    by +2 (batch/channel dims of the 5D patch tensor) and expanded as
    ``[c for i in range(len(mirror_axes)) for c in combinations(axes, i + 1)]``.
    """
    if not mirror_axes:
        return []
    shifted = [int(m) + 2 for m in mirror_axes]
    # Reference assert: max(mirror_axes) <= x.ndim - 3 (unshifted).
    if max(mirror_axes) > input_ndim - 3:
        raise ValueError("mirror_axes does not match the dimension of the input!")
    return [tuple(c) for i in range(len(shifted)) for c in combinations(shifted, i + 1)]


def mirror_and_predict(
    network: torch.nn.Module,
    x: torch.Tensor,
    mirror_combinations: Sequence[Tuple[int, ...]],
) -> torch.Tensor:
    """One TTA pass over a single patch: normal prediction first, then every
    allowed mirror permutation, accumulated with **sequential FP32 ``+=``**
    in the reference order (INF-003/INF-004).

    ``x``: ``(1, C, *patch)``. Returns ``(1, heads, *patch)`` in FP32. The
    network forward itself runs in FP16 under the caller's autocast; each
    forward output is cast to FP32 *before* accumulating, so the TTA summing
    never happens in FP16 (non-associative).
    """
    prediction = network(x).float()
    for axes in mirror_combinations:
        prediction += torch.flip(network(torch.flip(x, axes)), axes).float()
    prediction = prediction / (len(mirror_combinations) + 1)
    return prediction


def _sliding_window_slicers(
    image_size: Sequence[int], patch_size: Sequence[int], tile_step_size: float
) -> List[Tuple[slice, ...]]:
    """Window slicers in the reference order (sx outer, sy, sz inner).

    Replica of ``nnUNetPredictor._internal_get_sliding_window_slicers`` using
    nnUNet's own ``compute_steps_for_sliding_window`` (see module docstring on
    why MONAI's step generator is not used).
    """
    from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window

    steps = compute_steps_for_sliding_window(tuple(image_size), tuple(patch_size), tile_step_size)
    slicers: List[Tuple[slice, ...]] = []
    for sx, sy, sz in product(steps[0], steps[1], steps[2]):
        slicers.append(
            tuple([slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), patch_size)]])
        )
    return slicers


def sliding_window_predict(bundle: ModelBundle, data: torch.Tensor) -> torch.Tensor:
    """One fold's sliding-window inference with TTA, on GPU, FP32 accumulators.

    ``data``: ``(C, X, Y, Z)`` CUDA tensor (4D, no batch dim — the reference
    ``predict_sliding_window_return_logits`` contract). Returns logits
    ``(heads, X, Y, Z)`` FP32 on CUDA with the same spatial shape.
    """
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    assert data.ndim == 4, "input_image must be a 4D tensor (c, x, y, z)"
    assert data.device.type == bundle.device.type and (
        bundle.device.index is None or data.device.index == bundle.device.index
    ), f"expected data on {bundle.device}, got {data.device}"

    patch_size = tuple(bundle.patch_size)
    padded, slicer_revert_padding = pad_nd_image(data, patch_size, "constant", {"value": 0}, True, None)
    slicers = _sliding_window_slicers(padded.shape[1:], patch_size, bundle.tile_step_size)

    # FP32 accumulators (INF-004) — deliberately not nnUNet 2.8.1's FP16.
    predicted_logits = torch.zeros(
        (bundle.num_segmentation_heads, *padded.shape[1:]), dtype=torch.float32, device=bundle.device
    )
    n_predictions = torch.zeros(padded.shape[1:], dtype=torch.float32, device=bundle.device)

    if bundle.use_gaussian:
        gaussian = compute_gaussian(
            tuple(patch_size),
            sigma_scale=GAUSSIAN_SIGMA_SCALE,
            value_scaling_factor=GAUSSIAN_VALUE_SCALING_FACTOR,
            dtype=torch.float32,
            device=bundle.device,
        )
    else:
        gaussian = None

    mirror_combinations = (
        build_mirror_axis_combinations(bundle.mirror_axes, 5)
        if (bundle.use_mirroring and bundle.mirror_axes)
        else []
    )

    for sl in slicers:
        workon = padded[sl][None].contiguous()
        prediction = mirror_and_predict(bundle.network, workon, mirror_combinations)[0]
        if gaussian is not None:
            prediction = prediction * gaussian
        predicted_logits[sl] += prediction
        n_predictions[sl[1:]] += gaussian if gaussian is not None else 1

    torch.div(predicted_logits, n_predictions, out=predicted_logits)
    # Reference parity: the inf check raises — it is never swallowed.
    if torch.any(torch.isinf(predicted_logits)):
        raise RuntimeError(
            "Encountered inf in predicted array. Aborting... If this problem persists, "
            "reduce value_scaling_factor in compute_gaussian or increase the dtype of "
            "predicted_logits to fp32"
        )
    return predicted_logits[(slice(None), *slicer_revert_padding[1:])]


def predict_logits(bundle: ModelBundle, data: torch.Tensor) -> torch.Tensor:
    """Per-study logits: sequential per-fold accumulation (FP32, on GPU) then
    per-fold average — the reference ``predict_logits_from_preprocessed_data``
    order, minus the CPU round-trip (INF-001).

    Autocast scope (INF-011, reference parity): each fold's sliding-window
    inference runs in its own ``torch.autocast("cuda")`` block, and each
    fold's ``load_state_dict`` runs **outside** any active autocast — exactly
    like the reference, which loads ``params`` between separate per-fold
    autocast scopes. Loading weights inside an already-active autocast
    context measurably shifts the following forward (reproduced on torch
    2.13), so that arrangement is intentionally avoided. ``no_grad`` is
    owned by the caller (``compute``).
    """
    from nnunetv2.configuration import default_num_processes

    n_threads = torch.get_num_threads()
    torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
    try:
        prediction = None
        for params in bundle.fold_state_dicts:
            bundle.network.load_state_dict(params)
            with torch.autocast(device_type="cuda", enabled=True):
                fold_logits = sliding_window_predict(bundle, data)
            # Reference order: first fold assigns, later folds sequential +=.
            if prediction is None:
                prediction = fold_logits
            else:
                prediction += fold_logits
        if len(bundle.fold_state_dicts) > 1:
            prediction = prediction / len(bundle.fold_state_dicts)
        return prediction
    finally:
        torch.set_num_threads(n_threads)


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


class SlideWindowOperator(Operator):
    """Inference core: preprocessed GPU tensor -> per-config logits on GPU.

    The model (architecture + all fold weights) is loaded exactly once in
    ``setup()``; ``compute()`` performs inference only — no per-study cold
    start (INF-008).

    Named Inputs:
        preprocessed: zero-copy GPU tensor (``holoscan.core.Tensor``) with the
            preprocessed float32 volume, shape ``(C, X, Y, Z)`` or
            ``(1, C, X, Y, Z)`` in nnUNet post-transpose order.

    Named Outputs:
        logits: zero-copy GPU tensor (``holoscan.core.Tensor``) with the
            fold-averaged logits ``(heads, X, Y, Z)`` (FP32, CUDA).
    """

    INPUT_PREPROCESSED = "preprocessed"
    OUTPUT_LOGITS = "logits"

    def __init__(
        self,
        fragment: Any,
        *args: Any,
        model_path: Optional[Union[str, Any]] = None,
        config_name: str = "3d_fullres",
        checkpoint_name: Optional[str] = None,
        tile_step_size: float = DEFAULT_TILE_STEP_SIZE,
        use_gaussian: bool = True,
        use_mirroring: bool = True,
        device: str = "cuda",
        **kwargs: Any,
    ):
        """Create the operator.

        Args:
            fragment: the owning application (passed to ``Operator``).
            model_path: bundle path (bundle root with ``jsonpkls/`` + config
                dirs, or a per-config dir).
            config_name: plans.json configuration key (default ``3d_fullres``).
            checkpoint_name: explicit checkpoint filename; None = reference
                auto-order ``final_model.pt > best_model.pt > model.pt``.
            tile_step_size: sliding-window overlap ratio (reference: 0.5).
            use_gaussian: gaussian blending (reference: True).
            use_mirroring: TTA mirroring (reference: True; axes come from the
                checkpoint metadata).
            device: CUDA device for model + inference.
        """
        # NOTE: holoscan 4.2's Operator.__init__ invokes self.setup(spec) before
        # this constructor body finishes, and setup() loads the model — so all
        # state touched by setup must exist first.
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.model_path = model_path
        self.config_name = config_name
        self.checkpoint_name = checkpoint_name
        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        self.device = device
        self._bundle: Optional[ModelBundle] = None
        self.model_load_count = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        """Declare the I/O and load the model exactly once (INF-008).

        In holoscan-cu13 4.2 this runs from ``Operator.__init__`` (graph-build
        time); ``_load_model`` is idempotent, so an explicit ``setup`` call in
        tests is harmless.
        """
        spec.input(self.INPUT_PREPROCESSED)
        spec.output(self.OUTPUT_LOGITS)
        self._load_model()

    def _load_model(self) -> ModelBundle:
        if self._bundle is not None:
            return self._bundle
        if not self.model_path:
            raise RuntimeError("SlideWindowOperator requires model_path to load the model bundle.")
        # The model is CUDA-resident by contract — no silent CPU fallback
        # (INF-001/INF-005).
        assert_cuda_available()

        timing = GpuTiming("model_load")
        timing.start()
        self._bundle = load_model_bundle(
            self.model_path,
            self.config_name,
            self.checkpoint_name,
            use_mirroring=self.use_mirroring,
            use_gaussian=self.use_gaussian,
            tile_step_size=self.tile_step_size,
            device=self.device,
        )
        self.model_load_count += 1
        record = timing.stop()
        record.update(
            {
                "config": self._bundle.config_name,
                "trainer": self._bundle.trainer_name,
                "folds": len(self._bundle.fold_state_dicts),
                "patch_size": list(self._bundle.patch_size),
                "mirror_axes": list(self._bundle.mirror_axes or []),
            }
        )
        self._logger.info(
            "model loaded ONCE in setup (load #%d): %s", self.model_load_count, json.dumps(record)
        )
        return self._bundle

    @staticmethod
    def _to_preprocessed_4d(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize the incoming tensor to the reference ``(C, X, Y, Z)``."""
        if tensor.ndim == 5:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"SlideWindowOperator supports batch size 1, got batch {tensor.shape[0]}."
                )
            return tensor[0]
        if tensor.ndim == 4:
            return tensor
        raise ValueError(
            f"expected a (C, X, Y, Z) or (1, C, X, Y, Z) preprocessed tensor, got ndim={tensor.ndim}."
        )

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        """Inference only — the model was already loaded in setup()."""
        if self._bundle is None:
            raise RuntimeError(
                "SlideWindowOperator: model not loaded. setup() must run before compute(); "
                "per-study model loading is not allowed (INF-008)."
            )
        bundle = self._bundle

        with nvtx_range("inference"):
            timing = GpuTiming("inference")
            timing.start()

            # Entry guard: the pipeline is GPU-resident by contract (INF-005).
            assert_cuda_available()

            holo_tensor = op_input.receive(self.INPUT_PREPROCESSED)
            if holo_tensor is None:
                raise ValueError("SlideWindowOperator received no 'preprocessed' input.")

            tensor = torch.utils.dlpack.from_dlpack(holo_tensor)
            # Device invariant at the boundary: a CPU tensor raises here and
            # inference never silently runs on CPU (INF-001/INF-005).
            assert_on_gpu(tensor)
            data = self._to_preprocessed_4d(tensor.float())

            # Eager-mode inference: no_grad here; predict_logits opens one
            # torch.autocast("cuda") block per fold (reference parity — see
            # its docstring), all accumulation in FP32. Autocast is never
            # split across operator boundaries (INF-011). No OOM/CPU-fallback
            # handler anywhere in this path — a RuntimeError/OOM propagates
            # (INF-001/INF-005).
            with torch.no_grad():
                logits = predict_logits(bundle, data)

            # Exit guard: the emitted buffer must be CUDA-resident FP32
            # (INF-001/INF-005); to_holoscan_gpu_tensor asserts again at emit.
            assert_on_gpu(logits)
            op_output.emit(to_holoscan_gpu_tensor(logits), self.OUTPUT_LOGITS)

            record = timing.stop()
            record["study"] = get_study_id(self.fragment)
            StudyTimingCollector.record(self.fragment, record)
            self._logger.info("timing: %s", json.dumps(record))
