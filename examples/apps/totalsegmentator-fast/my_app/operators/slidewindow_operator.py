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
import os
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.deploy.conditions import CountCondition
from monai.deploy.core import Operator, OperatorSpec

try:  # package-style import (my_app.*)
    from my_app.config import (
        MODEL_PARTS,
        InferenceParams,
        detect_available_folds,
        load_inference_params,
        load_preprocess_params,
        resolve_checkpoint_name,
    )
    from my_app.operators.buffer_cache import _ShapeCache
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
    from config import (
        MODEL_PARTS,
        InferenceParams,
        detect_available_folds,
        load_inference_params,
        load_preprocess_params,
        resolve_checkpoint_name,
    )
    from operators.buffer_cache import _ShapeCache
    from operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        assert_on_gpu,
        get_study_id,
        nvtx_range,
    )
    from operators.preprocess_operator import to_holoscan_gpu_tensor

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

_logger = logging.getLogger(__name__)

# 26-01 (CFG-01 enabler): env-gated cascade zero one-hot stub. OFF by default;
# value "1" enables it. The app DAG has no lowres_seg producers, so a bundle
# whose config is 3d_cascade_fullres (25 input channels = 1 image + 24 one-hot)
# cannot be fed a real one-hot; this stub loads its conv0 with zero-padded
# input channels and expands a 1-channel input with 24 zero channels, letting
# the layout cells RUN in the config matrix (26-03/26-04). Input-side only —
# preprocess_operator's lowres_seg NotImplementedError is untouched.
SYNTHETIC_CASCADE_ENV = "HOLOSCAN_SYNTHETIC_CASCADE"


def synthetic_cascade_enabled() -> bool:
    """True iff HOLOSCAN_SYNTHETIC_CASCADE == "1" (strict; anything else is off)."""
    return os.environ.get(SYNTHETIC_CASCADE_ENV, "") == "1"


def _pad_conv0_for_synthetic_cascade(
    fold_state_dict: Dict[str, torch.Tensor],
    num_input_channels: int,
) -> Dict[str, torch.Tensor]:
    """26-01: pad in-1 ConvNd weights to ``num_input_channels`` with zero
    one-hot channels (24 zero channels for a 25-channel cascade config).

    Matching is by VALUE shape, not module name: every tensor entry with
    ndim 5 and ``weight.shape[1] == 1`` (a ConvNd weight with a single
    input channel) is padded — channel 0 keeps the original weights,
    channels 1..N are zero (the zero one-hot stub).

    26-04 fix: ALL matching entries are padded, not just the first. A
    PlainConvUNet checkpoint stores conv0 under MULTIPLE keys (the encoder
    conv0 plus the decoder's mirrored ``decoder.encoder`` conv0, each with
    conv/all_modules sub-entries); padding only the first left the mirrors
    at [O, 1, D, H, W] and ``load_state_dict`` failed with a size mismatch
    against the 25-channel network.

    Strict no-op (returns the SAME dict object) when the env is unset,
    ``num_input_channels <= 1``, or no entry matches. Otherwise returns a
    NEW dict; the input dict is never mutated; non-matching entries keep
    their object identity (biases and later convs untouched).
    """
    if not synthetic_cascade_enabled() or num_input_channels <= 1:
        return fold_state_dict
    # No matching in-1 conv weight -> exact no-op, same object returned.
    if not any(
        isinstance(value, torch.Tensor) and value.ndim == 5 and value.shape[1] == 1
        for value in fold_state_dict.values()
    ):
        return fold_state_dict
    new: Dict[str, torch.Tensor] = {}
    n_padded = 0
    for key, value in fold_state_dict.items():
        if isinstance(value, torch.Tensor) and value.ndim == 5 and value.shape[1] == 1:
            expanded = torch.zeros(
                (value.shape[0], num_input_channels, *value.shape[2:]),
                dtype=value.dtype,
                device=value.device,
            )
            expanded[:, 0] = value[:, 0]
            value = expanded
            n_padded += 1
        new[key] = value
    return new


def _expand_input_for_synthetic_cascade(data: torch.Tensor, num_input_channels: int) -> torch.Tensor:
    """26-01: expand a 1-channel model input to ``num_input_channels`` with
    zero one-hot channels (channel 0 = original image).

    Accepts the 4D ``preprocessed`` contract ``(C, X, Y, Z)`` and the batched
    5D ``(1, C, X, Y, Z)`` form. Strict no-op (same object) when the env is
    unset, the input has more than 1 channel, or ``num_input_channels <= 1``
    (non-cascade configs keep the byte-identical path).
    """
    if not synthetic_cascade_enabled() or num_input_channels <= 1 or data.shape[1 if data.ndim == 5 else 0] != 1:
        return data
    if data.ndim == 5:  # (1, C, ...)
        expanded = torch.zeros(
            (data.shape[0], num_input_channels, *data.shape[2:]),
            dtype=data.dtype,
            device=data.device,
        )
        expanded[:, 0] = data[:, 0]
    else:  # (C, X, Y, Z), C == 1
        expanded = torch.zeros(
            (num_input_channels, *data.shape[1:]),
            dtype=data.dtype,
            device=data.device,
        )
        expanded[0] = data[0]
    return expanded


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
    num_input_channels: int  # 26-01: bundle plans' input channel count (1 non-cascade / 25 cascade)
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

    # 26-01 (CFG-01 enabler): env-gated cascade zero one-hot stub — pad every
    # fold's conv0 in-1 weight to the plans' input channel count BEFORE the
    # first fold's load_state_dict initializes the network. Strict no-op when
    # HOLOSCAN_SYNTHETIC_CASCADE is unset (byte-identical to pre-26-01).
    if synthetic_cascade_enabled() and params.num_input_channels > 1:
        _padded: List[Dict[str, torch.Tensor]] = []
        _pad_applied = False
        for _sd in fold_state_dicts:
            _new_sd = _pad_conv0_for_synthetic_cascade(_sd, params.num_input_channels)
            if _new_sd is not _sd:
                _pad_applied = True
            _padded.append(_new_sd)
        fold_state_dicts = _padded
        if _pad_applied:
            _logger.info(
                f"SYNTHETIC_CASCADE: conv0 input channels 1 -> {params.num_input_channels} "
                f"(zero one-hot stub, env {SYNTHETIC_CASCADE_ENV}=1)"
            )

    # Initialize the network with the first fold's weights (reference behavior).
    network.load_state_dict(fold_state_dicts[0])
    network = network.to(device)
    network.eval()
    return network, fold_state_dicts


def load_model_bundle(
    model_path: Union[str, Path],
    config_name: str = "3d_fullres",
    checkpoint_name: Optional[str] = None,
    use_mirroring: Optional[bool] = None,
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

    ``use_mirroring`` (D-04, 26-01): ``None`` = plan-driven via
    ``configurations.<cfg>.use_mirroring`` / top-level ``use_mirroring``
    (default False — see ``my_app.config.resolve_use_mirroring``); an
    explicit True/False wins over the plan value (tests/dev keep full
    control). Shipped plans set neither key, so ``None`` keeps today's
    exact behavior (off) on every registry bundle.
    """
    device = torch.device(device)
    params = load_inference_params(model_path, config_name, checkpoint_name)
    if use_mirroring is None:
        use_mirroring = params.use_mirroring  # plan-driven (D-04, 26-01)
    network, fold_state_dicts = build_network_from_params(params, device)
    # Parity with nnUNetPredictor: it enables cudnn benchmark on cuda devices.
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and torch.cuda.memory.get_allocator_backend() == "pluggable":
        # RMM is active (INFR-01/D-14). torch 2.13's cudnn benchmark search
        # calls the pluggable allocator's unsupported cacheInfo() and raises
        # RuntimeError: "CUDAPluggableAllocator does not yet support
        # cacheInfo" on the first conv (reproduced 2026-08-19, independent of
        # tensor size). RMM wins over benchmark-mode parity with the
        # reference; cudnn falls back to its default algorithm selection.
        torch.backends.cudnn.benchmark = False

    return ModelBundle(
        config_name=params.config_name,
        trainer_name=params.trainer_name,
        network=network,
        fold_state_dicts=fold_state_dicts,
        mirror_axes=params.mirror_axes,
        patch_size=params.patch_size,
        num_segmentation_heads=params.num_segmentation_heads,
        num_input_channels=int(params.num_input_channels),
        use_mirroring=bool(use_mirroring),
        use_gaussian=bool(use_gaussian),
        tile_step_size=float(tile_step_size),
        device=device,
    )


# ---------------------------------------------------------------------------
# Inference core (compute-time; model is never (re-)loaded here)
# ---------------------------------------------------------------------------


def build_mirror_axis_combinations(mirror_axes: Sequence[int], input_ndim: int) -> List[Tuple[int, ...]]:
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
        slicers.append((slice(None), *(slice(si, si + ti) for si, ti in zip((sx, sy, sz), patch_size))))
    return slicers


def sliding_window_predict(
    bundle: ModelBundle,
    data: torch.Tensor,
    buf_cache: Optional[_ShapeCache] = None,
    gaussian: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One fold's sliding-window inference with TTA, on GPU, FP32 accumulators.

    ``data``: ``(C, X, Y, Z)`` CUDA tensor (4D, no batch dim — the reference
    ``predict_sliding_window_return_logits`` contract). Returns logits
    ``(heads, X, Y, Z)`` FP32 on CUDA with the same spatial shape.

    INFR-02/D-24 (both optional, default None = original fresh-allocation
    behavior, kept for standalone use/tests):
      ``buf_cache`` — shape-keyed torch-family ``_ShapeCache``; the big
      fixed-shape accumulators and the per-patch ``workon`` are borrowed
      from it instead of allocated fresh;
      ``gaussian``  — a precomputed blending kernel (identical every fold —
      see ``SlideWindowOperator.setup``); when None it is computed here as
      before.
    """
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    assert data.ndim == 4, "input_image must be a 4D tensor (c, x, y, z)"
    assert data.device.type == bundle.device.type and (
        bundle.device.index is None or data.device.index == bundle.device.index
    ), f"expected data on {bundle.device}, got {data.device}"

    patch_size = tuple(bundle.patch_size)
    # INFR-02 EXPLICIT NON-DECISION: ``padded`` (the F.pad result, ~64 MB)
    # is NOT cached — the extra copy_ into a cache buffer would cost about
    # as much as the allocation it saves at 16 MB/patch scale under RMM
    # (pool expansions only, never per-tile); the hot per-patch allocation
    # (``workon`` below) IS cached.
    padded, slicer_revert_padding = pad_nd_image(data, patch_size, "constant", {"value": 0}, True, None)
    slicers = _sliding_window_slicers(padded.shape[1:], patch_size, bundle.tile_step_size)

    if buf_cache is None:
        # FP32 accumulators (INF-004) — deliberately not nnUNet 2.8.1's FP16.
        predicted_logits = torch.zeros(
            (bundle.num_segmentation_heads, *padded.shape[1:]), dtype=torch.float32, device=bundle.device
        )
        n_predictions = torch.zeros(padded.shape[1:], dtype=torch.float32, device=bundle.device)
    else:
        # INFR-02: borrow with zero=True — the reference allocates FRESH
        # torch.zeros at both sites, so the borrow must re-zero (the cached
        # buffer holds the previous study's logits/counts).
        predicted_logits = buf_cache.get((bundle.num_segmentation_heads, *padded.shape[1:]), torch.float32, zero=True)
        n_predictions = buf_cache.get(padded.shape[1:], torch.float32, zero=True)

    if gaussian is None:
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
        build_mirror_axis_combinations(bundle.mirror_axes, 5) if (bundle.use_mirroring and bundle.mirror_axes) else []
    )

    for sl in slicers:
        if buf_cache is None:
            workon = padded[sl][None].contiguous()
        else:
            # INFR-02: the hottest allocation (16 MB x ~150 patches x 5
            # folds) — one cache entry borrowed per patch. copy_ fully
            # overwrites before the forward reads it (zero=False; a fresh
            # .contiguous() copy has no zeroed semantics to preserve).
            workon = buf_cache.get(padded[sl][None].shape, torch.float32)
            workon.copy_(padded[sl][None])
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


def predict_logits(
    bundle: ModelBundle,
    data: torch.Tensor,
    buf_cache: Optional[_ShapeCache] = None,
    gaussian: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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
                fold_logits = sliding_window_predict(bundle, data, buf_cache=buf_cache, gaussian=gaussian)
            # Reference order: first fold assigns, later folds sequential +=.
            if prediction is None:
                # INFR-02 aliasing rule: with the shape cache, fold_logits is a
                # VIEW of the per-fold cached predicted_logits buffer, which the
                # NEXT fold re-borrows with zero=True. Accumulating in place on
                # that view would be wiped by the next fold's zero_() (and fold
                # k would add the buffer to itself) — so the running sum must
                # live on a FRESH tensor the cache never touches.
                prediction = fold_logits.clone() if buf_cache is not None else fold_logits
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
        prev_part_seg (gated parts ONLY): the previous part's ``seg_argmax``
            output, consumed in compute() purely as a synchronization gate
            (values unused) so a part starts only after the previous part's
            postresample completed (P9 OOM fix — serialized 5-part chain).

    Named Outputs:
        logits: zero-copy GPU tensor (``holoscan.core.Tensor``) with the
            fold-averaged logits ``(heads, X, Y, Z)`` (FP32, CUDA).
    """

    INPUT_PREPROCESSED = "preprocessed"
    INPUT_PREV_SEG = "prev_part_seg"
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
        use_mirroring: Optional[bool] = None,
        device: str = "cuda",
        part: Optional[str] = None,
        normalize: bool = False,
        gated: bool = False,
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
            use_mirroring: TTA mirroring. D-04 (26-01): ``None`` (default)
                = plan-driven via ``configurations.<cfg>.use_mirroring`` /
                top-level ``use_mirroring`` in plans.json (default False);
                an explicit True/False wins over the plan value. Pre-26-01
                the app always passed False (oracle uses no TTA/mirroring).
                The mirror AXES themselves still come from the checkpoint
                metadata.
            device: CUDA device for model + inference.
            part: P9 (09-02) — model part name for the 5-part chain. When
                set, the NVTX range + timing label become
                ``inference_{config_name}_{part}`` so the 5 parts are
                distinct in the log/trace; ``None`` (single-part P8 path)
                keeps the byte-identical ``inference_{config_name}``.
            normalize: P9 (09-04) — apply this part's OWN CT normalization
                (clip/mean/std from its own bundle's plans.json) to the
                input in compute() before sliding-window inference. The
                5-part chain sets normalize=True on every swin because the
                shared PreprocessOperator then emits the UNNORMALIZED
                resampled+cropped volume (oracle parity: the oracle's
                per-part run_case_npy normalizes with each part's own CT
                properties — a shared single-property normalization upstream
                is wrong for 4 of 5 parts). The single-part P8 path keeps
                normalize=False (its PreprocessOperator already normalizes).
            gated: P9 OOM fix — DAG serialization gate for the 5-part chain.
                When True, the operator declares an extra input
                ``prev_part_seg`` (the previous part's postresample
                ``seg_argmax``), installs a 2-input CountCondition (the
                MergeRemapOperator constructor-arg pattern), and consumes the
                gate in compute() (values unused). Each part then starts
                only after the previous part's postresample — including its
                release_buffers() — has run, so per-part logit stacks
                (~4.4-4.7 GiB) are REUSED sequentially instead of all 5
                accumulating under the concurrent scheduler (31322 RMM pool
                OOM). Default False = the byte-for-byte previous single-input
                code path (single-part P8 mode unchanged).
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
        self._normalize = normalize
        self._ct_props: Optional[Tuple[float, float, float, float]] = None
        # P9 (09-02): per-part NVTX/timing tag (validated against MODEL_PARTS).
        if part is not None and part not in {p["name"] for p in MODEL_PARTS}:
            raise ValueError(f"Unknown model part {part!r}. Valid: {[p['name'] for p in MODEL_PARTS]}")
        self._part = part
        # P9 OOM fix: serialized-chain gate flag. Validated/assigned BEFORE
        # super().__init__ — Pitfall 7: holoscan 4.2's Operator.__init__
        # invokes self.setup(spec) before this constructor body finishes, and
        # setup() reads _gated to decide whether to declare INPUT_PREV_SEG.
        self._gated = gated
        self._bundle: Optional[ModelBundle] = None
        self._released = False
        self.model_load_count = 0
        self._bufs_cleared = False  # P9: defensive-log flag for release_buffers()
        # INFR-02/D-24: shape-keyed torch-family buffer cache (created
        # BEFORE super().__init__ — Pitfall 7 discipline). Per-site table:
        #   predicted_logits (512 MB)  CACHED zero=True  (reference = fresh torch.zeros).
        #                                   ALIASING RULE: sliding_window_predict
        #                                   returns a VIEW of this per-fold buffer;
        #                                   predict_logits clones the fold-1 result
        #                                   when a cache is active, because the
        #                                   running sum must not alias a buffer the
        #                                   next fold re-borrows with zero=True
        #                                   (otherwise the zero_() wipes the partial
        #                                   sum mid-accumulation).
        #   n_predictions    (64 MB)   CACHED zero=True  (reference = fresh torch.zeros)
        #   gaussian         (1 MB)    computed ONCE in setup() — identical
        #                                   every fold; read-only in the loop
        #                                   (prediction * gaussian, n_predictions +=
        #                                   gaussian — never written)
        #   per-patch workon (16 MB x ~150 patches x 5 folds) CACHED zero=False
        #                                   (copy_ overwrites; the hottest site)
        #   padded (F.pad result)      NOT CACHED (explicit non-decision in
        #                                   sliding_window_predict — copy_ cost
        #                                   ~= the RMM-pooled alloc it saves)
        # Emit boundary: logits cross to PostResample via DLPack — if a
        # cached buffer (or a view of one) ends up as the emitted tensor,
        # compute() emits a copy (Phase 1 DLPack ownership lesson).
        self._buf_cache = _ShapeCache(self.device, family="torch")
        # INFR-02: setup()-time gaussian (None = not computed, e.g.
        # use_gaussian=False or model not yet loaded).
        self._gaussian: Optional[torch.Tensor] = None
        if gated:
            # P9 OOM fix: 2-input CountCondition (preprocessed +
            # prev_part_seg) — compute() must NOT fire on the preprocessed
            # fan-out alone. Constructor-arg form, exactly like
            # MergeRemapOperator (this holoscan build has no
            # OperatorSpec.add_condition).
            super().__init__(fragment, CountCondition(fragment, 2), *args, **kwargs)
        else:
            super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        """Declare the I/O and load the model exactly once (INF-008).

        In holoscan-cu13 4.2 this runs from ``Operator.__init__`` (graph-build
        time); ``_load_model`` is idempotent, so an explicit ``setup`` call in
        tests is harmless.
        """
        spec.input(self.INPUT_PREPROCESSED)
        if self._gated:
            # P9 OOM fix: serialized-chain gate input. Declared ONLY when
            # gated — the non-gated (single-part P8) spec is byte-for-byte
            # the previous declaration set.
            spec.input(self.INPUT_PREV_SEG)
        spec.output(self.OUTPUT_LOGITS)
        self._load_model()
        if self._normalize:
            self._load_ct_props()
        # INFR-02/D-24: the gaussian blending kernel depends only on the
        # patch size + sigma/value-scaling constants — identical for every
        # fold of every study — so compute it ONCE here instead of once per
        # fold per study (RESEARCH §D-24 inventory: "gaussian 1 MB —
        # identical every fold: computable once in setup").
        bundle = self._bundle
        if bundle is not None and bundle.use_gaussian:
            from nnunetv2.inference.sliding_window_prediction import compute_gaussian

            self._gaussian = compute_gaussian(
                tuple(bundle.patch_size),
                sigma_scale=GAUSSIAN_SIGMA_SCALE,
                value_scaling_factor=GAUSSIAN_VALUE_SCALING_FACTOR,
                dtype=torch.float32,
                device=bundle.device,
            )

    def _load_ct_props(self) -> None:
        """P9 (09-04): load this part's OWN CT normalization constants.

        CTNormalization needs no data-dependent reduction (clip bounds,
        mean and std are dataset constants from plans.json), so only the
        single image channel's properties are loaded. Loaded in setup so
        compute() stays a pure inference path (INF-008 pattern).
        """
        params = load_preprocess_params(self.model_path, self.config_name)
        scheme = params.normalization_schemes[0]
        if scheme != "CTNormalization":
            raise ValueError(
                f"SlideWindowOperator(normalize=True) requires CTNormalization, "
                f"got {scheme!r} for channel 0 of {self.config_name}."
            )
        props = params.intensity_properties["0"]
        self._ct_props = (
            float(props["percentile_00_5"]),
            float(props["percentile_99_5"]),
            float(props["mean"]),
            float(props["std"]),
        )
        self._logger.info(
            "per-part CT normalization enabled: clip=[%r, %r] mean=%r std=%r "
            "(from own bundle plans.json — oracle parity)",
            *self._ct_props,
        )

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
        self._logger.info("model loaded ONCE in setup (load #%d): %s", self.model_load_count, json.dumps(record))
        if self._bundle.use_mirroring:
            # D-04 (26-01): TTA-on marker line, grepped by the 26-04 A/B gate driver.
            _axes = tuple(self._bundle.mirror_axes or ())
            self._logger.info(
                f"TTA on: config={self._bundle.config_name} mirror_axes={_axes} "
                f"combinations={len(build_mirror_axis_combinations(_axes, 5))}"
            )
        return self._bundle

    def release(self) -> None:
        """MEM-003/D-23: free this config's weights after the auxiliary
        fragment's terminal emit (wired by ``NnUnetConfigSubgraph.compose``
        through the PostResampleOperator ``release_fn`` callback — exactly
        once, exactly for the lowres_seg-emitting configuration).

        Safe no-op if already released or never loaded. Under the RMM
        pluggable allocator ``torch.cuda.empty_cache()`` may be a
        driver-level no-op (Open Q2 — measured in Phase 3 Plan 02 Task 2),
        but the pool-level handback of the weights is deterministic either
        way.
        """
        if self._bundle is None:
            return
        bundle = self._bundle
        n_folds = len(bundle.fold_state_dicts)
        self._bundle = None
        self._released = True
        del bundle.network, bundle.fold_state_dicts
        if self._gaussian is not None:
            del self._gaussian
            self._gaussian = None
        torch.cuda.empty_cache()  # RMM: may be a driver-level no-op (Open Q2)
        self._logger.info("weights released: %s (folds=%d) (MEM-003)", self.config_name, n_folds)

    def release_buffers(self) -> None:
        """P9 (09-02 / PIP-03 memory model): drop THIS part's cached
        logits/workon buffers after its argmax has consumed the logits —
        wired by app.py as the PostResampleOperator ``release_fn`` per part,
        so the 5 logit stacks of the 5-part chain never coexist.

        CACHES ONLY, by design:
        * ``self._bundle`` (weights) and ``self._released`` are untouched —
          all 5 parts stay weight-resident for the process lifetime
          (PIP-03 "load once"; the v1.1 weight-release pattern is
          deliberately NOT used in 5-part mode).
        * the setup()-time gaussian is NOT in the cache and is identical
          for every study — it is kept (recomputing would be pure waste).

        Safe no-op if the cache was never populated. A later compute()
        simply re-allocates (defensive log, once).
        """
        n = len(self._buf_cache.keys())
        self._buf_cache.clear()
        self._bufs_cleared = True
        self._logger.info(
            "buffers released: %s (cache entries=%d; weights kept resident)",
            self._part or self.config_name,
            n,
        )

    @staticmethod
    def _to_preprocessed_4d(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize the incoming tensor to the reference ``(C, X, Y, Z)``."""
        if tensor.ndim == 5:
            if tensor.shape[0] != 1:
                raise ValueError(f"SlideWindowOperator supports batch size 1, got batch {tensor.shape[0]}.")
            return tensor[0]
        if tensor.ndim == 4:
            return tensor
        raise ValueError(f"expected a (C, X, Y, Z) or (1, C, X, Y, Z) preprocessed tensor, got ndim={tensor.ndim}.")

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        """Inference only — the model was already loaded in setup()."""
        if self._bundle is None:
            if self._released:
                # Defensive: the DAG never schedules this (release fires
                # after the aux fragment's terminal emit). A hit means a DAG
                # ordering violation (MEM-003).
                raise RuntimeError(
                    f"compute() after release() — DAG ordering violation for " f"{self.config_name} (MEM-003)."
                )
            raise RuntimeError(
                "SlideWindowOperator: model not loaded. setup() must run before compute(); "
                "per-study model loading is not allowed (INF-008)."
            )
        bundle = self._bundle

        if self._bufs_cleared:
            self._logger.info(
                "compute() after release_buffers() — re-allocating cached buffers for %s",
                self._part or self.config_name,
            )
            self._bufs_cleared = False

        # INFR-005: the NVTX range + timing label carry the config so
        # multi-fragment traces/records stay unambiguous (RESEARCH Pitfall 9).
        # P9 (09-02): with ``part`` set, the label is per-part
        # (inference_{config_name}_{part}); part=None keeps the P8 name
        # byte-identical.
        _label = f"inference_{self.config_name}_{self._part}" if self._part else f"inference_{self.config_name}"
        with nvtx_range(_label):
            timing = GpuTiming(_label)
            timing.start()

            # Entry guard: the pipeline is GPU-resident by contract (INF-005).
            assert_cuda_available()

            # P9 OOM fix: gated parts consume the previous part's
            # seg_argmax purely as a synchronization gate (values unused).
            # The CountCondition(2) above guarantees this arrives before
            # compute() fires, i.e. the previous part's postresample —
            # including release_buffers() — has completed, so per-part
            # logit stacks are reused instead of accumulating.
            if self._gated:
                op_input.receive(self.INPUT_PREV_SEG)

            holo_tensor = op_input.receive(self.INPUT_PREPROCESSED)
            if holo_tensor is None:
                raise ValueError("SlideWindowOperator received no 'preprocessed' input.")

            tensor = torch.utils.dlpack.from_dlpack(holo_tensor)
            # Device invariant at the boundary: a CPU tensor raises here and
            # inference never silently runs on CPU (INF-001/INF-005).
            assert_on_gpu(tensor)
            data = self._to_preprocessed_4d(tensor.float())

            # 26-01 (CFG-01 enabler): env-gated cascade zero one-hot stub —
            # expand a 1-channel input to the bundle's input channel count
            # before anything else. Strict no-op when the env is unset or the
            # config is non-cascade (num_input_channels == 1).
            data = _expand_input_for_synthetic_cascade(data, bundle.num_input_channels)

            # P9 (09-04): per-part CT normalization (oracle parity — each
            # part's run_case_npy applies its OWN plans' CT properties; the
            # 5-part chain's shared PreprocessOperator emits unnormalized).
            # OUT-OF-PLACE fp32 element-wise ops: the shared fanout buffer
            # is never mutated. clip/sub/div with np.float32 constants is
            # bit-identical to the PreprocessOperator's CuPy path (all
            # IEEE-754 fp32 element-wise: min/max compare, subtract, divide).
            if self._ct_props is not None:
                _p005, _p995, _mean, _std = self._ct_props
                data = torch.clamp(data, _p005, _p995)
                data = data - torch.as_tensor(np.float32(_mean), device=data.device)
                data = data / torch.as_tensor(np.float32(max(_std, 1e-8)), device=data.device)

            # Eager-mode inference: no_grad here; predict_logits opens one
            # torch.autocast("cuda") block per fold (reference parity — see
            # its docstring), all accumulation in FP32. Autocast is never
            # split across operator boundaries (INF-011). No OOM/CPU-fallback
            # handler anywhere in this path — a RuntimeError/OOM propagates
            # (INF-001/INF-005).
            with torch.no_grad():
                logits = predict_logits(bundle, data, buf_cache=self._buf_cache, gaussian=self._gaussian)

            # INFR-02 emit-boundary rule (Phase 1 DLPack ownership lesson):
            # a cached buffer must never cross the operator boundary via
            # DLPack. Multi-fold bundles emit the fresh fold-average tensor;
            # a single-fold bundle would emit a VIEW of the cached
            # predicted_logits — in that case emit a copy. shares_storage
            # compares storage-base pointers, so offset views are caught.
            if self._buf_cache.shares_storage(logits):
                logits = logits.clone()

            # Exit guard: the emitted buffer must be CUDA-resident FP32
            # (INF-001/INF-005); to_holoscan_gpu_tensor asserts again at emit.
            assert_on_gpu(logits)
            op_output.emit(to_holoscan_gpu_tensor(logits), self.OUTPUT_LOGITS)

            record = timing.stop()
            record["study"] = get_study_id(self.fragment)
            record["config"] = self.config_name
            StudyTimingCollector.record(self.fragment, record)
            self._logger.info("timing: %s", json.dumps(record))
