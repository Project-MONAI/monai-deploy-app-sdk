# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PostResampleOperator: revert per-config logits to the original DICOM volume
as a probability volume, GPU-resident (POST-02).

Reference path replicated (``nnunetv2.inference.export_prediction`` /
``resample_and_save`` with ``save_probabilities=True``):

    resampling_fn_probabilities            (scipy/scikit-image, CPU)
    -> softmax over the class axis         (``torch.softmax(x, 0)``)
    -> revert cropping on probabilities    (background channel = 1 outside)
    -> revert transpose to original order

Reference order matters: the *logits* are resampled first, then softmax is
applied — softmax of resampled logits is NOT equal to resampled softmax of
logits, so the softmax runs AFTER the resample, exactly like the reference.

The resample itself runs on the reference CPU path (scipy
``map_coordinates`` / skimage ``resize``), per the Phase 0/1 decision that
resampling stays on the reference CPU path for pixel-exactness (GPU
resampling is deferred to v2). That single GPU->CPU->GPU round-trip is
therefore deliberate and scoped to this operator; the downstream
``PostprocessOperator`` boundary (POST-03) still performs exactly one
GPU->CPU transfer for the final seg.

All per-config values (target shape, spacings, crop bbox, transpose) come
from the ``preprocessed_meta`` dict emitted by ``PreprocessOperator``
(bit-exact replica of the reference ``run_case_npy`` properties).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cupy as cp  # D-22: gated GPU resample path (flag OFF = unused numerically)
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from skimage.transform import resize

from monai.deploy.core import Operator, OperatorSpec

try:  # package-style import (my_app.*)
    from my_app.config import MODEL_PARTS
    from my_app.operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        assert_on_gpu,
        get_study_id,
        nvtx_range,
    )
    from my_app.operators.gpu_zoom import gpu_resample_enabled, stock_gpu_resize
    from my_app.operators.part_postprocess import apply_part_postprocess, find_postprocessing_pkl
    from my_app.operators.preprocess_operator import _determine_do_sep_z_and_axis, to_holoscan_gpu_tensor
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from config import MODEL_PARTS
    from operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        assert_on_gpu,
        get_study_id,
        nvtx_range,
    )
    from operators.gpu_zoom import gpu_resample_enabled, stock_gpu_resize
    from operators.part_postprocess import apply_part_postprocess, find_postprocessing_pkl
    from operators.preprocess_operator import _determine_do_sep_z_and_axis, to_holoscan_gpu_tensor

__all__ = [
    "PostResampleOperator",
    "postresample_reference",
    "revert_crop_and_transpose_gpu",
    "revert_crop_gpu",
    "resample_probabilities_to_shape",
]

# Reference ``resampling_fn_probabilities`` kwargs from plans.json
# (resample_data_or_seg_to_shape, is_seg=False):
PROBABILITY_RESAMPLE_ORDER = 1
PROBABILITY_RESAMPLE_ORDER_Z = 0
PROBABILITY_RESAMPLE_FORCE_SEPARATE_Z = None


def _inverse_permutation(perm: Sequence[int]) -> List[int]:
    """Inverse of a permutation: ``inv[perm[i]] = i``."""
    perm = [int(i) for i in perm]
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def resample_probabilities_to_shape(
    data: np.ndarray,
    new_shape: Union[Sequence[int], np.ndarray],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    order: int = PROBABILITY_RESAMPLE_ORDER,
    order_z: int = PROBABILITY_RESAMPLE_ORDER_Z,
    force_separate_z: Optional[bool] = PROBABILITY_RESAMPLE_FORCE_SEPARATE_Z,
) -> np.ndarray:
    """Resample ``(C, X, Y, Z)`` logits/probabilities to a target shape.

    Bit-exact replica of the non-segmentation path (``is_seg=False``) of
    ``nnunetv2...default_resampling.resample_data_or_seg_to_shape`` /
    ``resample_data_or_seg``: ``skimage.transform.resize`` (mode='edge',
    anti_aliasing=False, given ``order``) plus the separate-z pass via
    ``scipy.ndimage.map_coordinates`` (align_corners=False, mode='nearest').
    """
    if data.ndim != 4:
        raise ValueError(f"data must be (c, x, y, z), got ndim={data.ndim}")

    do_separate_z, axis = _determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing)

    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)

    if np.any(shape != new_shape):
        if gpu_resample_enabled() and not do_separate_z:
            # D-22 (D-22a amended) gated GPU resample (HOLOSCAN_GPU_RESAMPLE=1):
            # per channel, the STOCK cupyx.scipy.ndimage mirror of the OFF-path
            # skimage chain (fp64 widening -> grid_mode zoom -> fp64 clip ->
            # fp32 cast; the custom RawKernel provenance in gpu_zoom.py is NOT
            # wired — D-22a). The result returns to CPU numpy because the
            # reference torch CPU softmax (thread-scoped, bit-exactness
            # decision) runs downstream — the resample span itself computes on
            # GPU. HOLOSCAN_GPU_RESAMPLE=0 is the verbatim Phase 2/3 path.
            data_gpu = cp.asarray(data, dtype=cp.float32)
            return np.ascontiguousarray(stock_gpu_resize(data_gpu, tuple(int(s) for s in new_shape), order).get())
        dtype_out = data.dtype
        data = data.astype(float, copy=False)
        reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
        resize_kwargs = {"mode": "edge", "anti_aliasing": False}

        if do_separate_z:
            # D-22: the separate-z map_coordinates branches stay scipy in
            # BOTH flag states (inactive in this bundle; near-isotropic
            # spacings) — the flag only gates the non-sep-z branch below.
            assert axis is not None, "if do_separate_z, we need to know what axis is anisotropic"
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            for c in range(data.shape[0]):
                tmp = new_shape.copy()
                tmp[axis] = shape[axis]
                reshaped_here = np.zeros(tmp)
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_here[slice_id] = resize(data[c, slice_id], new_shape_2d, order, **resize_kwargs)
                    elif axis == 1:
                        reshaped_here[:, slice_id] = resize(data[c, :, slice_id], new_shape_2d, order, **resize_kwargs)
                    else:
                        reshaped_here[:, :, slice_id] = resize(
                            data[c, :, :, slice_id], new_shape_2d, order, **resize_kwargs
                        )
                if shape[axis] != new_shape[axis]:
                    # align_corners=False coordinate map (reference replica)
                    rows, cols, dim = int(new_shape[0]), int(new_shape[1]), int(new_shape[2])
                    orig_rows, orig_cols, orig_dim = reshaped_here.shape
                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim
                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5
                    coord_map = np.array([map_rows, map_cols, map_dims])
                    reshaped_final[c] = map_coordinates(reshaped_here, coord_map, order=order_z, mode="nearest")
                else:
                    reshaped_final[c] = reshaped_here
        else:
            for c in range(data.shape[0]):
                reshaped_final[c] = resize(data[c], new_shape, order, **resize_kwargs)
        return reshaped_final
    else:
        # No resampling necessary — the reference returns the input unchanged.
        return data


def postresample_reference(logits: Union[np.ndarray, torch.Tensor], meta: Dict[str, Any]) -> np.ndarray:
    """Reference post-inference path on CPU: resample -> softmax -> probabilities.

    Replicates ``export_prediction.convert_predicted_logits_to_segmentation_with_correct_shape``
    up to (and including) ``apply_inference_nonlin``, i.e.:

    * ``resampling_fn_probabilities(logits, shape_after_cropping_and_before_resampling,
      current_spacing=plans spacing, target_spacing=[props['spacing'][i] for i in tf])``
      with the plans kwargs (is_seg=False, order=1, order_z=0, force_separate_z=None);
    * ``softmax_helper_dim0`` = ``torch.softmax(x, 0)`` on float32 — the
      reference runs this on CPU torch, so we keep it on CPU here for
      bit-exact parity.

    Thread-scope parity: the reference wraps this whole span in
    ``torch.set_num_threads(default_num_processes)``. That is not a no-op for
    bit-exactness — torch's CPU softmax is not bit-reproducible across thread
    counts (measured: 2-ulp flips at ~70-80 voxels on a 2x25x16x35 volume),
    so we replicate the reference's thread scope exactly.

    Args:
        logits: ``(C, X, Y, Z)`` inference logits in nnUNet post-transpose
            orientation (the SlideWindowOperator output), float32.
        meta: the ``preprocessed_meta`` dict from PreprocessOperator.

    Returns:
        ``(C, X, Y, Z)`` float32 numpy probabilities in the cropped
        (pre-resample) shape, still in nnUNet post-transpose orientation.
    """
    from nnunetv2.configuration import default_num_processes

    tf = [int(i) for i in meta["transpose_forward"]]
    # Reference spacing bookkeeping (post-transpose axis order throughout):
    current_spacing = [float(s) for s in meta["target_spacing"]]  # plans spacing
    target_spacing = [float(meta["original_spacing"][i]) for i in tf]  # [props['spacing'][i] for i in tf]
    new_shape = tuple(int(s) for s in meta["shape_after_cropping_and_before_resampling"])

    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    logits = np.ascontiguousarray(np.asarray(logits, dtype=np.float32))

    old_threads = torch.get_num_threads()
    torch.set_num_threads(default_num_processes)
    try:
        resampled = resample_probabilities_to_shape(logits, new_shape, current_spacing, target_spacing)
        # Reference: apply_inference_nonlin -> torch.from_numpy(...).float() -> softmax dim 0.
        probabilities = torch.nn.functional.softmax(torch.from_numpy(resampled).float(), dim=0)
        return probabilities.numpy()
    finally:
        torch.set_num_threads(old_threads)


def seg_argmax_to_original_reference(
    logits: Union[np.ndarray, torch.Tensor], meta: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """CPU reference for the ``resample_seg_to_original`` emit_argmax_dicom
    path (22-11, TS 2.18-faithful for ``resample is None`` cascade tasks).

    Replicates ``export_prediction.convert_predicted_logits_to_segmentation_with_correct_shape``
    for these bundles (non-region training, identity transpose_forward):

    * ``postresample_reference`` — resample the LOGITS to the crop-native
      shape with the plans ``resampling_fn_probabilities`` semantics
      (order 1 / order_z 0 / force_separate_z None), then softmax; the
      argmax of resampled logits equals the argmax of the resampled
      softmax (softmax is monotone per voxel);
    * argmax over the class axis (uint8, crop-native layout order);
    * revert cropping on the INTEGER seg (zeros, insert at
      ``bbox_used_for_cropping``) + transpose-back — the same revert
      ``revert_crop_gpu`` performs.

    For these bundles ``shape_after_cropping_and_before_resampling``
    equals the crop's native shape, so the returned ``seg_dicom`` is the
    crop-NATIVE-resolution segmentation in the original orientation —
    the TS pipeline's output space, which PasteOp's identity branch
    consumes without any label resample.

    Args:
        logits: ``(C, X, Y, Z)`` inference logits at plans spacing in
            nnUNet post-transpose order (SlideWindowOperator output).
        meta: the ``preprocessed_meta`` dict (same keys
            ``postresample_reference`` / ``revert_crop_gpu`` consume).

    Returns:
        ``(seg_model, seg_dicom)`` — both uint8 numpy: ``seg_model`` is
        the plans-spacing argmax (3D, post-transpose order — the same
        value the GPU fast path emits for gating/sink consumers);
        ``seg_dicom`` is the crop-native-resolution argmax in the
        original (pre-transpose) orientation.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    logits_np = np.ascontiguousarray(np.asarray(logits, dtype=np.float32))

    seg_model = np.ascontiguousarray(logits_np.argmax(axis=0)).astype(np.uint8)
    probabilities = postresample_reference(logits_np, meta)
    seg_crop = np.ascontiguousarray(np.asarray(probabilities).argmax(axis=0)).astype(np.uint8)

    original_shape = tuple(int(s) for s in meta["shape_before_cropping"])
    tb = _inverse_permutation(meta["transpose_forward"])
    full = np.zeros(original_shape, dtype=np.uint8)
    slicer = tuple(slice(int(lo), int(hi)) for lo, hi in meta["bbox_used_for_cropping"])
    full[slicer] = seg_crop
    seg_dicom = np.ascontiguousarray(np.transpose(full, tb))
    return seg_model, seg_dicom


def revert_crop_and_transpose_gpu(
    probabilities: Union[np.ndarray, torch.Tensor], meta: Dict[str, Any], device: str = "cuda"
) -> torch.Tensor:
    """Revert crop + transpose on GPU (bit-exact: fill + copy + transpose only).

    Replicates ``LabelManager.revert_cropping_on_probabilities`` (zeros,
    background channel = 1, insert the crop) and the reference transpose-back
    ``probs.transpose([0] + [i + 1 for i in transpose_backward])``.

    Returns:
        ``(C, *shape_before_cropping)`` float32 CUDA tensor in the original
        (pre-transpose) DICOM orientation.
    """
    if isinstance(probabilities, np.ndarray):
        probabilities = torch.as_tensor(probabilities)
    probabilities = probabilities.to(torch.device(device)).float()

    original_shape = tuple(int(s) for s in meta["shape_before_cropping"])
    tf = [int(i) for i in meta["transpose_forward"]]
    tb = _inverse_permutation(tf)

    full = torch.zeros(
        (probabilities.shape[0], *original_shape), dtype=probabilities.dtype, device=probabilities.device
    )
    # Reference: probs_reverted_cropping[0] = 1 (non-region training).
    full[0] = 1
    slicer = tuple(slice(int(lo), int(hi)) for lo, hi in meta["bbox_used_for_cropping"])
    full[(slice(None),) + slicer] = probabilities

    # Reference: .transpose([0] + [i + 1 for i in transpose_backward]) — the
    # reference applies that to a numpy array (permutation form); on torch the
    # equivalent is permute().
    return full.permute(0, *[i + 1 for i in tb])


def revert_crop_gpu(
    seg_crop: Union[np.ndarray, torch.Tensor], meta: Dict[str, Any], device: str = "cuda"
) -> torch.Tensor:
    """Revert crop + transpose on GPU for an INTEGER segmentation
    (bit-exact: zeros fill + insert + permute only).

    Mirrors the reference seg path of
    ``convert_predicted_logits_to_segmentation_with_correct_shape``
    (export_prediction.py): background-0 fill at the pre-crop shape, insert
    the crop at ``bbox_used_for_cropping``, transpose with
    ``transpose_backward``. The result is a 3D uint8 CUDA tensor in the
    original DICOM orientation — the same array order as
    ``image.asnumpy()`` — which is the orientation contract the cascade
    PreprocessOperator consumes (Task 2).

    NO connected-component cleanup: the reference cascade input is pre-CC
    (the reference KeepLargestCC runs only on the final output, never on
    the cascade input — verified in 02-CONTEXT D-09).
    """
    if isinstance(seg_crop, np.ndarray):
        seg_crop = torch.as_tensor(seg_crop)
    seg_crop = seg_crop.to(torch.device(device))
    if seg_crop.dtype != torch.uint8:
        seg_crop = seg_crop.to(torch.uint8)

    original_shape = tuple(int(s) for s in meta["shape_before_cropping"])
    tf = [int(i) for i in meta["transpose_forward"]]
    tb = _inverse_permutation(tf)

    full = torch.zeros(original_shape, dtype=torch.uint8, device=seg_crop.device)
    slicer = tuple(slice(int(lo), int(hi)) for lo, hi in meta["bbox_used_for_cropping"])
    full[slicer] = seg_crop
    # Reference: .transpose(transpose_backward) — for the 3D seg there is
    # no channel axis, so the permutation is ``transpose_backward`` as-is
    # (the probability revert's [0] + [i + 1 for i in ...] is its 4D form).
    return full.permute(*tb).contiguous()


class PostResampleOperator(Operator):
    """Post-inference head: per-config GPU logits -> per-config probability
    volume in original DICOM orientation (POST-02).

    Named Inputs:
        logits: zero-copy GPU tensor (``holoscan.core.Tensor``) with the
            per-config logits ``(C, X, Y, Z)`` (or ``(1, C, X, Y, Z)``) in
            nnUNet post-transpose order, from SlideWindowOperator.
        preprocessed_meta: the metadata dict from PreprocessOperator
            (crop bbox, pre-crop shape, spacings, transpose).

    Named Outputs:
        probabilities: zero-copy GPU tensor (``holoscan.core.Tensor``) with
            the per-config softmax probabilities ``(C, *original_shape)`` in
            original DICOM orientation (FP32, CUDA). Declared only when
            ``emit_probabilities`` (default True).
        lowres_seg: (optional, ``emit_lowres_seg=True`` for cascade producer
            fragments) zero-copy GPU tensor with the post-softmax argmax
            segmentation (uint8 CUDA, 3D, original DICOM orientation — the
            same array order as ``image.asnumpy()``, no connected-component
            cleanup) consumed by the cascade PreprocessOperator's
            ``lowres_seg`` input (D-09/D-10, zero disk I/O).
    """

    INPUT_LOGITS = "logits"
    INPUT_META = "preprocessed_meta"
    OUTPUT_PROBABILITIES = "probabilities"
    OUTPUT_LOWRES_SEG = "lowres_seg"
    OUTPUT_SEG_ARGMAX = "seg_argmax"
    OUTPUT_SEG_ARGMAX_DICOM = "seg_argmax_dicom"

    def __init__(
        self,
        fragment: Any,
        *args: Any,
        emit_lowres_seg: bool = False,
        emit_probabilities: bool = True,
        emit_argmax_seg: bool = False,
        config_name: Optional[str] = None,
        release_fn: Optional[Callable[[], None]] = None,
        part: Optional[str] = None,
        emit_argmax_dicom: bool = True,
        model_path: Optional[Union[str, Path]] = None,
        resample_seg_to_original: bool = False,
        **kwargs: Any,
    ):
        """Create the operator.

        Args:
            fragment: the owning application (passed to ``Operator``).
            emit_lowres_seg: declare/emit the extra ``lowres_seg`` output
                (post-softmax argmax, uint8, original DICOM orientation,
                no CC) for cascade producer fragments (3d_lowres). Plan 04
                wires it to the cascade PreprocessOperator's ``lowres_seg``
                input; defaults False = byte-for-byte Phase 1 behavior.
            emit_probabilities: declare/emit the ``probabilities`` output.
                Declared now (consumed by Plan 04's conditional wiring) so
                Plan 04 doesn't re-edit this file's init pattern;
                ``False`` simply omits the output declaration.
            emit_argmax_seg: P8 single-part E2E (plan 08-05) — declare/emit
                the extra ``seg_argmax`` (model-resolution, slidewindow-native
                order, pre-revert — the 8.5 gate target) and
                ``seg_argmax_dicom`` (crop/transpose-reverted, original DICOM
                orientation) outputs, both uint8 CUDA. The compute path is
                GPU argmax of the fold-averaged logits with NO softmax
                materialization, NO probability resample, and NO CPU hop
                (PIP-01: eliminates the oracle's .npz save->load->average->
                argmax round-trip). Mutually exclusive with
                ``emit_lowres_seg`` and ``emit_probabilities`` (one emit mode
                at a time; port discipline).
            config_name: plans.json configuration key — tags the NVTX range
                name and timing record so per-config observability survives
                sub-Fragments (INFR-005). ``None`` (default) keeps the bare
                ``"postresample"`` name.
            release_fn: MEM-003/D-23 — zero-arg callback invoked after the
                LAST emit of ``compute()`` (all emits done; nothing
                downstream touches the released bundle). The aux (lowres)
                subgraph wires it to its SlideWindowOperator.release();
                ``None`` (default) = no release, byte-for-byte prior
                behavior.
            part: P9 (09-02) — model part name for the 5-part chain. When
                set, the NVTX range + timing label become
                ``postresample_{config_name}_{part}``; ``None`` (single-part
                P8 path) keeps the byte-identical P8 label.
            emit_argmax_dicom: P9 (09-02, port discipline) — when False with
                ``emit_argmax_seg=True``, the ``seg_argmax_dicom`` output is
                NOT declared and the crop/transpose revert is NOT computed
                (pure overhead in 5-part mode, where only the model-space
                argmax feeds the merge). Default True = P8 behavior
                unchanged.
            model_path: P9 (09-03) — the part bundle root (contains
                ``jsonpkls/``). When set, the operator runs the oracle-parity
                per-part postprocess gate (``part_postprocess``): a setup-time
                ``jsonpkls/postprocessing.pkl`` existence check (cached — a
                deployed bundle's pkl cannot appear at runtime) and, on the
                emit_argmax_seg path, the hook at the oracle's exact
                placement (per-part, model-space, PRE-merge). With 0 pkls
                shipped (live path) the hook is a TRUE identity — no D2H, no
                copy, one log line per part. ``None`` (default) = no hook
                (byte-identical prior behavior; nothing new is logged).
        """
        # NOTE: holoscan 4.2's Operator.__init__ invokes self.setup(spec)
        # before this constructor body finishes — initialize all state first
        # (same pattern as EnsembleAverageOperator.emit_averaged_probabilities
        # and the emit flags above, RESEARCH Pitfall 7 discipline).
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._emit_lowres_seg = bool(emit_lowres_seg)
        self._emit_probabilities = bool(emit_probabilities)
        self._emit_argmax_seg = bool(emit_argmax_seg)
        if self._emit_argmax_seg and (self._emit_lowres_seg or self._emit_probabilities):
            raise ValueError(
                "emit_argmax_seg is mutually exclusive with emit_lowres_seg and "
                "emit_probabilities — one emit mode at a time (port discipline: a "
                "declared output must have a receiver; plan 08-05)."
            )
        self.config_name = config_name
        self._release_fn = release_fn
        # P9 (09-02): per-part tag (validated) + port-discipline flag.
        if part is not None and part not in {p["name"] for p in MODEL_PARTS}:
            raise ValueError(f"Unknown model part {part!r}. Valid: {[p['name'] for p in MODEL_PARTS]}")
        self._part = part
        self._emit_argmax_dicom = bool(emit_argmax_dicom)
        # 22-11 (audit S4): for `resample is None` cascade tasks the emit
        # path reverts through the plans probability resample so the
        # seg_argmax_dicom output is at CROP-NATIVE resolution (TS's
        # output space) instead of model resolution; PasteOp's identity
        # branch then pastes without its order-0 label zoom. Default
        # False = byte-identical pre-fix fast path (explicit-resample
        # tasks + the total chain are untouched).
        self._resample_seg_to_original = bool(resample_seg_to_original)
        # P9 (09-03): per-part postprocess gate — must exist before
        # super().__init__ (Operator.__init__ invokes setup).
        self._model_path = Path(model_path) if model_path is not None else None
        self._has_pp = False
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        """Declare the operator's I/O: logits + meta in; probabilities and
        (optionally) lowres_seg out, gated on the constructor flags.
        Port discipline (RESEARCH Pitfall 7): a declared output with no
        receiver is a GXF rejection, so both outputs are declared only when
        the owning fragment will wire them."""
        spec.input(self.INPUT_LOGITS)
        spec.input(self.INPUT_META)
        if self._emit_probabilities:
            spec.output(self.OUTPUT_PROBABILITIES)
        if self._emit_lowres_seg:
            spec.output(self.OUTPUT_LOWRES_SEG)
        if self._emit_argmax_seg:
            spec.output(self.OUTPUT_SEG_ARGMAX)
            # P9 (09-02): seg_argmax_dicom is declared ONLY when needed —
            # in 5-part mode (emit_argmax_dicom=False) the revert is pure
            # overhead and an unwired declared port would hang the DAG
            # (Pitfall 5). Default True = P8 wiring unchanged.
            if self._emit_argmax_dicom:
                spec.output(self.OUTPUT_SEG_ARGMAX_DICOM)
        # P9 (09-03): oracle-parity per-part postprocess gate — the pkl
        # existence stat happens HERE, once, at setup (a deployed bundle's
        # pkl cannot appear at runtime; the live per-study path performs no
        # filesystem access and no D2H when the pkl is absent).
        if self._model_path is not None:
            self._has_pp = find_postprocessing_pkl(self._model_path) is not None
            if self._has_pp:
                self._logger.info(
                    "PostProcess: postprocessing.pkl found at %s/jsonpkls — "
                    "per-part rules will run pre-merge (dormant path)",
                    self._model_path,
                )
            else:
                self._logger.info(
                    "PostProcess: no postprocessing.pkl at %s/jsonpkls — " "identity no-op (oracle parity)",
                    self._model_path,
                )

    @staticmethod
    def _to_4d(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize the incoming tensor to the reference ``(C, X, Y, Z)``."""
        if tensor.ndim == 5:
            if tensor.shape[0] != 1:
                raise ValueError(f"PostResampleOperator supports batch size 1, got batch {tensor.shape[0]}.")
            return tensor[0]
        if tensor.ndim == 4:
            return tensor
        raise ValueError(f"expected a (C, X, Y, Z) or (1, C, X, Y, Z) logits tensor, got ndim={tensor.ndim}.")

    def postresample(self, logits: torch.Tensor, meta: Dict[str, Any]):
        """Full post-resample: reference CPU resample+softmax, GPU revert.

        Args:
            logits: CUDA ``(C, X, Y, Z)`` float32 logits (post-transpose order).
            meta: PreprocessOperator metadata dict.

        Returns:
            CUDA ``(C, *original_shape)`` float32 probabilities in original
            DICOM orientation, or the tuple
            ``(probabilities_gpu, lowres_seg)`` when ``emit_lowres_seg`` is
            set (``probabilities_gpu`` is ``None`` when
            ``emit_probabilities`` is False; ``lowres_seg`` is the 3D uint8
            CUDA argmax segmentation in original DICOM orientation), or the
            tuple ``(seg_model, seg_dicom)`` when ``emit_argmax_seg`` is set
            (both uint8 CUDA; ``seg_model`` is the model-resolution
            slidewindow-native argmax — the 8.5 gate target; ``seg_dicom``
            is the crop/transpose-reverted original-DICOM-orientation form).
        """
        if self._emit_argmax_seg:
            # P8 (plan 08-05): D-09 lock — argmax is invariant under the
            # monotone per-voxel softmax, so argmax(logits) ==
            # argmax(softmax(logits)) EXACTLY. The oracle's .npz
            # save->load->average->argmax round-trip (tmp_total/3d_fullres)
            # is replaced by this single in-GPU argmax of the fold-averaged
            # logits — no softmax materialization, no probability resample,
            # no CPU hop (PIP-01). `logits` are already the fold-averaged
            # (C, X, Y, Z) in slidewindow-native (nnUNet post-transpose) order.
            seg_model = torch.argmax(logits, dim=0).to(torch.uint8)
            # P9 (09-02): the crop/transpose revert is computed ONLY when
            # the dicom output is actually declared (port discipline —
            # 5-part mode skips it as pure overhead).
            if self._emit_argmax_dicom:
                if self._resample_seg_to_original:
                    # 22-11 (audit S4): TS 2.18 resamples the LOGITS to the
                    # original (crop-native) shape with the plans probability
                    # kwargs (order 1 / order_z 0 / separate-z) and argmaxes
                    # THERE (export_prediction.py:30-47). The CPU reference
                    # runs the exact chain; seg_model stays the GPU
                    # model-resolution argmax for the unchanged gate/sink
                    # consumers.
                    _seg_model_ref, seg_dicom_np = seg_argmax_to_original_reference(logits, dict(meta))
                    seg_dicom = torch.from_numpy(seg_dicom_np).to(logits.device)
                else:
                    seg_dicom = revert_crop_gpu(seg_model, meta)
                return seg_model, seg_dicom
            return seg_model, None
        # The resample runs on the reference CPU path (project decision:
        # resampling stays on the reference scipy/scikit-image path for
        # pixel-exactness in Phases 1-2). This is the one deliberate
        # GPU->CPU hop of this operator.
        logits_cpu = logits.detach().cpu().numpy()
        probabilities_cpu = postresample_reference(logits_cpu, meta)
        if self._emit_lowres_seg:
            # Post-softmax argmax == the reference argmax-of-resampled-logits:
            # softmax is monotone per voxel, so argmax(softmax(p)) ==
            # argmax(p) exactly (D-09: the cascade input is the argmax seg).
            # NO connected-component cleanup — the reference cascade input is
            # pre-CC (the reference KeepLargestCC runs only on the final
            # output, verified).
            seg_crop = torch.argmax(torch.from_numpy(probabilities_cpu), dim=0).to(torch.uint8)
            seg_full = revert_crop_gpu(seg_crop, meta)
            if self._emit_probabilities:
                probabilities_gpu = revert_crop_and_transpose_gpu(probabilities_cpu, meta)
            else:
                probabilities_gpu = None
            return probabilities_gpu, seg_full
        return revert_crop_and_transpose_gpu(probabilities_cpu, meta)

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        """Resample probabilities + revert crop/transpose; emit GPU tensor."""
        # INFR-005: the NVTX range + timing label carry the config when set,
        # so multi-fragment traces/records stay unambiguous (RESEARCH Pitfall
        # 9); None keeps the bare Phase 1 name. P9 (09-02): with ``part``
        # set, the label is per-part (postresample_{config}_{part}); None
        # keeps the P8 name byte-identical.
        if self.config_name and self._part:
            _range_name = f"postresample_{self.config_name}_{self._part}"
        elif self.config_name:
            _range_name = f"postresample_{self.config_name}"
        else:
            _range_name = "postresample"
        with nvtx_range(_range_name):
            timing = GpuTiming(_range_name)
            timing.start()

            # Entry guard: the pipeline is GPU-resident by contract (INF-005).
            assert_cuda_available()

            holo_tensor = op_input.receive(self.INPUT_LOGITS)
            meta = op_input.receive(self.INPUT_META)
            if holo_tensor is None:
                raise ValueError("PostResampleOperator received no 'logits' input.")
            if not meta:
                raise ValueError("PostResampleOperator received no 'preprocessed_meta' input.")

            tensor = torch.utils.dlpack.from_dlpack(holo_tensor)
            # Device invariant at the boundary (INF-005).
            assert_on_gpu(tensor)
            data = self._to_4d(tensor.float())

            if self._emit_argmax_seg:
                seg_model, seg_dicom = self.postresample(data, dict(meta))
                # Exit guards: emitted buffers must be CUDA-resident uint8.
                assert_on_gpu(seg_model)
                # P9 (09-03): per-part postprocess at the oracle's exact
                # placement — per-part, model-space (SAR), PRE-merge (the
                # oracle's PostProcessNNUnet runs inside
                # _post_process_for_part, before the 5-part merge).
                # LIVE path (no pkl — all 5 shipped bundles): TRUE identity,
                # zero-copy — no D2H, no copy; seg_model flows on untouched.
                # DORMANT path (pkl present, exercised only by the synthetic
                # suite driving the pure function): D2H ->
                # apply_part_postprocess (GPU CC port) -> re-upload; the
                # emit contract (CUDA uint8 tensor) stays intact.
                if self._model_path is not None:
                    if self._has_pp:
                        seg_cpu = seg_model.detach().cpu().numpy()
                        seg_cpu, pp_info = apply_part_postprocess(seg_cpu, self._model_path)
                        seg_model = torch.from_numpy(np.ascontiguousarray(seg_cpu)).to(seg_model.device)
                        assert seg_model.dtype == torch.uint8
                    else:
                        pp_info = {"no_op": True, "pkl": None}
                    _pp_op = (
                        f"postprocess_{self._part}"
                        if self._part
                        else (f"postprocess_{self.config_name}" if self.config_name else "postprocess")
                    )
                    self._logger.info("postprocess record: op=%s no_op=%s", _pp_op, pp_info["no_op"])
                    assert_on_gpu(seg_model)
                op_output.emit(to_holoscan_gpu_tensor(seg_model), self.OUTPUT_SEG_ARGMAX)
                if seg_dicom is not None:
                    assert_on_gpu(seg_dicom)
                    op_output.emit(to_holoscan_gpu_tensor(seg_dicom), self.OUTPUT_SEG_ARGMAX_DICOM)
                probabilities = None
            elif self._emit_lowres_seg:
                probabilities, seg_full = self.postresample(data, dict(meta))
            else:
                probabilities = self.postresample(data, dict(meta))
                seg_full = None

            if probabilities is not None:
                # Exit guard: the emitted buffer must be CUDA-resident FP32.
                assert_on_gpu(probabilities)
                op_output.emit(to_holoscan_gpu_tensor(probabilities), self.OUTPUT_PROBABILITIES)
            if self._emit_lowres_seg:
                # Exit guard: the cascade input must be CUDA-resident uint8.
                assert_on_gpu(seg_full)
                op_output.emit(to_holoscan_gpu_tensor(seg_full), self.OUTPUT_LOWRES_SEG)

            record = timing.stop()
            record["study"] = get_study_id(self.fragment)
            record["config"] = self.config_name
            if probabilities is not None:
                record["probabilities_shape"] = list(probabilities.shape)
            if self._emit_argmax_seg:
                record["seg_argmax_shape"] = list(seg_model.shape)
                record["seg_argmax_dtype"] = "uint8"
                if seg_dicom is not None:
                    record["seg_argmax_dicom_shape"] = list(seg_dicom.shape)
            if self._emit_lowres_seg:
                record["lowres_seg_shape"] = list(seg_full.shape)
                record["lowres_seg_dtype"] = "uint8"
            StudyTimingCollector.record(self.fragment, record)
            self._logger.info("timing: %s", json.dumps(record))

            # MEM-003/D-23: release callback at the very end of compute() —
            # after ALL emits. The cascade consumes only the emitted
            # lowres_seg tensor, so nothing downstream touches the released
            # bundle (the aux SlideWindowOperator is never scheduled again).
            if self._release_fn is not None:
                self._release_fn()
