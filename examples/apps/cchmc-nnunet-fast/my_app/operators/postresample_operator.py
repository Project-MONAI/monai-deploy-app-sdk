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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from skimage.transform import resize

import cupy as cp  # D-22: gated GPU resample path (flag OFF = unused numerically)

from monai.deploy.core import Operator, OperatorSpec

try:  # package-style import (my_app.*)
    from my_app.operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        assert_on_gpu,
        get_study_id,
        nvtx_range,
    )
    from my_app.operators.gpu_zoom import gpu_resample_enabled, stock_gpu_resize
    from my_app.operators.preprocess_operator import _determine_do_sep_z_and_axis, to_holoscan_gpu_tensor
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        assert_on_gpu,
        get_study_id,
        nvtx_range,
    )
    from gpu_zoom import gpu_resample_enabled, stock_gpu_resize
    from preprocess_operator import _determine_do_sep_z_and_axis, to_holoscan_gpu_tensor

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
            return np.ascontiguousarray(
                stock_gpu_resize(
                    data_gpu, tuple(int(s) for s in new_shape), order
                ).get()
            )
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
                    reshaped_final[c] = map_coordinates(
                        reshaped_here, coord_map, order=order_z, mode="nearest"
                    )
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
        resampled = resample_probabilities_to_shape(
            logits, new_shape, current_spacing, target_spacing
        )
        # Reference: apply_inference_nonlin -> torch.from_numpy(...).float() -> softmax dim 0.
        probabilities = torch.nn.functional.softmax(torch.from_numpy(resampled).float(), dim=0)
        return probabilities.numpy()
    finally:
        torch.set_num_threads(old_threads)


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

    full = torch.zeros((probabilities.shape[0], *original_shape), dtype=probabilities.dtype, device=probabilities.device)
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

    def __init__(
        self,
        fragment: Any,
        *args: Any,
        emit_lowres_seg: bool = False,
        emit_probabilities: bool = True,
        config_name: Optional[str] = None,
        release_fn: Optional[Callable[[], None]] = None,
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
        """
        # NOTE: holoscan 4.2's Operator.__init__ invokes self.setup(spec)
        # before this constructor body finishes — initialize all state first
        # (same pattern as EnsembleAverageOperator.emit_averaged_probabilities
        # and the emit flags above, RESEARCH Pitfall 7 discipline).
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._emit_lowres_seg = bool(emit_lowres_seg)
        self._emit_probabilities = bool(emit_probabilities)
        self.config_name = config_name
        self._release_fn = release_fn
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

    @staticmethod
    def _to_4d(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize the incoming tensor to the reference ``(C, X, Y, Z)``."""
        if tensor.ndim == 5:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"PostResampleOperator supports batch size 1, got batch {tensor.shape[0]}."
                )
            return tensor[0]
        if tensor.ndim == 4:
            return tensor
        raise ValueError(
            f"expected a (C, X, Y, Z) or (1, C, X, Y, Z) logits tensor, got ndim={tensor.ndim}."
        )

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
            CUDA argmax segmentation in original DICOM orientation).
        """
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
        # 9); None keeps the bare Phase 1 name.
        _range_name = f"postresample_{self.config_name}" if self.config_name else "postresample"
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

            if self._emit_lowres_seg:
                probabilities, seg_full = self.postresample(data, dict(meta))
            else:
                probabilities = self.postresample(data, dict(meta))

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
