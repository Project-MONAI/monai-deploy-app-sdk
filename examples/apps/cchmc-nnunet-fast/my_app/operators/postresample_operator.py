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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from skimage.transform import resize

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
    from preprocess_operator import _determine_do_sep_z_and_axis, to_holoscan_gpu_tensor

__all__ = [
    "PostResampleOperator",
    "postresample_reference",
    "revert_crop_and_transpose_gpu",
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
        dtype_out = data.dtype
        data = data.astype(float, copy=False)
        reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
        resize_kwargs = {"mode": "edge", "anti_aliasing": False}

        if do_separate_z:
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
            original DICOM orientation (FP32, CUDA).
    """

    INPUT_LOGITS = "logits"
    INPUT_META = "preprocessed_meta"
    OUTPUT_PROBABILITIES = "probabilities"

    def __init__(self, fragment: Any, *args: Any, **kwargs: Any):
        # NOTE: holoscan 4.2's Operator.__init__ invokes self.setup(spec)
        # before this constructor body finishes — initialize all state first.
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        """Declare the operator's I/O: logits + meta in, probabilities out."""
        spec.input(self.INPUT_LOGITS)
        spec.input(self.INPUT_META)
        spec.output(self.OUTPUT_PROBABILITIES)

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

    def postresample(self, logits: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
        """Full post-resample: reference CPU resample+softmax, GPU revert.

        Args:
            logits: CUDA ``(C, X, Y, Z)`` float32 logits (post-transpose order).
            meta: PreprocessOperator metadata dict.

        Returns:
            CUDA ``(C, *original_shape)`` float32 probabilities in original
            DICOM orientation.
        """
        # The resample runs on the reference CPU path (project decision:
        # resampling stays on the reference scipy/scikit-image path for
        # pixel-exactness in Phases 1-2). This is the one deliberate
        # GPU->CPU hop of this operator.
        logits_cpu = logits.detach().cpu().numpy()
        probabilities_cpu = postresample_reference(logits_cpu, meta)
        probabilities_gpu = revert_crop_and_transpose_gpu(probabilities_cpu, meta)
        return probabilities_gpu

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        """Resample probabilities + revert crop/transpose; emit GPU tensor."""
        with nvtx_range("postresample"):
            timing = GpuTiming("postresample")
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

            probabilities = self.postresample(data, dict(meta))

            # Exit guard: the emitted buffer must be CUDA-resident FP32.
            assert_on_gpu(probabilities)
            op_output.emit(to_holoscan_gpu_tensor(probabilities), self.OUTPUT_PROBABILITIES)

            record = timing.stop()
            record["study"] = get_study_id(self.fragment)
            record["probabilities_shape"] = list(probabilities.shape)
            StudyTimingCollector.record(self.fragment, record)
            self._logger.info("timing: %s", json.dumps(record))
