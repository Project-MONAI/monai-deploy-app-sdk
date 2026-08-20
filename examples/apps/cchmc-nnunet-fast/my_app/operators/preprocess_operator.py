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

"""PreprocessOperator: GPU (CuPy) preprocessing path + GPU handoff.

Reproduces the reference nnUNet ``DefaultPreprocessor.run_case_npy`` pipeline
in order:

    astype(float32) -> transpose(transpose_forward) -> crop_to_nonzero
    -> per-channel normalize (before resample) -> resample to config spacing

Phase 2 (CuPy port): the layout and element-wise stages — the raw-volume
transpose (PREP-01), the crop-slice (PREP-04), and the element-wise per-channel
normalization (PREP-02) — run on GPU via CuPy in fp32, C-contiguous (D-12).
Two stages deliberately stay on the verified numpy/scipy CPU reference path:
the Z-score/CT mean-std *reductions* (CuPy reductions are not bit-identical to
numpy — reduction order differs) and the resample itself (PREP-03, D-13:
scipy/scikit-image reference path; the resulting GPU->CPU->GPU round-trip is
expected and accepted, ~64 MB fp32 per study). The module keeps
``preprocess_reference`` as the documented pure-CPU fallback path.

The final float32 volume is emitted as a zero-copy GPU tensor for the
downstream SlideWindowOperator.

Note on the handoff contract: holoscan-cu13 4.2's Python API exposes the
zero-copy GPU buffer primitive as ``holoscan.core.Tensor`` (built via DLPack,
``Tensor.device.device_type == kDLDeviceCUDA``) — the 4.2 equivalent of the
``MemoryData(DeviceType::GPU)`` contract from the plan.

All per-config parameters (spacing, transpose order, crop, normalization,
resampling) come from the model bundle ``jsonpkls/plans.json`` keyed by
``config_name`` — nothing is hard-coded (PREP-01..04).
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cupy as cp
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes, map_coordinates
from skimage.transform import resize

from monai.deploy.core import Image, Operator, OperatorSpec

try:  # package-style import (my_app.*)
    from my_app.config import PreprocessParams, load_preprocess_params
    from my_app.operators.buffer_cache import _ShapeCache
    from my_app.operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        assert_on_gpu,
        nvtx_range,
        set_study_id,
    )
    from my_app.operators.gpu_zoom import (
        gpu_resample_enabled,
        stock_gpu_resize,
        stock_gpu_zoom,
        zoom_factors_for,
    )
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from config import PreprocessParams, load_preprocess_params
    from buffer_cache import _ShapeCache
    from gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        assert_on_gpu,
        nvtx_range,
        set_study_id,
    )
    from gpu_zoom import (
        gpu_resample_enabled,
        stock_gpu_resize,
        stock_gpu_zoom,
        zoom_factors_for,
    )

__all__ = ["PreprocessOperator", "preprocess_reference", "to_holoscan_gpu_tensor"]

# Mirrors nnunetv2.configuration.ANISO_THRESHOLD (a framework constant, not a
# per-dataset parameter, so it is not present in plans.json).
ANISO_THRESHOLD = 3

# kDLDeviceCUDA value in the DLPack device-type enumeration.
_KDLDEVICE_CUDA = 2


# ---------------------------------------------------------------------------
# Reference CPU path primitives (bit-exact replicas of the nnUNet reference)
# ---------------------------------------------------------------------------


def _create_nonzero_mask(data: np.ndarray) -> np.ndarray:
    """True where the data is nonzero, with holes filled (reference replica)."""
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (X, Y, Z)"
    mask = data[0] != 0
    for c in range(1, data.shape[0]):
        mask |= data[c] != 0
    return binary_fill_holes(mask)


def _get_bbox_from_mask(mask: np.ndarray) -> List[List[int]]:
    """Bounding box as half-open intervals [[minz, maxz], [minx, maxx], [miny, maxy]].

    Replica of the ``acvl_utils`` implementation used by the reference so the
    crop matches bit-for-bit without taking a runtime dependency on acvl_utils.
    """
    z, x, y = mask.shape
    min_z, max_z, min_x, max_x, min_y, max_y = 0, z, 0, x, 0, y
    for i in range(z):
        if np.any(mask[i]):
            min_z = i
            break
    for i in range(z - 1, -1, -1):
        if np.any(mask[i]):
            max_z = i + 1
            break
    for i in range(x):
        if np.any(mask[:, i]):
            min_x = i
            break
    for i in range(x - 1, -1, -1):
        if np.any(mask[:, i]):
            max_x = i + 1
            break
    for i in range(y):
        if np.any(mask[:, :, i]):
            min_y = i
            break
    for i in range(y - 1, -1, -1):
        if np.any(mask[:, :, i]):
            max_y = i + 1
            break
    return [[min_z, max_z], [min_x, max_x], [min_y, max_y]]


def _compute_new_shape(
    old_shape: Sequence[int],
    old_spacing: Sequence[float],
    new_spacing: Sequence[float],
) -> np.ndarray:
    """Target voxel shape for a spacing change (reference replica)."""
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    return np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])


def _determine_do_sep_z_and_axis(
    force_separate_z: Optional[bool],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    anisotropy_threshold: float = ANISO_THRESHOLD,
) -> Tuple[bool, Optional[int]]:
    """Decide separate-z resampling and the anisotropic axis (reference replica)."""
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        axis = np.where(max(current_spacing) / np.array(current_spacing) == 1)[0] if force_separate_z else None
    else:
        cur_aniso = (np.max(current_spacing) / np.min(current_spacing)) > anisotropy_threshold
        new_aniso = (np.max(new_spacing) / np.min(new_spacing)) > anisotropy_threshold
        if cur_aniso:
            do_separate_z, axis = True, np.where(max(current_spacing) / np.array(current_spacing) == 1)[0]
        elif new_aniso:
            do_separate_z, axis = True, np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]
        else:
            do_separate_z, axis = False, None
    if axis is not None:
        if len(axis) == 3:
            do_separate_z, axis = False, None
        elif len(axis) == 2:
            do_separate_z, axis = False, None
        else:
            axis = axis[0]
    return do_separate_z, axis


def _normalize_channel(image: np.ndarray, params: PreprocessParams, channel: int, mask: Optional[np.ndarray]) -> np.ndarray:
    """Apply the per-channel normalization scheme from plans.json (reference replica)."""
    scheme = params.normalization_schemes[channel]
    image = image.astype(np.float32, copy=False)
    eps = 1e-8

    if scheme == "ZScoreNormalization":
        if params.use_mask_for_norm[channel] and mask is not None:
            # Reference: mask = seg >= 0 (synthetic seg encodes the crop region).
            m = mask
            mean = image[m].mean()
            std = image[m].std()
            image[m] = (image[m] - mean) / (max(std, eps))
        else:
            mean = image.mean()
            std = image.std()
            image -= mean
            image /= (max(std, eps))
        return image

    if scheme == "CTNormalization":
        props = params.intensity_properties[str(channel)]
        np.clip(image, props["percentile_00_5"], props["percentile_99_5"], out=image)
        image -= props["mean"]
        image /= max(props["std"], eps)
        return image

    if scheme == "NoNormalization":
        return image

    raise ValueError(f"unsupported normalization scheme {scheme!r} for channel {channel}")


def _resample_to_shape(
    data: np.ndarray,
    new_shape: Union[Sequence[int], np.ndarray],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    params: PreprocessParams,
) -> np.ndarray:
    """Resample ``(C, X, Y, Z)`` data to a target shape (reference replica).

    Same scipy/scikit-image reference path: ``skimage.transform.resize``
    (mode='edge', anti_aliasing=False) plus the separate-z pass via
    ``scipy.ndimage.map_coordinates``.
    """
    do_separate_z, axis = _determine_do_sep_z_and_axis(
        params.resample_force_separate_z, current_spacing, new_spacing
    )

    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)

    if np.any(shape != new_shape):
        if gpu_resample_enabled() and not do_separate_z:
            # D-22 (D-22a amended) gated GPU resample (HOLOSCAN_GPU_RESAMPLE=1):
            # the STOCK cupyx.scipy.ndimage path — per-channel exact mirror of
            # the flag-OFF chain (fp64 widening -> grid_mode zoom -> fp64
            # clip -> fp32 cast). Stays on GPU end-to-end; no D2H for the
            # resample span. Flag OFF (default) runs the verbatim scipy/skimage
            # reference below. (The custom RawKernel provenance in gpu_zoom.py
            # is NOT wired — D-22a.)
            return stock_gpu_resize(
                cp.asarray(data, dtype=cp.float32),
                tuple(int(s) for s in new_shape),
                params.resample_order,
            )
        dtype_out = data.dtype
        data = data.astype(float, copy=False)
        reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
        order = params.resample_order
        order_z = params.resample_order_z
        resize_kwargs = {"mode": "edge", "anti_aliasing": False}

        if do_separate_z:
            # D-22: the separate-z map_coordinates branches stay scipy in
            # BOTH flag states — inactive in this bundle (near-isotropic
            # spacings); order_z=0 was byte-identical on CuPy anyway, so no
            # port is needed. The flag only gates the non-sep-z branch.
            assert axis is not None, "if do_separate_z, we need to know what axis is anisotropic"
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            for c in range(data.shape[0]):
                tmp = deepcopy(new_shape)
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


def _resize_segmentation(segmentation: np.ndarray, new_shape: Sequence[int], order: int) -> np.ndarray:
    """Segmentation resizer — bit-exact replica of the vendored
    ``batchgenerators.augmentations.utils.resize_segmentation`` (2.8.1 era).

    This is the ``resize_fn`` of the SEGMENTATION path (``is_seg=True``) of
    ``nnunetv2...default_resampling.resample_data_or_seg`` — NOT the
    ``skimage.transform.resize`` of the data path: per-label multihot
    resize (skimage ``resize``, mode='edge', ``clip=True``,
    ``anti_aliasing=False``) with a >= 0.5 threshold, so interpolation
    cannot invent intermediate labels. ``np.unique`` on the raveled volume
    is exactly ``np.sort(pd.unique(...))`` for numeric arrays.
    """
    tpe = segmentation.dtype
    if gpu_resample_enabled() and segmentation.ndim == 3:
        # D-22 (D-22a amended) gated GPU multihot (HOLOSCAN_GPU_RESAMPLE=1):
        # per-label fp64 0/1 mask -> stock CuPy grid_mode zoom (exact mirror
        # of the OFF-path skimage chain: fp64 widening, fp64 clip) -> the
        # SAME >= 0.5 threshold / label-assignment / cast tail the scipy path
        # has. The 2D separate-z slice calls (and the flag-OFF path) stay on
        # the verbatim scipy/skimage reference.
        out_shape = tuple(int(s) for s in new_shape)
        base = cp.asarray(np.ascontiguousarray(segmentation), dtype=cp.float32)
        zf = zoom_factors_for(segmentation.shape, out_shape)
        if order == 0:
            # Reference early return: resize(..., 0, clip=True).astype(tpe) —
            # nearest copy of the (exactly representable) label values.
            m0 = stock_gpu_zoom(base, zf, 0, out_shape=out_shape)
            m0 = cp.clip(
                m0, float(cp.min(base).get()), float(cp.max(base).get())
            )
            return np.asarray(m0.get(), dtype=tpe)
        reshaped = np.zeros(out_shape, dtype=tpe)
        for c in np.unique(segmentation.ravel()):
            mask = cp.equal(base, np.float32(c)).astype(cp.float32)
            m = stock_gpu_zoom(mask, zf, order, out_shape=out_shape)
            m = cp.clip(m, 0.0, 1.0).get()
            reshaped[m >= 0.5] = c
        return reshaped
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    reshaped = np.zeros(new_shape, dtype=tpe)
    for c in np.unique(segmentation.ravel()):
        mask = segmentation == c
        reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
        reshaped[reshaped_multihot >= 0.5] = c
    return reshaped


def _resample_seg_to_shape(
    data: np.ndarray,
    new_shape: Union[Sequence[int], np.ndarray],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    params: PreprocessParams,
) -> np.ndarray:
    """Resample ``(C, X, Y, Z)`` segmentation to a target shape (bit-exact
    replica of the SEGMENTATION path (``is_seg=True``) of the vendored
    ``nnunetv2 2.8.1 default_resampling.resample_data_or_seg_to_shape`` /
    ``resample_data_or_seg``).

    Mirrors the vendored branch structure exactly, with the seg semantics
    (``default_resampling.py``):

    * ``resize_fn = resize_segmentation`` (see above), ``kwargs = {}`` —
      no ``mode='edge'``/``anti_aliasing=False`` kwargs at the call site:
      for ``is_seg`` anti-aliasing is disabled (it lives inside
      ``resize_segmentation``), unlike the data-path replica
      ``_resample_to_shape``;
    * ``order = params.resample_seg_order`` (1),
      ``order_z = params.resample_seg_order_z`` (0),
      ``force_separate_z = params.resample_seg_force_separate_z`` (None);
    * the intermediate cast to float before resize and the final cast back
      to the input dtype on assignment (the input stays integer — uint8
      in/uint8 out, as the reference does);
    * the same "no resampling necessary -> return input unchanged"
      short-circuit as the reference.

    Unit-tested ``np.array_equal`` against the vendored function on random
    uint8 volumes (scripts/test_cascade_config.py::test_seg_resample_replica).
    """
    do_separate_z, axis = _determine_do_sep_z_and_axis(
        params.resample_seg_force_separate_z, current_spacing, new_spacing
    )

    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)

    if np.any(shape != new_shape):
        dtype_out = data.dtype
        data = data.astype(float, copy=False)
        reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
        order = params.resample_seg_order
        order_z = params.resample_seg_order_z

        if do_separate_z:
            # D-22: the separate-z map_coordinates branches stay scipy in
            # BOTH flag states (inactive in this bundle) — see the comment
            # in _resample_to_shape.
            assert axis is not None, "if do_separate_z, we need to know what axis is anisotropic"
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            for c in range(data.shape[0]):
                tmp = deepcopy(new_shape)
                tmp[axis] = shape[axis]
                reshaped_here = np.zeros(tmp)
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_here[slice_id] = _resize_segmentation(
                            data[c, slice_id], new_shape_2d, order
                        )
                    elif axis == 1:
                        reshaped_here[:, slice_id] = _resize_segmentation(
                            data[c, :, slice_id], new_shape_2d, order
                        )
                    else:
                        reshaped_here[:, :, slice_id] = _resize_segmentation(
                            data[c, :, :, slice_id], new_shape_2d, order
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
                    if order_z == 0:
                        # Reference: is_seg with order_z == 0 takes the same
                        # direct map_coordinates path as the data path.
                        reshaped_final[c] = map_coordinates(
                            reshaped_here, coord_map, order=order_z, mode="nearest"
                        )[None]
                    else:
                        # Not exercised by this bundle (order_z=0); kept for
                        # branch fidelity with the vendored function
                        # (np.unique == np.sort(pd.unique(...)) numerically).
                        for cl in np.unique(reshaped_here.ravel()):
                            reshaped_final[c][
                                np.round(
                                    map_coordinates(
                                        (reshaped_here == cl).astype(float),
                                        coord_map,
                                        order=order_z,
                                        mode="nearest",
                                    )
                                )
                                > 0.5
                            ] = cl
                else:
                    reshaped_final[c] = reshaped_here
        else:
            for c in range(data.shape[0]):
                reshaped_final[c] = _resize_segmentation(data[c], new_shape, order)
        return reshaped_final
    else:
        # No resampling necessary — the reference returns the input unchanged.
        return data


def preprocess_reference(
    data: np.ndarray,
    spacing_xyz: Sequence[float],
    params: PreprocessParams,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run the reference ``run_case_npy`` preprocessing path on CPU.

    Args:
        data: ``(C, X, Y, Z)`` volume in the app-transposed order
            (channel-first, post ``Transposed([0, 3, 2, 1])``), any integer or
            float dtype — it is cast to float32 internally, as the reference
            does. Must be **C-contiguous**: numpy's ``astype`` preserves the
            input's memory order, and the resample stage (skimage resize /
            map_coordinates) is not layout-invariant at float32, so an
            F-ordered input would diverge from the reference by 1 ulp over
            most voxels (the reference receives a fresh C-contiguous array).
        spacing_xyz: physical voxel spacing ``(sx, sy, sz)`` (LPS column
            norms), as derived from the series affine.
        params: per-config parameters from the bundle plans.

    Returns:
        ``(volume, properties)`` — the preprocessed float32 volume and the
        recorded properties (crop bbox, pre-crop shape, resampled shape,
        spacing metadata) needed by PostResample/Ensemble to revert.
    """
    data = data.astype(np.float32)  # creates a copy, as the reference does

    tf = params.transpose_forward
    # apply transpose_forward (this also needs to be applied to the spacing!)
    data = data.transpose([0, *[i + 1 for i in tf]])
    # Reference stores properties["spacing"] in the reversed (z, y, x) order.
    spacing = tuple(reversed(spacing_xyz))
    original_spacing = [spacing[i] for i in tf]

    # crop, remember the size before cropping!
    properties: Dict[str, Any] = {}
    properties["shape_before_cropping"] = tuple(int(s) for s in data.shape[1:])

    nonzero_mask = _create_nonzero_mask(data)
    bbox = _get_bbox_from_mask(nonzero_mask)
    slicer = tuple(slice(*i) for i in bbox)
    data = data[(slice(None),) + slicer]
    properties["bbox_used_for_cropping"] = bbox
    properties["shape_after_cropping_and_before_resampling"] = tuple(int(s) for s in data.shape[1:])

    # resample target shape
    new_shape = _compute_new_shape(data.shape[1:], original_spacing, params.spacing)
    properties["new_shape"] = tuple(int(s) for s in new_shape)
    properties["original_spacing"] = [float(s) for s in original_spacing]
    properties["target_spacing"] = [float(s) for s in params.spacing]
    properties["transpose_forward"] = [int(i) for i in tf]

    # normalization MUST happen before resampling (reference order, PREP-02)
    norm_mask = None
    if any(params.use_mask_for_norm):
        # synthetic seg mask (0 inside the crop, -1 outside) -> mask = seg >= 0
        norm_mask = nonzero_mask[slicer][None]

    for c in range(data.shape[0]):
        channel_mask = norm_mask[c][0] if norm_mask is not None else None
        data[c] = _normalize_channel(data[c], params, c, channel_mask)

    data = _resample_to_shape(data, new_shape, original_spacing, params.spacing, params)
    return data, properties


def to_holoscan_gpu_tensor(tensor: torch.Tensor):
    """Wrap a contiguous CUDA tensor into a zero-copy holoscan ``Tensor``.

    This is the holoscan-cu13 4.2 equivalent of emitting a
    ``MemoryData(DeviceType::GPU)``: the returned tensor shares the CUDA
    buffer with ``tensor`` (DLPack, no copy) and carries
    ``device.device_type == kDLDeviceCUDA``.
    """
    import holoscan.core as hc

    assert_on_gpu(tensor)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    holo_tensor = hc.Tensor.from_dlpack(tensor)
    if holo_tensor.device.device_type != _KDLDEVICE_CUDA:
        raise RuntimeError(
            f"to_holoscan_gpu_tensor: expected a CUDA-backed holoscan Tensor, "
            f"got device_type={holo_tensor.device.device_type}."
        )
    return holo_tensor


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


class PreprocessOperator(Operator):
    """Head operator: in-memory ``Image`` -> nnUNet-preprocessed float32 volume on GPU.

    Named Input:
        image: ``Image`` object produced by ``DICOMSeriesToVolumeOperator``
            (array index order DHW, ``nifti_affine_transform`` in metadata).

    Named Outputs:
        preprocessed: zero-copy GPU tensor (``holoscan.core.Tensor`` backed by
            a CUDA buffer) with the preprocessed float32 volume, shape
            ``(C, H, D, W)`` in nnUNet post-transpose order.
        preprocessed_meta: JSON-serializable dict with the recorded crop bbox,
            pre-crop shape, resampled shape, and spacing/transpose metadata for
            PostResample/Ensemble revert.
    """

    INPUT_IMAGE = "image"
    INPUT_LOWRES_SEG = "lowres_seg"
    OUTPUT_PREPROCESSED = "preprocessed"
    OUTPUT_META = "preprocessed_meta"

    def __init__(
        self,
        fragment: Any,
        *args: Any,
        model_path: Optional[Union[str, Any]] = None,
        config_name: str = "3d_fullres",
        **kwargs: Any,
    ):
        """Create the operator.

        Args:
            fragment: the owning application (passed to ``Operator``).
            model_path: bundle path containing ``jsonpkls/plans.json`` (or a
                ``models/`` folder that does).
            config_name: plans.json configuration key (default ``3d_fullres``).
        """
        # NOTE: holoscan 4.2's Operator.__init__ invokes self.setup(spec)
        # before this constructor body finishes — initialize all state first.
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.model_path = model_path
        self.config_name = config_name
        self._params: Optional[PreprocessParams] = None
        # INFR-02/D-24: shape-keyed CuPy buffer cache (closes the CuPy-side
        # gap — CuPy's LRU pool is INDEPENDENT of RMM, so its blocks bypass
        # the RMM pool entirely). Created BEFORE super().__init__ (Pitfall 7
        # discipline). Per-site cache decision table:
        #   vol upload (cp.array(raw))           UNTOUCHED — pure H2D input transfer
        #   vol transpose materialization (64 MB) CACHED    zero=False (copy overwrites)
        #   mask (uint8, 16 MB)                 CACHED    zero=False (comparison overwrites)
        #   vol_c crop materialization (64 MB)  CACHED    zero=False (copy overwrites)
        #   seg4 G2D copy + cascade seg layout   UNTOUCHED — input transfer; not in D-24 inventory
        #   per-channel normalize temporaries    UNTOUCHED — outside the D-24 inventory (the
        #                                              element-wise writes land in the cached
        #                                              vol_c; the temps ride CuPy's own LRU pool)
        #   one_hot channels (64 MB x labels)    CACHED    zero=False (cp.equal fills each channel)
        #   vol_gpu = cp.array(vol_out)          UNTOUCHED — pure H2D input transfer
        #   vol2 channel concat (128 MB)         CACHED    zero=False (channel copies overwrite)
        # Emit boundary: compute() builds the emitted tensor FRESH from the
        # D2H numpy (torch.as_tensor(...).to(cuda)) — no cached CuPy buffer
        # ever crosses the operator boundary, so no extra boundary copy is
        # needed (DLPack retention rule satisfied by construction).
        # D-13: the scipy CPU resample round-trips are UNTOUCHED (GPUP-01
        # belongs to Phase 3 Plan 04; the OFF path stays Phase 2 behavior).
        self._buf_cache = _ShapeCache("cuda", family="cupy")
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        """Declare the operator's I/O: Image in (plus lowres_seg for cascade
        configs), GPU tensor + meta out."""
        spec.input(self.INPUT_IMAGE)
        if self.model_path is not None:
            # Load params eagerly: the optional cascade input is declared
            # ONLY for cascade configs (plans.json ``previous_stage``).
            # Port discipline (RESEARCH Pitfall 7): a declared input with no
            # flow is a silent hang and a declared output with no receiver
            # is a GXF rejection — for non-cascade configs the lowres_seg
            # port simply does not exist.
            self._load_params()
            if self._params.previous_stage is not None:
                spec.input(self.INPUT_LOWRES_SEG)
        spec.output(self.OUTPUT_PREPROCESSED)
        spec.output(self.OUTPUT_META)

    def _load_params(self) -> PreprocessParams:
        if self._params is None:
            if not self.model_path:
                raise RuntimeError(
                    "PreprocessOperator requires model_path to load plans.json parameters."
                )
            self._params = load_preprocess_params(self.model_path, self.config_name)
            self._logger.info(
                "Loaded preprocessing params for config %s: spacing=%s transpose_forward=%s "
                "normalization=%s",
                self._params.config_name,
                list(self._params.spacing),
                list(self._params.transpose_forward),
                list(self._params.normalization_schemes),
            )
        return self._params

    @staticmethod
    def _derive_spacing_xyz(image: Image) -> Tuple[float, float, float]:
        """Physical (x, y, z) voxel spacing from the series affine.

        Matches the reference derivation: the wrapper reads the ``affine``
        metadata and takes the column norms (LPS physical axes).
        """
        metadata = image.metadata()
        affine = np.asarray(metadata.get("nifti_affine_transform"))
        if affine is None or affine.size == 0 or tuple(affine.shape) != (4, 4):
            raise ValueError(
                "Image metadata is missing a 4x4 'nifti_affine_transform'; "
                "cannot derive voxel spacing."
            )
        spacing_xyz = tuple(float(np.sqrt(np.sum(affine[:3, i] ** 2))) for i in range(3))
        return spacing_xyz

    def preprocess_image(
        self,
        image: Image,
        lowres_seg: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[np.ndarray, "cp.ndarray"], Dict[str, Any]]:
        """Run the CuPy preprocessing path on an in-memory Image.

        Transpose (PREP-01), crop-slice (PREP-04) and element-wise normalize
        (PREP-02) run on GPU (fp32, C-contiguous — D-12); the Z-score/CT
        reductions and the scipy resample (PREP-03) stay on the numpy/scipy
        CPU reference path (D-13). ``preprocess_reference`` (pure CPU) is kept
        in this module as the documented fallback.

        Cascade (PIPE-04, D-09/D-10): when ``lowres_seg`` (a CUDA uint8 3D
        tensor in the SAME array order as ``image.asnumpy()`` — the
        orientation contract set by PostResampleOperator's ``lowres_seg``
        output) is provided, it is cropped with the IMAGE-derived bbox,
        resampled on the CPU reference seg path (``_resample_seg_to_shape``,
        PREP-03/D-13), one-hotted on GPU, and concatenated as extra
        channel(s) AFTER the image channel (reference
        ``np.vstack((data, seg_onehot))`` order) — zero disk I/O. The
        one-hot channel is NEVER normalized. The 1-channel path is
        byte-for-byte the Plan 01 flow.

        Returns:
            ``(volume, properties)`` — the preprocessed volume (CPU float32
            ``(C, H, D, W)`` post-transpose for the 1-channel path; GPU
            float32 ``(1 + n_labels, H, D, W)`` for the cascade path) and
            the recorded properties.
        """
        params = self._load_params()
        arr = np.asarray(image.asnumpy())
        if arr.ndim != 3:
            raise ValueError(f"expected a 3D volume (D, H, W), got shape {arr.shape}")
        spacing_xyz = self._derive_spacing_xyz(image)

        # ------------------------------------------------------------------
        # 1. ONE host->device transfer of the raw volume (D-13: the GPU<->CPU
        #    round-trips around the scipy resample are expected and accepted).
        #    Reference app chain: asnumpy() (DHW) -> .T (WHW->WDH) -> channel-
        #    first -> Transposed([0, 3, 2, 1]) -> (1, D, H, W), cast to
        #    float32. The C-contiguous raw buffer is handed to CuPy (a plain
        #    copy via the CUDA array interface — deliberately NOT
        #    cp.from_dlpack, which transfers buffer ownership) and the
        #    integer->fp32 cast happens on-device (element-wise casts are
        #    bit-identical to numpy).
        # ------------------------------------------------------------------
        raw = np.ascontiguousarray(arr.T, dtype=arr.dtype)
        # INFR-02 site "vol upload": UNTOUCHED — pure H2D input transfer
        # (see the per-site decision table at the cache creation).
        vol_upload = cp.array(raw, dtype=cp.float32)
        # PREP-01: channel-first + (0, 3, 2, 1) transpose, materialized —
        # CuPy transpose returns a view; D-12 requires C-contiguous fp32
        # (Phase 1 measured 1-ulp divergence from a single contiguity slip).
        # INFR-02 site "vol transpose materialization": CACHED, zero=False
        # (the copy fully overwrites the buffer before any read).
        vol_t_shape = (1, *reversed(raw.shape))  # (1, D, H, W) -> (1, W, H, D)
        vol = self._buf_cache.get(vol_t_shape, cp.float32)
        vol[...] = vol_upload[None, ...].transpose(0, 3, 2, 1)

        tf = params.transpose_forward
        # apply transpose_forward as a VIEW (mirrors preprocess_reference;
        # this also needs to be applied to the spacing!)
        vol_t = vol.transpose(0, *[i + 1 for i in tf])
        # Reference stores properties["spacing"] in the reversed (z, y, x) order.
        spacing = tuple(reversed(spacing_xyz))
        original_spacing = [spacing[i] for i in tf]

        # 2. crop, remember the size before cropping! (PREP-04)
        properties: Dict[str, Any] = {}
        properties["shape_before_cropping"] = tuple(int(s) for s in vol_t.shape[1:])

        # The mask MATH stays on the scipy CPU reference path (binary_fill_
        # holes is scipy); only the cheap per-channel OR runs on GPU, then
        # the small uint8 mask round-trips to CPU.
        # INFR-02 site "mask": CACHED, zero=False (the initial comparison
        # fully overwrites the buffer before any read; multi-channel ORs
        # only modify it in place afterwards).
        mask = self._buf_cache.get(vol_t[0].shape, cp.uint8)
        cp.not_equal(vol_t[0], 0, out=mask)
        for c in range(1, vol_t.shape[0]):
            mask |= (vol_t[c] != 0).astype(cp.uint8)
        mask_np = mask.get()
        nonzero_mask = binary_fill_holes(mask_np.astype(bool))
        bbox = _get_bbox_from_mask(nonzero_mask)
        slicer = tuple(slice(*i) for i in bbox)
        # CuPy transpose/slice return views — materialize (D-12).
        # INFR-02 site "vol_c crop materialization": CACHED, zero=False
        # (the crop copy fully overwrites the buffer before any read).
        vol_c_src = vol_t[(slice(None),) + slicer]
        vol_c = self._buf_cache.get(vol_c_src.shape, cp.float32)
        vol_c[...] = vol_c_src
        properties["bbox_used_for_cropping"] = bbox
        properties["shape_after_cropping_and_before_resampling"] = tuple(
            int(s) for s in vol_c.shape[1:]
        )

        # Cascade (PIPE-04): the lowres seg arrives in the same array order
        # as image.asnumpy() (orientation contract), so it takes the EXACT
        # same layout chain as the image raw upload (same orientation =>
        # the image-derived bbox applies to it verbatim).
        seg_c: Optional[Union[np.ndarray, "cp.ndarray"]] = None
        if lowres_seg is not None:
            seg4 = cp.array(lowres_seg)  # uint8 G2G copy via __cuda_array_interface__ (NOT cp.from_dlpack — ownership pitfall)
            seg4 = cp.ascontiguousarray(seg4.transpose(2, 1, 0))
            seg4 = cp.ascontiguousarray(seg4[None, ...].transpose(0, 3, 2, 1))
            seg4_t = seg4.transpose(0, *[i + 1 for i in tf])
            # Crop with the IMAGE-derived bbox (reference
            # DefaultPreprocessor.run_case_npy crops the seg with the
            # image's crop_to_nonzero box; identical image => identical box)
            # — GPU, layout-only, bit-exact on 0/1 labels.
            seg_c = cp.ascontiguousarray(seg4_t[(slice(None),) + slicer])
            # D2H (~16 MB uint8 — D-13 accepted): the seg resample runs on
            # the CPU reference path (PREP-03/D-13).
            seg_c = seg_c.get()

        # resample target shape
        new_shape = _compute_new_shape(vol_c.shape[1:], original_spacing, params.spacing)
        properties["new_shape"] = tuple(int(s) for s in new_shape)
        properties["original_spacing"] = [float(s) for s in original_spacing]
        properties["target_spacing"] = [float(s) for s in params.spacing]
        properties["transpose_forward"] = [int(i) for i in tf]

        # 3. per-channel normalization on GPU (PREP-02) — MUST happen before
        #    resampling (reference order). The mean/std REDUCTIONS stay on the
        #    numpy reference path: CuPy reductions are not bit-identical to
        #    numpy (reduction order differs — RESEARCH Pitfall 4); the element-
        #    wise subtract/divide (and clip) are bit-identical in fp32.
        vol_np = vol_c.get()  # D2H fp32 (~64 MB) — D-13 accepted round-trip
        eps = 1e-8
        norm_mask = None
        if any(params.use_mask_for_norm):
            # synthetic seg mask cropped with the image bbox -> mask = seg >= 0
            norm_mask = nonzero_mask[slicer]
        for c in range(vol_c.shape[0]):
            scheme = params.normalization_schemes[c]
            ch = vol_c[c]
            ch_np = vol_np[c]
            if scheme == "ZScoreNormalization":
                if params.use_mask_for_norm[c] and norm_mask is not None:
                    # Reference: only the masked voxels are normalized.
                    m = norm_mask
                    mean = np.float32(ch_np[m].mean())
                    std = np.float32(ch_np[m].std())
                    m_gpu = cp.asarray(m)
                    ch[m_gpu] = (
                        ch[m_gpu] - cp.asarray(mean)
                    ) / cp.asarray(np.float32(max(std, eps)))
                else:
                    mean = np.float32(ch_np.mean())
                    std = np.float32(ch_np.std())
                    vol_c[c] = (
                        ch - cp.asarray(mean)
                    ) / cp.asarray(np.float32(max(std, eps)))
            elif scheme == "CTNormalization":
                props = params.intensity_properties[str(c)]
                # element-wise clip/sub/divide with the plan constants — the
                # numpy path's weak-scalar casts match np.float32(x) exactly.
                vol_c[c] = cp.clip(
                    ch, props["percentile_00_5"], props["percentile_99_5"]
                )
                vol_c[c] = vol_c[c] - cp.asarray(np.float32(props["mean"]))
                vol_c[c] = vol_c[c] / cp.asarray(np.float32(max(props["std"], eps)))
            elif scheme == "NoNormalization":
                pass
            else:
                raise ValueError(
                    f"unsupported normalization scheme {scheme!r} for channel {c}"
                )
        # Materialize after per-channel writes (D-12) before leaving the GPU.
        vol_c = cp.ascontiguousarray(vol_c)

        # 4. resample (PREP-03) — flag-gated (D-22 / D-22a): flag ON runs the
        #    stock cupyx.scipy.ndimage mirror on GPU (no D2H for the span);
        #    flag OFF (default) is the UNCHANGED scipy/skimage CPU reference
        #    path: device->host of the normalized volume, then the exact
        #    Phase 1 _resample_to_shape. Entering it requires a C-contiguous
        #    fp32 buffer (skimage.resize/map_coordinates are not
        #    layout-invariant at float32 — the Phase 1 1-ulp lesson).
        if gpu_resample_enabled():
            vol_out = _resample_to_shape(
                vol_c, new_shape, original_spacing, params.spacing, params
            )
        else:
            vol_out = np.ascontiguousarray(vol_c.get())  # D2H fp32 — D-13 accepted
            assert vol_out.dtype == np.float32 and vol_out.flags["C_CONTIGUOUS"]
            vol_out = _resample_to_shape(
                vol_out, new_shape, original_spacing, params.spacing, params
            )
        if lowres_seg is None:
            return vol_out, properties

        # ------------------------------------------------------------------
        # 5. cascade 2-channel input (PIPE-04, D-09): seg resample on the
        #    CPU reference path with the cascade seg kwargs, one-hot on
        #    GPU, channel concat with the image volume FIRST (reference
        #    np.vstack((data, seg_onehot)) order). The one-hot channel is
        #    NEVER normalized (the image channel was normalized in step 3
        #    only). Zero disk I/O between the configs.
        # ------------------------------------------------------------------
        seg_r = _resample_seg_to_shape(
            seg_c, new_shape, original_spacing, params.spacing, params
        )
        # One-hot on GPU (bit-exact vs the vendored
        # convert_labelmap_to_one_hot — unit-tested in
        # scripts/test_cascade_config.py::test_one_hot_vs_reference),
        # channels in foreground_labels order. The reference one-hots
        # ``seg[0]`` (the 3D volume — ``seg`` is (1, *spatial)); one-hotting
        # the 4D array would stack to 5D and mismatch the image channels.
        seg3 = cp.array(seg_r)[0]
        # INFR-02 site "one_hot channels": CACHED, zero=False (cp.equal
        # fills every channel before any read). The bool->fp32 out-cast is
        # exact (0/1), so the channel values are identical to the previous
        # (seg3 == lbl).astype(cp.float32) elements; the stack result buffer
        # (64 MB per label set) is the cached one, no per-label temps.
        one_hot = self._buf_cache.get(
            (len(params.foreground_labels), *seg3.shape), cp.float32
        )
        for _i, lbl in enumerate(params.foreground_labels):
            cp.equal(seg3, int(lbl), out=one_hot[_i])
        vol_gpu = cp.array(vol_out)  # image channel(s), already resampled
        # INFR-02 site "vol2 channel concat": CACHED, zero=False (both
        # channel-block copies fully overwrite the buffer before any read;
        # filling the blocks directly also avoids the 128 MB concatenate
        # temporary the previous code allocated).
        vol2 = self._buf_cache.get(
            (vol_gpu.shape[0] + one_hot.shape[0], *seg3.shape), cp.float32
        )
        vol2[: vol_gpu.shape[0]] = vol_gpu
        vol2[vol_gpu.shape[0]:] = one_hot
        return vol2, properties

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        """Main compute: preprocess on the CPU reference path, then GPU handoff."""
        # INFR-005: the NVTX range + timing label carry the config so
        # multi-fragment traces/records stay unambiguous (RESEARCH Pitfall 9).
        with nvtx_range(f"preprocess_{self.config_name}"):
            timing = GpuTiming(f"preprocess_{self.config_name}")
            timing.start()

            # Entry guard: the pipeline is GPU-resident by contract — never
            # silently fall back to CPU (INF-005).
            assert_cuda_available()

            image = op_input.receive(self.INPUT_IMAGE)
            if image is None:
                raise ValueError("PreprocessOperator received no 'image' input.")

            # Study identity for the structured timing records: register it
            # for the fragment so the downstream GPU operators (which receive
            # tensors, not the Image) can include it in their records.
            _meta = image.metadata()
            study = str(_meta.get("StudyInstanceUID") or _meta.get("SeriesInstanceUID") or "unknown")
            set_study_id(self.fragment, study)

            # Cascade (PIPE-04): the optional lowres_seg input is declared
            # only for cascade configs (setup() gates it on
            # params.previous_stage), so it is received only when the port
            # exists — a declared-but-unwired input would hang silently.
            lowres_seg: Optional[torch.Tensor] = None
            params = self._load_params()
            if params.previous_stage is not None:
                seg_ht = op_input.receive(self.INPUT_LOWRES_SEG)
                if seg_ht is None:
                    raise ValueError("PreprocessOperator received no 'lowres_seg' input.")
                seg_tensor = torch.utils.dlpack.from_dlpack(seg_ht)
                # Device invariant at the boundary (INF-005).
                assert_on_gpu(seg_tensor)
                # D-10 (locked anti-decision): the cascade input must be the
                # post-softmax argmax segmentation, NEVER raw probabilities
                # — enforce an integer dtype at the boundary.
                if seg_tensor.is_floating_point():
                    raise ValueError(
                        f"D-10: 'lowres_seg' must be an integer (argmax) segmentation, "
                        f"got floating dtype {seg_tensor.dtype} (probabilities are forbidden)."
                    )
                lowres_seg = seg_tensor

            volume, properties = self.preprocess_image(image, lowres_seg)

            # Move the final float32 tensor to CUDA (1 channel for the
            # non-cascade path — byte-for-byte the Plan 01 flow; 2 channels
            # image+one-hot for cascade configs; SlideWindow is already
            # config-driven off load_inference_params.num_input_channels).
            if isinstance(volume, cp.ndarray):
                volume = volume.get()  # D2H fp32 (cascade 2-channel volume)
            tensor = torch.as_tensor(volume, dtype=torch.float32).to(torch.device("cuda"))

            # Exit guard: the emitted buffer must be CUDA-resident.
            assert_on_gpu(tensor)
            gpu_tensor = to_holoscan_gpu_tensor(tensor)

            op_output.emit(gpu_tensor, self.OUTPUT_PREPROCESSED)
            op_output.emit(properties, self.OUTPUT_META)

            record = timing.stop()
            record["study"] = study
            record["config"] = self.config_name
            record["bbox_used_for_cropping"] = properties["bbox_used_for_cropping"]
            record["shape_before_cropping"] = list(properties["shape_before_cropping"])
            record["new_shape"] = list(properties["new_shape"])
            if lowres_seg is not None:
                record["lowres_seg"] = True
                record["lowres_seg_shape"] = list(lowres_seg.shape)
            StudyTimingCollector.record(self.fragment, record)
            self._logger.info("timing: %s", json.dumps(record))
