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
    from my_app.operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        assert_on_gpu,
        nvtx_range,
        set_study_id,
    )
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from config import PreprocessParams, load_preprocess_params
    from gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        assert_on_gpu,
        nvtx_range,
        set_study_id,
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
        dtype_out = data.dtype
        data = data.astype(float, copy=False)
        reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
        order = params.resample_order
        order_z = params.resample_order_z
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
        super().__init__(fragment, *args, **kwargs)
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.model_path = model_path
        self.config_name = config_name
        self._params: Optional[PreprocessParams] = None

    def setup(self, spec: OperatorSpec) -> None:
        """Declare the operator's I/O: Image in, GPU tensor + meta out."""
        spec.input(self.INPUT_IMAGE)
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

    def preprocess_image(self, image: Image) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run the CuPy preprocessing path on an in-memory Image.

        Transpose (PREP-01), crop-slice (PREP-04) and element-wise normalize
        (PREP-02) run on GPU (fp32, C-contiguous — D-12); the Z-score/CT
        reductions and the scipy resample (PREP-03) stay on the numpy/scipy
        CPU reference path (D-13). ``preprocess_reference`` (pure CPU) is kept
        in this module as the documented fallback.

        Returns:
            ``(volume, properties)`` — the preprocessed CPU float32 volume
            (shape ``(C, H, D, W)`` post-transpose) and the recorded properties.
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
        vol = cp.array(raw, dtype=cp.float32)
        # PREP-01: channel-first + (0, 3, 2, 1) transpose, materialized —
        # CuPy transpose returns a view; D-12 requires C-contiguous fp32
        # (Phase 1 measured 1-ulp divergence from a single contiguity slip).
        vol = cp.ascontiguousarray(vol[None, ...].transpose(0, 3, 2, 1))

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
        mask = (vol_t[0] != 0).astype(cp.uint8)
        for c in range(1, vol_t.shape[0]):
            mask |= (vol_t[c] != 0).astype(cp.uint8)
        mask_np = mask.get()
        nonzero_mask = binary_fill_holes(mask_np.astype(bool))
        bbox = _get_bbox_from_mask(nonzero_mask)
        slicer = tuple(slice(*i) for i in bbox)
        # CuPy transpose/slice return views — materialize (D-12).
        vol_c = cp.ascontiguousarray(vol_t[(slice(None),) + slicer])
        properties["bbox_used_for_cropping"] = bbox
        properties["shape_after_cropping_and_before_resampling"] = tuple(
            int(s) for s in vol_c.shape[1:]
        )

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

        # 4. resample (PREP-03) on the UNCHANGED scipy/skimage CPU reference
        #    path (D-13): device->host of the normalized volume, then the
        #    exact Phase 1 _resample_to_shape. Entering it requires a
        #    C-contiguous fp32 buffer (skimage.resize/map_coordinates are not
        #    layout-invariant at float32 — the Phase 1 1-ulp lesson).
        vol_out = np.ascontiguousarray(vol_c.get())  # D2H fp32 — D-13 accepted
        assert vol_out.dtype == np.float32 and vol_out.flags["C_CONTIGUOUS"]
        vol_out = _resample_to_shape(
            vol_out, new_shape, original_spacing, params.spacing, params
        )
        return vol_out, properties

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        """Main compute: preprocess on the CPU reference path, then GPU handoff."""
        with nvtx_range("preprocess"):
            timing = GpuTiming("preprocess")
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

            volume, properties = self.preprocess_image(image)

            # Move the final float32 tensor to CUDA (channel count preserved:
            # C=1 for 3D_fullres; C=2 one-hot reserved for cascade in Phase 2).
            tensor = torch.as_tensor(volume, dtype=torch.float32).to(torch.device("cuda"))

            # Exit guard: the emitted buffer must be CUDA-resident.
            assert_on_gpu(tensor)
            gpu_tensor = to_holoscan_gpu_tensor(tensor)

            op_output.emit(gpu_tensor, self.OUTPUT_PREPROCESSED)
            op_output.emit(properties, self.OUTPUT_META)

            record = timing.stop()
            record["study"] = study
            record["bbox_used_for_cropping"] = properties["bbox_used_for_cropping"]
            record["shape_before_cropping"] = list(properties["shape_before_cropping"])
            record["new_shape"] = list(properties["new_shape"])
            StudyTimingCollector.record(self.fragment, record)
            self._logger.info("timing: %s", json.dumps(record))
