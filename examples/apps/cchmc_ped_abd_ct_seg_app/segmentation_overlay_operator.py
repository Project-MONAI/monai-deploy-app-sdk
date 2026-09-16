# Copyright 2021-2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, List, Union

import matplotlib
import numpy as np
import torch

from monai.deploy.utils.importutil import optional_import

cupy, has_cupy = optional_import("cupy")
cupyx_scipy_ndimage, has_cupyx_scipy = optional_import("cupyx.scipy.ndimage")

from monai.deploy.core import Fragment, Operator, OperatorSpec
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries
from monai.deploy.core.domain.image import Image


class SegmentationOverlayOperator(Operator):
    """
    This operator generates an RGB overlay image by blending a segmentation mask with the corresponding input scan.

    The overlay highlights segmented regions on top of the grayscale or intensity image, using alpha blending for visualization.
    GPU acceleration is used if available and enabled.

    VOI LUT Module tags are extracted from the source DICOM series when available to apply
    appropriate windowing to the Secondary Capture.

    See DICOM PS3.3 C.11.2 for details on VOI LUT Module.

    Windowing behaviour by case:

    1. CT with VOI LUT tags present — WindowCenter/WindowWidth (Hounsfield Units) are read
       from the source series. If the values are identical across all instances (typical for
       CT) a single scalar window is used. If they vary per instance the per-slice path is
       taken (see case 3).

    2. CT with no VOI LUT tags — the class-level soft-tissue HU defaults
       (CENTER=40 HU, WIDTH=400 HU) are applied as a scalar window.

    3. MR (or any non-CT modality) with VOI LUT tags present — per-instance
       WindowCenter/WindowWidth values are collected from every SOP instance, sorted
       ascending by dot(slice_normal, ImagePositionPatient) to match the axis-0 ordering
       produced by DICOMSeriesToVolumeOperator, then inspected for uniformity (±0.5
       threshold). If uniform, a single scalar window is used; if varying, per-slice
       ndarrays of shape (N_instances,) are passed to the windowing path, where they are
       broadcast along the detected slice axis of the image volume.

    4. MR (or any non-CT modality) with no VOI LUT tags — the window is auto-computed
       from the 1st/99th percentile of non-zero pixel values so that arbitrary scanner
       signal units are mapped correctly to the [0, 255] display range.

    Named Input:
        segmentation_mask: Segmentation mask as a tensor, numpy array, or Image object.
        input_scan: Input scan/image as a tensor, numpy array, or Image object.
        study_selected_series_list: The DICOM series from which the segmentation mask was derived.
    Named Output:
        overlay: RGB overlay image (same type as input, typically Image) with segmentation regions highlighted.
    """

    # Default CT soft-tissue display window (Hounsfield Units) - range [-160, 240] HU
    # Used only when the source series modality is CT and no DICOM VOI LUT tags are present
    DEFAULT_WINDOW_CENTER = float(40.0)
    DEFAULT_WINDOW_WIDTH = float(400.0)

    # Default VOI LUT Function - LINEAR (modality agnostic)
    DEFAULT_VOI_LUT_FUNCTION = "LINEAR"

    def __init__(self, fragment: Fragment, *args, use_gpu: bool = True, alpha: float = 0.7, **kwargs):
        """Create an instance for a containing application object.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            use_gpu (bool): If True and GPU is available, use CuPy for GPU acceleration. Default is True.
            alpha (float): Alpha blending factor for overlay (0.0 to 1.0). Default is 0.7.
        """

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.input_name_seg_mask = "segmentation_mask"
        self.input_name_scan = "input_scan"
        self.input_name_study_series = "study_selected_series_list"
        self.output_name_overlay = "overlay"
        self.use_gpu = use_gpu and has_cupy
        self.alpha = alpha
        self.window_center_default = SegmentationOverlayOperator.DEFAULT_WINDOW_CENTER
        self.window_width_default = SegmentationOverlayOperator.DEFAULT_WINDOW_WIDTH
        self.voi_lut_function_default = SegmentationOverlayOperator.DEFAULT_VOI_LUT_FUNCTION

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Set up the named input(s), and output(s) if applicable, aka ports.

        Args:
            spec (OperatorSpec): The Operator specification for inputs and outputs etc.
        """

        spec.input(self.input_name_seg_mask)
        spec.input(self.input_name_scan)
        spec.input(self.input_name_study_series)
        spec.output(self.output_name_overlay)

    def compute(self, op_input, op_output, context):
        """Performs computation for this operator and handles I/O."""

        # Receive inputs
        segmentation_mask = op_input.receive(self.input_name_seg_mask)
        scan = op_input.receive(self.input_name_scan)
        study_selected_series_list = op_input.receive(self.input_name_study_series)

        # Try to extract VOI window & LUT function from the source DICOM series
        # All SOP instances are iterated: per-instance WindowCenter/WindowWidth values
        # are collected, sorted by slice distance (matching DICOMSeriesToVolumeOperator
        # axis-0 order), and returned as scalar floats (uniform series, e.g. CT) or
        # per-slice ndarrays (varying series, e.g. MR). CT soft-tissue HU defaults are
        # used when tags are absent; non-CT modalities fall back to auto-windowing.
        window_center, window_width, voi_lut_function = self._extract_dicom_window(
            study_selected_series_list,
            default_window_center=self.window_center_default,
            default_window_width=self.window_width_default,
            default_voi_lut_function=self.voi_lut_function_default,
        )

        # create overlay
        overlay = self.create_overlay(
            segmentation_mask,
            scan,
            window_center=window_center,
            window_width=window_width,
            voi_lut_function=voi_lut_function,
        )

        # Emit output
        op_output.emit(overlay, self.output_name_overlay)

    def _extract_dicom_window(
        self,
        study_selected_series_list,
        default_window_center: float,
        default_window_width: float,
        default_voi_lut_function: str,
    ):
        """Extract WindowCenter, WindowWidth, and VOILUTFunction from the source DICOM series.

        Iterates every SOP instance in the series, collecting per-instance WindowCenter and
        WindowWidth values. Each instance is assigned a sort key equal to
        dot(slice_normal, ImagePositionPatient) — the exact signed distance used by
        DICOMSeriesToVolumeOperator.prepare_series to order volume slices — so that the
        returned window arrays are in the same axis-0 order as the stacked image volume.

        When multiple preset windows are stored per instance (e.g. soft-tissue and bone
        windows on a CT), the first entry is used, which conventionally is the scanner's
        preferred display window.

        VOILUTFunction is read once from the first instance that carries the tag; it is
        assumed to be constant across a series.

        Args:
            study_selected_series_list: Selected source DICOM series.
            default_window_center: Value used when no WindowCenter tag is found (CT only).
            default_window_width: Value used when no WindowWidth tag is found (CT only).
            default_voi_lut_function: Value used when no VOILUTFunction tag is found.

        Returns:
            Tuple (window_center, window_width, voi_lut_function).
            - CT (or unknown modality) with uniform tags: scalar floats.
            - MR (or any modality) with uniform tags: scalar floats.
            - MR with per-instance varying tags: np.ndarray of shape (N_instances,) for both
              window_center and window_width, sorted in the same axis-0 order as the image
              volume produced by DICOMSeriesToVolumeOperator (ascending dot(slice_normal,
              ImagePositionPatient)).
            - When no tags are found, CT returns HU-based scalar defaults; non-CT returns
              (None, None, ...) so the caller can auto-compute from pixel data.
        """

        modality = None
        voi_fn = default_voi_lut_function
        voi_fn_set = False

        # List of (sort_key, wc, ww) tuples for per-instance collection.
        # sort_key is dot(slice_normal, ImagePositionPatient), mirroring the distance
        # used by DICOMSeriesToVolumeOperator.prepare_series to order volume slices.
        # Falls back to InstanceNumber, then iteration index, if geometry tags are absent.
        per_instance: List = []

        try:
            for study_selected_series in study_selected_series_list or []:
                if not isinstance(study_selected_series, StudySelectedSeries):
                    continue
                for selected_series in study_selected_series.selected_series:
                    for iter_idx, sop_instance in enumerate(selected_series.series.get_sop_instances()):
                        native = sop_instance.get_native_sop_instance()

                        # Read Modality once for fallback decision
                        if modality is None and hasattr(native, "Modality"):
                            modality = str(native.Modality).upper().strip()

                        # VOILUTFunction stays constant across a series; read from first instance
                        if not voi_fn_set and hasattr(native, "VOILUTFunction") and native.VOILUTFunction:
                            voi_fn = str(native.VOILUTFunction).upper().strip()
                            voi_fn_set = True

                        # WindowCenter / WindowWidth may be scalar or a list (multiple preset windows)
                        if hasattr(native, "WindowCenter") and hasattr(native, "WindowWidth"):
                            wc_raw = native.WindowCenter
                            ww_raw = native.WindowWidth
                            # pydicom can return DSfloat, a list, or a MultiValue; take the first window
                            wc = float(wc_raw[0] if hasattr(wc_raw, "__iter__") else wc_raw)
                            ww = float(ww_raw[0] if hasattr(ww_raw, "__iter__") else ww_raw)

                            # Compute sort key matching DICOMSeriesToVolumeOperator.prepare_series:
                            #   slice_normal = cross(row_cosines, col_cosines) from ImageOrientationPatient
                            #   distance     = dot(slice_normal, ImagePositionPatient)
                            sort_key = self._compute_slice_distance(native, fallback=iter_idx)
                            per_instance.append((sort_key, wc, ww))
        except Exception as exc:
            self._logger.warning(f"Could not read VOI LUT module from source DICOM: {exc}")

        if per_instance:
            # Sort ascending by slice distance to match image volume axis-0 ordering
            per_instance.sort(key=lambda x: x[0])
            wc_arr = np.array([v[1] for v in per_instance], dtype=np.float64)
            ww_arr = np.array([v[2] for v in per_instance], dtype=np.float64)

            _units = "HU" if modality == "CT" else "signal units"
            # Detect whether windowing is effectively uniform across the series
            wc_range = wc_arr.max() - wc_arr.min()
            ww_range = ww_arr.max() - ww_arr.min()
            is_uniform = wc_range < 0.5 and ww_range < 0.5

            if is_uniform:
                self._logger.info(
                    f"Uniform VOI LUT across {len(wc_arr)} instances "
                    f"(Modality={modality}): Center={wc_arr[0]:.1f} {_units}, "
                    f"Width={ww_arr[0]:.1f} {_units}, VOILUTFunction={voi_fn}. "
                    f"Uniform windowing will be applied"
                )
                return float(wc_arr[0]), float(ww_arr[0]), voi_fn
            else:
                self._logger.info(
                    f"Non-uniform VOI LUT found across {len(wc_arr)} instances "
                    f"(Modality={modality}, VOILUTFunction={voi_fn}): "
                    f"Center {_units} range [{wc_arr.min():.1f}, {wc_arr.max():.1f}], "
                    f"Width {_units} range [{ww_arr.min():.1f}, {ww_arr.max():.1f}]. "
                    f"Per-slice windowing will be applied (sorted by slice distance)"
                )
                return wc_arr, ww_arr, voi_fn

        # No VOI LUT tags found in source series
        # For CT (or unknown modality), fall back to the class-level HU-based soft-tissue defaults
        # For MR and other non-CT modalities, return None so the caller auto-computes the window
        # from the actual pixel data, since signal values are scanner-dependent
        is_ct = (modality == "CT") if modality is not None else True
        if is_ct:
            self._logger.info(
                f"No VOI LUT DICOM tags found (Modality={modality or 'unknown'}); "
                f"using CT soft tissue defaults: Center={default_window_center:.1f} HU, "
                f"Width={default_window_width:.1f} HU, VOILUTFunction={default_voi_lut_function}"
            )
            return default_window_center, default_window_width, default_voi_lut_function
        else:
            self._logger.info(
                f"No VOI LUT DICOM tags found (Modality={modality}); "
                f"window will be auto-computed from pixel data percentiles"
            )
            return None, None, default_voi_lut_function

    def create_overlay(
        self,
        segmentation_mask: Union[np.ndarray, torch.Tensor, Image],
        scan: Union[np.ndarray, torch.Tensor, Image],
        window_center: Union[float, np.ndarray, None],
        window_width: Union[float, np.ndarray, None],
        voi_lut_function: str,
    ) -> Union[np.ndarray, Image]:
        """Create overlay image from segmentation mask and scan with CuPy acceleration when possible.

        Args:
            segmentation_mask: Segmentation mask as numpy array, torch tensor, or Image object
            scan: Input scan/image as numpy array, torch tensor, or Image object
            window_center: VOI window center - scalar float, per-slice np.ndarray (MR with
                instance-varying tags), or None for non-CT modalities (auto-computed below).
            window_width: VOI window width - scalar float, per-slice np.ndarray (MR with
                instance-varying tags), or None for non-CT modalities (auto-computed below).
            voi_lut_function: VOI LUT function string - from source series or LINEAR default.

        Returns:
            RGB overlay image with the same type as input (numpy array or Image)
        """

        # Handle different input types
        original_type = type(segmentation_mask)
        metadata = None

        if isinstance(segmentation_mask, Image):
            mask_data = segmentation_mask.asnumpy()
            metadata = segmentation_mask.metadata()
        elif isinstance(segmentation_mask, torch.Tensor):
            # Check if tensor is on GPU to avoid unnecessary CPU transfer
            if segmentation_mask.is_cuda and self.use_gpu and has_cupy:
                # Convert directly to CuPy without going through CPU
                mask_data = cupy.asarray(segmentation_mask.detach())
            else:
                # For CPU tensors, convert to numpy (will be transferred to GPU later if use_gpu=True)
                mask_data = segmentation_mask.detach().numpy()
        else:
            # NumPy arrays or other types - will be transferred to GPU later if use_gpu=True
            mask_data = segmentation_mask

        if isinstance(scan, Image):
            scan_data = scan.asnumpy()
        elif isinstance(scan, torch.Tensor):
            # Check if tensor is on GPU to avoid unnecessary CPU transfer
            if scan.is_cuda and self.use_gpu and has_cupy:
                # Convert directly to CuPy without going through CPU
                scan_data = cupy.asarray(scan.detach())
            else:
                # For CPU tensors, convert to numpy (will be transferred to GPU later if use_gpu=True)
                scan_data = scan.detach().numpy()
        else:
            # NumPy arrays or other types - will be transferred to GPU later if use_gpu=True
            scan_data = scan

        # Remove channel dimension if present
        if mask_data.ndim == 4 and mask_data.shape[0] == 1:
            mask_data = mask_data[0]
        if scan_data.ndim == 4 and scan_data.shape[0] == 1:
            scan_data = scan_data[0]

        # Auto-compute window for non-CT modalities when DICOM VOI tags are absent
        if window_center is None or window_width is None:
            _scan_np = (
                cupy.asnumpy(scan_data) if (has_cupy and isinstance(scan_data, cupy.ndarray)) else np.asarray(scan_data)
            )
            window_center, window_width = self._auto_window_from_data(_scan_np)
            self._logger.info(
                f"Auto-computed VOI window from pixel data: Center={window_center:.1f}, "
                f"Width={window_width:.1f} (VOILUTFunction={voi_lut_function})"
            )

        # Check if we should use GPU
        use_cupy = self.use_gpu and has_cupy

        if use_cupy:
            try:
                # Ensure both arrays are on GPU independently
                if not isinstance(mask_data, cupy.ndarray):
                    mask_data = cupy.asarray(mask_data)
                if not isinstance(scan_data, cupy.ndarray):
                    scan_data = cupy.asarray(scan_data)
                overlay_image = self._create_overlay_cupy(
                    scan_data, mask_data, window_center, window_width, voi_lut_function
                )
                # Transfer back to CPU
                overlay_image = cupy.asnumpy(overlay_image)
            except Exception as e:
                self._logger.warning(f"CuPy processing failed, falling back to CPU: {e}")
                if isinstance(mask_data, cupy.ndarray):
                    mask_data = cupy.asnumpy(mask_data)
                if isinstance(scan_data, cupy.ndarray):
                    scan_data = cupy.asnumpy(scan_data)
                overlay_image = self._create_overlay_numpy(
                    scan_data, mask_data, window_center, window_width, voi_lut_function
                )
        else:
            overlay_image = self._create_overlay_numpy(
                scan_data, mask_data, window_center, window_width, voi_lut_function
            )

        # Return in original format
        if original_type == Image:
            return Image(overlay_image, metadata=metadata)
        else:
            return np.asarray(overlay_image)

    def _compute_slice_distance(self, native_ds, fallback: int = 0) -> float:
        """Compute the signed slice distance used by DICOMSeriesToVolumeOperator.prepare_series.

        Replicates the exact formula from prepare_series so that per-instance VOI window arrays
        are sorted in the same axis-0 order as the stacked image volume:

            slice_normal = cross(row_cosines, col_cosines)   # from ImageOrientationPatient
            distance = dot(slice_normal, ImagePositionPatient)

        Args:
            native_ds: pydicom Dataset for one SOP instance.
            fallback: Value returned when required geometry tags are absent.

        Returns:
            Scalar float distance along the slice normal, or the fallback value.
        """
        try:
            iop = native_ds.ImageOrientationPatient  # [rx, ry, rz, cx, cy, cz]
            ipp = native_ds.ImagePositionPatient  # [x, y, z]
            cosines = [float(v) for v in iop]
            pos = [float(v) for v in ipp]
            # Cross product of row and column direction cosines gives the slice normal
            n0 = cosines[1] * cosines[5] - cosines[2] * cosines[4]
            n1 = cosines[2] * cosines[3] - cosines[0] * cosines[5]
            n2 = cosines[0] * cosines[4] - cosines[1] * cosines[3]
            # Dot product with ImagePositionPatient = signed distance from origin
            return n0 * pos[0] + n1 * pos[1] + n2 * pos[2]
        except Exception:
            # Fall back to InstanceNumber if available, else iteration index
            try:
                return float(native_ds.InstanceNumber)
            except Exception:
                return float(fallback)

    def _auto_window_from_data(self, image_data: np.ndarray):
        """Compute a VOI window (center, width) from pixel-data percentiles.

        Used as a fallback when DICOM VOI LUT tags are absent and the modality is not CT.
        MR signal values are scanner- and sequence-dependent; the 1st and 99th percentiles
        of non-zero voxels give a robust tissue-based window without being biased by
        background air / zero-padding voxels.

        Args:
            image_data: N-dimensional image array (any numeric dtype).

        Returns:
            Tuple (window_center, window_width) as floats.
        """
        flat = image_data.ravel().astype(np.float32)
        nonzero = flat[flat > 0]
        if nonzero.size == 0:
            nonzero = flat  # all-zero volume; prevent empty-sequence error
        low = float(np.percentile(nonzero, 1))
        high = float(np.percentile(nonzero, 99))
        center = (low + high) / 2.0
        width = max(high - low, 1.0)  # PS3.3 C.11.2.1.3: width must be > 0
        return center, width

    def _create_overlay_cupy(
        self,
        image_volume: Any,
        label_volume: Any,
        window_center: Union[float, np.ndarray],
        window_width: Union[float, np.ndarray],
        voi_lut_function: str,
    ) -> Any:
        """Create overlay using CuPy for GPU acceleration.

        Args:
            image_volume: Image volume as CuPy array (3D: D, H, W)
            label_volume: Label volume as CuPy array (3D: D, H, W)
            window_center: Scalar or per-slice ndarray of VOI window centers.
            window_width: Scalar or per-slice ndarray of VOI window widths.
            voi_lut_function: VOI LUT function string - from source series or LINEAR default.

        Returns:
            RGB overlay image as CuPy array (3, D, H, W)
        """

        # Log scan range (sampled on CPU to keep it lightweight)
        _s = cupy.asnumpy(image_volume)
        _fn = voi_lut_function.upper().strip()

        if isinstance(window_center, np.ndarray):
            _ww_arr = np.asarray(window_width)
            self._logger.info(
                f"Scan Value Range: [{_s.min():.1f}, {_s.max():.1f}]  "
                f"VOI Window: per-slice ({len(window_center)} slices), "
                f"Center range [{window_center.min():.1f}, {window_center.max():.1f}], "
                f"Width range [{_ww_arr.min():.1f}, {_ww_arr.max():.1f}], "
                f"Function={_fn}"
            )
        else:
            if _fn == "SIGMOID":
                _window_str = f"SIGMOID (no hard clip, inflection={window_center:.1f}, scale={window_width:.1f})"
            elif _fn == "LINEAR_EXACT":
                _low = window_center - (window_width / 2.0)
                _high = window_center + (window_width / 2.0)
                _window_str = f"→ Range [{_low:.1f}, {_high:.1f}]"
            else:  # LINEAR (default)
                _low = window_center - 0.5 - ((window_width - 1) / 2.0)
                _high = window_center - 0.5 + ((window_width - 1) / 2.0)
                _window_str = f"→ Range [{_low:.1f}, {_high:.1f}]"
            self._logger.info(
                f"Scan Value Range: [{_s.min():.1f}, {_s.max():.1f}]  "
                f"VOI Window: Center={window_center:.1f}, Width={window_width:.1f}, "
                f"Function={_fn} {_window_str}"
            )

        del _s

        # Convert image and label to RGB
        image_rgb = self._convert_to_rgb_cupy(image_volume, window_center, window_width, voi_lut_function)
        label_rgb = self._apply_jet_colormap_cupy(label_volume)

        # Create alpha-blended overlay
        overlay = image_rgb.copy()
        mask = label_volume > 0

        # Apply overlay where mask is present
        for i in range(3):  # For each color channel
            overlay[i][mask] = (self.alpha * label_rgb[i][mask] + (1 - self.alpha) * overlay[i][mask]).astype(
                cupy.uint8
            )

        return overlay

    def _create_overlay_numpy(
        self,
        image_volume: np.ndarray,
        label_volume: np.ndarray,
        window_center: Union[float, np.ndarray],
        window_width: Union[float, np.ndarray],
        voi_lut_function: str,
    ) -> np.ndarray:
        """Create overlay using NumPy for CPU processing.

        Args:
            image_volume: Image volume as NumPy array (3D: D, H, W)
            label_volume: Label volume as NumPy array (3D: D, H, W)
            window_center: Scalar or per-slice ndarray of VOI window centers.
            window_width: Scalar or per-slice ndarray of VOI window widths.
            voi_lut_function: VOI LUT function string - from source series or LINEAR default.

        Returns:
            RGB overlay image as NumPy array (3, H, W, D)
        """

        # Log scan range once so the effective window can be verified against actual data
        _fn = voi_lut_function.upper().strip()

        if isinstance(window_center, np.ndarray):
            _ww_arr = np.asarray(window_width)
            self._logger.info(
                f"Scan Value Range: [{image_volume.min():.1f}, {image_volume.max():.1f}]  "
                f"VOI Window: per-slice ({len(window_center)} slices), "
                f"Center range [{window_center.min():.1f}, {window_center.max():.1f}], "
                f"Width range [{_ww_arr.min():.1f}, {_ww_arr.max():.1f}], "
                f"Function={_fn}"
            )
        else:
            if _fn == "SIGMOID":
                _window_str = f"SIGMOID (no hard clip, inflection={window_center:.1f}, scale={window_width:.1f})"
            elif _fn == "LINEAR_EXACT":
                _low = window_center - (window_width / 2.0)
                _high = window_center + (window_width / 2.0)
                _window_str = f"→ Range [{_low:.1f}, {_high:.1f}]"
            else:  # LINEAR (default)
                _low = window_center - 0.5 - ((window_width - 1) / 2.0)
                _high = window_center - 0.5 + ((window_width - 1) / 2.0)
                _window_str = f"→ Range [{_low:.1f}, {_high:.1f}]"
            self._logger.info(
                f"Scan Value Range: [{image_volume.min():.1f}, {image_volume.max():.1f}]  "
                f"VOI Window: Center={window_center:.1f}, Width={window_width:.1f}, "
                f"Function={_fn} {_window_str}"
            )

        # Convert image and label to RGB
        image_rgb = self._convert_to_rgb_numpy(image_volume, window_center, window_width, voi_lut_function)
        label_rgb = self._apply_jet_colormap_numpy(label_volume)

        # Create alpha-blended overlay
        overlay = image_rgb.copy()
        mask = label_volume > 0

        # Ensure shapes match
        if not (overlay.shape[1:] == label_rgb.shape[1:] == mask.shape):
            raise ValueError(
                f"Shape mismatch: overlay {overlay.shape}, label_rgb {label_rgb.shape}, mask {mask.shape}.\n"
                f"image_volume shape: {image_volume.shape}, label_volume shape: {label_volume.shape}"
            )

        # Apply overlay where mask is present
        for i in range(3):  # For each color channel
            overlay[i][mask] = (self.alpha * label_rgb[i][mask] + (1 - self.alpha) * overlay[i][mask]).astype(np.uint8)

        return overlay

    def _apply_jet_colormap_cupy(self, label_volume: Any) -> Any:
        """Apply Jet colormap to label volume using CuPy.

        Args:
            label_volume: 3D label volume (D, H, W)

        Returns:
            RGB label volume (3, D, H, W)
        """

        # Normalize to 0-255 range
        max_val = cupy.max(label_volume)
        if max_val > 0:
            label_normalized = (label_volume / max_val) * 255.0
        else:
            label_normalized = cupy.zeros_like(label_volume, dtype=cupy.float32)

        label_uint8 = label_normalized.astype(cupy.uint8)

        # Apply Jet colormap manually (since matplotlib is CPU-only)
        val = label_uint8.astype(cupy.float32) / 255.0

        # Red channel
        r = cupy.clip((1.5 - cupy.abs(4.0 * val - 3.0)) * 255, 0, 255).astype(cupy.uint8)
        # Green channel
        g = cupy.clip((1.5 - cupy.abs(4.0 * val - 2.0)) * 255, 0, 255).astype(cupy.uint8)
        # Blue channel
        b = cupy.clip((1.5 - cupy.abs(4.0 * val - 1.0)) * 255, 0, 255).astype(cupy.uint8)

        # Stack to create RGB volume (3, D, H, W)
        label_rgb = cupy.stack([r, g, b], axis=0)

        return label_rgb

    def _apply_jet_colormap_numpy(self, label_volume: np.ndarray) -> np.ndarray:
        """Apply Jet colormap to label volume using NumPy.

        Args:
            label_volume: 3D label volume (D, H, W)

        Returns:
            RGB label volume (3, D, H, W)
        """

        # Normalize to 0-255 range
        max_val = np.max(label_volume)
        if max_val > 0:
            label_normalized = (label_volume / max_val) * 255.0
        else:
            label_normalized = np.zeros_like(label_volume, dtype=np.float32)

        label_uint8 = label_normalized.astype(np.uint8)

        # Apply Jet colormap
        jet_colormap = matplotlib.colormaps["jet"].resampled(256)
        label_rgb = np.asarray(jet_colormap(label_uint8))[:, :, :, :3]  # Take only RGB channels

        # Convert to uint8 and rearrange to (3, D, H, W)
        label_rgb = (label_rgb * 255).astype(np.uint8)
        label_rgb = np.transpose(label_rgb, (3, 0, 1, 2))

        return label_rgb

    def _convert_to_rgb_cupy(
        self,
        image_volume: Any,
        window_center: Union[float, np.ndarray],
        window_width: Union[float, np.ndarray],
        voi_lut_function: str,
    ) -> Any:
        """Convert grayscale image to RGB using CuPy with VOI windowing applied.

        Applies the configured VOI window so that pixel values map to the [0, 255] display range,
        preserving clinically relevant contrast. For CT the input values are Hounsfield Units;
        for MR they are scanner signal units. Implements all three DICOM PS3.3 C.11.2 VOI LUT
        Functions (LINEAR, LINEAR_EXACT, SIGMOID).

        For per-slice windowing (MR), the slice axis is detected dynamically as the axis
        whose size matches the number of DICOM instances, so this method handles both
        (D, H, W) and (H, W, D) volume orderings correctly.

        Args:
            image_volume: 3D grayscale image volume (D, H, W from this pipeline)
            window_center: Scalar float (CT / uniform MR) or per-slice ndarray sorted by
                dot(slice_normal, ImagePositionPatient), matching image volume axis-0 order.
            window_width: Scalar float or per-slice ndarray (same ordering as window_center).
            voi_lut_function: VOI LUT function string - from source series or LINEAR default.

        Returns:
            RGB volume (3, D, H, W) with VOI windowing applied
        """

        img = image_volume.astype(cupy.float32)
        fn = voi_lut_function.upper().strip()

        if isinstance(window_center, np.ndarray):
            # Per-slice windowing (common in MR).
            # The scan volume axis order varies (D,H,W) or (H,W,D) depending on the upstream
            # transform pipeline. Detect the slice axis as the one whose size matches the
            # number of DICOM instances. If multiple axes match (or none), fall back to the
            # smallest axis, which is almost always the slice dimension in a medical volume.
            n_win = len(window_center)
            match_axes = [ax for ax, s in enumerate(img.shape) if s == n_win]
            if len(match_axes) == 1:
                slice_axis = match_axes[0]
            else:
                slice_axis = int(np.argmin(img.shape))
            self._logger.debug(f"Per-slice window: {n_win} windows, img shape {img.shape}, slice_axis={slice_axis}")

            wc = window_center.astype(np.float64)
            ww = np.asarray(window_width).astype(np.float64)
            d = img.shape[slice_axis]
            if d != n_win:
                idx = np.linspace(0, n_win - 1, d)
                wc = np.interp(idx, np.arange(n_win), wc)
                ww = np.interp(idx, np.arange(n_win), ww)

            # Build a shape that broadcasts the 1-D window array along slice_axis only
            bc_shape = [1, 1, 1]
            bc_shape[slice_axis] = -1
            wc_gpu = cupy.asarray(wc.reshape(bc_shape).astype(np.float32))
            ww_gpu = cupy.asarray(ww.reshape(bc_shape).astype(np.float32))
            if fn == "SIGMOID":
                image_normalized = 1.0 / (1.0 + cupy.exp(-4.0 * (img - wc_gpu) / ww_gpu))
            elif fn == "LINEAR_EXACT":
                low = wc_gpu - (ww_gpu / 2.0)
                high = wc_gpu + (ww_gpu / 2.0)
                image_normalized = cupy.clip((img - low) / (high - low), 0.0, 1.0)
            else:  # LINEAR
                low = wc_gpu - 0.5 - ((ww_gpu - 1) / 2.0)
                high = wc_gpu - 0.5 + ((ww_gpu - 1) / 2.0)
                image_normalized = cupy.clip((img - low) / (high - low), 0.0, 1.0)
        else:
            # Scalar windowing (CT, or uniform MR)
            if fn == "SIGMOID":
                # PS3.3 C.11.2.1.3.1: y = 1 / (1 + exp(-4*(x-c)/w))
                image_normalized = 1.0 / (1.0 + cupy.exp(-4.0 * (img - window_center) / window_width))
            elif fn == "LINEAR_EXACT":
                # PS3.3 C.11.2.1.3.2: floor = c-w/2, ceiling = c+w/2
                low = float(window_center - (window_width / 2.0))
                high = float(window_center + (window_width / 2.0))
                image_normalized = cupy.clip((img - low) / (high - low), 0.0, 1.0)
            else:  # LINEAR
                # Default when VOILUTFunction tag is absent
                # PS3.3 C.11.2.1.2.1: floor = c - 0.5 - (w-1)/2,  ceiling = c - 0.5 + (w-1)/2
                low = float(window_center - 0.5 - ((window_width - 1) / 2.0))
                high = float(window_center - 0.5 + ((window_width - 1) / 2.0))
                image_normalized = cupy.clip((img - low) / (high - low), 0.0, 1.0)

        # Replicate to 3-channel RGB (3, D, H, W)
        image_rgb = cupy.stack([image_normalized] * 3, axis=0)
        image_rgb = (image_rgb * 255).astype(cupy.uint8)

        return image_rgb

    def _convert_to_rgb_numpy(
        self,
        image_volume: np.ndarray,
        window_center: Union[float, np.ndarray],
        window_width: Union[float, np.ndarray],
        voi_lut_function: str,
    ) -> np.ndarray:
        """Convert grayscale image to RGB using NumPy with VOI windowing applied.

        Applies the configured VOI window so that pixel values map to the [0, 255] display range,
        preserving clinically relevant contrast. For CT the input values are Hounsfield Units;
        for MR they are scanner signal units. Implements all three DICOM PS3.3 C.11.2 VOI LUT
        Functions (LINEAR, LINEAR_EXACT, SIGMOID).

        For per-slice windowing (MR), the slice axis is detected dynamically as the axis
        whose size matches the number of DICOM instances, so this method handles both
        (D, H, W) and (H, W, D) volume orderings correctly.

        Args:
            image_volume: 3D grayscale image volume (D, H, W from this pipeline)
            window_center: Scalar float (CT / uniform MR) or per-slice ndarray sorted by
                dot(slice_normal, ImagePositionPatient), matching image volume axis-0 order.
            window_width: Scalar float or per-slice ndarray (same ordering as window_center).
            voi_lut_function: VOI LUT function string - from source series or LINEAR default.

        Returns:
            RGB volume (3, D, H, W) with VOI windowing applied
        """

        img = image_volume.astype(np.float32)
        fn = voi_lut_function.upper().strip()

        if isinstance(window_center, np.ndarray):
            # Per-slice windowing (common in MR).
            # The scan volume axis order varies (D,H,W) or (H,W,D) depending on the upstream
            # transform pipeline. Detect the slice axis as the one whose size matches the
            # number of DICOM instances. If multiple axes match (or none), fall back to the
            # smallest axis, which is almost always the slice dimension in a medical volume.
            n_win = len(window_center)
            match_axes = [ax for ax, s in enumerate(img.shape) if s == n_win]
            if len(match_axes) == 1:
                slice_axis = match_axes[0]
            else:
                slice_axis = int(np.argmin(img.shape))
            self._logger.debug(f"Per-slice window: {n_win} windows, img shape {img.shape}, slice_axis={slice_axis}")

            wc = window_center.astype(np.float64)
            ww = np.asarray(window_width).astype(np.float64)
            d = img.shape[slice_axis]
            if d != n_win:
                idx = np.linspace(0, n_win - 1, d)
                wc = np.interp(idx, np.arange(n_win), wc)
                ww = np.interp(idx, np.arange(n_win), ww)

            # Build a shape that broadcasts the 1-D window array along slice_axis only
            bc_shape = [1, 1, 1]
            bc_shape[slice_axis] = -1
            wc = wc.reshape(bc_shape).astype(np.float32)
            ww = ww.reshape(bc_shape).astype(np.float32)
            if fn == "SIGMOID":
                image_normalized = 1.0 / (1.0 + np.exp(-4.0 * (img - wc) / ww))
            elif fn == "LINEAR_EXACT":
                low = wc - (ww / 2.0)
                high = wc + (ww / 2.0)
                image_normalized = np.clip((img - low) / (high - low), 0.0, 1.0)
            else:  # LINEAR
                low = wc - 0.5 - ((ww - 1) / 2.0)
                high = wc - 0.5 + ((ww - 1) / 2.0)
                image_normalized = np.clip((img - low) / (high - low), 0.0, 1.0)
        else:
            # Scalar windowing (CT, or uniform MR)
            if fn == "SIGMOID":
                # PS3.3 C.11.2.1.3.1: y = 1 / (1 + exp(-4*(x-c)/w))
                image_normalized = 1.0 / (1.0 + np.exp(-4.0 * (img - window_center) / window_width))
            elif fn == "LINEAR_EXACT":
                # PS3.3 C.11.2.1.3.2: floor = c-w/2, ceiling = c+w/2
                low_s = window_center - (window_width / 2.0)
                high_s = window_center + (window_width / 2.0)
                image_normalized = np.clip((img - low_s) / (high_s - low_s), 0.0, 1.0)
            else:  # LINEAR
                # Default when VOILUTFunction tag is absent
                # PS3.3 C.11.2.1.2.1: floor = c - 0.5 - (w-1)/2,  ceiling = c - 0.5 + (w-1)/2
                low_s = window_center - 0.5 - ((window_width - 1) / 2.0)
                high_s = window_center - 0.5 + ((window_width - 1) / 2.0)
                image_normalized = np.clip((img - low_s) / (high_s - low_s), 0.0, 1.0)

        # Replicate to 3-channel RGB (3, D, H, W)
        image_rgb = np.stack([np.asarray(image_normalized)] * 3, axis=0)
        image_rgb = (image_rgb * 255).astype(np.uint8)

        return image_rgb


def test():
    """Test function for the SegmentationOverlayOperator."""
    import time

    import numpy as np

    from monai.deploy.core import Fragment
    from monai.deploy.core.domain.image import Image

    print("Testing SegmentationOverlayOperator...")
    print("=" * 60)

    # Create synthetic data: 100x100x100 volume
    rng = np.random.default_rng(42)
    scan_data = rng.random((100, 100, 100)) * 100  # Random intensities 0-100 (HU-like)
    seg_data = np.zeros((100, 100, 100), dtype=np.int32)

    # Create labeled regions
    seg_data[20:50, 20:80, 20:80] = 1  # Label 1: liver (large region)
    seg_data[60:90, 30:70, 30:70] = 2  # Label 2: spleen (medium region)
    seg_data[10:15, 10:15, 10:15] = 3  # Label 3: kidney (small region)

    scan_image = Image(scan_data, metadata={"spacing": [1.0, 1.0, 1.0]})
    seg_image = Image(seg_data)

    # Test 1: CPU path with scalar window (CT soft-tissue defaults)
    print("\n[Test 1] Running create_overlay() with CPU, scalar window...")
    fragment1 = Fragment()
    operator_cpu = SegmentationOverlayOperator(fragment1, use_gpu=False, alpha=0.7)
    start_time = time.time()
    overlay_cpu = operator_cpu.create_overlay(
        seg_image,
        scan_image,
        window_center=40.0,
        window_width=400.0,
        voi_lut_function="LINEAR",
    )
    cpu_time = time.time() - start_time
    print(f"CPU Time: {cpu_time:.4f} seconds")
    overlay_arr = overlay_cpu.asnumpy() if isinstance(overlay_cpu, Image) else np.asarray(overlay_cpu)
    print(f"Output shape: {overlay_arr.shape}, dtype: {overlay_arr.dtype}")
    assert overlay_arr.ndim == 4 and overlay_arr.shape[0] == 3, f"Expected (3, D, H, W), got {overlay_arr.shape}"

    # Test 2: CPU path with LINEAR_EXACT
    print("\n[Test 2] Running create_overlay() with CPU, LINEAR_EXACT window...")
    overlay_le = operator_cpu.create_overlay(
        seg_image,
        scan_image,
        window_center=50.0,
        window_width=200.0,
        voi_lut_function="LINEAR_EXACT",
    )
    arr_le = overlay_le.asnumpy() if isinstance(overlay_le, Image) else np.asarray(overlay_le)
    assert arr_le.shape == overlay_arr.shape, "LINEAR_EXACT output shape mismatch"
    print(f"Output shape: {arr_le.shape} — OK")

    # Test 3: CPU path with SIGMOID
    print("\n[Test 3] Running create_overlay() with CPU, SIGMOID window...")
    overlay_sig = operator_cpu.create_overlay(
        seg_image,
        scan_image,
        window_center=50.0,
        window_width=400.0,
        voi_lut_function="SIGMOID",
    )
    arr_sig = overlay_sig.asnumpy() if isinstance(overlay_sig, Image) else np.asarray(overlay_sig)
    assert arr_sig.shape == overlay_arr.shape, "SIGMOID output shape mismatch"
    print(f"Output shape: {arr_sig.shape} — OK")

    # Test 4: CPU path with per-slice window (MR-like)
    print("\n[Test 4] Running create_overlay() with CPU, per-slice window (MR)...")
    n_slices = scan_data.shape[0]
    wc_arr = np.linspace(30.0, 50.0, n_slices)
    ww_arr = np.full(n_slices, 300.0)
    overlay_ps = operator_cpu.create_overlay(
        seg_image,
        scan_image,
        window_center=wc_arr,
        window_width=ww_arr,
        voi_lut_function="LINEAR",
    )
    arr_ps = overlay_ps.asnumpy() if isinstance(overlay_ps, Image) else np.asarray(overlay_ps)
    assert arr_ps.shape == overlay_arr.shape, "Per-slice output shape mismatch"
    print(f"Output shape: {arr_ps.shape} — OK")

    # Test 5: GPU path (if CuPy is available)
    print("\n[Test 5] Running create_overlay() with GPU (if available)...")
    fragment2 = Fragment()
    operator_gpu = SegmentationOverlayOperator(fragment2, use_gpu=True, alpha=0.7)
    if operator_gpu.use_gpu:
        start_time = time.time()
        overlay_gpu = operator_gpu.create_overlay(
            seg_image,
            scan_image,
            window_center=40.0,
            window_width=400.0,
            voi_lut_function="LINEAR",
        )
        gpu_time = time.time() - start_time
        arr_gpu = overlay_gpu.asnumpy() if isinstance(overlay_gpu, Image) else np.asarray(overlay_gpu)
        print(f"GPU Time: {gpu_time:.4f} seconds, Output shape: {arr_gpu.shape}")
        assert arr_gpu.shape == overlay_arr.shape, "GPU output shape mismatch"
        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        print(f"Speedup vs CPU: {speedup:.2f}x")
    else:
        print("CuPy not available — skipping GPU test")

    print("\n" + "=" * 60)
    print("Test completed successfully!")


if __name__ == "__main__":
    test()
