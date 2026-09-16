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
from time import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from scipy import ndimage

from monai.deploy.utils.importutil import optional_import

cupy, has_cupy = optional_import("cupy")
cupyx_scipy_ndimage, has_cupyx_scipy = optional_import("cupyx.scipy.ndimage")

from monai.data import MetaTensor
from monai.deploy.core import ConditionType, Fragment, Operator, OperatorSpec
from monai.deploy.core.domain.image import Image


class SegmentationMetricsOperator(Operator):
    """This operator computes segmentation metrics for predicted segmentation masks.

    The computed metrics include volume/area, slice information, pixel counts, and intensity statistics
    for each labeled region in the segmentation mask.

    Named Input:
        segmentation_mask: Segmentation mask as tensor, numpy array, or Image object.
        input_scan: Input scan/image as tensor, or Image object.
        label_dict: Dictionary mapping label names to their corresponding mask indices.
        segmentation_metatensor: Optional MetaTensor version of segmentation mask for GPU processing.
        use_gpu: If True and GPU is available, use CuPy for GPU acceleration.
    Named Output:
        metrics_dict: Dictionary containing metrics for each label.
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        compute_components: bool = True,
        labels_dict: Optional[dict] = None,
        use_gpu: Optional[bool] = True,
        **kwargs,
    ):
        """Create an instance for a containing application object.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            compute_components (bool): If True, computes connected components for each labeled region > 5 pixels
                and outputs in the metrics dictionary. Set to False if not needed. Default is True.
            labels_dict (dict): Dictionary mapping label names to their corresponding mask indices.
                Provide only labels for which metrics are desired.
            use_gpu (bool): If True and GPU is available, use CuPy for GPU acceleration. Default is True.
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.input_name_seg_mask = "segmentation_mask"
        self.input_name_scan = "input_scan"
        self.input_name_labels = labels_dict if labels_dict is not None else {"organ1": 1}

        self.output_name_metrics = "metrics_dict"
        self.use_gpu = use_gpu and has_cupy
        self.compute_components = compute_components

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_seg_mask)
        spec.input(self.input_name_scan)
        spec.output(self.output_name_metrics).condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        """Performs computation for this operator and handles I/O."""

        # Receive inputs
        segmentation_mask = op_input.receive(self.input_name_seg_mask)
        input_scan = op_input.receive(self.input_name_scan)

        # Log type of inputs
        self._logger.info(f"Received segmentation mask of type: {type(segmentation_mask).__name__}")
        self._logger.info(f"Received input scan of type: {type(input_scan).__name__}")

        label_dict = self.input_name_labels

        # Validate inputs
        if label_dict is None or not isinstance(label_dict, dict):
            raise ValueError("label_dict must be a dictionary mapping label names to mask indices")

        # Calculate metrics
        # log if calculate_metrics is using GPU or CPU
        in_gpu = segmentation_mask.device.type == "cuda" if hasattr(segmentation_mask, "device") else False
        backend = "GPU" if self.use_gpu else "CPU"
        self._logger.info(
            f"Calculating segmentation metrics using {backend} backend | "
            f"Segmentation_mask is in GPU already: {in_gpu}."
        )
        # Time the calculate_metrics function
        start_time = time()
        metrics = self.calculate_metrics(segmentation_mask, input_scan, label_dict)
        end_time = time()
        self._logger.info(f"Segmentation metrics calculation took {end_time - start_time:.4f} seconds.")
        # Emit output
        op_output.emit(metrics, self.output_name_metrics)

    def _get_spacing(self, image_obj: Union[torch.Tensor, np.ndarray, Image]) -> Optional[Tuple[float, ...]]:
        """Extract spacing from Image object metadata.

            Args:
                image_obj: Image object that must contain spacing metadata.
        Returns:
            Tuple of spacing values.

        Raises:
            ValueError: If spacing cannot be extracted.
        """
        if not isinstance(image_obj, Image):
            raise ValueError("Spacing required: input must be an Image with metadata containing spacing.")

        # Type of Image object
        self._logger.info(f"Extracting spacing from image metadata for image type: {type(image_obj).__name__} ")
        metadata = image_obj.metadata() or {}

        spacing = None
        if metadata:
            # Try common spacing keys in order of preference
            spacing = metadata.get("spacing") or metadata.get("pixdim") or metadata.get("pixel_spacing")

            # If not found, try DICOM-specific pixel spacing keys
            if spacing is None:
                row_spacing = metadata.get("row_pixel_spacing")
                col_spacing = metadata.get("col_pixel_spacing")
                depth_spacing = metadata.get("depth_pixel_spacing")

                if row_spacing is not None and col_spacing is not None and depth_spacing is not None:
                    spacing = (float(row_spacing), float(col_spacing), float(depth_spacing))

            if spacing is not None and not isinstance(spacing, (list, tuple, np.ndarray)):
                raise ValueError(f"Spacing required: expected list/tuple/ndarray, got {type(spacing).__name__}.")

        if spacing is None:
            affine = getattr(image_obj, "affine", None)
            if affine is not None:
                affine_arr = np.asarray(affine)
                if affine_arr.shape[0] < 3 or affine_arr.shape[1] < 3:
                    raise ValueError("Spacing required: affine matrix missing spatial axes.")
                spacing = (
                    float(np.linalg.norm(affine_arr[:3, 0])),
                    float(np.linalg.norm(affine_arr[:3, 1])),
                    float(np.linalg.norm(affine_arr[:3, 2])),
                )
            else:
                raise ValueError(
                    "Spacing required: metadata missing and affine attribute not available for spacing extraction."
                )

        return tuple(spacing)

    def _compute_volume_or_area(
        self, pixel_count: Union[int, Any], spacing: Optional[Tuple[float, ...]], is_3d: bool, xp: Any
    ) -> float:
        """Compute volume (3D) or area (2D) from pixel count and spacing.

        Args:
            pixel_count: Number of pixels in the mask.
            spacing: Pixel/voxel spacing in mm.
            is_3d: Whether the data is 3D or 2D.
            xp: numpy or cupy module.

        Returns:
            Volume in mL (3D) or area in cm² (2D).
        """
        if spacing is None:
            # Return pixel/voxel count if spacing is not available
            return float(pixel_count)

        if is_3d:
            # Volume = pixel_count * spacing_x * spacing_y * spacing_z (in mm³)
            # Convert mm³ to mL: 1 mL = 1000 mm³
            volume_per_voxel_mm3 = spacing[0] * spacing[1] * spacing[2]
            volume_ml = float(pixel_count * volume_per_voxel_mm3) / 1000.0
            return volume_ml
        else:
            # Area = pixel_count * spacing_x * spacing_y (in mm²)
            # Convert mm² to cm²: 1 cm² = 100 mm²
            area_per_pixel_mm2 = spacing[0] * spacing[1]
            area_cm2 = float(pixel_count * area_per_pixel_mm2) / 100.0
            return area_cm2

    def calculate_metrics(
        self,
        segmentation_mask: Union[Image, MetaTensor],
        input_scan: Image,
        label_dict: Dict[str, int],
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate segmentation metrics for each label.

        Args:
            segmentation_mask: Segmentation mask (Image or MetaTensor).
            input_scan: Input scan/image (Image).
            label_dict: Dictionary mapping label names to mask indices.

        Returns:
            Dictionary with metrics for each label:
                - volume (3D) or area (2D): Volume in mL or area in cm² of the segmented region
                - num.slices: Number of slices containing the organ
                - slice.range: Tuple (first_slice, last_slice) containing the organ
                - pixel.count: Number of pixels/voxels with this label
                - mean.intensity.hu: Mean intensity in HU of pixels in the mask region
                - std.intensity.hu: Standard deviation of intensity in HU in the mask region
        """

        # Get spacing from input scan
        spacing = self._get_spacing(input_scan)

        scan_array = input_scan.asnumpy()

        xp = np  # Default to numpy
        if self.use_gpu and has_cupy:
            xp = cupy

        # Parameter to Determine if 3D or 2D
        is_3d = False

        # Process segmentation mask to array, check if 3D or 2D, and if cupy or numpy array
        if isinstance(segmentation_mask, Image):
            seg_array = segmentation_mask.asnumpy()
            if len(seg_array.shape) == 3:
                is_3d = True
        else:
            try:
                seg_array = xp.asarray(segmentation_mask)
            except Exception as e:
                seg_array = (
                    segmentation_mask.cpu().numpy()
                )  # Fallback to CPU numpy array, applies if self.use_gpu is False or CuPy not available

            seg_array = seg_array[0] if seg_array.shape[0] == 1 else seg_array  # Remove batch dimension if present
            # Align orientation with scan: only transpose if shapes don't already match
            # (e.g., MR MetaTensor in WHD order needs transpose to DHW; CT MetaTensor is already DHW)
            if seg_array.ndim == 3 and seg_array.shape != scan_array.shape:
                if has_cupy and isinstance(seg_array, cupy.ndarray):
                    transposed = cupy.transpose(seg_array, (2, 1, 0))
                else:
                    transposed = np.transpose(seg_array, (2, 1, 0))
                if transposed.shape == scan_array.shape:
                    seg_array = transposed
                    self._logger.info("Transposed segmentation array to match scan orientation (WHD -> DHW).")
            if len(seg_array.shape) == 3:
                is_3d = True

        if (
            self.use_gpu and has_cupy and not isinstance(seg_array, cupy.ndarray)
        ):  # If input segmentation mask is not already on GPU, move it there, applies when input is on CPU
            self._logger.info("Moving segmentation mask from CPU to GPU for processing.")
            seg_array = xp.asarray(seg_array)

        if seg_array.shape != scan_array.shape:
            raise ValueError(
                f"Segmentation shape {seg_array.shape} does not match scan shape {scan_array.shape}. "
                "Inputs must already be spatially aligned before metric computation."
            )

        # Print mean, max min for scan_array for debugging
        self._logger.info(
            f"Input scan array stats - mean: {xp.mean(scan_array):.2f}, "
            f"max: {xp.max(scan_array):.2f}, min: {xp.min(scan_array):.2f}"
        )

        # Initialize results dictionary
        results: Dict[str, Dict[str, Any]] = {}

        # Calculate metrics for each label
        for label_name, label_idx in label_dict.items():
            try:
                label_mask = seg_array == label_idx

                # Pixel count
                pixel_count = xp.sum(label_mask)

                # Skip if label not present
                if pixel_count == 0:
                    results[label_name] = {
                        "volume" if is_3d else "area": 0.0,
                        "num.slices": 0,
                        "slice.range": None,
                        "pixel.count": 0,
                        "mean.intensity.hu": 0.0,
                        "std.intensity.hu": 0.0,
                    }
                    if self.compute_components:
                        results[label_name]["num.connected.components"] = 0
                    continue

                # Compute volume or area
                volume_or_area = self._compute_volume_or_area(pixel_count, spacing, is_3d, xp)

                # Slice information (assumes first dimension is depth/slices for 3D)
                if is_3d:
                    # Find which slices contain the label
                    slices_with_label = xp.any(label_mask, axis=(1, 2))
                    slice_indices = xp.where(slices_with_label)[0]
                    num_slices = len(slice_indices)
                    slice_range = (int(slice_indices[0]), int(slice_indices[-1])) if num_slices > 0 else None
                else:
                    # For 2D, there's only one "slice"
                    num_slices = 1
                    slice_range = (0, 0)

                # Print the device of label_mask using is_cuda
                if has_cupy and isinstance(label_mask, cupy.ndarray):
                    self._logger.info(f"Label mask for {label_name!r} is on GPU.")
                    masked_intensities = scan_array[label_mask.get()]
                else:
                    self._logger.info(f"Label mask for {label_name!r} is on CPU.")
                    masked_intensities = scan_array[label_mask]

                # Intensity statistics (mean and std of pixels within the mask)
                mean_intensity = float(xp.mean(masked_intensities))
                std_intensity = float(xp.std(masked_intensities))

                # Store results for this label
                results[label_name] = {
                    "volume" if is_3d else "area": float(volume_or_area),
                    "num.slices": int(num_slices),
                    "slice.range": slice_range,
                    "pixel.count": int(pixel_count),
                    "mean.intensity.hu": float(mean_intensity),
                    "std.intensity.hu": float(std_intensity),
                }

                # Connected components analysis (> 5 pixels) - optional
                if self.compute_components:
                    num_components = self._count_connected_components(label_mask, min_size=5)
                    results[label_name]["num.connected.components"] = int(num_components)

            except Exception as e:
                self._logger.error(f"Error calculating metrics for label {label_name!r} (index {label_idx}): {e}")
                results[label_name] = {"error": str(e)}

        self._logger.info("Segmentation metrics calculation completed.")
        self._logger.info(f"Metrics results: {results}")

        return results

    def _count_connected_components(self, binary_mask: Union[np.ndarray, Any], min_size: int = 5) -> int:
        """Count connected components with size greater than min_size pixels.

        Connected components analysis is performed on CPU as it's faster than GPU
        for typical medical imaging segmentation tasks.

        Args:
            binary_mask: Binary mask array (numpy or cupy).
            min_size: Minimum component size in pixels to count. Default is 5.

        Returns:
            Number of connected components with size > min_size.
        """
        # Always use CPU for connected components (faster for typical sizes)
        if has_cupy and isinstance(binary_mask, cupy.ndarray):
            binary_mask = cupy.asnumpy(binary_mask)

        # Convert to numpy if it's a different array type
        binary_mask = np.asarray(binary_mask)

        # Label connected components using scipy
        labeled_array, num_features = ndimage.label(binary_mask)

        if num_features == 0:
            return 0

        # Use bincount for efficient counting
        component_sizes = np.bincount(labeled_array.ravel())

        # Skip index 0 (background) and count components > min_size
        num_large_components = int(np.sum(component_sizes[1:] > min_size))

        return num_large_components


def test():
    """Test function for the SegmentationMetricsOperator."""

    import numpy as np

    from monai.deploy.core import Fragment
    from monai.deploy.core.domain.image import Image

    # Create a larger 3D test case for timing comparison
    print("Testing SegmentationMetricsOperator...")
    print("=" * 60)

    # Create synthetic data: 100x100x100 volume for better timing comparison
    rng = np.random.default_rng(42)
    scan_data = rng.random((100, 100, 100)) * 100  # Random intensities 0-100
    seg_data = np.zeros((100, 100, 100), dtype=np.int32)

    # Create multiple labeled regions with various sizes
    seg_data[20:50, 20:80, 20:80] = 1  # Label 1: liver (large region)
    seg_data[60:90, 30:70, 30:70] = 2  # Label 2: spleen (medium region)
    seg_data[10:15, 10:15, 10:15] = 3  # Label 3: kidney (small region)

    # Add some fragmentation to test connected components
    seg_data[25:28, 25:28, 25:28] = 2  # Small isolated spleen fragment
    seg_data[85:88, 85:88, 85:88] = 2  # Another small spleen fragment

    # Create Image objects with spacing metadata
    scan_image = Image(scan_data, metadata={"spacing": [1.0, 1.0, 1.0]})  # 1mm spacing (mL = mm³/1000)
    seg_image = Image(seg_data)

    # Define label dictionary
    label_dict = {
        "liver": 1,
        "spleen": 2,
        "kidney": 3,
    }

    # Helper classes to simulate op_input and op_output for compute()
    class MockOpInput:
        def __init__(self, seg_mask, scan):
            self.data = {"segmentation_mask": seg_mask, "input_scan": scan}

        def receive(self, name):
            return self.data[name]

    class MockOpOutput:
        def __init__(self):
            self.outputs = {}

        def emit(self, value, name):
            self.outputs[name] = value

    # Test: Operator compute() with CPU
    print("\n[Test] Running SegmentationMetricsOperator.compute() with CPU...")
    fragment = Fragment()
    operator = SegmentationMetricsOperator(fragment, use_gpu=False, labels_dict=label_dict)
    op_input = MockOpInput(seg_image, scan_image)
    op_output = MockOpOutput()
    operator.compute(op_input, op_output, context=None)
    metrics = op_output.outputs[operator.output_name_metrics]

    print("\n" + "=" * 60)
    print("Segmentation Metrics Results:")
    print("=" * 60)
    for label_name, label_metrics in metrics.items():
        print(f"\n{label_name}:")
        for metric_name, metric_value in label_metrics.items():
            print(f"  {metric_name}: {metric_value}")
    print("\n" + "=" * 60)
    print("Test completed successfully!")


if __name__ == "__main__":
    test()
