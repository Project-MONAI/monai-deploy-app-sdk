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
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from scipy import ndimage

from monai.deploy.utils.importutil import optional_import

cupy, has_cupy = optional_import("cupy")
cupyx_scipy_ndimage, has_cupyx_scipy = optional_import("cupyx.scipy.ndimage")

from monai.deploy.core import ConditionType, Fragment, Operator, OperatorSpec
from monai.deploy.core.domain.image import Image


class SegmentationMetricsOperator(Operator):
    """This operator computes segmentation metrics for predicted segmentation masks.

    The computed metrics include volume/area, slice information, pixel counts, and intensity statistics
    for each labeled region in the segmentation mask.

    Named Input:
        segmentation_mask: Segmentation mask as tensor, numpy array, or Image object.
        input_scan: Input scan/image as tensor, numpy array, or Image object.
        label_dict: Dictionary mapping label names to their corresponding mask indices.
    Named Output:
        metrics_dict: Dictionary containing metrics for each label.
    """

    def __init__(self, fragment: Fragment, *args, use_gpu: bool = True, compute_components: bool = True, **kwargs):
        """Create an instance for a containing application object.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            use_gpu (bool): If True and GPU is available, use CuPy for GPU acceleration. Default is True.
            compute_components (bool): If True, compute connected components. Set to False if not needed. Default is True.
        """

        self.input_name_seg_mask = "segmentation_mask"
        self.input_name_scan = "input_scan"
        self.input_name_labels = "label_dict"
        self.output_name_metrics = "metrics_dict"
        self.use_gpu = use_gpu and has_cupy
        self.compute_components = compute_components
        
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_seg_mask)
        spec.input(self.input_name_scan)
        spec.input(self.input_name_labels)
        spec.output(self.output_name_metrics).condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        """Performs computation for this operator and handles I/O."""

        # Receive inputs
        segmentation_mask = op_input.receive(self.input_name_seg_mask)
        input_scan = op_input.receive(self.input_name_scan)
        label_dict = op_input.receive(self.input_name_labels)

        # Validate inputs
        if label_dict is None or not isinstance(label_dict, dict):
            raise ValueError("label_dict must be a dictionary mapping label names to mask indices")

        # Calculate metrics
        metrics = self.calculate_metrics(segmentation_mask, input_scan, label_dict)
        
        # Emit output
        op_output.emit(metrics, self.output_name_metrics)

    def _to_array(self, data: Union[torch.Tensor, np.ndarray, Image], preserve_gpu: bool = True) -> Union[np.ndarray, "cupy.ndarray"]:
        """Convert input data to numpy or cupy array.
        
        Args:
            data: Input data as tensor, numpy array, or Image object.
            preserve_gpu: If True, keep data on GPU if it's already there. If False, convert to numpy.
            
        Returns:
            Array in numpy or cupy format.
        """
        # Extract array from Image object if needed
        if isinstance(data, Image):
            data = data.asnumpy()
        
        # Convert torch tensor - check if on GPU
        if isinstance(data, torch.Tensor):
            if preserve_gpu and data.is_cuda and has_cupy:
                # Convert CUDA tensor to CuPy array directly
                return cupy.asarray(data.detach())
            else:
                return data.detach().cpu().numpy()
        
        # Check if already a CuPy array
        if has_cupy and isinstance(data, cupy.ndarray):
            if preserve_gpu:
                return data
            else:
                return cupy.asnumpy(data)
        
        # Check if it's numpy array
        if isinstance(data, np.ndarray):
            if preserve_gpu and self.use_gpu and has_cupy:
                # User requested GPU but data is on CPU - transfer
                try:
                    return cupy.asarray(data)
                except Exception as e:
                    logging.warning(f"Failed to convert to cupy array, using numpy: {e}")
                    return data
            return data
        
        # Fallback to numpy
        return np.asarray(data)

    def _get_spacing(self, image_obj: Union[torch.Tensor, np.ndarray, Image]) -> Optional[Tuple[float, ...]]:
        """Extract spacing from Image object metadata.
        
        Args:
            image_obj: Image object that may contain spacing metadata.
            
        Returns:
            Tuple of spacing values or None if not available.
        """
        if isinstance(image_obj, Image):
            metadata = image_obj.metadata()
            if metadata:
                # Try common spacing keys
                spacing = metadata.get("spacing") or metadata.get("pixdim") or metadata.get("pixel_spacing")
                if spacing is not None:
                    return tuple(spacing) if isinstance(spacing, (list, tuple, np.ndarray)) else None
        return None

    def _compute_volume_or_area(
        self, 
        pixel_count: Union[int, "cupy.ndarray"],
        spacing: Optional[Tuple[float, ...]],
        is_3d: bool,
        xp
    ) -> float:
        """Compute volume (3D) or area (2D) from pixel count and spacing.
        
        Args:
            pixel_count: Number of pixels in the mask.
            spacing: Pixel/voxel spacing.
            is_3d: Whether the data is 3D or 2D.
            xp: numpy or cupy module.
            
        Returns:
            Volume in mm³ (3D) or area in mm² (2D).
        """
        if spacing is None:
            # Return pixel/voxel count if spacing is not available
            return float(pixel_count)
        
        if is_3d:
            # Volume = pixel_count * spacing_x * spacing_y * spacing_z
            volume_per_voxel = spacing[0] * spacing[1] * spacing[2]
            return float(pixel_count * volume_per_voxel)
        else:
            # Area = pixel_count * spacing_x * spacing_y
            area_per_pixel = spacing[0] * spacing[1]
            return float(pixel_count * area_per_pixel)

    def calculate_metrics(
        self,
        segmentation_mask: Union[torch.Tensor, np.ndarray, Image],
        input_scan: Union[torch.Tensor, np.ndarray, Image],
        label_dict: Dict[str, int]
    ) -> Dict[str, Dict[str, Union[float, int, Tuple[int, int]]]]:
        """Calculate segmentation metrics for each label.
        
        Args:
            segmentation_mask: Segmentation mask (tensor, numpy array, or Image).
            input_scan: Input scan/image (tensor, numpy array, or Image).
            label_dict: Dictionary mapping label names to mask indices.
            
        Returns:
            Dictionary with metrics for each label:
                - volume_mm3 (3D) or area_mm2 (2D): Volume or area of the segmented region
                - num_slices: Number of slices containing the organ
                - slice_range: Tuple (first_slice, last_slice) containing the organ
                - pixel_count: Number of pixels/voxels with this label
                - mean_intensity: Mean intensity of pixels in the mask region
                - std_intensity: Standard deviation of intensity in the mask region
        """
        # Convert inputs to arrays, preserving GPU location if present
        seg_array = self._to_array(segmentation_mask, preserve_gpu=True)
        scan_array = self._to_array(input_scan, preserve_gpu=True)
        
        # Auto-detect if data is on GPU
        data_on_gpu = has_cupy and isinstance(seg_array, cupy.ndarray)
        
        # Determine which library to use (numpy or cupy) based on actual data location
        xp = cupy if data_on_gpu else np
        
        if data_on_gpu:
            logging.debug("Using GPU for metrics computation (data already on GPU)")
        else:
            logging.debug("Using CPU for metrics computation")
        
        # Get spacing for volume/area calculation
        spacing = self._get_spacing(input_scan)
        
        # Determine if 3D or 2D
        is_3d = len(seg_array.shape) == 3 or (len(seg_array.shape) == 4 and seg_array.shape[0] == 1)
        
        # Remove batch dimension if present
        if len(seg_array.shape) == 4 and seg_array.shape[0] == 1:
            seg_array = seg_array[0]
        if len(scan_array.shape) == 4 and scan_array.shape[0] == 1:
            scan_array = scan_array[0]
        
        # Ensure scan and segmentation have same shape
        if seg_array.shape != scan_array.shape:
            logging.warning(
                f"Segmentation shape {seg_array.shape} doesn't match scan shape {scan_array.shape}. "
                "Attempting to proceed but results may be incorrect."
            )
        
        # Initialize results dictionary
        results = {}
        
        # Calculate metrics for each label
        for label_name, label_idx in label_dict.items():
            try:
                # Create binary mask for this label
                label_mask = (seg_array == label_idx)
                
                # Pixel count
                pixel_count = xp.sum(label_mask)
                
                # Skip if label not present
                if pixel_count == 0:
                    results[label_name] = {
                        "volume_mm3" if is_3d else "area_mm2": 0.0,
                        "num_slices": 0,
                        "slice_range": None,
                        "pixel_count": 0,
                        "mean_intensity": 0.0,
                        "std_intensity": 0.0,
                    }
                    if self.compute_components:
                        results[label_name]["num_connected_components"] = 0
                    continue
                
                # Compute volume or area
                volume_or_area = self._compute_volume_or_area(pixel_count, spacing, is_3d, xp)
                
                # Slice information (assumes first dimension is depth/slices for 3D)
                if is_3d:
                    # Find which slices contain the label
                    slices_with_label = xp.any(label_mask, axis=(1, 2))
                    if self.use_gpu and has_cupy:
                        slices_with_label = cupy.asnumpy(slices_with_label)
                    
                    slice_indices = np.where(slices_with_label)[0]
                    num_slices = len(slice_indices)
                    slice_range = (int(slice_indices[0]), int(slice_indices[-1])) if num_slices > 0 else None
                else:
                    # For 2D, there's only one "slice"
                    num_slices = 1
                    slice_range = (0, 0)
                
                # Intensity statistics (mean and std of pixels within the mask)
                masked_intensities = scan_array[label_mask]
                mean_intensity = float(xp.mean(masked_intensities))
                std_intensity = float(xp.std(masked_intensities))
                
                # Store results for this label
                results[label_name] = {
                    "volume_mm3" if is_3d else "area_mm2": volume_or_area,
                    "num_slices": int(num_slices),
                    "slice_range": slice_range,
                    "pixel_count": int(pixel_count),
                    "mean_intensity": mean_intensity,
                    "std_intensity": std_intensity,
                }
                
                # Connected components analysis (> 5 pixels) - optional
                if self.compute_components:
                    num_components = self._count_connected_components(label_mask, min_size=5)
                    results[label_name]["num_connected_components"] = int(num_components)
                
            except Exception as e:
                logging.error(f"Error calculating metrics for label '{label_name}' (index {label_idx}): {e}")
                results[label_name] = {
                    "error": str(e)
                }
        
        return results

    def _count_connected_components(
        self, 
        binary_mask: Union[np.ndarray, "cupy.ndarray"],
        min_size: int = 5
    ) -> int:
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
    import time
    import numpy as np
    from monai.deploy.core import Fragment
    from monai.deploy.core.domain.image import Image

    # Create a larger 3D test case for timing comparison
    print("Testing SegmentationMetricsOperator...")
    print("=" * 60)
    
    # Create synthetic data: 100x100x100 volume for better timing comparison
    scan_data = np.random.rand(100, 100, 100) * 100  # Random intensities 0-100
    seg_data = np.zeros((100, 100, 100), dtype=np.int32)
    
    # Create multiple labeled regions with various sizes
    seg_data[20:50, 20:80, 20:80] = 1  # Label 1: liver (large region)
    seg_data[60:90, 30:70, 30:70] = 2  # Label 2: spleen (medium region)
    seg_data[10:15, 10:15, 10:15] = 3  # Label 3: kidney (small region)
    
    # Add some fragmentation to test connected components
    seg_data[25:28, 25:28, 25:28] = 2  # Small isolated spleen fragment
    seg_data[85:88, 85:88, 85:88] = 2  # Another small spleen fragment
    
    # Create Image objects with spacing metadata
    scan_image = Image(scan_data, metadata={"spacing": [1.0, 1.0, 1.0]})  # 1mm spacing
    seg_image = Image(seg_data)
    
    # Define label dictionary
    label_dict = {
        "liver": 1,
        "spleen": 2,
        "kidney": 3,
    }
    
    # Test 1: Without GPU (CPU only)
    print("\n[Test 1] Running with CPU (use_gpu=False)...")
    fragment1 = Fragment()
    operator_cpu = SegmentationMetricsOperator(fragment1, use_gpu=False)
    
    start_time = time.time()
    metrics_cpu = operator_cpu.calculate_metrics(seg_image, scan_image, label_dict)
    cpu_time = time.time() - start_time
    
    print(f"CPU Time: {cpu_time:.4f} seconds")
    
    # Test 2: With GPU (if available) - data already on GPU
    print("\n[Test 2] Running with GPU (data already on GPU, with components)...")
    fragment2 = Fragment()
    operator_gpu = SegmentationMetricsOperator(fragment2, use_gpu=True, compute_components=True)
    
    # Check if GPU is actually available
    if has_cupy:
        try:
            import cupy
            print(f"CuPy detected: {cupy.__version__}")
            
            # Transfer data to GPU BEFORE timing
            print("Transferring data to GPU...")
            transfer_start = time.time()
            scan_data_gpu = cupy.asarray(scan_data)
            seg_data_gpu = cupy.asarray(seg_data)
            scan_image_gpu = Image(scan_data_gpu, metadata={"spacing": [1.0, 1.0, 1.0]})
            seg_image_gpu = Image(seg_data_gpu)
            transfer_time = time.time() - transfer_start
            print(f"Transfer Time: {transfer_time:.4f} seconds")
            
            gpu_available = True
        except Exception as e:
            print(f"CuPy not available: {e}")
            gpu_available = False
            scan_image_gpu = scan_image
            seg_image_gpu = seg_image
    else:
        print("CuPy not installed - will use CPU")
        gpu_available = False
        scan_image_gpu = scan_image
        seg_image_gpu = seg_image
    
    # Time only the computation (data already on GPU)
    start_time = time.time()
    metrics_gpu = operator_gpu.calculate_metrics(seg_image_gpu, scan_image_gpu, label_dict)
    gpu_compute_time = time.time() - start_time
    
    if gpu_available:
        print(f"GPU Compute Time (with components): {gpu_compute_time:.4f} seconds")
        speedup = cpu_time / gpu_compute_time
        print(f"Speedup vs CPU: {speedup:.2f}x")
    else:
        print(f"Fallback CPU Time: {gpu_compute_time:.4f} seconds")
        print("(No GPU available, used CPU backend)")
    
    # Test 3: GPU without connected components (pure GPU performance)
    if gpu_available:
        print("\n[Test 3] Running with GPU (without connected components for max speed)...")
        fragment3 = Fragment()
        operator_gpu_fast = SegmentationMetricsOperator(fragment3, use_gpu=True, compute_components=False)
        
        start_time = time.time()
        metrics_gpu_fast = operator_gpu_fast.calculate_metrics(seg_image_gpu, scan_image_gpu, label_dict)
        gpu_fast_time = time.time() - start_time
        
        print(f"GPU Compute Time (no components): {gpu_fast_time:.4f} seconds")
        speedup_fast = cpu_time / gpu_fast_time
        print(f"Speedup vs CPU: {speedup_fast:.2f}x")
        print(f"GPU speedup from skipping components: {gpu_compute_time / gpu_fast_time:.2f}x")
    
    # Display results from GPU run (or CPU fallback)
    print("\n" + "=" * 60)
    print("Segmentation Metrics Results:")
    print("=" * 60)
    for label_name, label_metrics in metrics_gpu.items():
        print(f"\n{label_name}:")
        for metric_name, metric_value in label_metrics.items():
            print(f"  {metric_name}: {metric_value}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")


if __name__ == "__main__":
    test()
