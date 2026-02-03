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


class SegmentationOverlayOperator(Operator):
    """
    This operator generates an RGB overlay image by blending a segmentation mask with the corresponding input scan.

    The overlay highlights segmented regions on top of the grayscale or intensity image, using alpha blending for visualization. GPU acceleration is used if available and enabled.

    Named Input:
        segmentation_mask: Segmentation mask as a tensor, numpy array, or Image object.
        input_scan: Input scan/image as a tensor, numpy array, or Image object.
    Named Output:
        overlay: RGB overlay image (same type as input, typically Image) with segmentation regions highlighted.
    """

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
        
        self.output_name_overlay = "overlay"
        self.use_gpu = use_gpu and has_cupy
        self.alpha = alpha
        
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_seg_mask)
        spec.input(self.input_name_scan)
        spec.output(self.output_name_overlay)

    def compute(self, op_input, op_output, context):
        """Performs computation for this operator and handles I/O."""

        # Receive inputs
        segmentation_mask = op_input.receive(self.input_name_seg_mask)
        scan = op_input.receive(self.input_name_scan)

        # Calculate metrics
        overlay = self.create_overlay(segmentation_mask, scan)
        
        # Emit output
        op_output.emit(overlay, self.output_name_overlay)

    def create_overlay(
        self,
        segmentation_mask: Union[np.ndarray, torch.Tensor, Image],
        scan: Union[np.ndarray, torch.Tensor, Image],
    ) -> Union[np.ndarray, Image]:
        """Create overlay image from segmentation mask and scan with CuPy acceleration when possible.

        Args:
            segmentation_mask: Segmentation mask as numpy array, torch tensor, or Image object
            scan: Input scan/image as numpy array, torch tensor, or Image object

        Returns:
            RGB overlay image with the same type as input (numpy array or Image)
        """
        # Handle different input types
        original_type = type(segmentation_mask)
        metadata = None
        is_on_gpu = False
        
        if isinstance(segmentation_mask, Image):
            mask_data = segmentation_mask.asnumpy()
            metadata = segmentation_mask.metadata()
        elif isinstance(segmentation_mask, torch.Tensor):
            # Check if tensor is on GPU to avoid unnecessary CPU transfer
            if segmentation_mask.is_cuda and self.use_gpu and has_cupy:
                # Convert directly to CuPy without going through CPU
                mask_data = cupy.asarray(segmentation_mask.detach())
                is_on_gpu = True
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
                is_on_gpu = True
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
        
        # Check if we should use GPU
        use_cupy = self.use_gpu and has_cupy
        
        if use_cupy:
            try:
                # Transfer to GPU only if not already there
                if not is_on_gpu:
                    mask_data = cupy.asarray(mask_data)
                    scan_data = cupy.asarray(scan_data)
                overlay_image = self._create_overlay_cupy(scan_data, mask_data)
                # Transfer back to CPU
                overlay_image = cupy.asnumpy(overlay_image)
            except Exception as e:
                self._logger.warning(f"CuPy processing failed, falling back to CPU: {e}")
                if isinstance(mask_data, cupy.ndarray):
                    mask_data = cupy.asnumpy(mask_data)
                if isinstance(scan_data, cupy.ndarray):
                    scan_data = cupy.asnumpy(scan_data)
                overlay_image = self._create_overlay_numpy(scan_data, mask_data)
        else:
            overlay_image = self._create_overlay_numpy(scan_data, mask_data)
        
        # Return in original format
        if original_type == Image:
            return Image(overlay_image, metadata=metadata)
        else:
            return overlay_image

    def _create_overlay_cupy(self, image_volume: "cupy.ndarray", label_volume: "cupy.ndarray") -> "cupy.ndarray":
        """Create overlay using CuPy for GPU acceleration.

        Args:
            image_volume: Image volume as CuPy array (3D: H, W, D)
            label_volume: Label volume as CuPy array (3D: H, W, D)

        Returns:
            RGB overlay image as CuPy array (3, H, W, D)
        """
        # Convert image and label to RGB
        image_rgb = self._convert_to_rgb_cupy(image_volume)
        label_rgb = self._apply_jet_colormap_cupy(label_volume)
        
        # Create alpha-blended overlay
        overlay = image_rgb.copy()
        mask = label_volume > 0
        
        # Apply overlay where mask is present
        for i in range(3):  # For each color channel
            overlay[i][mask] = (self.alpha * label_rgb[i][mask] + (1 - self.alpha) * overlay[i][mask]).astype(cupy.uint8)
        
        return overlay

    def _create_overlay_numpy(self, image_volume: np.ndarray, label_volume: np.ndarray) -> np.ndarray:
        """Create overlay using NumPy for CPU processing.

        Args:
            image_volume: Image volume as NumPy array (3D: H, W, D)
            label_volume: Label volume as NumPy array (3D: H, W, D)

        Returns:
            RGB overlay image as NumPy array (3, H, W, D)
        """
        # Convert image and label to RGB
        image_rgb = self._convert_to_rgb_numpy(image_volume)
        label_rgb = self._apply_jet_colormap_numpy(label_volume)
        
        # Create alpha-blended overlay
        overlay = image_rgb.copy()
        mask = label_volume > 0

        # Ensure shapes match
        if not (overlay.shape[1:] == label_rgb.shape[1:] == mask.shape):
            raise ValueError(f"Shape mismatch: overlay {overlay.shape}, label_rgb {label_rgb.shape}, mask {mask.shape}.\n"
                             f"image_volume shape: {image_volume.shape}, label_volume shape: {label_volume.shape}")
        
        # Apply overlay where mask is present
        for i in range(3):  # For each color channel
            overlay[i][mask] = (self.alpha * label_rgb[i][mask] + (1 - self.alpha) * overlay[i][mask]).astype(np.uint8)
        
        return overlay

    def _apply_jet_colormap_cupy(self, label_volume: "cupy.ndarray") -> "cupy.ndarray":
        """Apply Jet colormap to label volume using CuPy.

        Args:
            label_volume: 3D label volume (H, W, D)

        Returns:
            RGB label volume (3, H, W, D)
        """
        # Normalize to 0-255 range
        max_val = cupy.max(label_volume)
        if max_val > 0:
            label_normalized = (label_volume / max_val) * 255.0
        else:
            label_normalized = cupy.zeros_like(label_volume, dtype=cupy.float32)
        
        label_uint8 = label_normalized.astype(cupy.uint8)
        
        # Apply Jet colormap manually (since matplotlib is CPU-only)
        # Jet colormap approximation: Blue -> Cyan -> Green -> Yellow -> Red
        r = cupy.zeros_like(label_uint8, dtype=cupy.uint8)
        g = cupy.zeros_like(label_uint8, dtype=cupy.uint8)
        b = cupy.zeros_like(label_uint8, dtype=cupy.uint8)
        
        # Simplified Jet colormap
        val = label_uint8.astype(cupy.float32) / 255.0
        
        # Red channel
        r = cupy.clip((1.5 - cupy.abs(4.0 * val - 3.0)) * 255, 0, 255).astype(cupy.uint8)
        # Green channel  
        g = cupy.clip((1.5 - cupy.abs(4.0 * val - 2.0)) * 255, 0, 255).astype(cupy.uint8)
        # Blue channel
        b = cupy.clip((1.5 - cupy.abs(4.0 * val - 1.0)) * 255, 0, 255).astype(cupy.uint8)
        
        # Stack to create RGB volume (3, H, W, D)
        label_rgb = cupy.stack([r, g, b], axis=0)
        
        return label_rgb

    def _apply_jet_colormap_numpy(self, label_volume: np.ndarray) -> np.ndarray:
        """Apply Jet colormap to label volume using NumPy.

        Args:
            label_volume: 3D label volume (H, W, D)

        Returns:
            RGB label volume (3, H, W, D)
        """
        # Import matplotlib colormap
        from matplotlib import cm
        
        # Normalize to 0-255 range
        max_val = np.max(label_volume)
        if max_val > 0:
            label_normalized = (label_volume / max_val) * 255.0
        else:
            label_normalized = np.zeros_like(label_volume, dtype=np.float32)
        
        label_uint8 = label_normalized.astype(np.uint8)
        
        # Apply Jet colormap
        jet_colormap = cm.get_cmap("jet", 256)
        label_rgb = jet_colormap(label_uint8)[:, :, :, :3]  # Take only RGB channels
        
        # Convert to uint8 and rearrange to (3, H, W, D)
        label_rgb = (label_rgb * 255).astype(np.uint8)
        label_rgb = np.transpose(label_rgb, (3, 0, 1, 2))
        
        return label_rgb

    def _convert_to_rgb_cupy(self, image_volume: "cupy.ndarray") -> "cupy.ndarray":
        """Convert grayscale image to RGB using CuPy.

        Args:
            image_volume: 3D grayscale volume (H, W, D)

        Returns:
            RGB volume (3, H, W, D)
        """
        # Normalize to 0-1 range
        min_val = cupy.min(image_volume)
        max_val = cupy.max(image_volume)
        
        if max_val > min_val:
            image_normalized = (image_volume - min_val) / (max_val - min_val)
        else:
            image_normalized = cupy.zeros_like(image_volume, dtype=cupy.float32)
        
        # Stack to create 3-channel RGB
        image_rgb = cupy.stack([image_normalized] * 3, axis=0)
        image_rgb = (image_rgb * 255).astype(cupy.uint8)
        
        return image_rgb

    def _convert_to_rgb_numpy(self, image_volume: np.ndarray) -> np.ndarray:
        """Convert grayscale image to RGB using NumPy.

        Args:
            image_volume: 3D grayscale volume (H, W, D)

        Returns:
            RGB volume (3, H, W, D)
        """
        # Normalize to 0-1 range
        min_val = np.min(image_volume)
        max_val = np.max(image_volume)
        
        if max_val > min_val:
            image_normalized = (image_volume - min_val) / (max_val - min_val)
        else:
            image_normalized = np.zeros_like(image_volume, dtype=np.float32)
        
        # Stack to create 3-channel RGB
        image_rgb = np.stack([image_normalized] * 3, axis=0)
        image_rgb = (image_rgb * 255).astype(np.uint8)
        
        return image_rgb
   

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
    scan_image = Image(scan_data, metadata={"spacing": [1.0, 1.0, 1.0]})  # 1mm spacing (mL = mm³/1000)
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
