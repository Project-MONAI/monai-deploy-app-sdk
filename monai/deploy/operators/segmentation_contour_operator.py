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


import torch
import numpy as np

from monai.deploy.utils.importutil import optional_import

from monai.deploy.core import ConditionType, Fragment, Operator, OperatorSpec
from monai.deploy.core.domain.image import Image
from monai.data import MetaTensor
from monai.transforms import LabelToContour

cupy, has_cupy = optional_import("cupy")

import copy


class SegmentationContourOperator(Operator):
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

    def __init__(self, fragment: Fragment, *args, labels_dict: dict = {'organ1': 1}, **kwargs):
        """Create an instance for a containing application object.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            labels_dict (dict): Dictionary mapping label names to their corresponding mask indices.
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.input_name_seg_mask = "segmentation_mask"
        self.input_labels = labels_dict
        self.output_name_contour = "contour"
        
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_seg_mask)
        spec.output(self.output_name_contour)

    def compute(self, op_input, op_output, context):
        """Performs computation for this operator and handles I/O."""

        # Receive inputs
        segmentation_mask = op_input.receive(self.input_name_seg_mask)
        
        # Validate inputs
        if self.input_labels is None or not isinstance(self.input_labels, dict):
            raise ValueError("label_dict must be a dictionary mapping label names to mask indices")

        # Calculate metrics
        contour = self.create_contour(segmentation_mask, self.input_labels)
        
        # Emit output
        op_output.emit(contour, self.output_name_contour)

    def create_contour(
        self,
        segmentation_mask: Union[np.ndarray, torch.Tensor, Image],
        label_dict: Dict[str, int],
    ) -> Union[np.ndarray, Image]:
        """Create contours from segmentation mask with CuPy acceleration when possible.

        Args:
            segmentation_mask: Segmentation mask as numpy array, torch tensor, or Image object
            label_dict: Dictionary mapping label names to their mask indices

        Returns:
            Contour image with the same type as input (numpy array or Image)
        """
        
        label_image = copy.deepcopy(segmentation_mask)
        metadata = segmentation_mask.meta if isinstance(segmentation_mask, MetaTensor) else segmentation_mask.metadata() if isinstance(segmentation_mask, Image) else None
        
        self._logger.info(f"Segmentation_mask is of type: {type(segmentation_mask)}")
        
        xp = np # Cupy does not work with LabelToContour currently, so we use numpy for now
        if isinstance(label_image,Image):
            label_image = label_image.asnumpy()
            # Transpose DWH to HWD
            if label_image.ndim == 3:
                label_image = np.transpose(label_image, (2,1,0))
            elif label_image.ndim == 2:
                label_image = np.transpose(label_image, (1,0))
            else:
                raise ValueError(f"Unsupported number of dimensions in label image: {label_image.ndim}  Expected 2 or 3.")
            # Unsqueeze to add channel dimension
            self._logger.info(f"Label image shape before expand dims: {label_image.shape}")
            label_image = np.expand_dims(label_image, axis=0)
            self._logger.info(f"Label image shape after transpose and expand dims: {label_image.shape}")
        elif isinstance(label_image, MetaTensor):
            try:
                label_image = xp.asarray(label_image) # Direct conversion to cupy array if possible
            except Exception as e:
                label_image = label_image.cpu().numpy() # Fallback to CPU numpy array - if MetaTensor is on CPU
        
        # Does not apply - using numpy for now
        # Move to GPU if cupy is available and not already a cupy array 
        # if has_cupy and not isinstance(label_image, cupy.ndarray):
        #     label_image = cupy.asarray(label_image)

        self._logger.info(f"Label image shape: {label_image.shape}, dtype: {label_image.dtype}")

        # Initialize the contour image with the same shape as the label image
        contour_image = xp.zeros_like(label_image)

        if label_image.ndim == 4:  # Check if the label image is 4D with a channel dimension
            # Process each 2D slice independently along the last axis (z-axis)
            for i in range(label_image.shape[-1]):
                slice_image = label_image[:, :, :, i]

                # Extract unique labels excluding background (assumed to be 0)
                unique_labels = xp.unique(slice_image)
                unique_labels = unique_labels[unique_labels != 0]

                slice_contour = xp.zeros_like(slice_image)

                # Generate contours for each label in the slice
                for label in unique_labels:
                    # skip contour generation for labels that are not in output_labels
                    if label not in label_dict.values():
                        continue

                    # Create a binary mask for the current label
                    binary_mask = xp.zeros_like(slice_image)
                    binary_mask[slice_image == label] = 1.0

                    # Apply LabelToContour to the 2D slice (replace this with actual contour logic)
                    self._logger.info(f"Shape of binary mask for label {label}: {binary_mask.shape}, dtype: {binary_mask.dtype}")
                    # Squeeze the channel dimension
                    thick_edges = LabelToContour()(binary_mask.astype(xp.float32))

                    # Assign the label value to the contour image at the edge positions
                    slice_contour[thick_edges > 0] = label

                # Stack the processed slice back into the 4D contour image
                contour_image[:, :, :, i] = slice_contour
        else:
            # If the label image is not 4D, process it directly
            unique_labels = xp.unique(label_image)
            unique_labels = unique_labels[unique_labels != 0]

            for label in unique_labels:
                if label not in label_dict.values():
                    continue
                
                binary_mask = xp.zeros_like(label_image)
                binary_mask[label_image == label] = 1.0

                thick_edges = LabelToContour()(binary_mask.astype(xp.float32))
                contour_image[thick_edges > 0] = label
        
        self._logger.info(f"Contour image shape: {contour_image.shape}, dtype: {contour_image.dtype}")
        contour_image = self._MT_array_to_Image(contour_image, metadata)
        
        return contour_image
        
    
    def _MT_array_to_Image(self, out_ndarray, input_img_metadata):
        """
        Converts a MetaTensor or ndarray output to an Image object with correct shape and metadata.
        Squeezes channel dimension, transposes to DHW, and casts to uint8.
        Args:
            out_ndarray: The output array (typically from post-transforms, shape [C, W, H, D] or [1, W, H, D]).
            input_img_metadata: Metadata dictionary for the Image object.
        Returns:
            seg_image: Image object with correct shape and metadata.
        """
        # make sure out_ndarray is a numpy array by converting from cupy if needed
        if has_cupy and isinstance(out_ndarray, cupy.ndarray):
            out_ndarray = cupy.asnumpy(out_ndarray)
    
        # Need to squeeze out the channel dim first
        out_ndarray = np.squeeze(out_ndarray, 0)
        # Transpose to DHW (see note in original code)
        out_ndarray = out_ndarray.T.astype(np.uint8)
        self._logger.info(
            f"Output Seg image numpy array of type {type(out_ndarray)} shape: {out_ndarray.shape}"
        )
        self._logger.info(f"Output Seg image pixel max value: {np.amax(out_ndarray)}")
        seg_image = Image(out_ndarray, input_img_metadata)
        
        return seg_image

