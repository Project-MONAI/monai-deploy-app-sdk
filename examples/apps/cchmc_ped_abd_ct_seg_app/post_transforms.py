# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from typing import List

import matplotlib.cm as cm
import numpy as np

from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import LabelToContour, MapTransform


# Calculate segmentation volumes in ml
class CalculateVolumeFromMaskd(MapTransform):
    """
    Dictionary-based transform to calculate the volume of predicted organ masks.

    Args:
        keys (list): The keys corresponding to the predicted organ masks in the dictionary.
       label_names (list): The list of organ names corresponding to the masks.
    """

    def __init__(self, keys, label_names):
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        super().__init__(keys)
        self.label_names = label_names

    def __call__(self, data):
        # Initialize a dictionary to store the volumes of each organ
        pred_volumes = {}

        for key in self.keys:
            for label_name in self.label_names.keys():
                # self._logger.info('Key: ', key, ' organ_name: ', label_name)
                if label_name != "background":
                    # Get the predicted mask from the dictionary
                    pred_mask = data[key]
                    # Calculate the voxel size in cubic millimeters (voxel size should be in the metadata)
                    # Assuming the metadata contains 'spatial_shape' with voxel dimensions in mm
                    if hasattr(pred_mask, "affine"):
                        voxel_size = np.abs(np.linalg.det(pred_mask.affine[:3, :3]))
                    else:
                        raise ValueError("Affine transformation matrix with voxel spacing information is required.")

                    # Calculate the volume in cubic millimeters
                    label_volume_mm3 = np.sum(pred_mask == self.label_names[label_name]) * voxel_size

                    # Convert to milliliters (1 ml = 1000 mm^3)
                    label_volume_ml = label_volume_mm3 / 1000.0

                    # Store the result in the pred_volumes dictionary
                    # convert to int - radiologists prefer whole number with no decimals
                    pred_volumes[label_name] = int(round(label_volume_ml, 0))

                # Add the calculated volumes to the data dictionary
                key_name = key + "_volumes"

                data[key_name] = pred_volumes
            # self._logger.info('pred_volumes: ', pred_volumes)
        return data


class LabelToContourd(MapTransform):
    def __init__(self, keys: KeysCollection, output_labels: list, allow_missing_keys: bool = False):

        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        super().__init__(keys, allow_missing_keys)

        self.output_labels = output_labels

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label_image = d[key]
            assert isinstance(label_image, MetaTensor), "Input image must be a MetaTensor."

            # Initialize the contour image with the same shape as the label image
            contour_image = np.zeros_like(label_image.cpu().numpy())

            if label_image.ndim == 4:  # Check if the label image is 4D with a channel dimension
                # Process each 2D slice independently along the last axis (z-axis)
                for i in range(label_image.shape[-1]):
                    slice_image = label_image[:, :, :, i].cpu().numpy()

                    # Extract unique labels excluding background (assumed to be 0)
                    unique_labels = np.unique(slice_image)
                    unique_labels = unique_labels[unique_labels != 0]

                    slice_contour = np.zeros_like(slice_image)

                    # Generate contours for each label in the slice
                    for label in unique_labels:
                        # skip contour generation for labels that are not in output_labels
                        if label not in self.output_labels:
                            continue

                        # Create a binary mask for the current label
                        binary_mask = np.zeros_like(slice_image)
                        binary_mask[slice_image == label] = 1.0

                        # Apply LabelToContour to the 2D slice (replace this with actual contour logic)
                        thick_edges = LabelToContour()(binary_mask)

                        # Assign the label value to the contour image at the edge positions
                        slice_contour[thick_edges > 0] = label

                    # Stack the processed slice back into the 4D contour image
                    contour_image[:, :, :, i] = slice_contour
            else:
                # If the label image is not 4D, process it directly
                slice_image = label_image.cpu().numpy()
                unique_labels = np.unique(slice_image)
                unique_labels = unique_labels[unique_labels != 0]

                for label in unique_labels:
                    binary_mask = np.zeros_like(slice_image)
                    binary_mask[slice_image == label] = 1.0

                    thick_edges = LabelToContour()(binary_mask)
                    contour_image[thick_edges > 0] = label

            # Convert the contour image back to a MetaTensor with the original metadata
            contour_image_meta = MetaTensor(contour_image, meta=label_image.meta)  # , affine=label_image.affine)

            # Store the contour MetaTensor in the output dictionary
            d[key] = contour_image_meta

        return d


class OverlayImageLabeld(MapTransform):
    def __init__(
        self,
        image_key: KeysCollection,
        label_key: str,
        overlay_key: str = "overlay",
        alpha: float = 0.7,
        allow_missing_keys: bool = False,
    ):

        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        super().__init__(image_key, allow_missing_keys)

        self.image_key = image_key
        self.label_key = label_key
        self.overlay_key = overlay_key
        self.alpha = alpha
        self.jet_colormap = cm.get_cmap("jet", 256)  # Get the Jet colormap with 256 discrete colors

    def apply_jet_colormap(self, label_volume):
        """
        Apply the Jet colormap to a 3D label volume using matplotlib's colormap.
        """
        assert label_volume.ndim == 3, "Label volume should have 3 dimensions (H, W, D) after removing channel."

        label_volume_normalized = (label_volume / label_volume.max()) * 255.0
        label_volume_uint8 = label_volume_normalized.astype(np.uint8)

        # Apply the colormap to each label
        label_rgb = self.jet_colormap(label_volume_uint8)[:, :, :, :3]  # Only take the RGB channels

        label_rgb = (label_rgb * 255).astype(np.uint8)
        # Rearrange axes to get (3, H, W, D)
        label_rgb = np.transpose(label_rgb, (3, 0, 1, 2))

        assert label_rgb.shape == (
            3,
            *label_volume.shape,
        ), f"Label RGB shape should be (3,H, W, D) but got {label_rgb.shape}"

        return label_rgb

    def convert_to_rgb(self, image_volume):
        """
        Convert a single-channel grayscale 3D image to an RGB 3D image.
        """
        assert image_volume.ndim == 3, "Image volume should have 3 dimensions (H, W, D) after removing channel."

        image_volume_normalized = (image_volume - image_volume.min()) / (image_volume.max() - image_volume.min())
        image_rgb = np.stack([image_volume_normalized] * 3, axis=0)
        image_rgb = (image_rgb * 255).astype(np.uint8)

        assert image_rgb.shape == (
            3,
            *image_volume.shape,
        ), f"Image RGB shape should be (3,H, W, D) but got {image_rgb.shape}"

        return image_rgb

    def _create_overlay(self, image_volume, label_volume):
        # Convert the image volume and label volume to RGB
        image_rgb = self.convert_to_rgb(image_volume)
        label_rgb = self.apply_jet_colormap(label_volume)

        # Create an alpha-blended overlay
        overlay = image_rgb.copy()
        mask = label_volume > 0

        # Apply the overlay where the mask is present
        for i in range(3):  # For each color channel
            overlay[i, mask] = (self.alpha * label_rgb[i, mask] + (1 - self.alpha) * overlay[i, mask]).astype(np.uint8)

        assert (
            overlay.shape == image_rgb.shape
        ), f"Overlay shape should match image RGB shape: {overlay.shape} vs {image_rgb.shape}"

        return overlay

    def __call__(self, data):
        d = dict(data)

        # Get the image and label tensors
        image = d[self.image_key]  # Expecting shape (1, H, W, D)
        label = d[self.label_key]  # Expecting shape (1, H, W, D)

        # uncomment when running pipeline with mask (non-contour) outputs, i.e. LabelToContourd transform absent
        # if image.device.type == "cuda":
        #     image = image.cpu()
        #     d[self.image_key] = image
        # if label.device.type == "cuda":
        #     label = label.cpu()
        #     d[self.label_key] = label
        # # -----------------------

        # Ensure that the input has the correct dimensions
        assert image.shape[0] == 1 and label.shape[0] == 1, "Image and label must have a channel dimension of 1."
        assert image.shape == label.shape, f"Image and label must have the same shape: {image.shape} vs {label.shape}"

        # Remove the channel dimension for processing
        image_volume = image[0]  # Shape: (H, W, D)
        label_volume = label[0]  # Shape: (H, W, D)

        # Convert to 3D overlay
        overlay = self._create_overlay(image_volume, label_volume)

        # Add the channel dimension back
        # d[self.overlay_key] = np.expand_dims(overlay, axis=0)  # Shape: (1, H, W, D, 3)
        d[self.overlay_key] = MetaTensor(overlay, meta=label.meta, affine=label.affine)  # Shape: (3, H, W, D)

        # Assert the final output shape
        # assert d[self.overlay_key].shape == (1, *image_volume.shape, 3), \
        #     f"Final overlay shape should be (1, H, W, D, 3) but got {d[self.overlay_key].shape}"

        assert d[self.overlay_key].shape == (
            3,
            *image_volume.shape,
        ), f"Final overlay shape should be (3, H, W, D) but got {d[self.overlay_key].shape}"

        # Log the overlay creation (debugging)
        self._logger.info(f"Overlay created with shape: {overlay.shape}")
        # self._logger.info(f"Dictionary keys: {d.keys()}")

        # self._logger.info('overlay_image shape: ',  d[self.overlay_key].shape)
        return d


class SaveData(MapTransform):
    """
    Save the output dictionary into JSON files.

    The name of the saved file will be `{key}_{output_postfix}.json`.

    Args:
        keys: keys of the corresponding items to be saved in the dictionary.
        output_dir: directory to save the output files.
        output_postfix: a string appended to all output file names, default is `data`.
        separate_folder: whether to save each file in a separate folder. Default is `True`.
        print_log: whether to print logs when saving. Default is `True`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        namekey: str = "image",
        output_dir: str = "./",
        output_postfix: str = "data",
        separate_folder: bool = False,
        print_log: bool = True,
        allow_missing_keys: bool = False,
    ):
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        super().__init__(keys, allow_missing_keys)
        self.output_dir = output_dir
        self.output_postfix = output_postfix
        self.separate_folder = separate_folder
        self.print_log = print_log
        self.namekey = namekey

    def __call__(self, data):
        d = dict(data)
        image_name = os.path.basename(d[self.namekey].meta["filename_or_obj"]).split(".")[0]
        for key in self.keys:
            # Get the data
            output_data = d[key]

            # Determine the file name
            file_name = f"{image_name}_{self.output_postfix}.json"
            if self.separate_folder:
                file_path = os.path.join(self.output_dir, image_name, file_name)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            else:
                file_path = os.path.join(self.output_dir, file_name)

            # Save the dictionary as a JSON file
            with open(file_path, "w") as f:
                json.dump(output_data, f)

            if self.print_log:
                self._logger.info(f"Saved data to {file_path}")

        return d


# custom transform (not in original post_transforms.py in bundle):
class ExtractVolumeToTextd(MapTransform):
    """
    Custom transform to extract volume information from the segmentation results and format it as a textual summary.
    Filters organ volumes based on output_labels for DICOM SR write, while including all organs for MongoDB write.
    The upstream CalculateVolumeFromMaskd transform calculates organ volumes and stores them in the dictionary
    under the pred_key + '_volumes' key. The input dictionary is outputted unchanged as to not affect downstream operators.

    Args:
        keys: keys of the corresponding items to be saved in the dictionary.
        label_names: dictionary mapping organ names to their corresponding label indices.
        output_labels: list of target label indices for organs to include in the DICOM SR output.
    """

    def __init__(
        self,
        keys: KeysCollection,
        label_names: dict,
        output_labels: List[int],
        allow_missing_keys: bool = False,
    ):
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        super().__init__(keys, allow_missing_keys)

        self.label_names = label_names
        self.output_labels = output_labels

        # create separate result_texts for DICOM SR write (target organs) and MongoDB write (all organs)
        self.result_text_dicom_sr: str = ""
        self.result_text_mongodb: str = ""

    def __call__(self, data):
        d = dict(data)
        # use the first key in `keys` to access the volume data (e.g., pred_key + '_volumes')
        volumes_key = self.keys[0]
        organ_volumes = d.get(volumes_key, None)

        if organ_volumes is None:
            raise ValueError(f"Volume data not found for key {volumes_key}.")

        # create the volume text outputs
        volume_text_dicom_sr = []
        volume_text_mongodb = []

        # loop through calculated organ volumes
        for organ, volume in organ_volumes.items():

            # append all organ volumes for MongoDB entry
            volume_entry = f"{organ.capitalize()} Volume: {volume} mL"
            volume_text_mongodb.append(volume_entry)

            # if the organ's label index is in output_labels
            label_index = self.label_names.get(organ, None)
            if label_index in self.output_labels:
                # append organ volume for DICOM SR entry
                volume_text_dicom_sr.append(volume_entry)

        self.result_text_dicom_sr = "\n".join(volume_text_dicom_sr)
        self.result_text_mongodb = "\n".join(volume_text_mongodb)

        # not adding result_text to dictionary; return dictionary unchanged as to not affect downstream operators
        return d
