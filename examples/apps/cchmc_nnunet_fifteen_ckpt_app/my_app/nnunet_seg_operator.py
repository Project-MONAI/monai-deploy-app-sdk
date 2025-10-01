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

import logging
from pathlib import Path
from typing import Dict, List

import torch
from numpy import int16, uint8

# Import custom transforms
from post_transforms import CalculateVolumeFromMaskd, ExtractVolumeToTextd, LabelToContourd, OverlayImageLabeld

# Import from MONAI deploy
from monai.deploy.utils.importutil import optional_import

Dataset, _ = optional_import("monai.data", name="Dataset")
DataLoader, _ = optional_import("monai.data", name="DataLoader")
import os

# Try importing from local version first, then fall back to MONAI if not available
# This approach works regardless of how the file is imported (as module or script)
import sys

# Add current directory to path to ensure the local module is found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # Try local version first
    from nnunet_bundle import EnsembleProbabilitiesToSegmentation, get_nnunet_monai_predictors_for_ensemble
except ImportError:
    # Fall back to MONAI version if local version fails
    from monai.apps.nnunet.nnunet_bundle import (
        get_nnunet_monai_predictors_for_ensemble,
        EnsembleProbabilitiesToSegmentation,
    )

from monai.deploy.core import AppContext, Fragment, Model, Operator, OperatorSpec
from monai.deploy.operators.monai_seg_inference_operator import InMemImageReader

# Import MONAI transforms
from monai.transforms import Compose, KeepLargestConnectedComponentd, Lambdad, LoadImaged, SaveImaged, Transposed


class NNUnetSegOperator(Operator):
    """
    Operator that performs segmentation inference with nnU-Net ensemble models.

    This operator loads and runs multiple nnU-Net models in an ensemble fashion,
    processes the results, and outputs segmentation masks, volume measurements,
    and visualization overlays.
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        app_context: AppContext,
        model_path: Path,
        output_folder: Path = Path.cwd() / "output",
        output_labels: List[int] = None,
        model_list: List[str] = None,
        model_name: str = "best_model.pt",
        save_probabilities: bool = False,
        save_files: bool = False,
        **kwargs,
    ):
        """
        Initialize the nnU-Net segmentation operator.

        Args:
            fragment: The fragment this operator belongs to
            app_context: The application context
            model_path: Path to the nnU-Net model directory
            output_folder: Directory to save output files
            output_labels: List of label indices to include in outputs
            model_list: List of nnU-Net model types to use in ensemble
            model_name: Name of the model checkpoint file
            save_probabilities: Whether to save probability maps
            save_files: Whether to save intermediate files
        """
        # Initialize logger
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")

        # Set up data keys
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"

        # Model configuration
        self.model_path = self._find_model_file_path(model_path)
        self.model_list = model_list or ["3d_fullres", "3d_lowres", "3d_cascade_fullres"]
        self.model_name = model_name
        self.save_probabilities = save_probabilities
        self.save_files = save_files
        self.prediction_keys = [f"pred_{model}" for model in self.model_list]

        # Output configuration
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.output_labels = output_labels if output_labels is not None else [1]

        # Store app context
        self.app_context = app_context

        # I/O names for operator
        self.input_name_image = "image"
        self.output_name_seg = "seg_image"
        self.output_name_text = "result_text"
        self.output_name_sc_path = "dicom_sc_dir"

        # Call parent constructor
        super().__init__(fragment, *args, **kwargs)

    def _find_model_file_path(self, model_path: Path) -> Path:
        """
        Validates and returns the model directory path.

        Args:
            model_path: Path to the model directory

        Returns:
            Validated Path object to the model directory

        Raises:
            ValueError: If model_path is invalid or doesn't exist
        """
        # When executing as MAP, model_path is typically a directory (/opt/holoscan/models)
        # nnU-Net expects a directory structure with model subdirectories
        if not model_path:
            raise ValueError("Model path not provided")

        if not model_path.is_dir():
            raise ValueError(f"Model path should be a directory, got: {model_path}")

        return model_path

    def _load_nnunet_models(self):
        """
        Loads nnU-Net ensemble models using MONAI's nnU-Net bundle functionality
        and registers them in the app_context.

        Raises:
            RuntimeError: If model loading fails
        """
        # Determine device based on availability
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._logger.info(f"Loading nnU-Net ensemble models from: {self.model_path} on {_device}")

        try:
            # Get nnU-Net ensemble predictors (returns tuple of ModelnnUNetWrapper objects)
            network_def = get_nnunet_monai_predictors_for_ensemble(
                model_list=self.model_list, model_path=str(self.model_path), model_name=self.model_name
            )

            # Move models to device and set to evaluation mode
            ensemble_predictors = []
            for predictor in network_def:
                predictor.to(_device)
                predictor.eval()
                ensemble_predictors.append(predictor)

            # Create a MONAI Model object to encapsulate the ensemble
            loaded_model = Model(self.model_path, name="nnunet_ensemble")
            loaded_model.predictor = ensemble_predictors

            # Register the loaded Model object in the application context
            self.app_context.models = loaded_model

            self._logger.info(f"Successfully loaded {len(ensemble_predictors)} nnU-Net models: {self.model_list}")

        except Exception as e:
            self._logger.error(f"Failed to load nnU-Net models: {str(e)}")
            raise

    def setup(self, spec: OperatorSpec):
        """
        Sets up the operator by configuring input and output specifications.

        Args:
            spec: The operator specification to configure
        """
        # Define input - expects a DICOM image
        spec.input(self.input_name_image)

        # Define outputs:
        # 1. Segmentation output (for DICOM SEG)
        spec.output(self.output_name_seg)

        # 2. Measurement results text (for DICOM SR)
        spec.output(self.output_name_text)

        # 3. Directory path for visualization overlays (for DICOM SC)
        spec.output(self.output_name_sc_path)

    def _convert_dicom_metadata_datatype(self, metadata: Dict) -> Dict:
        """
        Converts pydicom-specific metadata types to corresponding native Python types.

        This addresses an issue with pydicom types in metadata for images converted from DICOM series.
        Reference issue: https://github.com/Project-MONAI/monai-deploy-app-sdk/issues/185

        Args:
            metadata: Dictionary containing image metadata

        Returns:
            Dictionary with converted metadata types
        """
        if not metadata:
            return metadata

        # Convert known metadata attributes to appropriate Python types
        known_conversions = {"SeriesInstanceUID": str, "row_pixel_spacing": float, "col_pixel_spacing": float}

        for key, conversion_func in known_conversions.items():
            if key in metadata:
                try:
                    metadata[key] = conversion_func(metadata[key])
                except Exception:
                    self._logger.warning(f"Failed to convert {key} to {conversion_func.__name__}")

        # Log converted metadata at debug level
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug("Converted Image object metadata:")
            for k, v in metadata.items():
                self._logger.debug(f"{k}: {v}, type {type(v)}")

        return metadata

    def compute(self, op_input, op_output, context):
        """
        Main compute method that processes input, runs inference, and emits outputs.
        """
        # Get input image
        input_image = op_input.receive(self.input_name_image)
        if not input_image:
            raise ValueError("Input image is not found.")

        # Load nnU-Net ensemble models
        self._logger.info("Loading nnU-Net ensemble models")
        self._load_nnunet_models()

        # Perform inference using our custom implementation
        data_dict = self.compute_impl(input_image, context)[0]

        # Squeeze the batch dimension
        data_dict[self._pred_dataset_key] = data_dict[self._pred_dataset_key].squeeze(0)
        data_dict[self._input_dataset_key] = data_dict[self._input_dataset_key].squeeze(0)

        # Squeeze the batch dimension of affine meta data
        data_dict[self._pred_dataset_key].affine = data_dict[self._pred_dataset_key].affine.squeeze(0)
        data_dict[self._input_dataset_key].affine = data_dict[self._input_dataset_key].affine.squeeze(0)

        # Log shape information
        self._logger.info(f"Segmentation prediction shape: {data_dict[self._pred_dataset_key].shape}")
        self._logger.info(f"Segmentation image shape: {data_dict[self._input_dataset_key].shape}")

        # Get post transforms for MAP outputs
        post_transforms = self.post_process_stage2()

        # Apply postprocessing transforms for MAP outputs
        data_dict = post_transforms(data_dict)

        self._logger.info(
            f"Segmentation prediction shape after post processing: {data_dict[self._pred_dataset_key].shape}"
        )

        # DICOM SEG output
        op_output.emit(data_dict[self._pred_dataset_key].squeeze(0).numpy().astype(uint8), self.output_name_seg)

        # DICOM SR output - extract result text
        result_text = self.get_result_text_from_transforms(post_transforms)
        if not result_text:
            raise ValueError("Result text could not be generated.")

        self._logger.info(f"Calculated Organ Volumes: {result_text}")
        op_output.emit(result_text, self.output_name_text)

        # DICOM SC output
        dicom_sc_dir = self.output_folder / "temp"
        self._logger.info(f"Temporary DICOM SC saved at: {dicom_sc_dir}")
        op_output.emit(dicom_sc_dir, self.output_name_sc_path)

    def pre_process(self, img_reader) -> Compose:
        """Composes transforms for preprocessing the input image before predicting on nnU-Net models."""
        my_key = self._input_dataset_key

        return Compose(
            [
                LoadImaged(keys=my_key, reader=img_reader, ensure_channel_first=True),
                Transposed(keys=my_key, indices=[0, 3, 2, 1]),
            ]
        )

    def compute_impl(self, input_image, context) -> List[Dict]:
        """
        Performs the actual nnU-Net ensemble inference using ModelnnUNetWrapper.
        This function handles the complete inference pipeline including preprocessing,
        ensemble prediction, and postprocessing.
        """

        if not input_image:
            raise ValueError("Input is None.")

        # Need to try to convert the data type of a few metadata attributes.
        # input_img_metadata = self._convert_dicom_metadata_datatype(input_image.metadata())
        # Need to give a name to the image as in-mem Image obj has no name.
        img_name = "Img_in_context"

        # This operator gets an in-memory Image object, so a specialized ImageReader is needed.
        _reader = InMemImageReader(input_image)

        # Apply preprocessing transforms
        pre_transforms = self.pre_process(_reader)

        # Create data dictionary
        data_dict = {self._input_dataset_key: img_name}

        # Create dataset and dataloader
        dataset = Dataset(data=[data_dict], transform=pre_transforms)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        out_dict = []
        for d in dataloader:
            preprocessed_image = d[self._input_dataset_key]
            self._logger.info(f"Input shape: {preprocessed_image.shape}")

            # Get the loaded ensemble models from app context
            if not hasattr(self.app_context, "models") or self.app_context.models is None:
                raise RuntimeError("nnU-Net models not loaded. Call _load_nnunet_models first.")

            ensemble_predictors = self.app_context.models.predictor

            # Perform ensemble inference
            self._logger.info("Running nnU-Net ensemble inference...")

            for i, predictor in enumerate(ensemble_predictors):
                model_key = self.prediction_keys[i]
                self._logger.info(f"Running inference with model: {model_key}")

                # Run inference with individual model
                prediction = predictor(preprocessed_image)
                d[model_key] = prediction

            self._logger.info("Inference complete, applying postprocessing...")

            # Apply postprocessing transforms (includes ensemble combination)
            post_transforms1 = self.post_process_stage1()
            d = post_transforms1(d)
            out_dict.append(d)
        return out_dict

    def post_process_stage1(self) -> Compose:
        """Composes transforms for postprocessing the nnU-Net prediction results."""
        pred_key = self._pred_dataset_key
        return Compose(
            [
                # nnU-Net ensemble post-processing
                EnsembleProbabilitiesToSegmentation(
                    keys=self.prediction_keys,
                    dataset_json_path=str(self.model_path / "jsonpkls/dataset.json"),
                    plans_json_path=str(self.model_path / "jsonpkls/plans.json"),
                    output_key=pred_key,
                ),
                # Add batch dimension to final prediction
                Lambdad(keys=[pred_key], func=lambda x: x.unsqueeze(0)),
                # Transpose dimensions back to original format
                Transposed(keys=[self._input_dataset_key, pred_key], indices=(0, 1, 4, 3, 2)),
            ]
        )

    def post_process_stage2(self) -> Compose:
        """Composes transforms for postprocessing MAP outputs"""
        pred_key = self._pred_dataset_key

        # Define labels for the segmentation output
        labels = {"background": 0, "airway": 1}

        return Compose(
            [
                # Keep only largest connected component for each label
                KeepLargestConnectedComponentd(keys=pred_key, applied_labels=[1]),
                # Calculate volume from segmentation mask
                CalculateVolumeFromMaskd(keys=pred_key, label_names=labels),
                # Extract volume data to text format
                ExtractVolumeToTextd(
                    keys=[pred_key + "_volumes"], label_names=labels, output_labels=self.output_labels
                ),
                # Convert labels to contours
                LabelToContourd(keys=pred_key, output_labels=self.output_labels),
                # Create overlay of image and contours
                OverlayImageLabeld(image_key=self._input_dataset_key, label_key=pred_key, overlay_key="overlay"),
                # Save overlays as DICOM SC
                SaveImaged(
                    keys="overlay",
                    output_ext=".dcm",
                    output_dir=self.output_folder / "temp",
                    separate_folder=False,
                    output_dtype=int16,
                ),
            ]
        )

    def get_result_text_from_transforms(self, post_transforms: Compose) -> str:
        """
        Extracts result_text from the ExtractVolumeToTextd transform in the transform pipeline.

        Args:
            post_transforms: Composed transforms that include ExtractVolumeToTextd

        Returns:
            The extracted result text or None if not found
        """
        for transform in post_transforms.transforms:
            if isinstance(transform, ExtractVolumeToTextd):
                return transform.result_text
        return None
