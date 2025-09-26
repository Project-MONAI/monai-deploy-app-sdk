# Copyright 2024 MONAI Consortium
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
from typing import List, Optional, Union

import torch

from monai.bundle import ConfigParser
from monai.deploy.core import AppContext, Fragment, Image, Operator, OperatorSpec
from monai.deploy.utils.importutil import optional_import
from monai.transforms import Compose

# Dynamic class imports to match MONAI model loader behavior
monai, _ = optional_import("monai")
torchvision, _ = optional_import("torchvision")

globals_dict = {
    "torch": torch,
    "monai": monai,
    "torchvision": torchvision,
}


class MonaiClassificationOperator(Operator):
    """Operator for MONAI classification models that use Python model definitions.

    This operator handles models like TorchVisionFCModel that require:
    1. Loading a Python class definition
    2. Instantiating the model
    3. Loading state dict weights

    It supports models from MONAI bundles that don't use TorchScript.
    """

    DEFAULT_PRE_PROC_CONFIG = ["preprocessing", "transforms"]
    DEFAULT_POST_PROC_CONFIG = ["postprocessing", "transforms"]

    def __init__(
        self,
        fragment: Fragment,
        *args,
        app_context: AppContext,
        bundle_path: Union[str, Path],
        config_names: Optional[Union[List[str], str]] = None,
        **kwargs,
    ):
        """Initialize the operator.

        Args:
            fragment: Fragment instance
            app_context: Application context
            bundle_path: Path to the MONAI bundle
            config_names: Names of configs to use
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._executing = False

        # Set attributes before calling super().__init__ since setup() is called from there
        self.app_context = app_context
        self.bundle_path = Path(bundle_path)
        self.config_names = config_names or []

        super().__init__(fragment, *args, **kwargs)

        # Will be loaded during setup
        self._model = None
        self._pre_processor = None
        self._post_processor = None
        self._inference_config = None

    def setup(self, spec: OperatorSpec):
        """Set up the operator."""
        spec.input("image")
        spec.output("pred")

    def _load_bundle(self):
        """Load the MONAI bundle configuration and model."""
        # Load inference config
        inference_path = self.bundle_path / "configs" / "inference.json"
        if not inference_path.exists():
            raise FileNotFoundError(f"Inference config not found: {inference_path}")

        self._logger.info(f"Loading inference config from: {inference_path}")
        parser = ConfigParser()
        parser.read_config(str(inference_path))

        # Set up global imports for dynamic loading
        parser.globals = globals_dict

        # Store raw config for later use
        self._inference_config = parser.config

        # Load preprocessing - get the transforms directly
        if "preprocessing" in parser.config and "transforms" in parser.config["preprocessing"]:
            pre_transforms = parser.get_parsed_content("preprocessing#transforms")
            # Skip LoadImaged since our image is already loaded
            filtered_transforms = []
            for t in pre_transforms:
                if type(t).__name__ not in ["LoadImaged", "LoadImage"]:
                    filtered_transforms.append(t)
                else:
                    self._logger.info(f"Skipping {type(t).__name__} transform as image is already loaded")
            if filtered_transforms:
                self._pre_processor = Compose(filtered_transforms)
                self._logger.info(f"Loaded preprocessing transforms: {[type(t).__name__ for t in filtered_transforms]}")

        # Load model
        self._load_model(parser)

        # Load postprocessing - get the transforms directly
        if "postprocessing" in parser.config and "transforms" in parser.config["postprocessing"]:
            post_transforms = parser.get_parsed_content("postprocessing#transforms")
            self._post_processor = Compose(post_transforms)
            self._logger.info(f"Loaded postprocessing transforms: {[type(t).__name__ for t in post_transforms]}")

    def _load_model(self, parser: ConfigParser):
        """Load the model from the bundle."""
        # Get model definition - parse it to instantiate the model
        try:
            model = parser.get_parsed_content("network_def")
            if model is None:
                raise ValueError("Failed to parse network_def")
            self._logger.info(f"Loaded model: {type(model).__name__}")
        except Exception as e:
            self._logger.error(f"Error loading model definition: {e}")
            raise

        # Load model weights
        model_path = self.bundle_path / "models" / "model.pt"
        if not model_path.exists():
            # Try alternative paths
            alt_paths = [
                self.bundle_path / "models" / "model.pth",
                self.bundle_path / "model.pt",
                self.bundle_path / "model.pth",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    model_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Model file not found. Looked in: {model_path} and alternatives")

        self._logger.info(f"Loading model weights from: {model_path}")

        # Detect device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load state dict
        # Use weights_only=True for security (requires PyTorch 1.13+)
        try:
            state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
        except TypeError:
            self._logger.warning("Using torch.load without weights_only restriction - ensure model files are trusted")
            state_dict = torch.load(str(model_path), map_location=device)

        # Handle different state dict formats
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        # Load weights into model
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        self._model = model
        self._device = device
        self._logger.info(f"Model loaded successfully on device: {device}")

    def compute(self, op_input, op_output, context):
        """Run inference on the input image."""
        input_image = op_input.receive("image")
        if input_image is None:
            raise ValueError("No input image received")

        # Ensure we're not processing multiple times
        if self._executing:
            self._logger.warning("Already executing, skipping")
            return

        self._executing = True

        try:
            # Lazy load model if not already loaded
            if self._model is None:
                self._logger.info("Loading model on first compute call")
                self._load_bundle()

            # Convert Image to tensor format expected by MONAI
            if isinstance(input_image, Image):
                # Image data is already in CHW format from ImageFileLoader
                image_tensor = torch.from_numpy(input_image.asnumpy()).float()
            else:
                image_tensor = input_image

            self._logger.info(f"Input tensor shape: {image_tensor.shape}")

            # Move to device first
            image_tensor = image_tensor.to(self._device)

            # Apply preprocessing
            if self._pre_processor:
                # MONAI dict transforms expect dict format with key "image"
                # Since all our transforms end with 'd', we need dict format
                data = {"image": image_tensor}
                data = self._pre_processor(data)
                image_tensor = data["image"]
                self._logger.info(f"After preprocessing shape: {image_tensor.shape}")

            # Add batch dimension if needed (after preprocessing)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

            # Run inference
            with torch.no_grad():
                pred = self._model(image_tensor)

            # Apply postprocessing
            if self._post_processor:
                data = {"pred": pred}
                data = self._post_processor(data)
                pred = data["pred"]

            # Convert to dict format for output
            if isinstance(pred, torch.Tensor):
                # For classification, output is typically probabilities per class
                pred_dict = {}
                if pred.dim() == 2 and pred.shape[0] == 1:
                    # Single batch, multiple classes
                    pred = pred.squeeze(0)

                # Create dict with class probabilities
                for i, prob in enumerate(pred.cpu().numpy()):
                    pred_dict[f"class_{i}"] = float(prob)

                # Add predicted class
                pred_dict["predicted_class"] = int(torch.argmax(pred).item())

                result = pred_dict
            else:
                result = pred

            # Emit the result
            op_output.emit(result, "pred")
            self._logger.info(f"Inference completed. Result: {result}")

        except Exception as e:
            self._logger.error(f"Error during inference: {e}")
            raise
        finally:
            self._executing = False
