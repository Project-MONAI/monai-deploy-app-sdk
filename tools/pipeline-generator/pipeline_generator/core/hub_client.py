# Copyright 2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HuggingFace Hub client for fetching model information."""

import logging
from typing import Any, List, Optional

from huggingface_hub import HfApi, list_models, model_info
from huggingface_hub.utils import HfHubHTTPError

from ..config import Endpoint, Settings
from .models import ModelInfo

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """Client for interacting with HuggingFace Hub."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the HuggingFace Hub client.

        Args:
            settings: Pipeline generator settings containing supported extensions
        """
        self.api = HfApi()
        self.settings = settings
        self.supported_extensions = settings.supported_models if settings else [".ts", ".pt", ".pth", ".safetensors"]

    def list_models_from_organization(self, organization: str) -> List[ModelInfo]:
        """List all models from a HuggingFace organization.

        Args:
            organization: HuggingFace organization name (e.g., 'MONAI')

        Returns:
            List of ModelInfo objects
        """
        models = []

        try:
            # Use the HuggingFace API to list models
            for model in list_models(author=organization):
                model_data = self._extract_model_info(model)
                models.append(model_data)

        except Exception as e:
            logger.error(f"Error listing models from {organization}: {e}")

        return models

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific model.

        Args:
            model_id: Model ID (e.g., 'MONAI/spleen_ct_segmentation')

        Returns:
            ModelInfo object or None if not found
        """
        try:
            model = model_info(model_id)
            return self._extract_model_info(model)
        except HfHubHTTPError as e:
            logger.error(f"Model {model_id} not found: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching model {model_id}: {e}")
            return None

    def list_models_from_endpoints(
        self, endpoints: List[Endpoint], fetch_details_for_bundles: bool = False
    ) -> List[ModelInfo]:
        """List models from all configured endpoints.

        Args:
            endpoints: List of endpoint configurations
            fetch_details_for_bundles: If True, fetch detailed info for potential MONAI Bundles to get accurate extension info

        Returns:
            List of ModelInfo objects from all endpoints
        """
        all_models = []

        for endpoint in endpoints:
            if endpoint.organization:
                # List all models from organization
                logger.info(f"Fetching models from organization: {endpoint.organization}")
                models = self.list_models_from_organization(endpoint.organization)

                if fetch_details_for_bundles:
                    # For MONAI organization, fetch detailed info for all models to get accurate extension data
                    # This is needed because bulk API doesn't provide file information (siblings=None)
                    if endpoint.organization == "MONAI":
                        enhanced_models = []
                        for model in models:
                            # Fetch detailed model info to get file extensions and accurate MONAI Bundle detection
                            detailed_model = self.get_model_info(model.model_id)
                            enhanced_models.append(detailed_model if detailed_model else model)
                        models = enhanced_models
                    else:
                        # For non-MONAI organizations, only fetch details for models that might be bundles
                        enhanced_models = []
                        for model in models:
                            if any("monai" in tag.lower() for tag in model.tags):
                                # Fetch detailed model info to get file extensions
                                detailed_model = self.get_model_info(model.model_id)
                                enhanced_models.append(detailed_model if detailed_model else model)
                            else:
                                enhanced_models.append(model)
                        models = enhanced_models

                all_models.extend(models)

            elif endpoint.model_id:
                # Get specific model
                logger.info(f"Fetching model: {endpoint.model_id}")
                model_info = self.get_model_info(endpoint.model_id)
                if model_info:
                    all_models.append(model_info)

        return all_models

    def _detect_model_extensions(self, model_data: Any) -> List[str]:
        """Detect model file extensions in a HuggingFace repository.

        Args:
            model_data: Model data from HuggingFace API

        Returns:
            List of detected model file extensions
        """
        extensions = []

        try:
            if hasattr(model_data, "siblings") and model_data.siblings is not None:
                file_names = [f.rfilename for f in model_data.siblings]
                for filename in file_names:
                    for ext in self.supported_extensions:
                        if filename.endswith(ext):
                            if ext not in extensions:
                                extensions.append(ext)
        except Exception as e:
            logger.debug(f"Could not detect extensions for {getattr(model_data, 'modelId', 'unknown')}: {e}")

        return extensions

    def list_torchscript_models(self, endpoints: List[Endpoint]) -> List[ModelInfo]:
        """List models that have TorchScript (.ts) files.

        This method fetches detailed information for each model individually to
        check for TorchScript files, which is slower than bulk listing but accurate.

        Args:
            endpoints: List of endpoint configurations

        Returns:
            List of ModelInfo objects that contain .ts files
        """
        torchscript_models = []

        for endpoint in endpoints:
            if endpoint.organization:
                # List all models from organization first (bulk)
                logger.info(f"Checking TorchScript models from organization: {endpoint.organization}")
                try:
                    for model in list_models(author=endpoint.organization):
                        # Fetch detailed model info to get file information
                        detailed_model = self.get_model_info(model.modelId)
                        if detailed_model and detailed_model.has_torchscript:
                            torchscript_models.append(detailed_model)
                except Exception as e:
                    logger.error(f"Error checking TorchScript models from {endpoint.organization}: {e}")

            elif endpoint.model_id:
                # Get specific model
                logger.info(f"Checking TorchScript model: {endpoint.model_id}")
                model = self.get_model_info(endpoint.model_id)
                if model and model.has_torchscript:
                    torchscript_models.append(model)

        return torchscript_models

    def _extract_model_info(self, model_data: Any) -> ModelInfo:
        """Extract ModelInfo from HuggingFace model data.

        Args:
            model_data: Model data from HuggingFace API

        Returns:
            ModelInfo object
        """
        # Detect model extensions
        model_extensions = self._detect_model_extensions(model_data)

        # Check if this is a MONAI Bundle - defined as having TorchScript (.ts) files
        is_monai_bundle = ".ts" in model_extensions
        bundle_metadata = None

        tags = getattr(model_data, "tags", [])

        # Extract description from cardData if available
        description = None
        card_data = getattr(model_data, "cardData", None)
        if card_data and isinstance(card_data, dict):
            description = card_data.get("description")
        if not description:
            description = getattr(model_data, "description", None)

        return ModelInfo(
            model_id=model_data.modelId,
            name=getattr(model_data, "name", model_data.modelId),
            author=getattr(model_data, "author", None),
            description=description,
            downloads=getattr(model_data, "downloads", None),
            likes=getattr(model_data, "likes", None),
            created_at=getattr(model_data, "created_at", None),
            updated_at=getattr(model_data, "lastModified", None),
            tags=tags,
            is_monai_bundle=is_monai_bundle,
            bundle_metadata=bundle_metadata,
            model_extensions=model_extensions,
        )
