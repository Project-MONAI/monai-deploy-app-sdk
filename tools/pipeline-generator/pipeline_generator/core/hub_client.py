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

from typing import List, Optional, Any
import logging

from huggingface_hub import HfApi, model_info, list_models
from huggingface_hub.utils import HfHubHTTPError

from .models import ModelInfo
from ..config import Endpoint


logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """Client for interacting with HuggingFace Hub."""

    def __init__(self) -> None:
        """Initialize the HuggingFace Hub client."""
        self.api = HfApi()

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

    def list_models_from_endpoints(self, endpoints: List[Endpoint]) -> List[ModelInfo]:
        """List models from all configured endpoints.

        Args:
            endpoints: List of endpoint configurations

        Returns:
            List of ModelInfo objects from all endpoints
        """
        all_models = []

        for endpoint in endpoints:
            if endpoint.organization:
                # List all models from organization
                logger.info(
                    f"Fetching models from organization: {endpoint.organization}"
                )
                models = self.list_models_from_organization(endpoint.organization)
                all_models.extend(models)

            elif endpoint.model_id:
                # Get specific model
                logger.info(f"Fetching model: {endpoint.model_id}")
                model = self.get_model_info(endpoint.model_id)
                if model:
                    all_models.append(model)

        return all_models

    def _extract_model_info(self, model_data: Any) -> ModelInfo:
        """Extract ModelInfo from HuggingFace model data.

        Args:
            model_data: Model data from HuggingFace API

        Returns:
            ModelInfo object
        """
        # Check if this is a MONAI Bundle
        is_monai_bundle = False
        bundle_metadata = None

        # Check tags for MONAI-related tags
        tags = getattr(model_data, "tags", [])
        if any("monai" in tag.lower() for tag in tags):
            is_monai_bundle = True

        # Check if metadata.json exists in the model files
        try:
            if hasattr(model_data, "siblings"):
                file_names = [f.rfilename for f in model_data.siblings]
                if any("metadata.json" in f for f in file_names):
                    is_monai_bundle = True
        except Exception:
            pass

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
        )
