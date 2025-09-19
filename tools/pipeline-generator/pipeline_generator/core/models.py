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

"""Data models for Pipeline Generator."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Model information from HuggingFace."""

    model_id: str = Field(..., description="Model ID (e.g., 'MONAI/spleen_ct_segmentation')")
    name: str = Field(..., description="Model name")
    author: Optional[str] = Field(None, description="Model author/organization")
    description: Optional[str] = Field(None, description="Model description")
    downloads: Optional[int] = Field(None, description="Number of downloads")
    likes: Optional[int] = Field(None, description="Number of likes")
    created_at: Optional[datetime] = Field(None, description="Creation date")
    updated_at: Optional[datetime] = Field(None, description="Last update date")
    tags: List[str] = Field(default_factory=list, description="Model tags")
    is_monai_bundle: bool = Field(False, description="Whether this is a MONAI Bundle")
    bundle_metadata: Optional[Dict[str, Any]] = Field(None, description="MONAI Bundle metadata if available")
    model_extensions: List[str] = Field(default_factory=list, description="Detected model file extensions")

    @property
    def display_name(self) -> str:
        """Get a display-friendly name for the model.

        Returns the model's name if available, otherwise generates a
        human-readable name from the model ID by removing the organization
        prefix and converting underscores to spaces.

        Returns:
            str: Display-friendly model name
        """
        if self.name:
            return self.name
        return self.model_id.split("/")[-1].replace("_", " ").title()

    @property
    def short_id(self) -> str:
        """Get the short model ID without the organization prefix.

        Example:
            'MONAI/spleen_ct_segmentation' -> 'spleen_ct_segmentation'

        Returns:
            str: Model ID without organization prefix
        """
        return self.model_id.split("/")[-1]

    @property
    def has_torchscript(self) -> bool:
        """Check if model has TorchScript (.ts) files.

        Returns:
            bool: True if model has .ts files, False otherwise
        """
        return ".ts" in self.model_extensions

    @property
    def primary_extension(self) -> Optional[str]:
        """Get the primary model extension for display.

        Returns the first extension found, prioritizing .ts files.

        Returns:
            str: Primary model extension or None if no extensions found
        """
        if not self.model_extensions:
            return None

        # Prioritize .ts extension
        if ".ts" in self.model_extensions:
            return ".ts"

        return self.model_extensions[0]
