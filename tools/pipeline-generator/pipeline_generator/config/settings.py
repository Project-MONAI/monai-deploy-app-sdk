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

"""Settings and configuration management for Pipeline Generator."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml  # type: ignore
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    model_id: str = Field(..., description="Model ID (e.g., 'MONAI/spleen_ct_segmentation')")
    input_type: str = Field("nifti", description="Input data type: 'nifti', 'dicom', 'image'")
    output_type: str = Field(
        "nifti",
        description="Output data type: 'nifti', 'dicom', 'json', 'image_overlay'",
    )
    configs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        None,
        description="Additional template configs per model (dict or list of dicts)",
    )
    dependencies: Optional[List[str]] = Field(
        default=[],
        description="Additional pip requirement specifiers to include in generated requirements.txt",
    )


class Endpoint(BaseModel):
    """Model endpoint configuration."""

    organization: Optional[str] = Field(None, description="HuggingFace organization name")
    model_id: Optional[str] = Field(None, description="Specific model ID")
    base_url: str = Field("https://huggingface.co", description="Base URL for the endpoint")
    description: str = Field("", description="Endpoint description")
    model_type: Optional[str] = Field(
        None,
        description="Model type: segmentation, pathology, multimodal, multimodal_llm",
    )
    models: List[ModelConfig] = Field(default_factory=list, description="Tested models with known data types")


class Settings(BaseModel):
    """Application settings."""

    supported_models: List[str] = Field(
        default_factory=lambda: [".ts", ".pt", ".pth", ".safetensors"], description="Supported model file extensions"
    )
    endpoints: List[Endpoint] = Field(default_factory=list)
    additional_models: List[Endpoint] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        """Load settings from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Settings object initialized from YAML data
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_all_endpoints(self) -> List[Endpoint]:
        """Get all endpoints including additional models.

        Combines the main endpoints list with additional_models to provide
        a single list of all configured endpoints.

        Returns:
            List of all Endpoint configurations
        """
        return self.endpoints + self.additional_models

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration for a specific model ID.

        Searches through all endpoints' model configurations to find
        the configuration for the specified model ID.

        Args:
            model_id: The model ID to search for

        Returns:
            ModelConfig if found, None otherwise
        """
        for endpoint in self.get_all_endpoints():
            for model in endpoint.models:
                if model.model_id == model_id:
                    return model
        return None


def load_config(config_path: Optional[Path] = None) -> Settings:
    """Load configuration from file or use defaults.

    Attempts to load configuration from the specified path, falling back to
    a config.yaml in the package directory, or finally to default settings
    if no config file is found.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Settings object with loaded or default configuration
    """
    if config_path is None:
        # Try to find config.yaml in package
        package_dir = Path(__file__).parent
        config_path = package_dir / "config.yaml"

    if config_path.exists():
        return Settings.from_yaml(config_path)

    # Return default settings if no config file found
    return Settings(
        endpoints=[
            Endpoint(
                organization="MONAI",
                model_id=None,
                base_url="https://huggingface.co",
                description="Official MONAI organization models",
                model_type=None,
            )
        ]
    )
