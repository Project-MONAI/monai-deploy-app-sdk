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

"""Tests for settings and configuration."""

import tempfile
from pathlib import Path

from pipeline_generator.config.settings import Endpoint, Settings, load_config


class TestEndpoint:
    """Test Endpoint model."""

    def test_endpoint_with_organization(self):
        """Test creating endpoint with organization."""
        endpoint = Endpoint(
            organization="MONAI",
            base_url="https://huggingface.co",
            description="MONAI models",
        )

        assert endpoint.organization == "MONAI"
        assert endpoint.model_id is None
        assert endpoint.base_url == "https://huggingface.co"

    def test_endpoint_with_model_id(self):
        """Test creating endpoint with specific model ID."""
        endpoint = Endpoint(model_id="Project-MONAI/test", description="Test model")

        assert endpoint.organization is None
        assert endpoint.model_id == "Project-MONAI/test"
        assert endpoint.base_url == "https://huggingface.co"  # default value


class TestSettings:
    """Test Settings model."""

    def test_empty_settings(self):
        """Test creating empty settings."""
        settings = Settings()

        assert settings.endpoints == []
        assert settings.additional_models == []
        assert settings.get_all_endpoints() == []

    def test_settings_with_endpoints(self):
        """Test settings with endpoints."""
        endpoint1 = Endpoint(organization="MONAI")
        endpoint2 = Endpoint(model_id="test/model")

        settings = Settings(endpoints=[endpoint1], additional_models=[endpoint2])

        assert len(settings.endpoints) == 1
        assert len(settings.additional_models) == 1
        assert len(settings.get_all_endpoints()) == 2

    def test_from_yaml(self):
        """Test loading settings from YAML file."""
        yaml_content = """
endpoints:
  - organization: "MONAI"
    base_url: "https://huggingface.co"
    description: "Official MONAI models"

additional_models:
  - model_id: "Project-MONAI/test"
    description: "Test model"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            settings = Settings.from_yaml(Path(f.name))

            assert len(settings.endpoints) == 1
            assert settings.endpoints[0].organization == "MONAI"
            assert len(settings.additional_models) == 1
            assert settings.additional_models[0].model_id == "Project-MONAI/test"

        Path(f.name).unlink()


class TestLoadConfig:
    """Test load_config function."""

    def test_load_config_with_file(self):
        """Test loading config from specified file."""
        yaml_content = """
endpoints:
  - organization: "TestOrg"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            settings = load_config(Path(f.name))
            assert len(settings.endpoints) == 1
            assert settings.endpoints[0].organization == "TestOrg"

        Path(f.name).unlink()

    def test_load_config_default(self):
        """Test loading config with default values when no file exists."""
        # Use a path that doesn't exist
        settings = load_config(Path("/nonexistent/config.yaml"))

        assert len(settings.endpoints) == 1
        assert settings.endpoints[0].organization == "MONAI"
        assert settings.endpoints[0].base_url == "https://huggingface.co"
