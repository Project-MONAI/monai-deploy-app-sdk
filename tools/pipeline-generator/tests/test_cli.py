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

"""Tests for CLI commands."""

from click.testing import CliRunner
from unittest.mock import Mock, patch
from pipeline_generator.cli.main import cli
from pipeline_generator.core.models import ModelInfo


class TestCLI:
    """Test CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Pipeline Generator" in result.output
        assert "Generate MONAI Deploy and Holoscan pipelines" in result.output

    def test_cli_version(self):
        """Test CLI version command."""
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    @patch("pipeline_generator.cli.main.HuggingFaceClient")
    @patch("pipeline_generator.cli.main.load_config")
    def test_list_command_table_format(self, mock_load_config, mock_client_class):
        """Test list command with table format."""
        # Mock the configuration
        mock_settings = Mock()
        mock_settings.get_all_endpoints.return_value = [Mock(organization="MONAI")]
        mock_settings.endpoints = []  # Add empty endpoints list
        mock_load_config.return_value = mock_settings

        # Mock the HuggingFace client
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock model data
        test_models = [
            ModelInfo(
                model_id="MONAI/test_model1",
                name="Test Model 1",
                downloads=100,
                likes=10,
                is_monai_bundle=True,
            ),
            ModelInfo(
                model_id="MONAI/test_model2",
                name="Test Model 2",
                downloads=200,
                likes=20,
                is_monai_bundle=False,
            ),
        ]
        mock_client.list_models_from_endpoints.return_value = test_models

        # Run command
        result = self.runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "Fetching models from HuggingFace" in result.output
        assert "MONAI/test_model1" in result.output
        assert "MONAI/test_model2" in result.output
        assert "Total models: 2" in result.output
        assert "MONAI Bundles: 1" in result.output

    @patch("pipeline_generator.cli.main.HuggingFaceClient")
    @patch("pipeline_generator.cli.main.load_config")
    def test_list_command_bundles_only(self, mock_load_config, mock_client_class):
        """Test list command with bundles-only filter."""
        # Mock setup
        mock_settings = Mock()
        mock_settings.get_all_endpoints.return_value = [Mock(organization="MONAI")]
        mock_settings.endpoints = []  # Add empty endpoints list
        mock_load_config.return_value = mock_settings

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock model data with mixed bundle status
        test_models = [
            ModelInfo(model_id="MONAI/bundle1", name="Bundle 1", is_monai_bundle=True),
            ModelInfo(model_id="MONAI/model1", name="Model 1", is_monai_bundle=False),
            ModelInfo(model_id="MONAI/bundle2", name="Bundle 2", is_monai_bundle=True),
        ]
        mock_client.list_models_from_endpoints.return_value = test_models

        # Run command with bundles-only filter
        result = self.runner.invoke(cli, ["list", "--bundles-only"])

        assert result.exit_code == 0
        assert "MONAI/bundle1" in result.output
        assert "MONAI/bundle2" in result.output
        assert "MONAI/model1" not in result.output
        assert "Total models: 2" in result.output  # Only bundles shown

    @patch("pipeline_generator.cli.main.HuggingFaceClient")
    @patch("pipeline_generator.cli.main.load_config")
    def test_list_command_simple_format(self, mock_load_config, mock_client_class):
        """Test list command with simple format."""
        # Mock setup
        mock_settings = Mock()
        mock_settings.get_all_endpoints.return_value = [Mock(organization="MONAI")]
        mock_settings.endpoints = []  # Add empty endpoints list
        mock_load_config.return_value = mock_settings

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        test_models = [
            ModelInfo(model_id="MONAI/test", name="Test", is_monai_bundle=True)
        ]
        mock_client.list_models_from_endpoints.return_value = test_models

        # Run command with simple format
        result = self.runner.invoke(cli, ["list", "--format", "simple"])

        assert result.exit_code == 0
        assert "📦 MONAI/test" in result.output

    def test_list_command_with_config(self):
        """Test list command with custom config file."""
        with self.runner.isolated_filesystem():
            # Create a test config file
            with open("test_config.yaml", "w") as f:
                f.write("""
endpoints:
  - organization: "TestOrg"
    description: "Test organization"
""")

            # Run command with config file
            with patch(
                "pipeline_generator.cli.main.HuggingFaceClient"
            ) as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                mock_client.list_models_from_endpoints.return_value = []

                result = self.runner.invoke(
                    cli, ["--config", "test_config.yaml", "list"]
                )

                assert result.exit_code == 0

    @patch("pipeline_generator.cli.main.HuggingFaceClient")
    @patch("pipeline_generator.cli.main.load_config")
    def test_list_command_json_format(self, mock_load_config, mock_client_class):
        """Test list command with JSON format output."""
        import json

        # Mock setup
        mock_settings = Mock()
        mock_settings.endpoints = []
        mock_load_config.return_value = mock_settings

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        test_models = [
            ModelInfo(
                model_id="MONAI/test",
                name="Test Model",
                is_monai_bundle=True,
                downloads=100,
                likes=10,
                tags=["medical", "segmentation"],
            )
        ]
        mock_client.list_models_from_endpoints.return_value = test_models

        # Run command with JSON format
        result = self.runner.invoke(cli, ["list", "--format", "json"])

        assert result.exit_code == 0

        # Extract JSON from output (skip header line)
        lines = result.output.strip().split("\n")
        json_start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("["):
                json_start = i
                break

        if json_start >= 0:
            json_text = "\n".join(lines[json_start:])
            if "\nTotal models:" in json_text:
                json_text = json_text[: json_text.rfind("\nTotal models:")]

            data = json.loads(json_text)
            assert len(data) == 1
            assert data[0]["model_id"] == "MONAI/test"
            assert data[0]["is_monai_bundle"] is True

    @patch("pipeline_generator.cli.main.HuggingFaceClient")
    @patch("pipeline_generator.cli.main.load_config")
    def test_list_command_no_models(self, mock_load_config, mock_client_class):
        """Test list command when no models are found."""
        # Mock setup
        mock_settings = Mock()
        mock_settings.endpoints = []
        mock_load_config.return_value = mock_settings

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.list_models_from_endpoints.return_value = []

        result = self.runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "No models found" in result.output or "Total models: 0" in result.output

    @patch("pipeline_generator.cli.main.HuggingFaceClient")
    @patch("pipeline_generator.cli.main.load_config")
    def test_list_command_tested_only(self, mock_load_config, mock_client_class):
        """Test list command with tested-only filter."""
        # Mock setup
        mock_settings = Mock()

        # Create tested models in settings
        tested_model = Mock()
        tested_model.model_id = "MONAI/tested_model"

        mock_endpoint = Mock()
        mock_endpoint.models = [tested_model]
        mock_settings.endpoints = [mock_endpoint]

        mock_load_config.return_value = mock_settings

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock the list response
        test_models = [
            ModelInfo(
                model_id="MONAI/tested_model", name="Tested Model", is_monai_bundle=True
            ),
            ModelInfo(
                model_id="MONAI/untested_model",
                name="Untested Model",
                is_monai_bundle=True,
            ),
        ]
        mock_client.list_models_from_endpoints.return_value = test_models

        # Test with tested-only filter
        result = self.runner.invoke(cli, ["list", "--tested-only"])

        assert result.exit_code == 0
        assert "MONAI/tested_model" in result.output
        assert "MONAI/untested_model" not in result.output

    @patch("pipeline_generator.cli.main.AppGenerator")
    @patch("pipeline_generator.cli.main.load_config")
    def test_gen_command_error_handling(self, mock_load_config, mock_generator_class):
        """Test gen command error handling."""
        mock_settings = Mock()
        mock_load_config.return_value = mock_settings

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        # Make generate_app raise an exception
        mock_generator.generate_app.side_effect = Exception("Test error")

        with patch("pipeline_generator.cli.main.logger") as mock_logger:
            result = self.runner.invoke(cli, ["gen", "MONAI/test_model"])

            # Should log the exception
            assert mock_logger.exception.called
            assert result.exit_code != 0
