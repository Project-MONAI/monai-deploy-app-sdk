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

"""Tests for the gen command."""

from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner
from pipeline_generator.cli.main import cli


class TestGenCommand:
    """Test the gen command functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("pipeline_generator.cli.main.AppGenerator")
    def test_gen_command_success(self, mock_generator_class, tmp_path):
        """Test successful application generation."""
        # Mock the generator
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_app.return_value = tmp_path / "output"

        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["gen", "MONAI/spleen_ct_segmentation"])

        assert result.exit_code == 0
        assert "Generating MONAI Deploy application" in result.output
        assert "✓ Application generated successfully!" in result.output
        mock_generator.generate_app.assert_called_once()

    @patch("pipeline_generator.cli.main.AppGenerator")
    def test_gen_command_with_custom_output(self, mock_generator_class, tmp_path):
        """Test gen command with custom output directory."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_app.return_value = tmp_path / "custom_output"

        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli,
                ["gen", "MONAI/spleen_ct_segmentation", "--output", "custom_output"],
            )

        assert result.exit_code == 0
        assert "Output directory: custom_output" in result.output

        # Verify the generator was called with correct parameters
        call_args = mock_generator.generate_app.call_args
        assert call_args[1]["output_dir"] == Path("custom_output")

    @patch("pipeline_generator.cli.main.AppGenerator")
    def test_gen_command_with_app_name(self, mock_generator_class, tmp_path):
        """Test gen command with custom app name."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_app.return_value = tmp_path / "output"

        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli,
                ["gen", "MONAI/spleen_ct_segmentation", "--app-name", "MyCustomApp"],
            )

        assert result.exit_code == 0

        # Verify the generator was called with custom app name
        call_args = mock_generator.generate_app.call_args
        assert call_args[1]["app_name"] == "MyCustomApp"

    @patch("pipeline_generator.cli.main.AppGenerator")
    def test_gen_command_with_format(self, mock_generator_class, tmp_path):
        """Test gen command with specific format."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_app.return_value = tmp_path / "output"

        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["gen", "MONAI/spleen_ct_segmentation", "--format", "nifti"])

        assert result.exit_code == 0
        assert "Format: nifti" in result.output

        # Verify the generator was called with format
        call_args = mock_generator.generate_app.call_args
        assert call_args[1]["data_format"] == "nifti"

    def test_gen_command_existing_directory_without_force(self):
        """Test gen command when output directory exists without force."""
        with self.runner.isolated_filesystem():
            # Create existing output directory with a file
            output_dir = Path("output")
            output_dir.mkdir()
            (output_dir / "existing_file.txt").write_text("test")

            result = self.runner.invoke(cli, ["gen", "MONAI/spleen_ct_segmentation"])

        assert result.exit_code == 1
        assert "Error: Output directory" in result.output
        assert "already exists" in result.output

    @patch("pipeline_generator.cli.main.AppGenerator")
    def test_gen_command_existing_directory_with_force(self, mock_generator_class, tmp_path):
        """Test gen command when output directory exists with force."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_app.return_value = tmp_path / "output"

        with self.runner.isolated_filesystem():
            # Create existing output directory
            output_dir = Path("output")
            output_dir.mkdir()
            (output_dir / "existing_file.txt").write_text("test")

            result = self.runner.invoke(cli, ["gen", "MONAI/spleen_ct_segmentation", "--force"])

        assert result.exit_code == 0
        assert "✓ Application generated successfully!" in result.output

    @patch("pipeline_generator.cli.main.AppGenerator")
    def test_gen_command_bundle_download_error(self, mock_generator_class):
        """Test gen command when bundle download fails."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_app.side_effect = RuntimeError("Failed to download bundle")

        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["gen", "MONAI/nonexistent_model"])

        assert result.exit_code == 1
        assert "Error generating application" in result.output

    @patch("pipeline_generator.cli.main.AppGenerator")
    def test_gen_command_generation_error(self, mock_generator_class):
        """Test gen command when generation fails."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_app.side_effect = Exception("Generation failed")

        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["gen", "MONAI/spleen_ct_segmentation"])

        assert result.exit_code == 1
        assert "Error generating application" in result.output

    @patch("pipeline_generator.cli.main.AppGenerator")
    def test_gen_command_shows_generated_files(self, mock_generator_class):
        """Test that gen command shows list of generated files."""

        with self.runner.isolated_filesystem():
            # Create output directory with files
            output_dir = Path("output")
            output_dir.mkdir()
            (output_dir / "app.py").write_text("# app")
            (output_dir / "requirements.txt").write_text("monai")
            (output_dir / "README.md").write_text("# README")
            model_dir = output_dir / "model"
            model_dir.mkdir()
            (model_dir / "model.pt").write_text("model")

            # Mock the generator to return our prepared directory
            mock_generator = Mock()
            mock_generator_class.return_value = mock_generator
            mock_generator.generate_app.return_value = output_dir

            result = self.runner.invoke(
                cli,
                [
                    "gen",
                    "MONAI/spleen_ct_segmentation",
                    "--force",
                ],  # Use force since dir exists
            )

        assert result.exit_code == 0
        assert "Generated files:" in result.output
        assert "• app.py" in result.output
        assert "• requirements.txt" in result.output
        assert "• README.md" in result.output
        assert "• model/model.pt" in result.output

    @patch("pipeline_generator.cli.main.AppGenerator")
    def test_gen_command_shows_next_steps(self, mock_generator_class, tmp_path):
        """Test that gen command shows next steps."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_app.return_value = tmp_path / "output"

        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["gen", "MONAI/spleen_ct_segmentation"])

        assert result.exit_code == 0
        assert "Next steps:" in result.output
        assert "Option 1: Run with uv (recommended)" in result.output
        assert "Option 2: Run with pg directly" in result.output
        assert "pg run output" in result.output
        assert "Option 3: Run manually" in result.output
        assert "cd output" in result.output
        assert "pip install -r requirements.txt" in result.output
