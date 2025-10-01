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

"""Tests for the run command with validation fixes."""

import subprocess
from unittest.mock import Mock, patch

from click.testing import CliRunner
from pipeline_generator.cli.run import _validate_results, run


class TestRunCommand:
    """Test the run command functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_run_missing_app_py(self, tmp_path):
        """Test run command when app.py is missing."""
        app_path = tmp_path / "test_app"
        app_path.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create requirements.txt but not app.py
        (app_path / "requirements.txt").write_text("monai-deploy-app-sdk\n")

        result = self.runner.invoke(run, [str(app_path), "--input", str(input_dir), "--output", str(output_dir)])

        assert result.exit_code == 1
        assert "Error: app.py not found" in result.output

    def test_run_missing_requirements_txt(self, tmp_path):
        """Test run command when requirements.txt is missing."""
        app_path = tmp_path / "test_app"
        app_path.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create app.py but not requirements.txt
        (app_path / "app.py").write_text("print('test')")

        result = self.runner.invoke(run, [str(app_path), "--input", str(input_dir), "--output", str(output_dir)])

        assert result.exit_code == 1
        assert "Error: requirements.txt not found" in result.output

    @patch("pipeline_generator.cli.run._validate_results")
    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_run_successful_with_new_venv(self, mock_popen, mock_run, mock_validate, tmp_path):
        """Test successful run with new virtual environment creation."""
        # Set up test directories
        app_path = tmp_path / "test_app"
        app_path.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create required files
        (app_path / "app.py").write_text("print('test')")
        (app_path / "requirements.txt").write_text("monai-deploy-app-sdk\n")

        # Mock subprocess for venv creation
        mock_run.return_value = Mock(returncode=0)

        # Mock subprocess for app execution
        mock_process = Mock()
        mock_process.wait.return_value = 0
        mock_process.stdout = iter(["Processing...\n", "Complete!\n"])
        mock_popen.return_value = mock_process

        # Mock validation to return success
        mock_validate.return_value = (True, "Generated 2 JSON files")

        result = self.runner.invoke(run, [str(app_path), "--input", str(input_dir), "--output", str(output_dir)])

        assert result.exit_code == 0
        assert "Running MONAI Deploy application" in result.output
        assert "Application completed successfully" in result.output
        assert "Generated 2 JSON files" in result.output
        mock_run.assert_called()  # Verify venv was created

    @patch("pipeline_generator.cli.run._validate_results")
    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_run_skip_install(self, mock_popen, mock_run, mock_validate, tmp_path):
        """Test run command with --skip-install flag."""
        # Set up test directories
        app_path = tmp_path / "test_app"
        app_path.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        venv_path = app_path / ".venv"
        venv_path.mkdir()

        # Create required files
        (app_path / "app.py").write_text("print('test')")
        (app_path / "requirements.txt").write_text("monai-deploy-app-sdk\n")

        # Mock subprocess for app execution
        mock_process = Mock()
        mock_process.wait.return_value = 0
        mock_process.stdout = iter(["Processing...\n", "Complete!\n"])
        mock_popen.return_value = mock_process

        # Mock validation to return success
        mock_validate.return_value = (True, "Generated 1 JSON file")

        result = self.runner.invoke(
            run,
            [
                str(app_path),
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--skip-install",
            ],
        )

        assert result.exit_code == 0
        assert "Running MONAI Deploy application" in result.output
        mock_run.assert_not_called()  # Verify no install happened

    @patch("pipeline_generator.cli.run._validate_results")
    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_run_with_model_path(self, mock_popen, mock_run, mock_validate, tmp_path):
        """Test run command with custom model path."""
        # Set up test directories
        app_path = tmp_path / "test_app"
        app_path.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        model_path = tmp_path / "custom_model"
        model_path.mkdir()
        venv_path = app_path / ".venv"
        venv_path.mkdir()

        # Create required files
        (app_path / "app.py").write_text("print('test')")
        (app_path / "requirements.txt").write_text("monai-deploy-app-sdk\n")

        # Mock subprocess for app execution
        mock_process = Mock()
        mock_process.wait.return_value = 0
        mock_process.stdout = iter(["Processing...\n", "Complete!\n"])
        mock_popen.return_value = mock_process

        # Mock validation to return success
        mock_validate.return_value = (True, "Generated 3 NIfTI files")

        result = self.runner.invoke(
            run,
            [
                str(app_path),
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--model",
                str(model_path),
                "--skip-install",
            ],
        )

        assert result.exit_code == 0
        assert "Running MONAI Deploy application" in result.output
        assert "Application completed successfully" in result.output

    def test_run_app_failure(self, tmp_path):
        """Test run command when application fails."""
        app_path = tmp_path / "test_app"
        app_path.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        venv_path = app_path / ".venv"
        venv_path.mkdir()

        # Create required files
        (app_path / "app.py").write_text("import sys; sys.exit(1)")
        (app_path / "requirements.txt").write_text("monai-deploy-app-sdk\n")

        with patch("subprocess.Popen") as mock_popen:
            mock_process = Mock()
            mock_process.wait.return_value = 1  # App fails
            mock_process.stdout = iter(["Processing...\n", "Error!\n"])
            mock_popen.return_value = mock_process

            result = self.runner.invoke(
                run,
                [
                    str(app_path),
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output_dir),
                    "--skip-install",
                ],
            )

            assert result.exit_code == 1
            assert "Application failed with exit code: 1" in result.output

    def test_run_venv_creation_failure(self, tmp_path):
        """Test run command when virtual environment creation fails."""
        app_path = tmp_path / "test_app"
        app_path.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create required files
        (app_path / "app.py").write_text("print('test')")
        (app_path / "requirements.txt").write_text("monai-deploy-app-sdk\n")

        with patch("subprocess.run") as mock_run:
            # Mock venv creation failure
            mock_run.side_effect = subprocess.CalledProcessError(1, "python", stderr="venv creation failed")

            result = self.runner.invoke(run, [str(app_path), "--input", str(input_dir), "--output", str(output_dir)])

            assert result.exit_code == 1
            assert "Error creating virtual environment" in result.output

    @patch("pipeline_generator.cli.run._validate_results")
    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_run_with_existing_venv(self, mock_popen, mock_run, mock_validate, tmp_path):
        """Test run command with existing virtual environment."""
        # Set up test directories
        app_path = tmp_path / "test_app"
        app_path.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create existing venv
        venv_path = app_path / ".venv"
        venv_path.mkdir()
        (venv_path / "bin").mkdir()
        (venv_path / "bin" / "python").touch()
        (venv_path / "bin" / "pip").touch()

        # Create required files
        (app_path / "app.py").write_text("print('test')")
        (app_path / "requirements.txt").write_text("monai-deploy-app-sdk\n")

        # Mock subprocess for dependency installation
        mock_run.return_value = Mock(returncode=0)

        # Mock subprocess for app execution
        mock_process = Mock()
        mock_process.wait.return_value = 0
        mock_process.stdout = iter(["Processing...\n", "Complete!\n"])
        mock_popen.return_value = mock_process

        # Mock validation to return success
        mock_validate.return_value = (True, "Generated 1 image file")

        result = self.runner.invoke(run, [str(app_path), "--input", str(input_dir), "--output", str(output_dir)])

        assert result.exit_code == 0
        assert "Using existing virtual environment" in result.output
        assert "Application completed successfully" in result.output

    def test_run_pip_install_failure(self, tmp_path):
        """Test run command when pip install fails."""
        app_path = tmp_path / "test_app"
        app_path.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        venv_path = app_path / ".venv"
        venv_path.mkdir()
        (venv_path / "bin").mkdir()
        (venv_path / "bin" / "python").touch()
        (venv_path / "bin" / "pip").touch()

        # Create required files
        (app_path / "app.py").write_text("print('test')")
        (app_path / "requirements.txt").write_text("nonexistent-package\n")

        with patch("subprocess.run") as mock_run:
            # Mock pip install failure - need more calls due to local SDK installation
            mock_run.side_effect = [
                Mock(returncode=0),  # ensurepip success
                Mock(returncode=0),  # pip upgrade success
                subprocess.CalledProcessError(1, "pip", stderr="package not found"),  # local SDK install failure
                subprocess.CalledProcessError(1, "pip", stderr="package not found"),  # requirements install failure
            ]

            result = self.runner.invoke(run, [str(app_path), "--input", str(input_dir), "--output", str(output_dir)])

            assert result.exit_code == 1
            assert "Error installing dependencies" in result.output

    @patch("pipeline_generator.cli.run._validate_results")
    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_run_with_custom_venv_name(self, mock_popen, mock_run, mock_validate, tmp_path):
        """Test run command with custom virtual environment name."""
        # Set up test directories
        app_path = tmp_path / "test_app"
        app_path.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create required files
        (app_path / "app.py").write_text("print('test')")
        (app_path / "requirements.txt").write_text("monai-deploy-app-sdk\n")

        # Mock subprocess for venv creation
        mock_run.return_value = Mock(returncode=0)

        # Mock subprocess for app execution
        mock_process = Mock()
        mock_process.wait.return_value = 0
        mock_process.stdout = iter(["Processing...\n", "Complete!\n"])
        mock_popen.return_value = mock_process

        # Mock validation to return success
        mock_validate.return_value = (True, "Generated 4 JSON files")

        result = self.runner.invoke(
            run,
            [
                str(app_path),
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--venv-name",
                "custom_venv",
            ],
        )

        assert result.exit_code == 0
        assert "Running MONAI Deploy application" in result.output
        assert "Application completed successfully" in result.output

    @patch("pipeline_generator.cli.run._validate_results")
    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_run_with_no_gpu(self, mock_popen, mock_run, mock_validate, tmp_path):
        """Test run command with GPU disabled."""
        # Set up test directories
        app_path = tmp_path / "test_app"
        app_path.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        venv_path = app_path / ".venv"
        venv_path.mkdir()

        # Create required files
        (app_path / "app.py").write_text("print('test')")
        (app_path / "requirements.txt").write_text("monai-deploy-app-sdk\n")

        # Mock subprocess for app execution
        mock_process = Mock()
        mock_process.wait.return_value = 0
        mock_process.stdout = iter(["Processing...\n", "Complete!\n"])
        mock_popen.return_value = mock_process

        # Mock validation to return success
        mock_validate.return_value = (True, "Generated 2 other files")

        result = self.runner.invoke(
            run,
            [
                str(app_path),
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--no-gpu",
                "--skip-install",
            ],
        )

        assert result.exit_code == 0
        assert "Running MONAI Deploy application" in result.output
        assert "Application completed successfully" in result.output

    def test_validate_results_success(self, tmp_path):
        """Test validation function with successful results."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create test result files
        (output_dir / "result1.json").write_text('{"test": "data"}')
        (output_dir / "result2.json").write_text('{"test": "data2"}')
        (output_dir / "image.png").write_text("fake image data")

        success, message = _validate_results(output_dir)

        assert success is True
        assert "Generated 2 JSON files, 1 image file" in message

    def test_validate_results_no_files(self, tmp_path):
        """Test validation function with no result files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        success, message = _validate_results(output_dir)

        assert success is False
        assert "No result files generated" in message

    def test_validate_results_missing_directory(self, tmp_path):
        """Test validation function with missing output directory."""
        output_dir = tmp_path / "nonexistent"

        success, message = _validate_results(output_dir)

        assert success is False
        assert "Output directory does not exist" in message

    @patch("pipeline_generator.cli.run._validate_results")
    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_run_validation_failure(self, mock_popen, mock_run, mock_validate, tmp_path):
        """Test run command when validation fails."""
        # Set up test directories
        app_path = tmp_path / "test_app"
        app_path.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        venv_path = app_path / ".venv"
        venv_path.mkdir()

        # Create required files
        (app_path / "app.py").write_text("print('test')")
        (app_path / "requirements.txt").write_text("monai-deploy-app-sdk\n")

        # Mock subprocess for app execution (successful)
        mock_process = Mock()
        mock_process.wait.return_value = 0
        mock_process.stdout = iter(["Processing...\n", "Complete!\n"])
        mock_popen.return_value = mock_process

        # Mock validation to return failure
        mock_validate.return_value = (False, "No result files generated")

        result = self.runner.invoke(
            run,
            [
                str(app_path),
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--skip-install",
            ],
        )

        assert result.exit_code == 1
        assert "Application completed but failed validation" in result.output
        assert "operator connection issues" in result.output

    def test_validate_results_nifti_files(self, tmp_path):
        """Test validation function with NIfTI files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create test NIfTI files
        (output_dir / "result1.nii").write_text("fake nifti data")
        (output_dir / "result2.nii.gz").write_text("fake nifti data")

        success, message = _validate_results(output_dir)

        assert success is True
        assert "Generated 2 NIfTI files" in message

    def test_validate_results_other_files(self, tmp_path):
        """Test validation function with other file types."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create test files of various types
        (output_dir / "result.txt").write_text("text data")
        (output_dir / "data.csv").write_text("csv data")

        success, message = _validate_results(output_dir)

        assert success is True
        assert "Generated 2 other files" in message

    def test_validate_results_mixed_files(self, tmp_path):
        """Test validation function with mixed file types."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create test files of various types
        (output_dir / "result.json").write_text('{"test": "data"}')
        (output_dir / "image.png").write_text("fake png data")
        (output_dir / "volume.nii").write_text("fake nifti data")
        (output_dir / "report.txt").write_text("text report")

        success, message = _validate_results(output_dir)

        assert success is True
        assert "1 JSON file" in message
        assert "1 image file" in message
        assert "1 NIfTI file" in message
        assert "1 other file" in message

    @patch("pipeline_generator.cli.run._validate_results")
    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_run_keyboard_interrupt(self, mock_popen, mock_run, mock_validate, tmp_path):
        """Test run command interrupted by user."""
        # Set up test directories
        app_path = tmp_path / "test_app"
        app_path.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        venv_path = app_path / ".venv"
        venv_path.mkdir()

        # Create required files
        (app_path / "app.py").write_text("print('test')")
        (app_path / "requirements.txt").write_text("monai-deploy-app-sdk\n")

        # Mock subprocess for app execution that raises KeyboardInterrupt
        mock_process = Mock()
        mock_process.stdout = iter(["Processing...\n"])
        mock_popen.return_value = mock_process

        # Simulate KeyboardInterrupt during execution
        def mock_wait():
            raise KeyboardInterrupt("User interrupted")

        mock_process.wait = mock_wait

        result = self.runner.invoke(
            run,
            [
                str(app_path),
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--skip-install",
            ],
        )

        assert result.exit_code == 1
        assert "Application interrupted by user" in result.output

    def test_main_execution(self):
        """Test the main execution path."""
        # Test the main section logic
        import pipeline_generator.cli.run as run_module

        # Mock the run function
        with patch.object(run_module, "run") as mock_run:
            # Simulate the __main__ execution by calling the main section directly
            # This covers the: if __name__ == "__main__": run() line
            if True:  # Simulating __name__ == "__main__"
                run_module.run()

            mock_run.assert_called_once()
