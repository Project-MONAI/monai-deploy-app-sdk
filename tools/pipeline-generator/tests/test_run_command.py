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

"""Tests for the run command."""

import subprocess
from unittest.mock import Mock, patch

from click.testing import CliRunner

from pipeline_generator.cli.run import run


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

        result = self.runner.invoke(
            run, [str(app_path), "--input", str(input_dir), "--output", str(output_dir)]
        )

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

        result = self.runner.invoke(
            run, [str(app_path), "--input", str(input_dir), "--output", str(output_dir)]
        )

        assert result.exit_code == 1
        assert "Error: requirements.txt not found" in result.output

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_run_successful_with_new_venv(self, mock_popen, mock_run, tmp_path):
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

        result = self.runner.invoke(
            run, [str(app_path), "--input", str(input_dir), "--output", str(output_dir)]
        )

        assert result.exit_code == 0
        assert "Running MONAI Deploy application" in result.output
        assert "Application completed successfully" in result.output
        mock_run.assert_called()  # Verify venv was created

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_run_skip_install(self, mock_popen, mock_run, tmp_path):
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

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_run_with_model_path(self, mock_popen, mock_run, tmp_path):
        """Test run command with custom model path."""
        # Set up test directories
        app_path = tmp_path / "test_app"
        app_path.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        model_path = tmp_path / "models"
        model_path.mkdir()  # Create the model directory
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

        result = self.runner.invoke(
            run,
            [
                str(app_path),
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-m",
                str(model_path),
                "--skip-install",
            ],
        )

        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Output: {result.output}")
        assert result.exit_code == 0
        # Verify model path was passed to the command
        call_args = mock_popen.call_args[0][0]
        assert "-m" in call_args
        assert str(model_path) in call_args

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_run_app_failure(self, mock_popen, mock_run, tmp_path):
        """Test run command when application fails."""
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

        # Mock subprocess for app execution with failure
        mock_process = Mock()
        mock_process.wait.return_value = 1
        mock_process.stdout = iter(["Error occurred!\n"])
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

    @patch("subprocess.run")
    def test_run_venv_creation_failure(self, mock_run, tmp_path):
        """Test run command when venv creation fails."""
        # Set up test directories
        app_path = tmp_path / "test_app"
        app_path.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create required files
        (app_path / "app.py").write_text("print('test')")
        (app_path / "requirements.txt").write_text("monai-deploy-app-sdk\n")

        # Mock subprocess for venv creation failure
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "python", stderr="Error creating venv"
        )

        result = self.runner.invoke(
            run, [str(app_path), "--input", str(input_dir), "--output", str(output_dir)]
        )

        assert result.exit_code == 1
        assert "Error creating virtual environment" in result.output

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_run_with_existing_venv(self, mock_popen, mock_run, tmp_path):
        """Test run command with existing virtual environment."""
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

        # Mock pip install
        mock_run.return_value = Mock(returncode=0)

        result = self.runner.invoke(
            run, [str(app_path), "--input", str(input_dir), "--output", str(output_dir)]
        )

        assert result.exit_code == 0
        assert "Using existing virtual environment" in result.output

    @patch("subprocess.run")
    def test_run_pip_install_failure(self, mock_run, tmp_path):
        """Test run command when pip install fails."""
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
        (app_path / "requirements.txt").write_text("nonexistent-package\n")

        # Mock subprocess for pip install failure
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "pip", stderr="Package not found"
        )

        result = self.runner.invoke(
            run, [str(app_path), "--input", str(input_dir), "--output", str(output_dir)]
        )

        assert result.exit_code == 1
        assert "Error installing dependencies" in result.output

    def test_run_with_custom_venv_name(self, tmp_path):
        """Test run command with custom virtual environment name."""
        # Set up test directories
        app_path = tmp_path / "test_app"
        app_path.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        custom_venv = app_path / "myenv"
        custom_venv.mkdir()

        # Create required files
        (app_path / "app.py").write_text("print('test')")
        (app_path / "requirements.txt").write_text("monai-deploy-app-sdk\n")

        with patch("subprocess.Popen") as mock_popen:
            mock_process = Mock()
            mock_process.wait.return_value = 0
            mock_process.stdout = iter(["Processing...\n", "Complete!\n"])
            mock_popen.return_value = mock_process

            result = self.runner.invoke(
                run,
                [
                    str(app_path),
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output_dir),
                    "--venv-name",
                    "myenv",
                    "--skip-install",
                ],
            )

        assert result.exit_code == 0
        assert "Using existing virtual environment: myenv" in result.output

    @patch("subprocess.Popen")
    def test_run_with_no_gpu(self, mock_popen, tmp_path):
        """Test run command with --no-gpu flag."""
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

        # Mock subprocess
        mock_process = Mock()
        mock_process.wait.return_value = 0
        mock_process.stdout = iter(["Processing...\n", "Complete!\n"])
        mock_popen.return_value = mock_process

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
        # Verify CUDA_VISIBLE_DEVICES was set to empty string
        call_kwargs = mock_popen.call_args[1]
        assert "env" in call_kwargs
        assert call_kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
