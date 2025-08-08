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

"""Run command for executing generated MONAI Deploy applications."""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

logger = logging.getLogger(__name__)
console = Console()

@click.command()
@click.argument(
    "app_path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--input",
    "-i",
    "input_dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input data directory",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(path_type=Path),
    default="./output",
    help="Output directory for results",
)
@click.option(
    "--model",
    "-m",
    "model_path",
    type=click.Path(exists=True, path_type=Path),
    help="Override model/bundle path",
)
@click.option("--venv-name", default=".venv", help="Virtual environment directory name")
@click.option("--skip-install", is_flag=True, help="Skip dependency installation")
@click.option("--gpu/--no-gpu", default=True, help="Enable/disable GPU support")
def run(app_path: str, input_dir: str, output_dir: str, model_path: Optional[str], venv_name: str, skip_install: bool, gpu: bool) -> None:
    """Run a generated MONAI Deploy application.

    This command automates the process of setting up and running a MONAI Deploy
    application by managing virtual environments, dependencies, and execution.

    Steps performed:
    1. Create a virtual environment if it doesn't exist
    2. Install dependencies from requirements.txt (unless --skip-install)
    3. Run the application with the specified input/output directories

    Args:
        app_path: Path to the generated application directory
        input_dir: Directory containing input data (DICOM or NIfTI files)
        output_dir: Directory where results will be saved
        model_path: Optional override for model/bundle path
        venv_name: Name of virtual environment directory (default: .venv)
        skip_install: Skip dependency installation if True
        gpu: Enable GPU support (default: True)

    Example:
        pg run ./my_app --input ./test_data --output ./results --no-gpu

    Raises:
        click.Abort: If app.py or requirements.txt not found, or if execution fails
    """
    app_path_obj = Path(app_path).resolve()
    input_dir_obj = Path(input_dir).resolve()
    output_dir_obj = Path(output_dir).resolve()

    # Check if app.py exists
    app_file = app_path_obj / "app.py"
    if not app_file.exists():
        console.print(f"[red]Error: app.py not found in {app_path}[/red]")
        raise click.Abort()

    # Check requirements.txt
    requirements_file = app_path_obj / "requirements.txt"
    if not requirements_file.exists():
        console.print(f"[red]Error: requirements.txt not found in {app_path}[/red]")
        raise click.Abort()

    venv_path = app_path_obj / venv_name

    console.print(f"[blue]Running MONAI Deploy application from: {app_path_obj}[/blue]")
    console.print(f"[blue]Input: {input_dir_obj}[/blue]")
    console.print(f"[blue]Output: {output_dir_obj}[/blue]")

    # Create output directory if it doesn't exist
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    # Step 1: Create virtual environment if needed
    if not venv_path.exists():
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Creating virtual environment...", total=None)
            try:
                subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                progress.update(task, description="[green]Virtual environment created")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Error creating virtual environment: {e.stderr}[/red]")
                raise click.Abort()
    else:
        console.print(f"[dim]Using existing virtual environment: {venv_name}[/dim]")

    # Determine python executable in venv
    if os.name == "nt":  # Windows
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:  # Unix/Linux/Mac
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"

    # Step 2: Install dependencies
    if not skip_install:
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Installing dependencies...", total=None)

            # Ensure pip/setuptools/wheel are up to date for Python 3.12+
            try:
                # Ensure pip is present and upgraded inside the venv
                subprocess.run(
                    [str(python_exe), "-m", "ensurepip", "--upgrade"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    [str(pip_exe), "install", "--upgrade", "pip", "setuptools", "wheel"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                console.print(
                    f"[yellow]Warning: Failed to upgrade pip/setuptools/wheel: {e.stderr}\nContinuing with dependency installation...[/yellow]"
                )

            # Detect local SDK checkout and install editable to expose local operators
            local_sdk_installed = False
            script_path = Path(__file__).resolve()
            sdk_path = script_path.parent.parent.parent.parent.parent
            if (sdk_path / "monai" / "deploy" ).exists() and (sdk_path / "setup.py").exists():
                console.print(f"[dim]Found local SDK at: {sdk_path}[/dim]")

                # Install local SDK first
                try:
                    subprocess.run(
                        [str(pip_exe), "install", "-e", str(sdk_path)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    local_sdk_installed = True
                except subprocess.CalledProcessError as e:
                    console.print(
                        f"[yellow]Warning: Failed to install local SDK: {e.stderr}[/yellow]"
                    )

            # Install requirements
            try:
                req_path_to_use = requirements_file
                temp_req_path = None

                if local_sdk_installed:
                    # Filter out SDK line to avoid overriding local editable install
                    try:
                        raw = requirements_file.read_text()
                        filtered_lines = []
                        for line in raw.splitlines():
                            s = line.strip()
                            if not s or s.startswith('#'):
                                filtered_lines.append(line)
                                continue
                            if s.lower().startswith('monai-deploy-app-sdk'):
                                continue
                            filtered_lines.append(line)
                        temp_req_path = app_path_obj / ".requirements.filtered.txt"
                        temp_req_path.write_text("\n".join(filtered_lines) + "\n")
                        req_path_to_use = temp_req_path
                        console.print("[dim]Using filtered requirements without monai-deploy-app-sdk[/dim]")
                    except Exception as fr:
                        console.print(f"[yellow]Warning: Failed to filter requirements: {fr}. Proceeding with original requirements.[/yellow]")
                        req_path_to_use = requirements_file

                subprocess.run(
                    [str(pip_exe), "install", "-r", str(req_path_to_use), "-q"],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                # Re-assert local editable SDK in case it was overridden
                if local_sdk_installed:
                    try:
                        subprocess.run(
                            [str(pip_exe), "install", "-e", str(sdk_path)],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                    except subprocess.CalledProcessError as re:
                        console.print(f"[yellow]Warning: Re-installing local SDK failed: {re.stderr}[/yellow]")

                progress.update(task, description="[green]Dependencies installed")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Error installing dependencies: {e.stderr}[/red]")
                raise click.Abort()

    # Step 3: Run the application
    console.print("\n[green]Starting application...[/green]\n")

    # Build command
    cmd = [str(python_exe), str(app_file), "-i", str(input_dir_obj), "-o", str(output_dir_obj)]

    # Add model path if provided
    if model_path:
        cmd.extend(["-m", str(model_path)])

    # Set environment variables
    env = os.environ.copy()
    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""

    try:
        # Run the application
        process = subprocess.Popen(
            cmd,
            cwd=str(app_path_obj),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output in real-time
        if process.stdout:
            for line in process.stdout:
                print(line, end="")

        # Wait for completion
        return_code = process.wait()

        if return_code == 0:
            console.print("\n[green]✓ Application completed successfully![/green]")
            console.print(f"[green]Results saved to: {output_dir_obj}[/green]")
        else:
            console.print(f"\n[red]✗ Application failed with exit code: {return_code}[/red]")
            raise click.Abort()

    except KeyboardInterrupt:
        console.print("\n[yellow]Application interrupted by user[/yellow]")
        process.terminate()
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error running application: {e}[/red]")
        raise click.Abort()


if __name__ == "__main__":
    run()
