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

"""Main CLI entry point for Pipeline Generator."""

import logging
from pathlib import Path
from typing import Optional, List, Set
import click
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

from ..config import load_config
from ..core import HuggingFaceClient, ModelInfo
from ..generator import AppGenerator
from .run import run as run_command


# Set up logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_time=False, show_path=False)],
)
logger = logging.getLogger(__name__)

console = Console()


@click.group()
@click.version_option()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to configuration file")
@click.pass_context
def cli(ctx: click.Context, config: Optional[str]) -> None:
    """Pipeline Generator - Generate MONAI Deploy and Holoscan pipelines from MONAI Bundles."""
    # Store config in context for subcommands
    ctx.ensure_object(dict)
    config_path = Path(config) if config else None
    ctx.obj["config_path"] = config_path
    
    # Load settings
    from ..config.settings import load_config
    settings = load_config(config_path)
    ctx.obj["settings"] = settings


@cli.command()
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "simple", "json"]),
    default="table",
    help="Output format",
)
@click.option("--bundles-only", "-b", is_flag=True, help="Show only MONAI Bundles")
@click.option("--tested-only", "-t", is_flag=True, help="Show only tested models")
@click.pass_context
def list(ctx: click.Context, format: str, bundles_only: bool, tested_only: bool) -> None:
    """List available models from configured endpoints.
    
    Args:
        ctx: Click context containing configuration
        format: Output format (table, simple, or json)
        bundles_only: If True, show only MONAI Bundles
        tested_only: If True, show only tested models
    
    Example:
        pg list --format table --bundles-only
    """

    # Load configuration
    config_path = ctx.obj.get("config_path")
    settings = load_config(config_path)

    # Get set of tested model IDs from configuration
    tested_models = set()
    for endpoint in settings.endpoints:
        for model in endpoint.models:
            tested_models.add(model.model_id)

    # Create HuggingFace client
    client = HuggingFaceClient()

    # Fetch models from all endpoints
    console.print("[blue]Fetching models from HuggingFace...[/blue]")
    models = client.list_models_from_endpoints(settings.get_all_endpoints())

    # Filter for bundles if requested
    if bundles_only:
        models = [m for m in models if m.is_monai_bundle]
    
    # Filter for tested models if requested
    if tested_only:
        models = [m for m in models if m.model_id in tested_models]

    # Sort models by name
    models.sort(key=lambda m: m.display_name)

    # Display results based on format
    if format == "table":
        _display_table(models, tested_models)
    elif format == "simple":
        _display_simple(models, tested_models)
    elif format == "json":
        _display_json(models, tested_models)

    # Summary
    bundle_count = sum(1 for m in models if m.is_monai_bundle)
    tested_count = sum(1 for m in models if m.model_id in tested_models)
    console.print(f"\n[green]Total models: {len(models)} (MONAI Bundles: {bundle_count}, Tested: {tested_count})[/green]")


@cli.command()
@click.argument("model_id")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./output",
    help="Output directory for generated app",
)
@click.option("--app-name", "-n", help="Custom application class name")
@click.option(
    "--format",
    type=click.Choice(["auto", "dicom", "nifti"]),
    default="auto",
    help="Input/output format (optional): auto (uses config for tested models), dicom, or nifti",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing output directory")
@click.pass_context
def gen(ctx: click.Context, model_id: str, output: str, app_name: Optional[str], format: str, force: bool) -> None:
    """Generate a MONAI Deploy application from a HuggingFace model.

    Downloads the specified model from HuggingFace and generates a complete
    MONAI Deploy application including app.py, app.yaml, requirements.txt,
    README.md, and the model files.

    Args:
        model_id: HuggingFace model ID (e.g., 'MONAI/spleen_ct_segmentation')
        output: Output directory for generated app (default: ./output)
        app_name: Custom application class name (default: derived from model)
        format: Input/output format - 'auto' (detect), 'dicom', or 'nifti'
        force: Overwrite existing output directory if True

    Example:
        pg gen MONAI/spleen_ct_segmentation --output my_app

    Raises:
        click.Abort: If output directory exists and force is False
    """
    output_path = Path(output)

    # Check if output directory exists
    if output_path.exists() and not force:
        if any(output_path.iterdir()):  # Directory is not empty
            console.print(
                f"[red]Error: Output directory '{output_path}' already exists and is not empty.[/red]"
            )
            console.print("Use --force to overwrite or choose a different output directory.")
            raise click.Abort()

    # Create generator with settings from context
    settings = ctx.obj.get("settings") if ctx.obj else None
    generator = AppGenerator(settings=settings)

    console.print(f"[blue]Generating MONAI Deploy application for model: {model_id}[/blue]")
    console.print(f"[blue]Output directory: {output_path}[/blue]")
    console.print(f"[blue]Format: {format}[/blue]")

    try:
        # Generate the application
        app_path = generator.generate_app(
            model_id=model_id, output_dir=output_path, app_name=app_name, data_format=format
        )

        console.print("\n[green]✓ Application generated successfully![/green]")
        console.print("\n[bold]Generated files:[/bold]")

        # List generated files
        for file in output_path.rglob("*"):
            if file.is_file():
                relative_path = file.relative_to(output_path)
                console.print(f"  • {relative_path}")

        console.print("\n[bold]Next steps:[/bold]")
        console.print("\n[green]Option 1: Run with poetry (recommended)[/green]")
        console.print(
            f"   [cyan]poetry run pg run {output_path} --input /path/to/input --output /path/to/output[/cyan]"
        )
        console.print("\n[green]Option 2: Run with pg directly[/green]")
        console.print(
            f"   [cyan]pg run {output_path} --input /path/to/input --output /path/to/output[/cyan]"
        )
        console.print("\n[dim]Option 3: Run manually[/dim]")
        console.print("   1. Navigate to the application directory:")
        console.print(f"      [cyan]cd {output_path}[/cyan]")
        console.print("   2. (Optional) Create and activate virtual environment:")
        console.print("      [cyan]python -m venv venv[/cyan]")
        console.print("      [cyan]source venv/bin/activate  # Linux/Mac[/cyan]")
        console.print("      [cyan]# or: venv\\Scripts\\activate  # Windows[/cyan]")
        console.print("   3. Install dependencies:")
        console.print("      [cyan]pip install -r requirements.txt[/cyan]")
        console.print("   4. Run the application:")
        console.print("      [cyan]python app.py -i /path/to/input -o /path/to/output[/cyan]")

    except Exception as e:
        console.print(f"[red]Error generating application: {e}[/red]")
        logger.exception("Generation failed")
        raise click.Abort()


def _display_table(models: List[ModelInfo], tested_models: Set[str]) -> None:
    """Display models in a rich table format.
    
    Args:
        models: List of ModelInfo objects to display
        tested_models: Set of tested model IDs
    """
    table = Table(title="Available Models", show_header=True, header_style="bold magenta")
    table.add_column("Model ID", style="cyan", width=40)
    table.add_column("Name", style="white")
    table.add_column("Type", style="green")
    table.add_column("Status", style="blue", width=10)
    table.add_column("Downloads", justify="right", style="yellow")
    table.add_column("Likes", justify="right", style="red")

    for model in models:
        model_type = "[green]MONAI Bundle[/green]" if model.is_monai_bundle else "Model"
        status = "[bold green]✓ Verified[/bold green]" if model.model_id in tested_models else ""
        table.add_row(
            model.model_id,
            model.display_name,
            model_type,
            status,
            str(model.downloads or "N/A"),
            str(model.likes or "N/A"),
        )

    console.print(table)


def _display_simple(models: List[ModelInfo], tested_models: Set[str]) -> None:
    """Display models in a simple list format.
    
    Shows each model with emoji indicators:
    - 📦 for MONAI Bundle, 📄 for regular model
    - ✓ for tested models
    
    Args:
        models: List of ModelInfo objects to display
        tested_models: Set of tested model IDs
    """
    for model in models:
        bundle_marker = "📦" if model.is_monai_bundle else "📄"
        tested_marker = " ✓" if model.model_id in tested_models else ""
        console.print(f"{bundle_marker} {model.model_id} - {model.display_name}{tested_marker}")


def _display_json(models: List[ModelInfo], tested_models: Set[str]) -> None:
    """Display models in JSON format.
    
    Outputs a JSON array of model information suitable for programmatic consumption.
    
    Args:
        models: List of ModelInfo objects to display
        tested_models: Set of tested model IDs
    """
    import json

    data = [
        {
            "model_id": m.model_id,
            "name": m.display_name,
            "is_monai_bundle": m.is_monai_bundle,
            "is_tested": m.model_id in tested_models,
            "downloads": m.downloads,
            "likes": m.likes,
            "tags": m.tags,
        }
        for m in models
    ]

    console.print_json(json.dumps(data, indent=2))


# Add the run command to CLI
cli.add_command(run_command)


if __name__ == "__main__":
    cli()
