# Pipeline Generator

A CLI tool for generating MONAI Deploy and Holoscan pipelines from MONAI Bundles.

## Features

- List available MONAI models from HuggingFace
- Generate complete MONAI Deploy applications from HuggingFace models
- Support for multiple model sources through configuration
- Automatic bundle download and analysis
- Template-based code generation with Jinja2
- Beautiful output formatting with Rich

## Installation

```bash
# Clone the repository
cd tools/pipeline-generator/

# Install with Poetry
poetry install
```

### Running Commands

With Poetry 2.0+, you can run commands in two ways:

**Option 1: Using `poetry run` (Recommended)**
```bash
poetry run pg --help
poetry run pg list
poetry run pg gen MONAI/model_name --output ./app
```

**Option 2: Activating the environment**
```bash
# On Linux/Mac
source $(poetry env info --path)/bin/activate

# On Windows
$(poetry env info --path)\Scripts\activate

# Then run commands directly
pg --help
```

> **Note**: Poetry 2.0 removed the `poetry shell` command. Use `poetry run` or activate the environment manually as shown above.

## Usage

### Complete Workflow Example

```bash
# 1. List available models
poetry run pg list

# 2. Generate an application from a model
poetry run pg gen MONAI/spleen_ct_segmentation --output my_app

# 3. Run the application
poetry run pg run my_app --input /path/to/test/data --output ./results
```

### List Available Models

List all models from configured endpoints:

```bash
poetry run pg list
```

Show only MONAI Bundles:

```bash
poetry run pg list --bundles-only
```

Show only tested models:

```bash
poetry run pg list --tested-only
```

Combine filters:

```bash
poetry run pg list --bundles-only --tested-only  # Show only tested MONAI Bundles
```

Use different output formats:

```bash
poetry run pg list --format simple  # Simple list format
poetry run pg list --format json    # JSON output
poetry run pg list --format table   # Default table format
```

Use a custom configuration file:

```bash
poetry run pg --config /path/to/config.yaml list
```

### Generate MONAI Deploy Application

Generate an application from a HuggingFace model:

```bash
poetry run pg gen MONAI/spleen_ct_segmentation --output my_app
```

Options:
- `--output, -o`: Output directory for generated app (default: ./output)
- `--app-name, -n`: Custom application class name (default: derived from model)
- `--format`: Input/output format (optional): auto, dicom, or nifti (default: auto)
  - For tested models, format is automatically detected from configuration
  - For untested models, attempts detection from model metadata
- `--force, -f`: Overwrite existing output directory

Generate with custom application class name:

```bash
poetry run pg gen MONAI/lung_nodule_ct_detection --output lung_app --app-name LungDetectorApp
```

Force overwrite existing directory:

```bash
poetry run pg gen MONAI/example_spleen_segmentation --output test_app --force
```

Override data format (optional - auto-detected for tested models):

```bash
# Force DICOM format instead of auto-detection
poetry run pg gen MONAI/some_model --output my_app --format dicom
```

### Run Generated Application

Run a generated application with automatic environment setup:

```bash
poetry run pg run my_app --input /path/to/input --output /path/to/output
```

The `run` command will:
1. Create a virtual environment if it doesn't exist
2. Install dependencies from requirements.txt
3. Run the application with the specified input/output

Options:
- `--input, -i`: Input data directory (required)
- `--output, -o`: Output directory for results (default: ./output)
- `--model, -m`: Override model/bundle path
- `--venv-name`: Virtual environment directory name (default: .venv)
- `--skip-install`: Skip dependency installation
- `--gpu/--no-gpu`: Enable/disable GPU support (default: enabled)

Examples:

```bash
# Skip dependency installation (if already installed)
poetry run pg run my_app --input test_data --output results --skip-install

# Run without GPU
poetry run pg run my_app --input test_data --output results --no-gpu

# Use custom model path
poetry run pg run my_app --input test_data --output results --model ./custom_model
```

## Configuration

The tool uses a YAML configuration file to define model sources. By default, it looks for `config.yaml` in the package directory.

Example configuration:

```yaml
# HuggingFace endpoints to scan for MONAI models
endpoints:
  - organization: "MONAI"
    base_url: "https://huggingface.co"
    description: "Official MONAI organization models"

# Additional specific models
additional_models:
  - model_id: "Project-MONAI/exaonepath"
    base_url: "https://huggingface.co"
    description: "ExaOnePath model"
```

## Generated Application Structure

When you run `pg gen`, it creates:

```
output/
├── app.py                  # Main application code
├── app.yaml               # Configuration for packaging
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── operators/            # Custom operators (if needed)
│   └── nifti_operators.py
└── model/                # Downloaded MONAI Bundle
    ├── configs/
    ├── models/
    └── docs/
```

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=pipeline_generator

# Run specific test file
poetry run pytest tests/test_cli.py
```

### Code Quality

```bash
# Format code
poetry run black pipeline_generator tests

# Lint code
poetry run flake8 pipeline_generator tests

# Type checking
poetry run mypy pipeline_generator
```

## Future Commands

The CLI is designed to be extensible. Planned commands include:

- `pg package <app>` - Package an application using holoscan-cli

## License

This project is part of the MONAI Deploy App SDK. 