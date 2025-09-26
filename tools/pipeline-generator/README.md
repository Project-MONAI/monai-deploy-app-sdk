# Pipeline Generator

A CLI tool for generating [MONAI Deploy](https://github.com/Project-MONAI/monai-deploy-app-sdk) application pipelines from [MONAI Bundles](https://docs.monai.io/en/stable/bundle_intro.html).

## Features

- List available MONAI models from HuggingFace
- Generate complete MONAI Deploy applications from HuggingFace models
- Support for multiple model sources through configuration
- Automatic bundle download and analysis
- Template-based code generation with Jinja2
- Beautiful output formatting with Rich (Python library for rich text and beautiful formatting)

## Platform Requirements

- **Linux/Unix operating systems only**
- Compatible with MONAI Deploy App SDK platform support
- Ubuntu 22.04+ recommended (aligned with main SDK requirements)

## Installation

```bash
# Clone the repository
cd tools/pipeline-generator/

# Install with uv (no virtualenv needed - uv manages it per command)
uv pip install -e ".[dev]"
```

### Running Commands

With uv, you can run commands directly without a prior "install" (pg is the Pipeline Generator command):

```bash
uv run pg --help
uv run pg list
uv run pg gen MONAI/spleen_ct_segmentation --output ./app
```

## Usage

### Complete Workflow Example

```bash
# 1. List available models
uv run pg list

# 2. Generate an application from a model
uv run pg gen MONAI/spleen_ct_segmentation --output my_app

# 3. Run the application
uv run pg run my_app --input /path/to/test/data --output ./results
```

> [!NOTE]  
> Input can be NIfTI files (.nii, .nii.gz) or DICOM series directories Refer to [config.yaml](tools/pipeline-generator/pipeline_generator/config/config.yaml) for details.

> [!NOTE]
> For DICOM input types, refer to the model documentation for DICOM series processing limitations.

### List Available Models

List all models from configured endpoints:

```bash
uv run pg list
```

Show only MONAI Bundles:

```bash
uv run pg list --bundles-only
```

Show only tested models:

```bash
uv run pg list --tested-only
```

Combine filters:

```bash
uv run pg list --bundles-only --tested-only # Show only tested MONAI Bundles
```

Use different output formats:

```bash
uv run pg list --format simple # Simple list format
uv run pg list --format json   # JSON output
uv run pg list --format table  # Default table format
```

Use a custom configuration file:

```bash
uv run pg --config /path/to/config.yaml list
```

### Generate MONAI Deploy Application

Generate an application from a HuggingFace model. Models are specified using the format `organization/model_name` (e.g., `MONAI/spleen_ct_segmentation`):

```bash
uv run pg gen MONAI/spleen_ct_segmentation --output my_app
```

Options:

- `--output, -o`: Output directory for generated app (default: ./output)
- `--app-name, -n`: Custom application class name (default: derived from model name)
- `--format`: Input/output data format (optional): auto, dicom, or nifti (default: auto)
  - For tested models, format is automatically detected from configuration
  - For untested models, attempts detection from model metadata
  - **DICOM Limitation**: Refer to the model documentation for multi-series support.
- `--force, -f`: Overwrite existing output directory

Generate with custom application class name:

```bash
uv run pg gen MONAI/lung_nodule_ct_detection --output lung_app --app-name LungDetectorApp
```

Force overwrite existing directory:

```bash
uv run pg gen MONAI/spleen_ct_segmentation --output test_app --force
```

Override data format (optional - auto-detected for tested models):

```bash
# Force DICOM format instead of auto-detection
uv run pg gen MONAI/some_model --output my_app --format dicom
```

### Run Generated Application

Run a generated application with automatic environment setup:

```bash
uv run pg run my_app --input /path/to/input --output /path/to/output
```

The `run` command will:

1. Create a virtual environment if it doesn't exist
1. Install dependencies from requirements.txt
1. Run the application with the specified input/output

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
uv run pg run my_app --input test_data --output results --skip-install

# Run without GPU
uv run pg run my_app --input test_data --output results --no-gpu

# Use custom model path
uv run pg run my_app --input test_data --output results --model ./custom_model
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

# Additional specific models not under the main organization
additional_models:
  - model_id: "Project-MONAI/exaonepath"
    base_url: "https://huggingface.co"
    description: "ExaOnePath model for digital pathology"
```

## Generated Application Structure

When you run `pg gen`, it creates:

```
output/
├── app.py                  # Main application code
├── app.yaml               # Configuration for MONAI Deploy packaging
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── operators/            # Custom operators (if needed)
│   └── nifti_operators.py
└── model/                 # Downloaded model files
    ├── configs/
    ├── models/
    └── docs/
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=pipeline_generator

# Run specific test file
uv run pytest tests/test_cli.py
```

### Code Quality

```bash
# Format code
uv run black pipeline_generator tests

# Lint code
uv run flake8 pipeline_generator tests

# Type checking
uv run mypy pipeline_generator
```

## Future Commands

The CLI is designed to be extensible. Planned commands include:

- `pg package <app>` - Package an application using the Holoscan CLI packaging tool

## License

This project is part of the MONAI Deploy App SDK and is licensed under the Apache License 2.0. See the main repository's LICENSE file for details.
