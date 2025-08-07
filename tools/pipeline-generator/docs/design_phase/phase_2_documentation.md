# Phase 2: Pipeline Generator CLI Tool

## Date: August 2025

## Overview

Successfully implemented a Pipeline Generator CLI tool with a `list` command that fetches available MONAI models from HuggingFace. The tool is designed to be extensible for future commands (generate, run, package).

## Implementation Decisions

### 1. Project Structure

Used Poetry for dependency management with a clean, modular structure:

```
tools/pipeline-generator/phase_2/
├── pipeline_generator/
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py          # CLI entry point
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.yaml      # Default configuration
│   │   └── settings.py      # Configuration models
│   └── core/
│       ├── __init__.py
│       ├── hub_client.py    # HuggingFace API client
│       └── models.py        # Data models
├── tests/
│   ├── __init__.py
│   ├── test_cli.py
│   ├── test_models.py
│   └── test_settings.py
├── pyproject.toml
└── README.md
```

### 2. Configuration System

**YAML Configuration Format:**
- Supports organization-level scanning
- Supports individual model references
- Extensible for Phase 7 additional models
- Default configuration includes MONAI organization

**Configuration Loading:**
- Loads from specified path via `--config` flag
- Falls back to package's config.yaml
- Defaults to MONAI organization if no config found

### 3. CLI Design

**Command Structure:**
```bash
pg [OPTIONS] COMMAND [ARGS]...
```

**Global Options:**
- `--config, -c`: Path to configuration file
- `--version`: Show version
- `--help`: Show help

**List Command Options:**
- `--format, -f`: Output format (table/simple/json)
- `--bundles-only, -b`: Show only MONAI Bundles

### 4. HuggingFace Integration

**Client Features:**
- Uses official `huggingface_hub` library
- Fetches models from organizations
- Fetches individual models by ID
- Detects MONAI Bundles by:
  - Checking for "monai" in tags
  - Looking for metadata.json in files

**Model Information Captured:**
- Model ID, name, author
- Downloads, likes
- Creation/update dates
- Tags
- Bundle detection

### 5. Output Formatting

**Rich Integration:**
- Beautiful table formatting
- Color-coded output
- Progress indicators
- JSON export capability

**Format Options:**
1. **Table** (default): Rich table with columns
2. **Simple**: One-line per model with emoji indicators
3. **JSON**: Machine-readable format

## Code Structure and Key Classes

### 1. Data Models (Pydantic)

**ModelInfo:**
- Represents a HuggingFace model
- Properties for display formatting
- Bundle detection flag

**Endpoint:**
- Configuration for model sources
- Supports organization or specific model ID

**Settings:**
- Main configuration container
- YAML loading capability
- Merges endpoints and additional models

### 2. HuggingFace Client

**HuggingFaceClient:**
- Wraps HuggingFace Hub API
- Lists models from organizations
- Fetches individual model info
- Processes all configured endpoints

### 3. CLI Implementation

**Click Framework:**
- Command group for extensibility
- Context passing for configuration
- Rich integration for output

## Testing Approach

### Unit Tests Coverage

1. **Model Tests** (`test_models.py`):
   - ModelInfo creation and properties
   - Display name generation
   - Short ID extraction

2. **Settings Tests** (`test_settings.py`):
   - Endpoint configuration
   - YAML loading
   - Default configuration

3. **CLI Tests** (`test_cli.py`):
   - Command invocation
   - Output formats
   - Filtering options
   - Configuration loading

### Test Strategy

- Used pytest with fixtures
- Mocked external API calls
- Tested all output formats
- Verified configuration handling

## Dependencies and Versions

**Main Dependencies:**
- Python: ^3.12
- click: ^8.2.1 (CLI framework)
- pyyaml: ^6.0.2 (Configuration)
- huggingface-hub: ^0.34.3 (API access)
- pydantic: ^2.11.7 (Data validation)
- rich: ^14.1.0 (Beautiful output)

**Development Dependencies:**
- pytest: ^8.4.1
- pytest-cov: ^6.2.1
- black: ^25.1.0
- flake8: ^7.3.0
- mypy: ^1.17.1
- types-pyyaml: ^6.0.12

## Extensibility for Future Phases

The CLI is designed to easily add new commands:

```python
@cli.command()
def gen():
    """Generate MONAI Deploy application."""
    pass

@cli.command()
def run():
    """Run generated application."""
    pass

@cli.command()
def package():
    """Package application."""
    pass
```

## Usage Examples

```bash
# List all models
pg list

# Show only MONAI Bundles
pg list --bundles-only

# Export as JSON
pg list --format json > models.json

# Use custom config
pg --config myconfig.yaml list
```

## Limitations and Assumptions

1. **API Rate Limits**: HuggingFace API has rate limits
2. **Bundle Detection**: Heuristic-based, may miss some bundles
3. **Network Dependency**: Requires internet connection
4. **Large Organizations**: May take time for organizations with many models

## Success Criteria Met

1. ✅ CLI tool called `pg` with `list` command
2. ✅ Fetches models from HuggingFace MONAI organization
3. ✅ YAML configuration for endpoints
4. ✅ Poetry for dependency management
5. ✅ Comprehensive unit tests
6. ✅ Extensible for future commands
7. ✅ Support for Phase 7 additional models

## Next Steps

This foundation enables:
- Phase 3: Generate command implementation
- Phase 4: Run command for generated apps
- Phase 5: Package command using holoscan-cli
- Phase 6: Holoscan SDK pipeline generation 