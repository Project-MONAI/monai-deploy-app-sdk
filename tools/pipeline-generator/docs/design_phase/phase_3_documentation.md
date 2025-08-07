# Phase 3: Generate Command Implementation

## Date: August 2025

## Overview

Successfully implemented the `gen` command for the Pipeline Generator CLI that generates complete MONAI Deploy applications from HuggingFace models. The command downloads MONAI Bundles and creates ready-to-run applications with all necessary files.

## Implementation Decisions

### 1. Architecture Overview

Created a modular generator system with:
- **BundleDownloader**: Downloads and analyzes MONAI Bundles from HuggingFace
- **AppGenerator**: Orchestrates the generation process using Jinja2 templates
- **Templates**: Separate templates for different application types (DICOM vs NIfTI)

### 2. Bundle Download Strategy

**HuggingFace Integration:**
- Uses `snapshot_download` to get all bundle files
- Downloads to `model/` subdirectory within output
- Preserves original bundle structure
- Caches downloads for efficiency

**Bundle Analysis:**
- Reads `metadata.json` for model information
- Reads `inference.json` for pipeline configuration
- Auto-detects model file location (.ts, .pt, .onnx)
- Handles various bundle directory structures

### 3. Template System

**Jinja2 Templates Created:**
1. `app_dicom.py.j2` - For CT/MR modalities using DICOM I/O
2. `app_nifti.py.j2` - For other modalities using NIfTI I/O
3. `app.yaml.j2` - Application configuration
4. `requirements.txt.j2` - Dependencies
5. `README.md.j2` - Documentation
6. `nifti_operators.py.j2` - Custom NIfTI operators

**Template Context:**
- Model metadata (name, version, task, modality)
- Extracted organ/structure name
- Input/output format decision
- Dynamic operator selection

### 4. Application Type Selection

**DICOM vs NIfTI Decision Logic:**
```python
use_dicom = modality in ['CT', 'MR', 'MRI']
```

**DICOM Applications Include:**
- DICOMDataLoaderOperator
- DICOMSeriesSelectorOperator
- DICOMSeriesToVolumeOperator
- DICOMSegmentationWriterOperator
- STLConversionOperator (for segmentation)

**NIfTI Applications Include:**
- Custom NiftiDataLoaderOperator
- Custom NiftiDataWriterOperator
- Dynamic output naming from bundle config

### 5. CLI Command Design

**Command Structure:**
```bash
pg gen <model_id> [OPTIONS]
```

**Options:**
- `--output, -o`: Output directory (default: ./output)
- `--app-name, -n`: Custom application class name
- `--force, -f`: Overwrite existing directory

**User Experience:**
- Progress indicators during download
- Clear error messages
- Helpful next steps after generation
- File listing of generated content

## Code Structure

### Generator Module
```
pipeline_generator/generator/
├── __init__.py
├── bundle_downloader.py    # HuggingFace download logic
└── app_generator.py        # Main generation orchestration
```

### Template Files
```
pipeline_generator/templates/
├── app_dicom.py.j2         # DICOM-based applications
├── app_nifti.py.j2         # NIfTI-based applications
├── app.yaml.j2             # Configuration
├── requirements.txt.j2     # Dependencies
├── README.md.j2            # Documentation
└── nifti_operators.py.j2   # Custom operators
```

## Key Features Implemented

### 1. Smart Bundle Analysis

- Automatic metadata extraction
- Fallback to sensible defaults
- Model file detection across various structures
- Task and modality identification

### 2. Dynamic Application Generation

- Appropriate I/O operators based on modality
- Organ-specific configurations
- Preserves bundle's inference configuration
- Follows MONAI Deploy best practices

### 3. Complete Application Package

Generated applications include:
- Executable `app.py` with proper pipeline
- Configuration `app.yaml` for packaging
- `requirements.txt` with all dependencies
- Comprehensive `README.md` with usage instructions
- Downloaded model files in `model/` directory

### 4. Template Flexibility

Templates support:
- Different tasks (segmentation, classification, etc.)
- Various modalities (CT, MR, etc.)
- Custom naming and branding
- Dynamic operator inclusion

## Testing Results

### Unit Tests

Created comprehensive tests for:
- BundleDownloader functionality
- AppGenerator logic
- Template rendering
- Context preparation

All 8 tests passing successfully.

### Integration Test

Successfully generated application for `MONAI/spleen_ct_segmentation`:
- Downloaded 14 files from HuggingFace
- Generated DICOM-based application
- Created all required files
- Proper organ detection (Spleen)
- Correct modality handling (CT)

## Generated Application Structure

```
output/
├── app.py                  # Main application
├── app.yaml               # Configuration
├── requirements.txt       # Dependencies
├── README.md             # Documentation
└── model/                # Downloaded bundle
    ├── configs/
    │   ├── metadata.json
    │   ├── inference.json
    │   └── ...
    ├── models/
    │   ├── model.ts
    │   └── model.pt
    └── docs/
        └── README.md
```

## Usage Example

```bash
# Generate application for spleen segmentation
pg gen MONAI/spleen_ct_segmentation --output my_app

# Generate with custom class name
pg gen MONAI/lung_nodule_ct_detection --output lung_app --app-name LungDetectorApp

# Force overwrite existing directory
pg gen MONAI/example_spleen_segmentation --output test_app --force
```

## Limitations and Assumptions

1. **Bundle Structure**: Assumes standard MONAI Bundle structure
2. **Model Format**: Prioritizes TorchScript (.ts) over other formats
3. **Metadata**: Falls back to defaults if metadata.json missing
4. **Organ Detection**: Limited to common organ names
5. **Task Support**: Optimized for segmentation tasks

## Dependencies Used

- **jinja2**: Template engine for code generation
- **huggingface-hub**: Already present for model downloading
- Existing Pipeline Generator infrastructure

## Next Steps

This implementation enables:
- Phase 4: `run` command to execute generated applications
- Phase 5: `package` command using holoscan-cli
- Phase 6: Holoscan SDK pipeline generation option

## Success Criteria Met

1. ✅ Generate app.py with end-to-end MONAI Deploy pipeline
2. ✅ Generate app.yaml with configurations
3. ✅ Download all model files from HuggingFace
4. ✅ Use Jinja2 for main code templates
5. ✅ Use Pydantic/dataclasses for configuration models
6. ✅ YAML library for configuration generation
7. ✅ Output structure matches specification 