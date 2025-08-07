# Phase 1: MONAI Deploy Application for Spleen CT Segmentation

## Date: July 2025

## Bundle Structure Analysis

### Overview
The spleen_ct_segmentation bundle from HuggingFace contains a complete MONAI Bundle with:
- Model files: Both PyTorch (.pt) and TorchScript (.ts) formats
- Configuration files: metadata.json, inference.json, and various training/evaluation configs
- Pre-computed evaluation results in the eval/ directory

### Key Files and Their Purpose

1. **Model Files** (`models/` directory):
   - `model.pt`: PyTorch state dict (18MB)
   - `model.ts`: TorchScript model (19MB) - **We'll use this for inference**

2. **Configuration Files** (`configs/` directory):
   - `metadata.json`: Bundle metadata, model specs, input/output formats
   - `inference.json`: Complete inference pipeline configuration with transforms

3. **Expected Output Structure** (`eval/` directory):
   - Individual folders for each test case (e.g., `spleen_1/`, `spleen_7/`)
   - Output files named: `{case_name}_trans.nii.gz`
   - Format: NIfTI segmentation masks (argmax applied, single channel)

### Model Specifications (from metadata.json)

**Input Requirements:**
- Type: CT image (Hounsfield units)
- Format: NIfTI
- Channels: 1 (grayscale)
- Patch size: [96, 96, 96]
- Dtype: float32
- Value range: [0, 1] (after normalization)

**Output Format:**
- Type: Segmentation mask
- Channels: 2 (background, spleen)
- Spatial shape: [96, 96, 96] patches
- Dtype: float32
- Value range: [0, 1] (probabilities before argmax)

**Model Architecture:**
- 3D UNet
- Channels: [16, 32, 64, 128, 256]
- Strides: [2, 2, 2, 2]
- Normalization: Batch normalization

### Preprocessing Pipeline (from inference.json)

1. **LoadImaged**: Load NIfTI files
2. **EnsureChannelFirstd**: Ensure channel-first format
3. **Orientationd**: Reorient to RAS coordinate system
4. **Spacingd**: Resample to [1.5, 1.5, 2.0] mm spacing
5. **ScaleIntensityRanged**: 
   - Window: [-57, 164] HU → [0, 1]
   - Clip values outside range
6. **EnsureTyped**: Convert to appropriate tensor type

### Inference Strategy

- **SlidingWindowInferer**:
  - ROI size: [96, 96, 96]
  - Batch size: 4
  - Overlap: 0.5 (50%)

### Postprocessing Pipeline

1. **Activationsd**: Apply softmax to get probabilities
2. **Invertd**: Invert preprocessing transforms (back to original space)
3. **AsDiscreted**: Apply argmax to get discrete labels
4. **SaveImaged**: Save as NIfTI with specific naming convention

## Implementation Decisions

### 1. Dynamic Configuration Loading
- **CRITICAL REQUIREMENT**: All configurations must be loaded from `inference.json` at runtime
- No hardcoded preprocessing/postprocessing parameters
- Parse transforms dynamically using MONAI Bundle ConfigParser
- Support for dynamic model loading based on bundle structure

### 2. Pure MONAI Deploy App SDK Usage
- **CRITICAL REQUIREMENT**: Use only MONAI Deploy SDK operators and APIs
- Cannot use MONAI Core transforms directly
- Must implement or extend MONAI Deploy operators for all functionality
- Create custom operators where existing ones don't meet requirements

### 3. Operator Architecture

#### Modified MonaiBundleInferenceOperator
The existing `MonaiBundleInferenceOperator` expects a ZIP file, but we need to support directory structure:
- Override `_init_config` to work with directory paths
- Skip ZIP extraction logic
- Load model directly from `models/model.ts`
- Parse transforms from `configs/inference.json`

#### Pipeline Structure
Following the standard pattern from design spec:
```
[Source/NiftiDataLoader] → [Preprocessing Op] → [Inference Op] → [Postprocessing Op] → [Sink/NiftiWriter]
```

### 4. Key Implementation Components

#### Custom Bundle Loader
```python
class DirectoryBundleLoader:
    """Loads MONAI Bundle from directory structure instead of ZIP"""
    - Parse metadata.json for model specifications
    - Load inference.json for transform configurations
    - Locate and load TorchScript model
```

#### Extended MonaiBundleInferenceOperator
```python
class ExtendedMonaiBundleInferenceOperator(MonaiBundleInferenceOperator):
    """Extends base operator to support directory bundles"""
    - Override bundle loading mechanism
    - Support directory path instead of ZIP path
    - Maintain compatibility with existing interfaces
```

#### Transform Mapping Strategy
Since we must use pure MONAI Deploy SDK:
- Map MONAI Core transform names to MONAI Deploy equivalents
- Create custom operators for transforms not available in Deploy SDK
- Ensure all transforms are loaded dynamically from config

### 5. Configuration Strategy

#### app.yaml Structure
```yaml
app:
  name: spleen_ct_segmentation
  version: 1.0.0

resources:
  bundle_path: "tools/pipeline-generator/phase_1/spleen_ct_segmentation"
  
operators:
  - name: nifti_loader
    args:
      input_dir: "/home/vicchang/Downloads/Task09_Spleen/Task09_Spleen/imagesTs"
  
  - name: bundle_inference
    args:
      bundle_path: "@resources.bundle_path"
      config_names: ["inference"]
      model_name: ""
      
  - name: nifti_writer
    args:
      output_dir: "output"
```

## Limitations and Assumptions

1. **Input Format**: Assumes all inputs are NIfTI files (.nii.gz)
2. **Single Model**: Designed for single TorchScript model inference
3. **Memory**: Sliding window inference helps with memory but still requires substantial GPU memory
4. **Batch Size**: Currently processes one volume at a time
5. **Transform Compatibility**: Some MONAI Core transforms may not have direct Deploy SDK equivalents

## Testing Approach

1. **Unit Tests**:
   - Test bundle loading from directory
   - Verify preprocessing pipeline matches inference.json
   - Check model loading and inference
   - Validate dynamic configuration parsing

2. **Integration Tests**:
   - Process test data from `/home/vicchang/Downloads/Task09_Spleen/Task09_Spleen/imagesTs`
   - Compare outputs with reference in `eval/` directory
   - Validate file naming and directory structure

3. **Validation Metrics**:
   - Dice score comparison with reference outputs
   - Visual inspection of segmentation masks
   - File size and format validation
   - Exact match of output directory structure

## Dependencies and Versions

Based on metadata.json:
- MONAI: 1.4.0
- PyTorch: 2.4.0
- NumPy: 1.24.4
- Additional:
  - nibabel: 5.2.1
  - pytorch-ignite: 0.4.11
  - MONAI Deploy App SDK: latest

## Next Steps

1. Implement directory-based bundle loader
2. Extend MonaiBundleInferenceOperator for directory support
3. Create transform mapping utilities
4. Build complete pipeline with pure Deploy SDK operators
5. Test with provided data
6. Compare outputs with reference results
7. Document any deviations or improvements

## Code Structure Plan

```
tools/pipeline-generator/phase_1/
├── spleen_seg_app/
│   ├── __init__.py
│   ├── app.py                     # Main application with pure Deploy SDK
│   ├── app.yaml                   # Configuration (dynamic loading)
│   ├── operators/
│   │   ├── __init__.py
│   │   ├── directory_bundle_inference_operator.py  # Extended operator
│   │   └── nifti_operators.py    # NIfTI I/O operators
│   └── utils/
│       ├── __init__.py
│       ├── bundle_parser.py       # Directory bundle parsing
│       └── transform_mapper.py    # Maps config transforms to Deploy SDK
└── test_results/
    └── comparison_report.md
```

## Key Implementation Notes

1. **Dynamic Loading**: All preprocessing/postprocessing parameters MUST come from inference.json
2. **Pure Deploy SDK**: No direct MONAI Core imports or transforms
3. **Directory Support**: Modify bundle loading to work with unpacked directory structure
4. **Transform Compatibility**: Create mapping layer for transforms not in Deploy SDK
5. **Output Matching**: Must exactly match reference output structure and naming

## Critical Success Criteria

1. ✓ Application loads all configurations from inference.json at runtime
2. ✓ Uses only MONAI Deploy App SDK operators and APIs
3. ✓ Processes test data correctly with dynamic transforms
4. ✓ Outputs match expected results in structure and content
5. ✓ No hardcoded preprocessing/postprocessing parameters 