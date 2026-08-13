# Architecture Map: MONAI Deploy App SDK

## Overview

MONAI Deploy App SDK is a Python framework for building medical imaging inference applications. It sits on top of NVIDIA Holoscan SDK and provides operators, data models, and tools to design DICOM-based AI inference pipelines for healthcare imaging.

**Branch:** `nnunet-fast` (forked from upstream `Project-MONAI/monai-deploy-app-sdk`)

## Core Modules

### `monai/deploy/core/` — Domain & Runtime

| Module | Purpose |
|--------|---------|
| `domain/` | Data models: `DICOMSeries`, `DICOMStudy`, `DICOMSOPInstance`, `Image`, `DataPath` |
| `datastores/` | Key-value abstraction for passing data between operators (`DataStore`, `MemoryDataStore`) |
| `models/` | Model abstraction layer: `Model`, `TorchModel`, `TritonModel`, `NamedModel` |
| `app_context.py` | Shared context passed across operators |
| `runtime_env.py` | Runtime environment configuration |
| `io_type.py` | I/O type definitions for operator inputs/outputs |

### `monai/deploy/operators/` — Pipeline Operators

Operators are the building blocks of inference pipelines. They connect via a DAG (Directed Acyclic Graph):

| Operator | Purpose |
|----------|---------|
| `DICOMDataLoaderOperator` | Loads DICOM data from filesystem |
| `DICOMSeriesSelectorOperator` | Selects relevant DICOM series from a study |
| `DICOMSeriesToVolumeOperator` | Converts DICOM series to 3D volume tensors |
| `MonaiSegInferenceOperator` | Runs in-process PyTorch segmentation inference |
| `MonaiBundleInferenceOperator` | Runs inference using MONAI Bundle models |
| `MonetBundleInferenceOperator` | Runs inference using MONAI Bundle with MONet (NVIDIA-accelerated) |
| `DICOMSegWriterOperator` | Writes segmentation results as DICOM-SEG |
| `DICOMTextSRWriterOperator` | Writes structured report as DICOM-SR |
| `STLConversionOperator` | Converts segmentation to STL mesh |
| `NIIDataLoaderOperator` | Loads NIfTI files (for testing/development) |
| `ClaraVizOperator` | Visualization via Clara Viz |
| `PublisherOperator` | Publishes results to external systems |
| `DecoderNVImgCodec` | GPU-accelerated DICOM decoding via nvidia-nvimgcodec |

### `monai/deploy/utils/` — Utilities

- `importutil.py` — Dynamic import helpers (highest PageRank file, 7K+ incoming refs)
- `deviceutil.py` — GPU/CUDA device utilities
- `fileutil.py` — Filesystem utilities
- `sizeutil.py` — Size/format utilities
- `argparse_types.py` — CLI argument type helpers
- `spinner.py` — Progress spinner

### `monai/deploy/conditions/` — Pipeline Conditions

Conditional logic for DAG-based execution.

### `monai/deploy/graphs/` — DAG Definitions

Pipeline graph definitions.

### `monai/deploy/exceptions/` — Error Types

`MONAIAppSdkError` base class with subclasses: `ItemAlreadyExistsError`, `ItemNotExistsError`, `IOMappingError`, `UnknownTypeError`, `UnsupportedOperationError`, `WrongValueError`.

## Vendored: nnUNet

**Location:** `nnUNet/` (no `.git`, vendored copy of `nnunetv2`)

nnU-Net v2 is a self-configuring semantic segmentation framework. The vendored copy is installed as an editable package alongside this SDK. Key components:

| Component | Purpose |
|-----------|---------|
| `nnunetv2/training/nnUNetTrainer/` | Training loop, trainer variants (Primus, Pretrained) |
| `nnunetv2/inference/predict_from_raw_data.py` | Inference pipeline (`nnUNetPredictor`) |
| `nnunetv2/utilities/plans_handling/` | Plans/configuration handling |

**Complexity hotspots:**
- `nnUNetTrainer` (score 160) — main training loop
- `nnUNetPredictor` (score 100) — inference pipeline
- `MonaiBundleInferenceOperator` (score 116) — SDK's MONAI Bundle integration

## Example Apps (`examples/apps/`)

54 example application files demonstrating various use cases:

- `simple_imaging_app` — Basic image processing pipeline
- `ai_livertumor_seg_app` — Liver tumor segmentation
- `ai_spleen_seg_app` — Spleen segmentation
- `ai_unetr_seg_app` — UNETR model segmentation
- `mednist_classifier_monaideploy` — MedNIST classification
- `ai_remote_infer_app` — Remote inference via Triton
- `cchmc_nnunet_fifteen_ckpt_app` — nnUNet with 15 checkpoint loading (this branch's focus)
- `convert_nnunet_ckpts.py` — nnUNet checkpoint conversion utility

## Tools (`tools/`)

- `pipeline-generator/` — CLI tool for generating pipeline scaffolds (21 files, `AppGenerator` class)

## Platform Adapters (`platforms/`)

- `nuance_pin/` — Nuance Power Intelligence Network integration (5 files)
- `aidoc/` — Aidoc integration (4 files)

## Key Architecture Patterns

1. **Operator DAG Model** — Applications are DAGs of operators connected via DataStore
2. **Model Abstraction** — Unified interface for local (PyTorch) and remote (Triton) inference
3. **DICOM-First** — All I/O is DICOM-based (DICOMDIR, DICOM-SEG, DICOM-SR)
4. **Holoscan Foundation** — Built on NVIDIA Holoscan SDK for GPU-accelerated pipeline execution
5. **Plugin Architecture** — Platform-specific adapters for deployment targets

## Data Flow (Typical Segmentation App)

```
DICOMDataLoaderOperator
  → DICOMSeriesSelectorOperator (selects target series)
  → DICOMSeriesToVolumeOperator (converts to 3D tensor)
  → MonaiSegInferenceOperator / nnUNetPredictor (runs inference)
  → DICOMSegWriterOperator (writes results as DICOM-SEG)
  → DICOMTextSRWriterOperator (writes structured report)
```

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| holoscan-cu13 | >=4.0.0, <4.3.0 | Core pipeline engine (GPU) |
| holoscan-cli | >=4.0.0, <4.3.0 | CLI tools |
| tritonclient[all] | >=2.53.0 | Remote inference via Triton |
| monai | (pinned) | Medical imaging deep learning framework |
| torch | (pinned) | Deep learning framework |
| nvidia-nvimgcodec-cu13 | (pinned) | GPU-accelerated image decoding |
| numpy | >=1.21.6 | Array operations |
| typeguard | >=3.0.0 | Runtime type checking |

## Entry Points

- `monai-deploy` CLI — packaging, running apps
- `tools/pipeline-generator/pipeline_generator/cli/main.py` — pipeline scaffold generation
- Various example app entry points in `examples/apps/`

---
*Mapped: 2025-07-19 via GSD codebase map*
