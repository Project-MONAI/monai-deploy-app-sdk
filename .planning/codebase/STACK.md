# Tech Stack: MONAI Deploy App SDK

## Language & Runtime

| Component | Choice | Notes |
|-----------|--------|-------|
| Language | Python 3.10–3.13 | Primary development language |
| GPU Runtime | CUDA 13.0+ | Required by Holoscan SDK |
| OS | Ubuntu 22.04 (glibc 2.35+) | Holoscan requirement |
| Package Manager | pip / setuptools | Build via `setup.py` + `versioneer` |

## Core Framework

| Component | Choice | Version | Notes |
|-----------|--------|---------|-------|
| Pipeline Engine | NVIDIA Holoscan SDK | 4.0–4.2.x | DAG-based execution, GPU operators |
| Medical Imaging | MONAI | (pinned by SDK) | Pre/post transforms, bundle inference |
| Deep Learning | PyTorch | (pinned by MONAI) | Model inference, training |
| DICOM I/O | Holoscan + PyDicom | — | DICOMDIR loading, DICOM-SEG/SR writing |
| GPU Decoding | nvidia-nvimgcodec-cu13 | — | Hardware-accelerated DICOM decode |
| Remote Inference | Triton Inference Server | client >=2.53.0 | Remote model hosting |

## nnUNet Integration

| Component | Choice | Notes |
|-----------|--------|-------|
| Framework | nnU-Net v2 (`nnunetv2`) | Vendored at `nnUNet/`, editable install |
| Purpose | State-of-the-art medical image segmentation | Self-configuring U-Net pipeline |
| Integration | In-proc Python import | SDK operators call nnUNet directly |
| Branch Focus | `nnunet-fast` | Multi-checkpoint loading (15 checkpoints) |

## Development Tools

| Component | Choice | Purpose |
|-----------|--------|---------|
| Code Formatting | Black | Line length 120 |
| Type Checking | Pyright, pytype | Static analysis |
| Testing | pytest | Unit + integration tests |
| CI/CD | GitHub Actions | PR validation, release |
| Documentation | Sphinx / ReadTheDocs | API docs, tutorials |
| Versioning | versioneer | Semantic versioning from git tags |
| Packaging | Docker (MONAI Application Package) | Portable deployment via `monai-deploy package` |

## File Formats

| Format | Role |
|--------|------|
| DICOMDIR | Input — study organization |
| DICOM-SEG | Output — segmentation results |
| DICOM-SR | Output — structured reports |
| NIfTI (.nii/.nii.gz) | Development/testing only |
| STL | Output — mesh for 3D visualization |
| JSON/YAML | App configuration (`app.yaml`) |

## Key Constraints

- **GPU required:** NVIDIA GPU with 8GB+ VRAM for inference
- **CUDA 13 only:** Holoscan 4.x requires CUDA 13 (not CUDA 12)
- **Ubuntu 22.04+:** glibc 2.35+ required by Holoscan
- **nnUNet is vendored:** Not installed from PyPI — local editable copy at `nnUNet/`
- **Python 3.10–3.13:** Supported range per `pyproject.toml`
- **GPU tensor handoff:** `MemoryData` does not exist in the holoscan-cu13 4.2 Python API — GPU buffers cross operator boundaries as `holoscan.core.Tensor` (DLPack, `device_type == kDLDeviceCUDA`). Downstream operators re-wrap via DLPack (decided 2026-08-18, Phase 1 Plan 01)

---
*Mapped: 2025-07-19 via GSD codebase map*
