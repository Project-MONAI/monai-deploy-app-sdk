# cchmc-nnunet-fast: Holoscan-Native nnUNet Inference

## What This Is

A new example app (`cchmc-nnunet-fast`) that replaces the current nnUNet wrapper approach with Holoscan-native GPU operators for medical image segmentation inference. The current `cchmc_nnunet_fifteen_ckpt_app` calls nnUNet as a Python wrapper — data bounces through CPU, intermediate `.npz` files hit disk, and Holoscan's GPU streaming/async capabilities go unused. This app rebuilds the inference pipeline so preprocessing, tile-based inference, ensemble averaging, and postprocessing run as connected Holoscan operators with zero-copy GPU data flow.

Target: clinical workflows where single-study latency matters most.

## Core Value

**Single-study inference latency without sacrificing correctness** — a CT study goes in, a pixel-identical DICOM-SEG comes out, faster than before, with every step between staying on GPU.

## Requirements

### Validated

(none yet — Phase 0 in progress)

### Active

- [ ] Phase 0 completion: reference corpus (≥5 CT studies), baseline benchmark + `.planning/baseline_results.csv`, RMM verification — blocked on host driver mismatch (CUDA 12.8 driver vs cu13 runtime; see `.planning/STATE.md`)
- [ ] Support all nnUNet model configurations (2D, 3D_fullres, 3D_lowres, 3D_cascade_fullres)
- [ ] Holoscan-native preprocessing operator (resampling, normalization on GPU)
- [ ] Holoscan-native inference operator (tile-based inference with GPU memory management)
- [ ] Holoscan-native ensemble operator (in-memory probability averaging, no disk I/O)
- [ ] Holoscan-native postprocessing operator (connected component cleanup on GPU)
- [ ] Maintain existing DICOM I/O operators (DICOM in → DICOM-SEG/SR/SC out)
- [ ] NVIDIA Nsight profiling integration for performance measurement
- [ ] **Pixel-exact output equivalence** — optimized app output must match original app output bit-for-bit (same DICOM-SEG pixel values, same SR measurements), validated via systematic comparison across representative test cases
- [ ] Performance profiling with NVIDIA tools (Nsight Systems, Nsight Compute) to measure end-to-end latency, GPU utilization, and identify bottlenecks
- [ ] Performance baseline vs current `cchmc_nnunet_fifteen_ckpt_app`

### Out of Scope

- Throughput optimization (concurrent multi-study processing) — Phase 2+
- GPU memory efficiency optimization — Phase 3+
- SDK-level changes to `monai/deploy/operators/` — example app only for now
- Modifications to existing nnUNet example apps
- Training pipeline (inference only)
- Remote/Triton inference

## Context

**Existing app:** `cchmc_nnunet_fifteen_ckpt_app` (on `nnunet-fast` branch)

Current bottleneck chain:
1. `DICOMSeriesToVolumeOperator` → CPU numpy array
2. `NNUnetSegOperator.compute()` calls nnUNet `predict_from_list_of_npy_arrays()`
3. nnUNet writes probability maps to `.npz` files on disk
4. `EnsembleProbabilitiesToSegmentation` reads those files back
5. Postprocessing runs on CPU
6. Result emitted to `DICOMSegmentationWriterOperator`

The nnUNet predictor internally does tiling and preprocessing, but it's not leveraging Holoscan's async execution, GPU memory pools, or streaming operators. All intermediate data touches CPU and disk.

**Holoscan SDK capabilities available:**
- GPU memory pool / zero-copy buffers
- Async operator execution with streaming
- ROS2-style message passing (already used by MONAI Deploy's DataStore)
- GPU-accelerated image processing primitives
- Pipeline-level parallelism

**nnUNet vendored:** `nnUNet/` directory, editable install. No `.git`, not from PyPI.

**Dependencies:** Holoscan cu13 4.0–4.2, CUDA 13+, Ubuntu 22.04, NVIDIA GPU 8GB+

## Constraints

- **GPU required** — NVIDIA GPU with 8GB+ VRAM, CUDA 13
- **Example app only** — new operators live in the app directory, not SDK core
- **Must support all nnUNet configs** — 2D, 3D_fullres, 3D_lowres, 3D_cascade_fullres, custom trainers
- **Output equivalence** — segmentation results must match current app (pixel-level agreement)
- **Holoscan cu13** — SDK pins holoscan-cu13 >=4.0.0, <4.3.0
- **Python 3.10–3.13** — per project pyproject.toml

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| New app, not SDK modification | Lower risk, easier to iterate, proves concept before upstreaming | - Pending |
| Latency first, throughput later | Clinical workflow needs fast single-study response | - Pending |
| Keep DICOM I/O as-is | Already optimized Holoscan operators, not the bottleneck | - Pending |
| nnUNet configs must all work | Generalization is a hard requirement | - Pending |

---
*Last updated: 2026-08-14 — progress review: Phase 0 acceptance status corrected (was marked validated 2025-08-13/commit 19b4a94 before driver regression surfaced)*

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state
