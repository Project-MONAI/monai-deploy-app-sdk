# cchmc-nnunet-fast: Holoscan-Native nnUNet Inference

## What This Is

A new example app (`cchmc-nnunet-fast`) that replaces the current nnUNet wrapper approach with Holoscan-native GPU operators for medical image segmentation inference. The current `cchmc_nnunet_fifteen_ckpt_app` calls nnUNet as a Python wrapper — data bounces through CPU, intermediate `.npz` files hit disk, and Holoscan's GPU streaming/async capabilities go unused. This app rebuilds the inference pipeline so preprocessing, tile-based inference, ensemble averaging, and postprocessing run as connected Holoscan operators with zero-copy GPU data flow.

Target: clinical workflows where single-study latency matters most.

## Core Value

**Single-study inference latency without sacrificing correctness** — a CT study goes in, a pixel-identical DICOM-SEG comes out, faster than before, with every step between staying on GPU.

## Requirements

### Validated

- [x] **Phase 0 — Foundation (2026-08-17, GSD-closed):** cu13 dependencies resolved in `/tmp/monai-env/.venv`; reference corpus (single airway MR study, 256 slices, SC/SEG/SR ground truth — ≥5-CT bar deferred to the final Phase 1 gate per TEST-01); baseline benchmark **169,747 ± 7,274 ms/study** (n=3) → `.planning/baseline_results.csv`; Nsight harness + demo trace (NVTX ranges verified); RMM active. Artifacts: `.planning/phases/00-foundation/`. Satisfies **TEST-006** + **TEST-007** (baseline); scaffold/feasibility for PIPE-01 / INFR-01 / INFR-005.
- [x] **Reference ground-truth reproduction (2026-08-17):** a fresh reference run (`testdata/current_output`) is **99.902% byte-identical** to the historical GT (`testdata/airway_output`) — the earlier "~45 mm / zero-overlap" concern was a thin-structure IoU decode artifact. The pixel-exact gate is de-risked; a freshly regenerated reference is a valid Phase 1 gate target. Re-run any time via `.planning/scripts/REFERENCE_RUN_GUIDE.md`.
- [x] **Phase 1 — Core Pipeline (2026-08-18, GSD-closed):** end-to-end Holoscan DAG (Preprocess → SlideWindow → PostResample → EnsembleAverage → Postprocess → DICOM-SEG/SR/SC) replaces `NNUnetSegOperator` with zero SDK core edits; 3d_fullres on the dev airway study is **pixel-exact vs a fresh 3d_fullres-only reference (99.9999% SEG byte-identity, 2 voxels at the documented FP16↔FP32 TTA-accumulation boundary; SR exact; SC bit-identical)**; GPU residency gate PASS (exactly 1 boundary `.cpu()`); NVTX ranges in Nsight trace; structured timing logs. **Baseline: ~61.8 s E2E vs 169.7 s reference baseline (in-study 42.1 s, inference 27.2 s dominant)** → `.planning/benchmarks/baseline-2026-08-18.csv`. Deviations: MONAI sliding-window utilities diverge from nnUNet 2.8.1 (reference utilities used); TTA accumulates FP32 per INF-004 while the reference uses FP16 (bit-for-bit logits unreachable by design). ≥5-study final gate deferred (corpus not yet supplied). Artifacts: `.planning/phases/01-core-pipeline/` (5 SUMMARYs + VERIFICATION).
- [x] **Holoscan-native preprocessing / inference / ensemble / postprocessing operators** (single-config `3d_fullres`; GPU port of resampling and all-config support is Phase 2)
- [x] **Maintain existing DICOM I/O operators** — DICOM in → DICOM-SEG/SR/SC out, unchanged SDK operators
- [x] **NVIDIA Nsight profiling integration + performance baseline** vs `cchmc_nnunet_fifteen_ckpt_app`

### Active

- [ ] Support all nnUNet model configurations (2D, 3D_fullres, 3D_lowres, 3D_cascade_fullres)
- [ ] GPU-accelerated preprocessing (resampling/transpose/crop on CuPy) and multi-config pipeline
- [ ] Final pixel-exact gate re-run on ≥5 studies once the corpus is supplied

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
| New app, not SDK modification | Lower risk, easier to iterate, proves concept before upstreaming | Phase 0 scaffold done |
| Latency first, throughput later | Clinical workflow needs fast single-study response | - Pending |
| Keep DICOM I/O as-is | Already optimized Holoscan operators, not the bottleneck | - Pending |
| nnUNet configs must all work | Generalization is a hard requirement | - Pending (Phase 2) |
| Pin all CUDA deps to cu13 (holoscan-cu13 4.2.0, cupy-cuda13x, rmm-cu13); no cupy-cuda12x | CUDA 13 requirement; mixed CUDA-12 packages corrupt the venv | Phase 0 validated |
| Resampling stays on reference CPU (scipy) path in Phases 1–2 | Pixel-exactness over speed; GPU resampling is hard to replicate exactly (GPUP deferred to v2) | - Pending |
| Docker build + container test deferred to post-Phase-3 | Optimize the pipeline first, then package + test the MAP container | logged 2026-08-17 |
| Baseline to beat = 169.7 ± 7.3 s/study (reference app) | The "before" number Phase 2/3 must improve on | Phase 0 |

---
*Last updated: 2026-08-18 — Phase 1 complete & verified (pixel-exact 3d_fullres pipeline, ~61.8 s vs 169.7 s baseline); ready for Phase 2 GPU Acceleration*

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
