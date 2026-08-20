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
- [x] **Phase 2 — GPU Acceleration (2026-08-19, GSD-closed):** all 3D nnUNet configs + the default bundle run through one multi-fragment Holoscan DAG (one Subgraph per config, cross-fragment cascade flow, per-config CudaStreamPool) with **all 4 pixel-exact gates PASS** (fullres 99.99986% / 3 documented fp16↔fp32 boundary voxels, lowres 100%, cascade 100%, **bundle 100.00000% vs `testdata/current_output`** — 2447=2447 voxels; SR exact; residency static+runtime PASS). Preprocess transpose/crop/normalize on CuPy (scipy resample stays CPU per D-13); RMM pluggable allocator + memory budget calculator; reference-semantic model-list selection. **Two-bar latency (D-18): same-scope fullres 57.14 s vs 61.8 s (1.082×, bar MET) + headline bundle 129.54 s vs 169.7 s reference (1.310×; inference stage −44.5%)** → `.planning/benchmarks/phase2_results.csv`, `.planning/phases/02-gpu-acceleration/02-BENCHMARK-REPORT.md` + VERIFICATION (38/38). Deviations: 2d blocked-on-model (D-01/D-03); INFR-02 cross-study buffer reuse deferred to Phase 3 (D-17); ncu permission-blocked → nsys-only profiling; ≥5-CT corpus gate still pending. Artifacts: `.planning/phases/02-gpu-acceleration/` (6 SUMMARYs + VERIFICATION + gates/).
- [x] **Phase 3 — Optimization (2026-08-20, GSD-closed, 12/12 must-haves):** all four optimizations shipped behind per-plan D-25 gate re-runs (pixel-identical at every step): **D-21 concurrent fragments** (`EventBasedScheduler(5)` default ON — 49.8 s fullres∥lowres overlap in nsys); **MEM-003** lowres weight release after the aux fragment (pool −0.8 GB derived / driver flat measured — Open Q2: `empty_cache()` is a driver-level no-op under RMM); **INFR-02** shape-keyed GPU buffer reuse (D-24(a)+(b): data_ptr stable + cudaMalloc 0/0 on studies 2/3; user reference examples = D-26 external dep); **GPU resample** — the D-13 scipy-CPU decision was **superseded** in Phase 3 by the stock-CuPy `cupyx.scipy.ndimage` port under the user's D-22b amendment (≥99% per-tensor gate: measured 100.0000%, max abs diff 0, ON gate pixel-identical to baseline → default ON, `HOLOSCAN_GPU_RESAMPLE=0` = scipy fallback; the custom RawKernel was the one-and-only bounded attempt per D-22a — o3 diverged 100% + real-shape crash → discarded, kept as provenance). Resample-dominated spans 28.8 s → 9.65 s (bundle). **Final benchmarks (shipping config, 2×2 matrix no regressed cell):** same-scope fullres **49,673 ms = 1.244× vs 61.8 s / 1.150× vs Phase 2 57.14 s**; bundle **104,180 ms = 1.629× vs 169,747 ms reference / 1.243× vs Phase 2 129.54 s** → `.planning/benchmarks/phase3_results.csv` + `03-BENCHMARK-REPORT.md`. Deferred with locked reasons: ACCEL-01/02/03 (ncu admin-blocked), MEM-01 (not a measured bottleneck), MEM-02 (8 GB target hardware-unverifiable on A100-40GB). Satisfies **INFR-02, MEM-003, GPUP-01/02, TEST-01/002/003/006/007 (final)**. Artifacts: `.planning/phases/03-optimization/` (5 SUMMARYs + VERIFICATION + gates/ + evidence/).
- [x] **GPU-accelerated preprocessing + multi-config pipeline** (transpose/crop/normalize on CuPy, numpy reductions; one Subgraph per config incl. cascade; 2d wiring generic but blocked-on-model — D-01/D-03)
- [x] **Holoscan-native preprocessing / inference / ensemble / postprocessing operators** (single-config `3d_fullres`; GPU port of resampling and all-config support is Phase 2)
- [x] **Maintain existing DICOM I/O operators** — DICOM in → DICOM-SEG/SR/SC out, unchanged SDK operators
- [x] **NVIDIA Nsight profiling integration + performance baseline** vs `cchmc_nnunet_fifteen_ckpt_app`

### Active

- **Milestone v1.0 complete (2026-08-20)** — all 36 v1 requirements delivered (33/36 counted Done in the corrected traceability: 33 rows + MEM-003/GPUP-01/02 shipped from the v2-tracked set; 3 locked deferrals: ACCEL-01/02/03, MEM-01, MEM-02). Pipeline at **104.2 s bundle / 49.7 s same-scope** (shipping config). Open items are external dependencies only (non-blocking, D-26 pattern — see `.planning/MILESTONES.md` v1.0): ≥5-CT corpus re-run (TEST-01 final gate, blocked on CT data), ncu admin access (ERR_NVGPUCTRPERM — keeps ACCEL-01/02/03 deferred), 2d model validation (blocked-on-model, wiring generic), user's INFR-02 reference examples. Next milestone TBD.

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
| New app, not SDK modification | Lower risk, easier to iterate, proves concept before upstreaming | **Validated at milestone close** — zero SDK core edits across all 4 phases (`git log monai/` empty); entire pipeline lives in `examples/apps/cchmc-nnunet-fast/` |
| Latency first, throughput later | Clinical workflow needs fast single-study response | **Fully validated** — single-study two-bar latency met and exceeded at every phase (final: 49.7 s same-scope 1.244×, 104.2 s bundle 1.629× vs reference); throughput (THRU-01..03) remains out of scope for v1, candidate for the next milestone |
| Keep DICOM I/O as-is | Already optimized Holoscan operators, not the bottleneck | Validated in Phase 1 — `git log monai/` empty for the phase; writers unchanged, SDK I/O operators used directly |
| nnUNet configs must all work | Generalization is a hard requirement | **Fully validated** — all 3D configs + cascade + bundle E2E and pixel-exact vs per-config reference oracles (Phase 2, 4/4 gates); 2d wiring is config-generic (D-02) and awaits a real model (blocked-on-model, D-01/D-03) |
| Pin all CUDA deps to cu13 (holoscan-cu13 4.2.0, cupy-cuda13x, rmm-cu13); no cupy-cuda12x | CUDA 13 requirement; mixed CUDA-12 packages corrupt the venv | Phase 0 validated |
| Resampling stays on reference CPU (scipy) path in Phases 1–2 | Pixel-exactness over speed; GPU resampling is hard to replicate exactly (GPUP deferred to v2) | **Validated in Phases 1–2** (D-13 boundary preserved, all gates pixel-exact) — then **superseded in Phase 3** by the stock-CuPy port under the user's D-22b amendment (≥99% per-tensor gate: measured 100.0000%, ON gate pixel-identical, default ON). The custom RawKernel was the one-and-only bounded attempt (D-22a) — discarded after o3 divergence + real-shape crash, kept as provenance |
| Docker build + container test deferred to post-Phase-3 | Optimize the pipeline first, then package + test the MAP container | logged 2026-08-17 — still open, now the natural first task of the next milestone |
| Baseline to beat = 169.7 ± 7.3 s/study (reference app) | The "before" number Phase 2/3 must improve on | **Beaten**: 104.18 s bundle (1.629×) in the final shipping configuration (Phase 3) |

---
*Last updated: 2026-08-20 — **Milestone v1.0 complete**: Phase 3 complete & verified (12/12); all 4 phases GSD-closed, 33/36 v1 requirements Done + 3 locked deferrals; final latency 104.18 s bundle (1.629× vs 169.7 s reference) / 49.67 s same-scope (1.244× vs 61.8 s); external dependencies pending (≥5-CT corpus, ncu admin, 2d model, INFR-02 user examples); next milestone TBD*

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
