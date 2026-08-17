---
phase: "0-foundation"
plan: "01"
name: "Foundation"
one-liner: "Scaffold, cu13 deps, baseline benchmark, Nsight + RMM tooling, and a validated single-study reference corpus for cchmc-nnunet-fast."
subsystem: "foundation"
tags: [baseline, profiling, rmm, scaffold]
provides:
  - "examples/apps/cchmc-nnunet-fast app scaffold"
  - ".planning/baseline_results.csv (169,747 ± 7,274 ms/study)"
  - ".planning/scripts/baseline_benchmark.py"
  - ".planning/scripts/REFERENCE_RUN_GUIDE.md"
  - "Nsight harness + demo trace"
  - "RMM active-allocator verification"
affects: [phase-1-core-pipeline]
tech-stack:
  added: [holoscan-cu13, cupy-cuda13x, rmm-cu13, monai 1.3.0, torch 2.13+cu130, nnunetv2 2.8.1 (editable)]
  patterns: []
key-files:
  created:
    - "examples/apps/cchmc-nnunet-fast/my_app/app.py"
    - "examples/apps/cchmc-nnunet-fast/pyproject.toml"
    - ".planning/scripts/baseline_benchmark.py"
    - ".planning/scripts/nsight_profile.sh"
    - ".planning/scripts/test_rmm.py"
    - ".planning/scripts/REFERENCE_RUN_GUIDE.md"
    - ".planning/baseline_results.csv"
  modified:
    - ".planning/STATE.md"
    - ".planning/REQUIREMENTS.md"
key-decisions:
  - "Pin every CUDA dep to cu13; cupy-cuda12x must not coexist"
  - "Resampling stays on reference CPU (scipy) path in Phases 1-2 for pixel-exactness"
  - "TEST-01 corpus deviation: single airway MR study; >=5-CT deferred to final Phase 1 gate"
  - "Docker build + container test deferred until after Phase 3 optimizations (2026-08-17)"
  - "Phase 0 validated by pythonic reference-app runs, not Docker (see REFERENCE_RUN_GUIDE.md)"
requirements-completed:
  - "TEST-006"
  - "TEST-007"
duration: "multi-session (2026-08-12..17)"
completed: 2026-08-17
---

# Phase 0: Foundation — Summary

Scaffold, cu13 dependency resolution, a validated single-study reference corpus, a
reproducible baseline (169,747 ± 7,274 ms/study), a working Nsight + NVTX harness, and
RMM active-allocator verification. Phase 0 executed ad-hoc and is now backfilled into the
GSD cycle (plan + summary + verification) so its work is credit-verified.

## Performance
- **Duration:** multi-session (2026-08-12 → 2026-08-17)
- **Tasks:** 7/7
- **Files modified:** see key-files above

## Accomplishments
- `cchmc-nnunet-fast` app scaffold (standard MAP layout) + cu13 pins resolved in `/tmp/monai-env/.venv`.
- Reference app `cchmc_nnunet_fifteen_ckpt_app` runs end-to-end on the airway corpus → SC/SEG/SR.
- Baseline benchmark: **169,747 ± 7,274 ms/study** (n=3, warmup excluded); per-stage: setup ~12.8 s / inference ~138 s / postprocess 9–23 s / write ~1.2 s → `.planning/baseline_results.csv`.
- Nsight harness + demo trace (NVTX preprocess/inference/postprocess ranges verified in trace).
- RMM pool allocator verified active (driver 610.57.04 / CUDA 13.3, A100-SXM4-40GB).
- **GT-mismatch carried finding RESOLVED:** a fresh reference run (`testdata/current_output`)
  reproduces the historical ground truth (`testdata/airway_output`) — SEG segment pixel data
  **99.902% byte-identical**, segment voxel counts 2430 (GT) vs 2447 (fresh), differences
  confined to the airway band (frames 106–225). The earlier "~45 mm / zero-overlap" note was a
  decoding artifact (thin 1-voxel airway → IoU hypersensitive to sub-voxel registration).

## Task Commits
- Scaffold + cu13 pins: 7d5e687
- Nsight/NVTX harness: d99deb4
- RMM + venv validation: 8910ae8, 205fcd3
- Reference-app env repairs + baseline + demo trace + close-out: 668ef0f..e2c38be
- Phase 0 GSD backfill + GT-match verification: (this commit)

## Files Created/Modified
- `.planning/baseline_results.csv` — baseline numbers
- `.planning/scripts/baseline_benchmark.py` — benchmark (TEST-006)
- `.planning/scripts/REFERENCE_RUN_GUIDE.md` — how to regenerate the reference (future runs)
- `.planning/scripts/nsight_profile.sh`, `nvtx_markers.py`, `test_rmm.py`
- `.planning/profiles/nsight_demo_target_20260817_111555.nsys-rep` (+ .sqlite)

## Decisions & Deviations
- **TEST-01 corpus deviation:** single airway MR series used; ≥5-CT bar deferred to final Phase 1 acceptance gate.
- **Docker deferred:** MAP Docker build + container testing deferred until after Phase 3 optimizations (decided 2026-08-17). Pythonic reference runs are the validation path until then.
- **Baseline provenance:** historical `testdata/airway_output` was regenerated and confirmed to match a fresh reference run, so it is a valid correctness reference for Phase 1.

## Next Phase Readiness (Phase 1: Core Pipeline)
- **Validation strategy (now unblocked):** gate `cchmc-nnunet-fast` vs a **freshly regenerated reference** (`testdata/current_output`), which is confirmed to match historical GT. Historical `testdata/airway_output` is a valid secondary reference.
- **Baseline to beat:** 169.7 ± 7.3 s/study (inference ≈ 138 s of that).
- **Models available:** 3d_fullres, 3d_lowres, 3d_cascade_fullres (2d absent — gates Phase 2/3 multi-config).
- **Env hazards:** `my_app` editable-install name collision (run apps from their app root); venv monai ptp patches (re-apply after monai reinstall); 32 MB stack requirement.
- **TEST-007** baseline ("before") established; the speedup-ratio comparison is completed in Phase 2/3.
- Requirements **INFR-01** (RMM), **INFR-005** (NVTX), **PIPE-01** (DAG) had feasibility/scaffold work done in Phase 0 but are **not** fully satisfied until Phase 1/2 wire them into the new pipeline (tracked, not credited here).
