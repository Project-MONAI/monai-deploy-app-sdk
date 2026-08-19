---
phase: 3-optimization
plan: "01"
subsystem: scheduling
tags: [d-21, eventbasedscheduler, concurrency, rmm, open-q1, initial-pool-size, pixel-exact-gate, nsys-overlap]

requires:
  - phase: 2-gpu-acceleration (plans 01-06)
    provides: multi-fragment DAG (per-config Subgraphs + CudaStreamPools), phase2_gate.py gate infrastructure (D-25 anchor), phase2 nsys traces (the §5 serial-contrast baseline), gpu_bootstrap RMM install
provides:
  - "HOLOSCAN_CONCURRENT_FRAGMENTS flag (default ON after step-5 flip; =0 restores byte-for-byte Phase 2 GreedyScheduler serial behavior) with EventBasedScheduler(worker_thread_number=5) wired at compose() tail"
  - "Phase 3 RMM baseline: initial_pool_size pinned to 4 GiB (live-measured 19.97 GiB default) — the correct INFR-02 churn baseline for Plan 03 (10 cudaMalloc / 1 cudaFree per bundle study, kernel counts unchanged)"
  - "Gate evidence: .planning/phases/03-optimization/gates/03-GATE-serial.json (exact Phase 2 reproduction) + 03-GATE-concurrent.json (fully green, pixel-identical numbers)"
  - "D-21 trace citation: .planning/profiles/phase3/overlap.md + overlap_concurrent_20260819_104410.* (rep/sqlite/nvtx_sum/nvtx_kern_sum) — 5 distinct worker tids, inference_fullres || inference_lowres 49.8 s overlap"
affects: [phase 3 plans 02-05 — all subsequent runs default to the concurrent scheduler; Plan 03 INFR-02 churn proof must baseline off the pinned 10-malloc number; Plan 05 final benchmark reports the shipping wall delta with the 2x2 matrix]

tech-stack:
  added: [pynvml 13.0.1 (measurement-only, in the existing /tmp/monai-env/.venv — Open Q1 protocol requires nvmlDeviceGetMemoryInfo)]
  patterns:
    - "Scheduler swap behind an env flag at compose() tail, default flip only after the full gate suite passes with concurrency ON (D-21 fallback clause)"
    - "pynvml before/after import probe to measure the RMM reinit reservation on a pinned known-free device (Pitfall 7)"
    - "phase2_gate.py reused verbatim via its existing --report arg for the Phase 3 JSON paths (env pass-through of the new flag already satisfied by os.environ.copy())"

key-files:
  created:
    - .planning/profiles/phase3/rmm_openq1.md
    - .planning/profiles/phase3/overlap.md
    - .planning/scripts/probes-phase3/probe_rmm_openq1.py
    - .planning/phases/03-optimization/gates/03-GATE-serial.json
    - .planning/phases/03-optimization/gates/03-GATE-concurrent.json
    - .planning/profiles/phase3/rmm_q1_baseline_20260819_101503.{nsys-rep,sqlite} + _cuda_api_sum.txt
    - .planning/profiles/phase3/rmm_q1_pinned_20260819_102101.{nsys-rep,sqlite} + _cuda_api_sum.txt
    - .planning/profiles/phase3/overlap_concurrent_20260819_104410.{nsys-rep,sqlite} + _nvtx_sum.txt + _nvtx_kern_sum.txt
  modified:
    - examples/apps/cchmc-nnunet-fast/my_app/app.py (EventBasedScheduler import + flag-gated scheduler at compose() tail, default ON)
    - examples/apps/cchmc-nnunet-fast/my_app/gpu_bootstrap.py (initial_pool_size=4 GiB pin + log line)

key-decisions:
  - "Open Q1 resolved: the live venv DOES carry the 20 GiB default (S = 19.97 GiB measured post-reinit, device 0, pynvml) — pinned initial_pool_size=4 GiB (~4.1x the airway bundle memory_budget total of 1,038,502,513 bytes from a fresh-run log; warm_pool still grows to plan.total_bytes per D-14)"
  - "D-21 ships concurrent-by-default (step-5 flip taken): the concurrent gate suite is fully green with pixel-identical numbers to serial, including the sensitive fullres 3-voxel fp16↔fp32 boundary gate (the set_num_threads interleave hazard did not materialize)"
  - "Measured wall delta (single bundle reps, same session, device 0): 120.4 s serial -> 110.4 s concurrent = -10.1 s (-8.4%) — recorded with the honest ceiling note: the GPU-saturated fullres||lowres inference pair time-shares (26 s -> ~50 s each, total GPU work conserved — exactly the research GPU-probe prediction) and the 9.4 s cascade preprocess cannot start until lowres_seg is emitted, so it hides under nothing"

patterns-established:
  - "Phase 3 gate runs reuse phase2_gate.py unchanged: new flags ride the os.environ.copy() pass-through; Phase 3 JSON paths go through the existing --report arg (no new arguments)"
  - "Device pinning + device id recorded in every gate JSON note (Pitfall 7 — devices 4-7 were tenant-occupied; device 0 used for all runs this plan)"

requirements-completed: [TEST-01, TEST-002, TEST-003]

duration: ~55min
completed: 2026-08-19
---

# Phase 3 Plan 01: D-21 Concurrent Fragments + Open-Q1 RMM Re-verification Summary

**The D-21 lever ships concurrent-by-default — `EventBasedScheduler(worker_thread_number=5)` behind `HOLOSCAN_CONCURRENT_FRAGMENTS` (default ON after the fully-green concurrent gate suite; `=0` restores byte-for-byte Phase 2 serial) — and the Open-Q1 RMM baseline is live-resolved: the scratch venv really does carry the 20 GiB default initial pool (19.97 GiB measured post-reinit), now pinned to 4 GiB, with the INFR-02 churn baseline for Plan 03 re-established at 10 cudaMalloc / 1 cudaFree per bundle study.**

## What was built

### Task 1 — Open Q1: RMM initial-pool re-verification + pin

- **Probe:** `.planning/scripts/probes-phase3/probe_rmm_openq1.py` — fresh subprocess, `ulimit -s unlimited`, `CUDA_VISIBLE_DEVICES=0` (known-free per nvidia-smi at run time), pynvml `nvmlDeviceGetMemoryInfo` before/after `import my_app.gpu_bootstrap` (the `rmm.reinitialize` site).
- **Finding:** S = **19.97 GiB** reserved immediately at reinit (0.485 → 20.450 GiB) — the 2026-08-19 research measurement holds against the live venv (rmm 26.2.0 default initial_pool_size = ½ total GPU memory). The Phase 2 trace's ~1.98 GB total cudaMalloc predates the venv drift (Pitfall 8). S ≥ 10 GiB branch taken → **pinned**.
- **The pin:** `gpu_bootstrap.py` passes `initial_pool_size=4 * 1024**3` (4 GiB ≈ 4.1× the fresh-run `memory_budget.total_bytes` = 1,038,502,513 bytes; `warm_pool` still grows the pool to the budget per D-14) + module-level `logging.info("rmm initial_pool_size: …")` at the reinitialize site; import-order invariant untouched. Post-pin probe: 4.41 GiB. Post-pin bundle run: exit 0, `memory_allocator_backend: pluggable`, `free_vram_bytes` 20.98 GB → 37.66 GB.
- **Churn baseline re-established** (full-process nsys, bundle rep, device 0): no-pin 9 cudaMalloc / 1 cudaFree / 540,700+87,480 kernel launches; **pinned 10 / 1 / identical kernel counts** (the +1 is a <2 ms pool expansion, never per-tile). Evidence: `.planning/profiles/phase3/rmm_openq1.md` (`initial_pool_size: pinned 4 GiB` token) + both rep/sqlite pairs + `cuda_api_sum` exports.

### Task 2 — D-21 scheduler swap, gates both ways, overlap evidence

- **Wiring:** `from holoscan.schedulers import EventBasedScheduler` + at the end of `compose()` (after `warm_pool`, before the End log): `HOLOSCAN_CONCURRENT_FRAGMENTS` flag-gated `self.scheduler(EventBasedScheduler(self, worker_thread_number=5, name="concurrent"))`. Per-config CudaStreamPools untouched (INFR-004 as-is); no per-fragment streams / green contexts (research anti-patterns). `phase2_gate.py` **unmodified** — its `run_fast_app`/residency env construction is `os.environ.copy()` + `HOLOSCAN_MODEL_LIST` management only (confirmation, not new work), and the Phase 3 JSON paths went through the existing `--report` arg (default still targets the Phase 2 JSON).
- **Gates (D-25 anchor, device 0, recorded in both JSON notes):**
  - **Serial (flag unset):** fullres **99.99986%/3** (documented fp16↔fp32 boundary class), lowres/cascade/bundle **100.00000%/0**, SR **0.0%** ×4, residency static+runtime **PASS** — exact Phase 2 reproduction → `03-GATE-serial.json`.
  - **Concurrent (=1):** **fully green, identical numbers** (the fullres 3-voxel gate did not flip) → `03-GATE-concurrent.json`.
- **Step-5 flip:** concurrent fully green → default ON: `os.environ.get("HOLOSCAN_CONCURRENT_FRAGMENTS", "1") != "0"`.
- **nsys overlap evidence** (`overlap_concurrent_20260819_104410.*`, full-process capture): **five distinct worker tids (…486–…490)**; `inference_3d_fullres` ∥ `inference_3d_lowres` **49.8 s overlap** (7.98→56.43 s) vs Phase 2 §5's single stream id 7 / back-to-back 0.2 ms gap; preprocess pair 5.36 s overlap; postresample pair 2.04 s; writer tail concurrent → `.planning/profiles/phase3/overlap.md` (the trace citation).
- **Measured wall delta (single reps, same session, non-nsys, device 0):** bundle process wall **120.4 s (serial) → 110.4 s (concurrent) = −10.1 s (−8.4%)**; in-study 108.5 → 100.9 s. Ceiling note recorded in overlap.md (GPU-conservation of the saturated inference pair + cascade preprocess gated on lowres_seg) — a ceiling note, NOT the fallback clause (no gate failed).
- **Shipping-config verification:** default (flag unset) bundle run exit 0 + `scheduler: EventBasedScheduler …` logged; `=0` run exit 0 + `scheduler: default GreedyScheduler (serial, Phase 2 behavior)` logged.

## Must-haves check

- [x] Concurrent execution wired as EventBasedScheduler in compose(), flag-gated; flag-OFF path byte-for-byte Phase 2 (03-GATE-serial.json exact reproduction)
- [x] RMM initial-pool re-verified against the live venv and pinned (19.97 GiB measured → `initial_pool_size: pinned 4 GiB`)
- [x] All 4 pixel gates + SR 0.1% + residency PASS with concurrency enabled (first branch — no fallback needed)
- [x] nsys trace of one concurrent bundle run with overlapping per-config NVTX spans (49.8 s inference∥inference) — recorded as the trace citation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed pynvml into the existing venv**
- **Found during:** Task 1
- **Issue:** the Open Q1 protocol (and the plan's task text) measures via pynvml `nvmlDeviceGetMemoryInfo`, but the scratch venv had no pynvml (ModuleNotFoundError).
- **Fix:** `pip install pynvml` (13.0.1, pulls nvidia-ml-py 13.610.43) into `/tmp/monai-env/.venv` — measurement-only dependency, no new venv, no CUDA-stack change.
- **Files modified:** (venv only)
- **Commit:** n/a (environment)

No other deviations — the plan executed as written (phase2_gate.py needed no changes, as the plan predicted).

## Known Stubs

None.

## Commits

- `911a63c` perf(03-01): pin RMM initial_pool_size to 4 GiB (Open Q1: live-measured 19.97 GiB default)
- `e33d007` perf(03-01): D-21 concurrent fragments default ON behind serial fallback (HOLOSCAN_CONCURRENT_FRAGMENTS=0)

## Self-Check: PASSED

All 12 referenced artifacts and both commits (911a63c, e33d007) verified present on 2026-08-19.
