---
phase: 2-gpu-acceleration
plan: 06
subsystem: benchmarks-profiling
tags: [two-bar-benchmark, per-operator, nsys, rmm-churn-check, nvtx, d-18, d-16, d-13, test-006, test-007, infr-005, phase-3-handoff]

requires:
  - phase: 2-gpu-acceleration (plan 04)
    provides: the multi-fragment bundle DAG (4 runnable HOLOSCAN_MODEL_LIST configurations) + config-tagged `timing: {json}` logs + one app-level `study_timing_summary` per run (Pitfall 9) — the CSV parsing this plan builds on
  - phase: 01-core-pipeline (plan 05)
    provides: baseline_benchmark.py harness pattern (fresh subprocess + stack rlimit + warmups), Phase 1 numbers (61.8 s center, in-study 42.1 s per-operator breakdown)
  - phase: 00-foundation
    provides: the 169,747 ± 7,274 ms reference baseline (.planning/baseline_results.csv) + the nsight_profile.sh nsys 2025.6.3 flag set

provides:
  - ".planning/benchmarks/phase2_results.csv — the roadmap-required benchmark CSV: per-rep, per-config operator columns for BOTH scopes (fullres same-scope, bundle headline), both speedups (vs 61.8 s / vs 169.7 s), mean±std summary rows"
  - "02-BENCHMARK-REPORT.md — the D-18 two-bar report: same-scope fullres 57.14±0.25 s vs 61.8 s (MET, 1.082×) with per-operator deltas vs Phase 1; headline bundle 129.54±0.90 s vs 169.7 s (1.310×) with per-stage deltas vs the reference; the scope-asymmetry deviation documented; profiling findings; ranked Phase 3 bottleneck list"
  - ".planning/profiles/phase2/ — nsys bundle trace (.nsys-rep + .sqlite) + cuda_api_sum / nvtx_sum / nvtx_kern_sum / cuda_gpu_kern_sum exports + ncu_status.txt (ERR_NVGPUCTRPERM) + README with the RMM-churn/NVTX/stream-overlap findings"

affects: [Phase 3 — receives the ranked trace-cited bottleneck list (scipy resample spans ~28.8 s, serial fragment scheduling ~25 s potential, ncu-blocked inference kernel work, postprocess, bootstrap) + carried items (INFR-02, ≥5-CT corpus, 2d model)]

tech-stack:
  added: []
  patterns:
    - "Benchmark CSVs parse the app's structured `timing: {json}` logs (config-tagged per Plan 04) instead of log-marker spans — one app-level `study_timing_summary` per run survives multi-fragment DAGs (Pitfall 9)"
    - "Append-safe multi-scope CSV: the header is the fixed UNION of both scopes' per-config columns; each scope invocation rewrites file = prior rows + this scope's rows + one mean±std summary row"
    - "RMM churn check = cudaMalloc/cudaFree counts + timestamps from the nsys sqlite (CUPTI_ACTIVITY_KIND_RUNTIME, names carry the `_v3020` suffix) relative to the first operator NVTX range — 9 pool-expansion allocations in a 629,929-kernel window = no per-tile churn"
    - "nsys full-process capture (NO --capture-range=cudaProfilerApi) is required when the RMM setup/warmup span must be in the trace"

key-files:
  created:
    - .planning/scripts/phase2_benchmark.py
    - .planning/benchmarks/phase2_results.csv
    - .planning/scripts/nsight_profile_phase2.sh
    - .planning/profiles/phase2/phase2_bundle_20260819_071235.nsys-rep (+ .sqlite sidecar)
    - .planning/profiles/phase2/phase2_bundle_cuda_api_sum.txt
    - .planning/profiles/phase2/phase2_bundle_nvtx_sum.txt
    - .planning/profiles/phase2/phase2_bundle_nvtx_kern_sum.txt
    - .planning/profiles/phase2/phase2_bundle_cuda_gpu_kern_sum.txt
    - .planning/profiles/phase2/ncu_status.txt
    - .planning/profiles/phase2/README.md
    - .planning/phases/02-gpu-acceleration/02-BENCHMARK-REPORT.md
  modified: []

key-decisions:
  - "Bundle speedup came in at 1.310× vs the 169.7 s reference — below the plan's ~2.0–2.8 expectation and far below the ~2.3–2.5× in the Context. Honest cause (report §4): the RESEARCH 70–80 s serial estimate excluded the ~20.8 s fresh-process bootstrap AND under-counted the three serial scipy-resample-dominated preprocess spans (22.2 s, D-13). The same-scope D-18 bar (the controlling positive-improvement bar) is MET at 1.082×. No bar was collapsed or massaged"
  - "Stream overlap (D-16) is NOT visible and is reported as such: all 629,929 kernels run on a single CUDA stream (id 7) and the three inference fragments are strictly back-to-back (0.2 ms gap) — the per-fragment CudaStreamPools are never concurrently active because the DAG schedules fragments serially; overlap exists only at light boundaries (postresample_3d_fullres || lowres inference)"
  - "cudaMalloc churn check is a PASS with a documented nuance: 9 cudaMallocs total (1 in setup, 8 pool-expansion events at operator-stage starts, each < 2 ms) vs 629,929 kernel launches — never per-tile; RMM (pluggable) covers the sliding-window loop"
  - "Profile artifacts committed per the Phase 0/1 convention (rep + sqlite + exports all tracked; Phase 1 precedent: 18 MB rep + 45 MB sqlite committed)"

requirements-completed: [TEST-006, TEST-007, INFR-005]

metrics:
  duration: ~55min
  tasks: 3/3
  commits: 3
  gates: "same-scope fullres 57,140.3 ± 250.4 ms E2E (1.082× vs Phase 1 61.8 s — D-18 positive-improvement bar MET; inference 26.16 s vs 27.2 s, no env drift); headline bundle 129,542.9 ± 896.4 ms (1.310× vs 169.7 s reference, −40.2 s / −23.7%; inference stage 76.7 s vs 138.2 s = −44.5%); RMM churn check PASS (no per-tile cudaMalloc); per-config NVTX ranges legible (14/14); ncu documented unavailable (ERR_NVGPUCTRPERM)"

completed: 2026-08-19
---

# Phase 2 Plan 06: Benchmarks (two-bar, per-operator) + nsys Profiling + Phase 3 Handoff Summary

**The roadmap-required benchmark CSV and the D-18 two-bar report are in: the same-scope fullres bar is MET (57.14 s vs Phase 1's 61.8 s — 1.082×, warmup-excluded mean±std over 3 fresh-process reps, per-operator deltas vs Phase 1), the headline bundle is 129.54 s vs the 169.7 s reference (1.310×, inference stage −44.5%), the bundle-vs-61.8 s scope asymmetry is documented as the D-18 deviation rather than hidden, the nsys trace proves no per-tile cudaMalloc churn under RMM + legible per-config NVTX + an honest no-overlap stream observation (D-16), ncu is documented unavailable (ERR_NVGPUCTRPERM), and Phase 3 receives a ranked, trace-cited bottleneck list.**

## What was built

### Task 1 — phase2_benchmark.py + both scopes → phase2_results.csv (commit fc7e977)

- `.planning/scripts/phase2_benchmark.py`: extends the Phase 0/1 `baseline_benchmark.py` spine (fresh subprocess per rep — cold start included, clinical single-study usage — 32 MB stack rlimit, sitecustomize INFO shim, 1 warmup + 3 measured). `--scope {fullres,bundle}`: fullres pins `HOLOSCAN_MODEL_LIST=3d_fullres`, bundle UNSETS it. Parses the fast app's config-tagged `timing: {json}` lines (Plan 04, Pitfall 9) + the app-level `study_timing_summary: {json}` into per-config columns (`preprocess_/inference_/postresample_ms_<cfg>`), `ensemble_ms`, `postprocess_ms`, `write_ms` (sum of write_seg/sr/sc), wall `total_ms`, and both speedups (constants 61,800 ms Phase 1 center / 169,747 ms reference mean documented in the header). Append-safe: fixed union header over both scopes, prior rows preserved, one `mean±std` summary row per scope. Per-rep consistency check printed (operator sum 68–84% of wall).
- `.planning/benchmarks/phase2_results.csv`: 10 rows — 2 scopes × (1 warmup + 3 measured) + 2 summary rows.
  - **fullres:** 57,425 (warmup), then 56,949 / 57,424 / 57,048 → **57,140.3 ± 250.4 ms**; speedup vs 61.8 s = 1.082×. `inference_ms_3d_fullres` 26,162.9 ± 15.2 ms ≈ Phase 1's 27.2 s (no environment drift — the plan's drift guard did not fire).
  - **bundle:** 128,925 (warmup), then 130,454 / 128,662 / 129,513 → **129,542.9 ± 896.4 ms**; speedup vs 169.7 s = 1.310×; all 9 per-config operator columns populated (14 timing records per run).

### Task 2 — nsys bundle profile → .planning/profiles/phase2/ (commit fb6cbea)

- `.planning/scripts/nsight_profile_phase2.sh`: sibling harness, same nsys 2025.6.3 flag set (`--trace=cuda,nvtx,osrt,cublas,cudnn` — no removed `cub`), full-process capture (NO `--capture-range` — the app never calls cudaProfilerStart and the RMM setup span must be in the trace), `env -u HOLOSCAN_MODEL_LIST` (unset = bundle; an empty value would be an explicit empty model list).
- `phase2_bundle_20260819_071235.nsys-rep` (72.5 MB) + `.sqlite` sidecar + four stats exports (`cuda_api_sum`, `nvtx_sum`, `nvtx_kern_sum`, `cuda_gpu_kern_sum`), committed per the Phase 0/1 convention (rep + sqlite tracked).
- **Findings (details in `README.md` + report §5):** RMM churn check PASS (9 cudaMalloc / 1 cudaFree vs 629,929 kernels; the 8 in-window ones are pool expansions at operator-stage starts, each < 2 ms); all 14 per-config NVTX ranges legible; **no stream overlap** (single stream 7, inference fragments back-to-back at 0.2 ms — serial fragment scheduling is the blocker, not the pools); top CPU-bound regions: scipy resample spans (preprocess 99.9% CPU: 7.6/5.0/9.4 s wall vs 2–3 ms kernels), postprocess (9.9 s wall, 5 ms GPU), bootstrap ≈20.8 s.
- `ncu_status.txt`: the prescribed ERR_NVGPUCTRPERM documentation (admin requirement); zero ncu kernel-metric numbers in any artifact.

### Task 3 — 02-BENCHMARK-REPORT.md (commit f405d83)

Six sections, every number traceable to `phase2_results.csv` or a profiling export:
1. Scope & method (fresh-process reps, 1+3, A100, both scopes, drift check).
2. **Bar (a) same-scope: MET** — fullres 57.14 s vs 61.83 s (−4.7 s, 1.082×) with the per-operator delta table vs Phase 1 (preprocess −1.87 s = the CuPy port, inference −1.04 s, bootstrap −1.6 s; in-study 39.06 s vs 42.1 s).
3. **Bar (b) headline: 1.310×** — bundle 129.54 s vs 169,747 ms with per-config table + per-stage deltas vs the reference (inference 76.7 s vs 138.2 s = −44.5% dominant; postprocess −85.8% within its 9–23 s variance).
4. **Bundle vs 61.8 s — D-18 deviation documented**: 0.477× printed WITH the scope note (single-config vs 3-config; same-scope bar is controlling; RESEARCH's 70–80 s serial estimate measured against 129.5 s, gap attributed to bootstrap + serial resample spans).
5. Profiling summary (churn PASS, NVTX PASS, CPU-bound ranking, D-16 no-overlap honest note, ncu unavailable).
6. Phase 3 handoff: ranked list (1. scipy resample ~28.8 s; 2. serial fragment scheduling ~25 s potential; 3. inference kernels — ncu-blocked; 4. postprocess 9.9 s; 5. bootstrap 20.8 s) + carried items (INFR-02, ≥5-CT corpus, 2d model).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] nsys harness path depth**
- **Found during:** Task 2 (first profile run: `--input: No such file/folder: /users/srv-mde/projects/testdata/airway_input`)
- **Issue:** the wrapper computed `REPO_ROOT` as three levels above `.planning/scripts/` (repo root is two levels up); the trace and its output dir landed in a stray `/users/srv-mde/projects/.planning/` (removed).
- **Fix:** corrected to `../..`; stray dir cleaned; re-ran successfully.
- **Files modified:** `.planning/scripts/nsight_profile_phase2.sh`
- **Commit:** fb6cbea

**2. [Rule 3 - Blocking] report arithmetic corrections before commit**
- **Found during:** Task 3 self-review of the delta tables
- **Issue:** three hand-summed values were off (bundle in-study total 108,648.8 → 108,734.7 ms; reference in-study ≈155,143 → ≈154,143 ms; bootstrap remainder 20.9 → 20.8 s, reflected in the profiles README too).
- **Fix:** recomputed from the CSV means; all report/README figures now re-add exactly.
- **Files modified:** `02-BENCHMARK-REPORT.md`, `.planning/profiles/phase2/README.md`
- **Commit:** f405d83

### Plan-observation notes (not deviations)

- **Bundle speedup 1.310× vs the plan's expected ~2.0–2.8:** recorded honestly (report §4) — the plan's expectation inherited RESEARCH's 70–80 s serial estimate, which excluded bootstrap and under-counted the serial D-13 resample spans. The acceptance criteria's expectation is superseded by the measured number with the attribution; the same-scope bar (the one the D-18 resolution designates as controlling) is MET.
- **cuda_api_sum runtime names carry a `_v3020` suffix** (`cudaMalloc_v3020`) in nsys 2025.6.3 sqlite — the churn check queried with LIKE patterns; the stats export itself shows the clean `cudaMalloc` label.

## Known Stubs

None. Every report figure is computed from `phase2_results.csv` (committed) or a profiling export (committed); no ncu numbers appear anywhere.

## Success criteria (plan)

- [x] TEST-006: E2E + per-operator benchmark for the fast app — fresh process per rep, warmup excluded, mean ± std over 3 measured reps, per-config columns for both scopes
- [x] TEST-007: speedup vs 169.7 s (1.310×) AND same-scope improvement vs 61.8 s (1.082×, MET), quantified in the roadmap CSV + the two-bar report
- [x] INFR-005: per-config NVTX ranges verified in a saved nsys trace (14/14 legible); cudaMalloc churn check recorded (PASS — no per-tile churn)
- [x] ncu documented unavailable (ncu_status.txt, ERR_NVGPUCTRPERM — no fake metrics); stream overlap noted honestly (not visible; D-16)
- [x] Phase 3 receives a ranked, trace-cited bottleneck list (report §6)

## Self-Check: PASSED
