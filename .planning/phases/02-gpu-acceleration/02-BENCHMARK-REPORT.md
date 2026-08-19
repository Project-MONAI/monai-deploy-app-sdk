# Phase 2 Benchmark Report — two-bar latency + per-operator deltas + profiling

**Date:** 2026-08-19 · **Host:** A100-SXM4-40GB (driver 610.57.04, CUDA 13.3) ·
**venv:** `/tmp/monai-env/.venv` (torch 2.13, monai 1.3.0, holoscan-cu13 4.2.0, rmm pluggable) ·
**Study:** `testdata/airway_input` (256 MR slices, 256×256) ·
**Harness:** `.planning/scripts/phase2_benchmark.py` (extends `baseline_benchmark.py`) ·
**Data:** `.planning/benchmarks/phase2_results.csv` (10 rows: 2 scopes × (1 warmup + 3 measured) + 2 summary)

## 1. Scope & method

- **Fresh process per rep** (cold model-load start included — the clinical
  single-study usage), launched from `examples/apps/cchmc-nnunet-fast` with a
  raised 32 MB stack limit; total = subprocess wall time.
- **1 warmup + 3 measured reps per scope** (warmups excluded from all
  statistics); mean ± std reported.
- Per-operator columns parsed from the fast app's structured logs
  (`timing: {json}` per operator, config-tagged per Plan 04 / INFR-005
  Pitfall 9; one app-level `study_timing_summary: {json}` per run — 8 records
  for the fullres scope, 14 for the bundle).
- **Two `HOLOSCAN_MODEL_LIST` scopes (the resolved D-18 two-bar):**
  - `fullres` — `HOLOSCAN_MODEL_LIST=3d_fullres` (the same scope Phase 1 was
    benchmarked in);
  - `bundle` — env var UNSET = reference default
    (3d_fullres + 3d_lowres + 3d_cascade_fullres, ensemble over
    fullres + cascade_fullres).
- Reproducibility checks: every rep exited 0; per-rep operator sums are
  68–84% of wall total (the rest is bootstrap: python start, DICOM load,
  model load, RMM warm — consistent across all 8 reps);
  `inference_ms_3d_fullres` = 26.16 s vs Phase 1's 27.2 s (within a few
  seconds — no environment drift detected).

## 2. Bar (a): same-scope fullres-only vs Phase 1 61.8 s

Phase 2 fast app, `HOLOSCAN_MODEL_LIST=3d_fullres`, n=3 measured (warmup
excluded). Deltas vs the Phase 1 fast-app numbers from
`.planning/benchmarks/baseline-2026-08-18.csv` (median 61,834 ms E2E;
in-study 42.1 s with the Plan 05 per-operator breakdown).

| Stage (ms) | Phase 1 fast (3d_fullres) | Phase 2 fast (3d_fullres) mean ± std | Δ (ms) | Δ (%) |
|---|---:|---:|---:|---:|
| preprocess_3d_fullres | 9,500 (cold¹) | 7,628.8 ± 57.6 | −1,871 | −19.7% |
| inference_3d_fullres | 27,200 | 26,162.9 ± 15.2 | −1,037 | −3.8% |
| postresample_3d_fullres | 1,700 | 1,680.4 ± 8.6 | −20 | −1.2% |
| ensemble_average | 9 | 8.0 ± 0.2 | −1 | — |
| postprocess | 2,700 | 2,530.4 ± 146.7 | −170 | −6.3% |
| writers (SEG+SR+SC) | 1,100 | 1,050.0 ± 60.7 | −50 | −4.5% |
| **in-study total** | **42,100** | **39,060.5** | **−3,040** | **−7.2%** |
| bootstrap (E2E − in-study) | ≈19,700 | ≈18,080 | ≈−1,620 | — |
| **E2E total** | **61,834** (61,205–69,536 range) | **57,140.3 ± 250.4** | **−4,694** | **−7.6%** |
| **speedup vs 61.8 s** | 1.000× | **1.082×** | | |

¹ Phase 1's preprocess was "9.5 s cold / 2.2 s warm" (Plan 05 summary); the
cold (fresh-process) value is the like-for-like comparison for
fresh-process reps.

**Verdict: the same-scope D-18 bar is MET** — a positive E2E improvement is
measured at identical scope: **57.14 s vs 61.83 s (−4.7 s, 1.082×)**, well
beyond the run-to-run noise (±0.25 s). Where the win comes from:

- **CuPy preprocess (Plan 01):** transpose/crop/normalize moved on-GPU →
  −1.9 s in `preprocess_3d_fullres` (19.7% faster). The span still ends at
  ~7.6 s because the D-13 scipy resample + accepted GPU↔CPU round-trip
  dominates it (see §5).
- **Inference:** −1.0 s (−3.8%) — within the fp32-accumulation
  re-measurement range, no drift.
- **Bootstrap:** ≈−1.6 s (RMM pre-allocation at setup vs ad-hoc growth).
- **Losses/unchanged:** the scipy-resample round-trips (preprocess +
  postresample ≈ 9.3 s combined) are untouched by design (D-13) — they are
  the next-largest in-study cost after inference.

## 3. Bar (b): headline bundle vs 169.7 s reference

Phase 2 fast app, default bundle (3 configs), n=3 measured. Reference
baseline: `.planning/baseline_results.csv` — **169,747 ± 7,274 ms**
(n=3, warmup excluded), per-stage means over the same n=3: setup 12,754 ms,
inference 138,236 ms, postprocess 9,335–23,343 ms (mean 14,727 ms), write
1,180 ms.

| Stage (ms) | Reference app (bundle) | Phase 2 fast app (bundle) mean ± std | Δ (ms) | Δ (%) |
|---|---:|---:|---:|---:|
| preprocess_3d_fullres | — | 7,739.7 ± 102.7 | — | — |
| inference_3d_fullres | — | 26,164.1 ± 11.3 | — | — |
| postresample_3d_fullres | — | 1,686.1 ± 10.9 | — | — |
| preprocess_3d_lowres | — | 5,084.2 ± 113.7 | — | — |
| inference_3d_lowres | — | 25,136.7 ± 11.9 | — | — |
| postresample_3d_lowres | — | 3,224.0 ± 36.5 | — | — |
| preprocess_3d_cascade_fullres | — | 9,382.8 ± 167.8 | — | — |
| inference_3d_cascade_fullres | — | 25,430.2 ± 15.7 | — | — |
| postresample_3d_cascade_fullres | — | 1,690.2 ± 4.3 | — | — |
| ensemble_average | — | 13.7 ± 1.6 | — | — |
| inference (all 3 configs, summed) | 138,236 | 76,731.0 | −61,505 | −44.5% |
| preprocess (all 3 configs, summed) | (inside reference inference/preproc stage) | 22,206.7 | — | — |
| postprocess | 14,727 (9,335–23,343) | 2,089.3 ± 529.6 | −12,638 | −85.8% |
| writers | 1,180 | 1,093.7 ± 188.1 | −86 | −7.3% |
| **in-study total** | ≈154,143 | **108,734.7** | — | — |
| **E2E total** | **169,747 ± 7,274** | **129,542.9 ± 896.4** | **−40,204** | **−23.7%** |
| **speedup vs 169.7 s** | 1.000× | **1.310×** | | |

The dominant win is the **inference stage: 76.7 s vs 138.2 s (−44.5%)** —
the fast app's single setup-time model load + FP32 sliding-window loop vs the
reference's per-stage nnUNet machinery; the secondary wins are postprocess
(GPU CC vs reference CPU postprocessing, −85.8% within its wide 9–23 s
variance) and the in-memory (zero-disk) cascade/ensemble handoff.

## 4. Bundle vs 61.8 s (documented deviation from literal D-18)

Printed next to the same-scope bar for completeness:

| | E2E (ms) |
|---|---:|
| Phase 1 fast, 3d_fullres only | 61,834 |
| Phase 2 fast, **bundle** (3 configs) | 129,542.9 ± 896.4 |
| ratio | 0.477× (2.10× slower) |

**Scope note (deviation from the literal D-18 wording):** Phase 1's 61.8 s
was a SINGLE-CONFIG (3d_fullres) run; the bundle runs ~2× the inference work
(three configs: 76.7 s of inference vs 26.2 s). Comparing them is
scope-asymmetric, so the **same-scope bar in §2 is the controlling
positive-improvement bar** (MET at 1.082×). This is the D-18 resolution
confirmed by the user 2026-08-19 (02-CONTEXT.md) and RESEARCH.md Open
Question 1 — the two-bar report, not a single literal bar. RESEARCH.md
estimated a serial 3-config bundle at **70–80 s** (in-study inference
extrapolation); **measured: 129.5 s E2E** (108.7 s in-study) — the estimate
excluded the ~20.8 s fresh-process bootstrap (python start, DICOM load,
3× model load, RMM warm) that the reference baseline's 12.8 s setup stage
also partially absorbs, and it under-counted the three serial preprocess
(22.2 s, scipy-resample-dominated) and postresample (6.6 s) spans. The
bundle-vs-61.8 s ratio is printed but is NOT the acceptance bar.

## 5. Profiling summary (nsys; ncu unavailable)

Trace: `.planning/profiles/phase2/` — `phase2_bundle_20260819_071235.nsys-rep`
(+ `.sqlite`), exports `phase2_bundle_cuda_api_sum.txt`,
`phase2_bundle_nvtx_sum.txt`, `phase2_bundle_nvtx_kern_sum.txt`,
`phase2_bundle_cuda_gpu_kern_sum.txt`, findings in `README.md`,
`ncu_status.txt`.

- **cudaMalloc churn (INFR-005): PASS — no per-tile churn.**
  `phase2_bundle_cuda_api_sum.txt`: 9 `cudaMalloc` (each < 2 ms; ≈1.98 GB
  total) + 1 `cudaFree` in a 117.5 s study window that launched 629,929 GPU
  kernels (~209k launches per inference fragment). One allocation precedes
  the study (t−9.9 s, setup/RMM warm pool); the other 8 sit at the start of
  later operator stages (t+1.8–2.1 s, +13.7 s, +79.1 s vs the first operator
  range) — RMM pool *expansion* events, not per-tile allocations. RMM
  (pluggable backend, Plan 02) is doing its job: the sliding-window loop
  allocates nothing per patch.
- **Per-config NVTX legibility (INFR-005): PASS.** All 14 operator ranges
  are legible in `phase2_bundle_nvtx_sum.txt` / `phase2_bundle_nvtx_kern_sum.txt`:
  `preprocess_/inference_/postresample_` × {3d_fullres, 3d_lowres,
  3d_cascade_fullres} + `ensemble_average`, `postprocess`,
  `write_seg`/`write_sr`/`write_sc`, with per-config boundaries matching the
  `timing:` JSON logs to the millisecond.
- **Top CPU-bound / GPU-idle regions (kernel timeline):**
  1. **scipy resample spans** — `preprocess_*` is 99.9% CPU (wall 7.6 / 5.0 /
     9.4 s vs 2–3 ms of kernels; D-13 round-trip) and `postresample_*`
     likewise (1.7–3.2 s, 0 ms kernels). In the bundle these sum to ~28.8 s
     of the 129.5 s E2E.
  2. **postprocess** — 9.9 s wall NVTX (5 ms GPU; the documented
     exactly-once D2H boundary + two-pass CC + contour/SR).
  3. **Bootstrap** — ≈20.9 s (E2E − in-study): DICOM load, 3× model load,
     RMM warm.
  4. **Inference spans are GPU-saturated** (24.3–24.6 s kernels on 25.1–26.7 s
     wall = 91–96% busy); further inference gains need kernel-level
     counter metrics.
- **Stream overlap (D-16, honest observation): NOT visible.** The three
  inference fragments run strictly back-to-back (`inference_3d_fullres` ends
  0.2 ms before `inference_3d_lowres` starts) and **every kernel in the run
  executes on a single CUDA stream (id 7)** — the per-fragment
  `CudaStreamPool`s (`streams_<cfg>`) are never concurrently active because
  the DAG schedules the fragments serially (cascade additionally waits on
  the lowres `lowres_seg` edge). Likely reason: serial fragment scheduling,
  not the stream pools. Overlap IS visible at light boundaries
  (`postresample_3d_fullres` runs while lowres inference is in flight), but
  never between the heavy inference spans.
- **ncu: UNAVAILABLE.** `ncu_status.txt`: Nsight Compute 2026.1.0 is
  installed but blocked by `ERR_NVGPUCTRPERM` (verified 2026-08-19 live
  probe); kernel-level counter profiling requires admin
  (`NVreg_RestrictProfilingToAdminUsers=0` or sudo). No ncu kernel-metric
  numbers appear anywhere in these artifacts.

## 6. Phase 3 handoff

Ranked bottleneck list (bundle, 129.5 s E2E / 108.7 s in-study), each with
its trace citation (all in `.planning/profiles/phase2/`,
`phase2_bundle_20260819_071235.nsys-rep` + `.sqlite`):

1. **scipy resample spans — ~28.8 s (22.2% of E2E).** GPU time in the
   spans: 2–3 ms of 5.0–9.4 s preprocess walls + 1.7–3.2 s postresample walls
   (`phase2_bundle_nvtx_kern_sum.txt`; D-13 locked them to CPU in Phase 2 —
   GPU resampling is the v2 GPUP-01 item). Largest single Phase 3 lever.
2. **Serial fragment scheduling — up to ~25 s potential.** No stream overlap
   (all kernels on stream 7; inference spans back-to-back, §5). If
   fullres + lowres inference could run concurrently (independent
   fragments), the 76.7 s inference block could approach ~52 s. D-16 note:
   the pools are wired but scheduling is the blocker — a DAG/executor-level
   change, not a stream-pool change.
3. **Inference kernels — 76.7 s at 91–96% GPU busy.** Saturated, but the
   per-kernel profile (memory-bound vs compute-bound, TTA cost) requires
   ncu — **blocked on `ERR_NVGPUCTRPERM` admin access** (`ncu_status.txt`);
   `phase2_bundle_cuda_gpu_kern_sum.txt` is the current ceiling for
   kernel-level insight without admin.
4. **postprocess — 9.9 s wall (GPU 5 ms)** — CC + contour + the exactly-once
   D2H boundary (`phase2_bundle_nvtx_sum.txt`).
5. **Bootstrap — ≈20.8 s** (DICOM load, 3× model load, RMM warm) —
   model-load caching / lazy RMM warm if the single-study clinical usage
   ever shifts to repeat studies.

**Carried items:**
- **INFR-02** cross-study buffer reuse — deferred to Phase 3 (D-17); the
  user is adding additional reference examples in Phase 3 to make it provable.
- **≥5-CT corpus re-run** — TEST-01 final gate, blocked on CT data (dev
  corpus deviation, carried from Phase 0/1).
- **2d model validation** — D-01/D-03/D-04: blocked-on-model; the fragment
  wiring is config-generic (D-02), so it is a test, not a code change.
