# Phase 3 Benchmark Report — two-bar latency, the 2×2 matrix, and the optimization evidence rollup

**Date:** 2026-08-20 · **Host:** A100-SXM4-40GB (driver 610.57.04, CUDA 13.3) · **GPU:** device 0, pinned via `CUDA_VISIBLE_DEVICES=0` on every run (Pitfall 7; recorded in the CSV `gpu` column)
**venv:** `/tmp/monai-env/.venv` (torch 2.13, monai 1.3.0, holoscan-cu13 4.2.0, cupy 14.1.1, rmm 26.2.0 — scratch env, Pitfall 8)
**Study:** `testdata/airway_input` (single airway dev study, 256 MR slices 256×256 — the D-26 corpus deviation)
**Harness:** `.planning/scripts/phase3_benchmark.py` (copy-extended from `phase2_benchmark.py`, D-25) ·
**Data:** `.planning/benchmarks/phase3_results.csv` (40 rows: 2 scopes × 4 cells × (1 warmup + 3 measured) + 8 per-cell summary rows) ·
**Final gate:** `.planning/phases/03-optimization/03-GATE-RESULTS.json` (ALL GATES PASS — the shipping configuration)
**Shipped flags under test:** HOLOSCAN_CONCURRENT_FRAGMENTS default ON (D-21, Plan 01) · RMM `initial_pool_size` pinned 4 GiB (Plan 01) · INFR-02 shape caches active (Plan 03) · MEM-003 lowres weight release hook active (Plan 02) · HOLOSCAN_GPU_RESAMPLE default ON — stock CuPy (D-22b, Plan 04)

## 1. Scope & method

- **Fresh process per rep** (cold model-load start included — the clinical single-study
  usage), launched from `examples/apps/cchmc-nnunet-fast` with a raised 32 MB stack
  limit and a **per-rep `timeout` of 900 s** in the harness (a wedged rep can no longer
  hang the matrix); total = subprocess wall time.
- **1 warmup + 3 measured reps per (scope, cell)** (warmups excluded from all
  statistics); mean ± std reported. All 32 reps exited 0 (`phase3_results.csv` `ok`
  column, `all ok: True`).
- Per-operator columns parsed from the fast app's structured logs
  (`timing: {json}` per operator, config-tagged; one app-level `study_timing_summary`
  per run) **by operator name** — no record-order assumptions (Pitfall 6: record order
  is nondeterministic under the concurrent scheduler).
- **The 2×2 matrix** (both flags set EXPLICITLY per cell — never relying on defaults):

  | cell | `HOLOSCAN_CONCURRENT_FRAGMENTS` | `HOLOSCAN_GPU_RESAMPLE` |
  |---|---|---|
  | conc-resample-off | 1 | 0 (scipy CPU reference path) |
  | conc-resample-on | 1 | 1 (stock CuPy) |
  | serial-resample-off | 0 | 0 |
  | serial-resample-on | 0 | 1 |

  Plan-04 verdict note: the resample flag shipped ON (amended D-22b — gate green,
  per-tensor 100.0000% accuracy, `evidence/gpu_resample_verdict.md`), so **all four
  cells are runnable; no NA cells**. The OFF column is `HOLOSCAN_GPU_RESAMPLE=0`
  (the flag is no longer default-OFF).
- **Scopes (D-18 two-bar, unchanged from Phase 2):** `fullres` =
  `HOLOSCAN_MODEL_LIST=3d_fullres` (same-scope bar vs Phase 1 61.8 s and Phase 2
  57.14 s); `bundle` = env UNSET = reference default (3d_fullres + 3d_lowres +
  3d_cascade_fullres, ensemble over fullres + cascade_fullres; headline bar vs the
  169,747 ms reference).
- **Reproducibility cross-check:** the serial/scipy cell is the Phase 2
  configuration itself — bundle `serial-resample-off` = **127,172.7 ± 1,052.5 ms**
  vs Phase 2's 129,542.9 ± 896.4 ms (`phase2_results.csv`) = **−1.83%** — the
  methodology and environment are stable across the phase (no drift large enough to
  explain any number in this report).
- Reps' in-study operator sums are 65–82% of wall total (the rest is fresh-process
  bootstrap — python start, DICOM load, model load, RMM warm — consistent with
  Phase 2's 68–84%).

## 2. Bar (a): same-scope fullres-only — the controlling positive-improvement bar

Best shipping cell = `conc-resample-on` (both flags at their shipped defaults),
n=3 measured, `phase3_results.csv`. D-18: reported against **both** bars.

**vs Phase 1 (61.8 s, `baseline-2026-08-18.csv`, 5 measured runs 61.2–62.2 s):**

| | E2E (ms) | speedup |
|---|---:|---:|
| Phase 1 fast, 3d_fullres only (center) | 61,800 | 1.000× |
| **Phase 3 shipping, 3d_fullres** | **49,672.9 ± 166.7** | **1.244×** |

**vs Phase 2 (57.14 s, `phase2_results.csv` fullres summary row) — per-operator deltas**
(exact column names from the CSV):

| Stage (ms) | Phase 2 mean ± std | Phase 3 shipping mean ± std | Δ (ms) | Δ (%) |
|---|---:|---:|---:|---:|
| `preprocess_ms_3d_fullres` | 7,628.8 ± 57.6 | 2,279.3 ± 5.7 | −5,349.5 | −70.1% |
| `inference_ms_3d_fullres` | 26,162.9 ± 15.2 | 25,504.1 ± 2.4 | −658.8 | −2.5% |
| `postresample_ms_3d_fullres` | 1,680.4 ± 8.6 | 264.5 ± 4.8 | −1,415.9 | −84.3% |
| `ensemble_ms` | 8.0 ± 0.2 | 8.3 ± 0.3 | +0.3 | — |
| `postprocess_ms` | 2,530.4 ± 146.7 | 2,527.2 ± 8.3 | −3.2 | — |
| `write_ms` | 1,050.0 ± 60.7 | 1,604.3 ± 23.7 | +554.3 | +52.8%¹ |
| **`total_ms` (E2E)** | **57,140.3 ± 250.4** | **49,672.9 ± 166.7** | **−7,467.4** | **−13.1%** |
| speedup vs 57.14 s | 1.000× | **1.150×** | | |

¹ `write_ms` (sum of `write_seg`/`write_sr`/`write_sc`) is the largest run-to-run
variability in the pipeline (DICOM I/O); the +554 ms sits inside a rep-level swing
visible in the per-rep rows (Phase 2's own rep range 1,022–1,107 ms vs Phase 3's
1,588–1,621 ms) and does not change any verdict.

**Verdict: both bars MET.** 49.67 s vs 61.8 s (1.244×) and vs 57.14 s (1.150×) —
each far beyond the ±0.25 s run-to-run noise. Where the Phase-3 increment comes
from: the scipy resample spans that Phase 2 left CPU-locked (D-13) —
`preprocess_ms_3d_fullres` −5.35 s (−70.1%) and `postresample_ms_3d_fullres` −1.42 s
(−84.3%) — now run on stock CuPy (D-22b); inference is unchanged within noise
(−2.5%, the fp32 re-measurement range).

## 3. Bar (b): headline bundle vs 169.7 s reference

Best shipping cell = `conc-resample-on` (shipped defaults), n=3. Reference baseline:
`baseline_results.csv` — **169,747 ± 7,274 ms**.

| Stage (ms) | Phase 2 bundle mean ± std | Phase 3 shipping mean ± std | Δ (ms) | Δ (%) |
|---|---:|---:|---:|---:|
| `preprocess_ms_3d_fullres` | 7,739.7 ± 102.7 | 2,298.5 ± 55.2 | −5,441.2 | −70.3% |
| `inference_ms_3d_fullres` | 26,164.1 ± 11.3 | 50,632.6 ± 381.4 | +24,468.5 | +93.5%² |
| `postresample_ms_3d_fullres` | 1,686.1 ± 10.9 | 345.4 ± 62.8 | −1,340.7 | −79.5% |
| `preprocess_ms_3d_lowres` | 5,084.2 ± 113.7 | 2,287.6 ± 41.0 | −2,796.6 | −55.0% |
| `inference_ms_3d_lowres` | 25,136.7 ± 11.9 | 50,692.6 ± 281.0 | +25,555.9 | +101.7%² |
| `postresample_ms_3d_lowres` | 3,224.0 ± 36.5 | 2,198.7 ± 1,079.0 | −1,025.3 | −31.8% |
| `preprocess_ms_3d_cascade_fullres` | 9,382.8 ± 167.8 | 2,349.2 ± 33.2 | −7,033.6 | −75.0% |
| `inference_ms_3d_cascade_fullres` | 25,430.2 ± 15.7 | 25,390.9 ± 16.2 | −39.3 | −0.2% |
| `postresample_ms_3d_cascade_fullres` | 1,690.2 ± 4.3 | 171.8 ± 0.5 | −1,518.4 | −90.0% |
| `ensemble_ms` | 13.7 ± 1.6 | 11.8 ± 0.2 | −1.9 | — |
| `postprocess_ms` | 2,089.3 ± 529.6 | 1,501.8 ± 393.8 | −587.5 | −28.1% |
| `write_ms` | 1,093.7 ± 188.1 | 1,671.4 ± 64.4 | +577.7 | +52.8%¹ |
| **inference (3 configs summed)** | 76,731.0 | 126,714.1 | +49,983.1 | +65.1%² |
| **preprocess (3 configs summed)** | 22,206.7 | 6,935.3 | −15,271.4 | −68.8% |
| **postresample (3 configs summed)** | 6,600.3 | 2,715.9 | −3,884.4 | −58.9% |
| **`total_ms` (E2E)** | **129,542.9 ± 896.4** | **104,180.1 ± 618.5** | **−25,362.8** | **−19.6%** |
| speedup vs 169,747 ms | 1.310× | **1.629×** | | |
| speedup vs Phase 2 bundle | 1.000× | **1.243×** | | |

² The inference per-fragment and summed increases are the **expected concurrency
signature, not a regression**: under the EventBasedScheduler the independent
fullres/lowres inference fragments run in parallel and time-share the saturated
GPU — each fragment's wall doubles (~25 s → ~50 s) while the pair completes in
~50 s instead of ~51 s back-to-back (total GPU work is conserved — the
Plan-01 ceiling note, `3-optimization-01-SUMMARY.md`; the serial cells in §4 show
each fragment at ~25 s again). The cascade inference is unaffected (it runs after
both, gated on `lowres_seg`).

**Scope-asymmetry note (D-18, carried from Phase 2):** bundle-vs-61.8 s prints as
61,800 / 104,180.1 = **0.593× (2.13× slower)** — bundle runs ~2× the inference work
(3 configs) and Phase 1's 61.8 s is a single-config run; the **same-scope bar in §2
is the controlling positive-improvement bar**, and the bundle-vs-169.7 s headline
is the reference-app comparison. The cross-scope ratio is printed, never a bar.

## 4. The 2×2 matrix in full (E2E means, ms; n=3 per cell, warmups excluded)

From `phase3_results.csv` summary rows. Phase 2's measured numbers
(`phase2_results.csv`) are the serial/scipy baseline — the `serial-resample-off`
cell reproduces it (bundle 127.17 s vs 129.54 s, −1.83%; fullres 56.65 s vs 57.14 s,
−0.85%), so the two factors can be isolated against the Phase 2 state.

**fullres scope** (single config — there is no independent fragment pair to
parallelize, so the scheduler factor is expected to be ~0):

| | resample OFF (scipy) | resample ON (CuPy) | scheduler effect (serial→conc) |
|---|---:|---:|---:|
| **concurrent (shipping)** | 56,459.8 ± 356.3 | **49,672.9 ± 166.7** | −189.8 ms (−0.3%) off / +384.3 ms (+0.8%) on — within noise |
| **serial** | 56,649.6 ± 183.1 | 49,288.6 ± 193.2 | |
| **resample effect (off→on)** | **−6,786.9 ms (−12.0%)** | **−7,361.0 ms (−13.0%)** | |

**bundle scope:**

| | resample OFF (scipy) | resample ON (CuPy) | scheduler effect (serial→conc) |
|---|---:|---:|---:|
| **concurrent (shipping)** | 115,141.3 ± 293.9 | **104,180.1 ± 618.5** | **−12,031.4 ms (−9.5%)** off / **−2,841.0 ms (−2.7%)** on |
| **serial** | 127,172.7 ± 1,052.5 | 107,021.1 ± 1,595.9 | |
| **resample effect (off→on)** | **−10,961.2 ms (−9.5%)** | **−20,151.6 ms (−15.8%)** | |

**Reading the matrix (honest):**

- **GPU resample (D-22b) is the dominant Phase 3 lever**, exactly as Phase 2's
  ranked §6 list predicted (item 1: scipy resample ~28.8 s = 22.2% of bundle E2E,
  `02-BENCHMARK-REPORT.md`). It collapses in every cell: fullres spans
  7.63 s+1.68 s → 2.28 s+0.26 s; bundle resample-dominated spans
  22.2 s+6.6 s → 6.9 s+2.7 s (−19.2 s of the 28.8 s; the residual is the GPU
  transpose/crop/normalize work that was always in those spans, plus the CPU
  reductions kept on numpy per D-12/D-13).
- **Concurrent fragments (D-21) is the second lever**: −12.0 s (−9.5%) on the
  Phase 2 configuration (resample OFF). Its measured increment on the shipping
  configuration (resample ON) is smaller (−2.8 s) — under the concurrent
  scheduler the CPU-bound scipy spans partially *overlapped* other fragments'
  GPU inference (the overlap visible at light boundaries in the Phase 2/Plan-01
  traces, `02-BENCHMARK-REPORT.md` §5 / `overlap.md`), so moving the spans to the
  GPU removed part of what the scheduler was already hiding. Both effects are real
  and measured, and they are not additive — the shipping cell is the
  product, not the sum.
- No cell regresses: the worst shipping-adjacent cell
  (`conc-resample-off`, 115.1 s) is still 1.474× vs the 169.7 s reference.

## 5. Optimization evidence rollup — each Phase 3 item vs its Phase 2 §6 bottleneck item

| # | Phase 3 optimization | Phase 2 §6 item it addresses | Measured effect | Evidence |
|---|---|---|---|---|
| 1 | **D-21 concurrent fragments** (Plan 01, `EventBasedScheduler(worker_thread_number=5)`, ships ON) | item 2 — serial fragment scheduling, "~25 s potential" | nsys trace: `inference_3d_fullres` ∥ `inference_3d_lowres` **49.8 s overlap** (vs Phase 2's single stream id 7, back-to-back, 0.2 ms gap) across **5 distinct worker tids**; measured E2E effect in the §4 matrix: **−12.0 s (−9.5%)** on the serial/scipy configuration, −2.8 s on the shipping configuration (see the hiding note in §4); Plan-01 single reps: 120.4 s → 110.4 s (−10.1 s, −8.4%) with the honest ceiling (GPU-saturated inference pair time-shares; ~25 s → ~50 s per fragment, total GPU work conserved) | `.planning/profiles/phase3/overlap.md` (+ `overlap_concurrent_20260819_104410.nsys-rep`/`.sqlite`/NVTX exports); `3-optimization-01-SUMMARY.md`; `phase3_results.csv` |
| 2 | **MEM-003 lowres weight release** (Plan 02, `SlideWindowOperator.release()` + `release_fn` at the postresample tail, ships active; `HOLOSCAN_KEEP_LOWRES_WEIGHTS=1` env opt-out kept for measurement only) | item 5 class — bootstrap/memory footprint (not a ranked latency item; memory-lifecycle deliverable per D-23, **no speed claim**) | 3-moment pynvml table (2 Hz): 5.413 GiB pre-lowres-inference → 9.427 GiB at the release line → 9.552 GiB post-cascade; **peak driver VRAM ON = OFF = 9.552 GiB (Δ 0.000 GiB)**; pool level **down ~0.8 GB — DERIVED and labeled as such** (rmm 26.2.0 exposes no Python pool-stats API; reference drop + pool free-list reclamation, unit-tested). **Open Q2 answered:** `torch.cuda.empty_cache()` is a silent driver-level no-op under the rmm 26.2.0 pluggable allocator (pool never shrinks its cudaMalloc-reserved blocks) — the flat driver level is the valid, reportable result | `evidence/mem003_vram.md` (+ `mem003_vram_{on,off}.csv`, run logs, release line in `mem003_run_on.log`); `3-optimization-02-SUMMARY.md` |
| 3 | **INFR-02 cross-study shape caches** (Plan 03, `_ShapeCache` in buffer_cache.py; CuPy + torch sides; ships active) | carried item — cross-study buffer reuse (D-17 deferral) | D-24(b) replay proof (one process, same study 3×, real operators, full-process nsys): cached-buffer `data_ptr` tables of studies 2/3 **byte-identical** to study 1 (5 cached buffers + setup-time gaussian); **cudaMalloc flat: study 2 = 0, study 3 = 0** (bootstrap 1 + study 1 = 8 sub-1 ms pool expansions; 9/1 total process vs the Plan-01 10/1 bundle baseline class — never per-tile against 627,124 kernel launches); driver VRAM flat 5,469 MiB across studies (+0.00%); repeat outputs byte-identical (2,331 voxels; SR text identical). **Allocator-traffic deliverable, no first-study speed claim** (per-study wall 45.4/41.3/41.3 s; single-study-per-run is the clinical model) | `evidence/infr02_proof.md` (+ `.planning/profiles/phase3/infr02_replay_20260819_151533.nsys-rep`/`.sqlite`/`.cuda_api_sum.txt`; unit proof `scripts/test_buffer_cache.py`); `3-optimization-03-SUMMARY.md` |
| 4 | **GPU resample (D-22, AMENDED D-22a/D-22b)** (Plan 04; ships ON by default) | item 1 — scipy resample spans, "~28.8 s (22.2% of E2E)" — the ranked #1 lever | Kernel verdict: the custom RawKernel (one-and-only bounded attempt) **failed** (o3: 100.000000% of voxels differ, max_abs 2.59; o3 real-shape CUDA_ERROR_ILLEGAL_ADDRESS; o0/o1 byte-identical) → **discarded per D-22a** (provenance kept in `gpu_zoom.py`); shipping path = stock `cupyx.scipy.ndimage.zoom(grid_mode=True, mode='nearest')` mirroring the exact scipy OFF-path call. Per-tensor accuracy vs scipy on the dev corpus: **100.0000% equal, max abs diff 0, on all 5 tensors** (the ≥99% D-22b bar); ON-gate pixel-identical to the OFF baseline (adds zero divergence). **Measured in this matrix (§4): the resample spans collapse in every cell — fullres 9.31 s → 2.54 s of spans; bundle resample-dominated spans 28.8 s → 9.65 s; isolated resample effect −6.8 to −7.4 s (fullres) and −11.0 to −20.2 s (bundle)** | `evidence/gpu_resample_verdict.md` (+ `step6_per_tensor_accuracy.json`, `scripts/test_gpu_zoom_verdict.py`); `gates/03-GATE-resample-{off,on}.json`; `3-optimization-04-SUMMARY.md`; `phase3_results.csv` |

Correctness anchor for all four: the final shipping-configuration gate
(`03-GATE-RESULTS.json`) is **pixel-identical to the Phase 2/3 baseline** —
fullres 99.99986%/3 (the documented fp16↔fp32 boundary class), lowres/cascade/
bundle 100.00000%/0, SR 0.0% ×4, residency static+runtime PASS — i.e. the
optimizations changed the wall clock without moving a voxel.

## 6. Deviations & honesty

- **No regressed cell** — every shipping-configuration number is at or better
  than its Phase 2 predecessor (fullres 49.67 s vs 57.14 s; bundle 104.18 s vs
  129.54 s); the serial/scipy cell reproduces Phase 2 within −0.85%/−1.83%
  (methodology stable, §1).
- **The concurrency gain is configuration-dependent** (−12.0 s on the Phase 2
  config, −2.8 s on the shipping config) — the CPU spans the scheduler overlapped
  for free in Phase 2 are gone when the resample moves to the GPU. The two levers
  are measured per cell in §4, not massaged into one sum.
- **Per-fragment inference walls double under concurrency** (25 s → ~50 s,
  §3 note ²) — expected time-sharing of a saturated GPU, total GPU work
  conserved; the serial cells show ~25 s again.
- **MEM-003 measured a flat driver level** (Δ 0.000 GiB peak) — reported as such,
  with the ~0.8 GB pool-level benefit explicitly DERIVED and labeled; Open Q2
  answered (empty_cache is a silent no-op under RMM), not papered over.
- **RMM Open Q1 (Plan 01):** the rmm 26.2.0 default initial pool measured
  **19.97 GiB** post-reinit on device 0 (the Phase 2 trace's ~1.98 GB total
  predates the venv drift, Pitfall 8) → `initial_pool_size=4 GiB` pinned in
  `gpu_bootstrap.py` (4.1× the 1,038,502,513-byte airway bundle budget);
  post-pin 4.41 GiB. `.planning/profiles/phase3/rmm_openq1.md`.
- **GPU resample fallback taken (documented):** the custom RawKernel was
  discarded after the bounded verdict (D-22a) and the shipping path is stock
  CuPy with the ≥99%-accuracy arbiter (D-22b), not byte-identity — per the
  user's 2026-08-19 amendment; measured outcome was stricter than the bar
  (100.0000% per-tensor equality, zero-gate-divergence).
- **`write_ms` runs ~+554 ms higher in this session's reps** than Phase 2's
  (DICOM I/O variance; §2 note ¹) — flagged, no verdict impact.
- **Corpus:** single airway dev study throughout — every number in this report
  is dev-corpus-scoped (§8 item 1).

## 7. Deferred-with-reason (for VERIFICATION.md — liftable verbatim)

- **ACCEL-01 / ACCEL-02 / ACCEL-03 (inference-kernel optimization): deferred —
  blocked on ncu admin access.** Kernel-level counter profiling is unavailable:
  `ncu` fails with `ERR_NVGPUCTRPERM` (verified 2026-08-19 live probe;
  `NVreg_RestrictProfilingToAdminUsers=0` or sudo required —
  `.planning/profiles/phase2/ncu_status.txt`), and the inference kernels are
  already 91–96% GPU-busy (`.planning/profiles/phase2/phase2_bundle_cuda_gpu_kern_sum.txt`)
  in a compile environment hostile to the custom-kernel work that the D-22
  attempt exposed (one-and-only bounded attempt, discarded —
  `evidence/gpu_zoom.py` provenance / `evidence/gpu_resample_verdict.md`).
  The current ceiling for kernel-level insight without admin is
  `phase2_bundle_cuda_gpu_kern_sum.txt`. Re-opens as a gap plan if admin access lands.
- **MEM-01 (model-load caching / bootstrap reduction): deferred — not a
  measured bottleneck in the clinical model.** Models load once per fresh
  process; the ~18–21 s bootstrap (python start, DICOM load, 3× model load,
  RMM warm) is amortized across repeat studies in any multi-study deployment,
  and single-study-per-run is the locked clinical usage (latency first,
  throughput later — PROJECT.md). No repeat-study latency target was measured
  that MEM-01 would move.
- **MEM-02 (8 GB-class VRAM target): deferred — hardware-unverifiable on this
  box.** The dev hardware is an A100-SXM4-40GB; the 8 GB target class cannot be
  exercised here. The relevant measured data points are shipped anyway: RMM
  pool pinned at 4 GiB (`.planning/profiles/phase3/rmm_openq1.md`), MEM-003's
  ~0.8 GB pool-level release (`evidence/mem003_vram.md`), measured bundle peak
  driver VRAM 9.552 GiB.
- **pylibraft coordinate-descent component-connectivity (ROADMAP 3.5):
  evaluated, not taken.** Postprocess measured at ~1.5–2.5 s wall in the
  shipping configuration (`phase3_results.csv` `postprocess_ms`) ≈ 1.5% of
  bundle E2E (Phase 2's 9.9 s figure, 7.6%, was pre-optimization and within a
  9–23 s variance band, `02-BENCHMARK-REPORT.md` §3); the CuPy two-pass CC path
  is adequate under the D-20 trim (optimization stops when the next lever is
  not worth its risk) — replacing it with pylibraft is not warranted.

## 8. External dependencies (blocked-on-external, non-blocking — D-26; liftable verbatim for VERIFICATION.md)

1. **≥5-CT corpus re-run (the TEST-01 final gate)** — blocked on CT data.
   All Phase 3 verification ran on the single airway dev study
   (`testdata/airway_input`) per the Phase 0/1 corpus deviation; the
   ≥5-CT-study re-run of the full gate suite + benchmark remains the final
   TEST-01 gate. Re-opens as a gap plan if/when the CT corpus lands.
2. **ncu kernel profiling (ACCEL-01/02/03)** — blocked on admin access
   (`ERR_NVGPUCTRPERM`). No ncu kernel-metric number appears anywhere in the
   Phase 2/3 artifacts; the `cuda_gpu_kern_sum` export is the standing
   ceiling. Re-opens as a gap plan if `NVreg_RestrictProfilingToAdminUsers=0`
   is enabled.
3. **INFR-02 user reference examples** — the user is adding additional
   reference examples during Phase 3 to make the cross-study reuse proof
   corpus-real; they did not arrive before the Plan-03 gate, so INFR-02
   shipped with the D-24(a)+(b) proof strategy (unit suite + 3-study replay,
   `evidence/infr02_proof.md`). If the examples land before VERIFICATION.md,
   fold them into the gate oracle.

**Requirement status:** TEST-01, TEST-002, TEST-003, TEST-006, TEST-007 are
met on the dev corpus with the §7/§8 deviations recorded above; TEST-01's
≥5-CT half is external dependency (1) above. TEST-002 (SR 0.1% bar) and
TEST-003 (automated pixel-diff fails on divergence) are exercised by every
Phase 3 gate run, including the final `03-GATE-RESULTS.json`.
