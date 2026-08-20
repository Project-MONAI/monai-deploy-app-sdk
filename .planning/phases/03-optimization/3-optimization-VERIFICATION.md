---
phase: 03-optimization
verified: 2026-08-20T13:41:33Z
status: passed
score: 12/12 must-haves verified
---

# Phase 3: Optimization Verification Report

**Phase Goal:** Push performance to the limit based on profiling data from Phase 2. Implement only optimizations validated by profiling as the actual bottleneck.
**Verified:** 2026-08-20T13:41:33Z (initial verification)
**Status:** PASSED
**Arbitration note:** Per D-22a/D-22b amendment (user directive 2026-08-19, `03-CONTEXT.md`), the GPU-resample arbiter is **≥99% per-tensor accuracy vs the CPU scipy reference**, not byte-identity. The custom RawKernel was the one-and-only bounded attempt, was DISCARDED (o3 diverged 100.000000% of voxels + CUDA_ERROR_ILLEGAL_ADDRESS at the real 256³ bundle shape), and is retained as provenance in `gpu_zoom.py` (docstring-marked NOT WIRED). Verification below applies the amended bar.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | D-21 concurrency wired as `EventBasedScheduler(worker_thread_number=5)` behind `HOLOSCAN_CONCURRENT_FRAGMENTS`; flag OFF = byte-for-byte Phase 2 serial; default ON | ✓ VERIFIED | `my_app/app.py:638-642` — `os.environ.get("HOLOSCAN_CONCURRENT_FRAGMENTS","1") != "0"` → `self.scheduler(EventBasedScheduler(self, worker_thread_number=5, name="concurrent"))`; default resolved ON programmatically; `gates/03-GATE-serial.json` all_gates_pass=true reproduces Phase 2 baseline exactly |
| 2 | RMM initial-pool reservation re-verified against live venv and pinned (Open Q1) | ✓ VERIFIED | `my_app/gpu_bootstrap.py:37-53` — `initial_pool_size=_INITIAL_POOL_SIZE` (4 GiB) with docstring citing the 20 GiB rmm-26.x default; `.planning/profiles/phase3/rmm_openq1.md` + probe sqlite/exports present |
| 3 | nsys trace of a concurrent bundle run shows overlapping per-config spans | ✓ VERIFIED | `.planning/profiles/phase3/overlap_concurrent_20260819_104410.{nsys-rep,sqlite,nvtx_sum.txt}` + `overlap.md` (49.8 s fullres∥lowres overlap, 5 worker tids) |
| 4 | MEM-003: 3d_lowres weights freed exactly once, only in the aux config, after the terminal emit; hook semantics pass headlessly | ✓ VERIFIED | `slidewindow_operator.py` `release()` (del network + fold_state_dicts, `_bundle=None`, guard on post-release compute); `postresample_operator.py:359-394,516-517` `release_fn` fires at compute() tail; single injection in `NnUnetConfigSubgraph.compose()`; **spot-check: `scripts/test_weight_release.py` exit 0** — release_fn called exactly once, guard raises on post-release compute, driver 5093→5093 MiB (RMM pool semantics, Open Q2) |
| 5 | MEM-003 VRAM delta measured at BOTH pool and driver levels | ✓ VERIFIED | `evidence/mem003_vram.md` + `mem003_vram_{on,off}.csv` + logs — driver 5.413 → 9.427 → 9.552 GiB; peak ON vs OFF = 9.552 vs 9.552 GiB (Δ 0.000, honest); pool level ~0.8 GB down, DERIVED + labeled; `torch.cuda.memory_stats` absent from the app (RMM-safe) |
| 6 | INFR-02: GPU buffers reused across compute() keyed on (shape, dtype); CuPy-side gap closed; torch big sites cached; unit suite passes | ✓ VERIFIED | `my_app/operators/buffer_cache.py` (144 lines — `_ShapeCache`, torch+cupy families, no LRU, `shares_storage`, zero-on-borrow semantics); wired in `preprocess_operator.py:644,755,776,788,913,923` (vol/mask/vol_c/one_hot/vol2) and `slidewindow_operator.py:534,688,697-698` + fold-1 clone at `:433` (the Rule-1 aliasing fix); **spot-check: `scripts/test_buffer_cache.py` PASS** — including the "WITHOUT the fold-1 clone the sum IS corrupted" regression guard |
| 7 | INFR-02 replay proof: data_ptr stable across studies 2/3 AND cudaMalloc flat (0/0) | ✓ VERIFIED | `evidence/infr02_proof.md` — assertion `[PASS] study 2/3: data_ptr table IDENTICAL to study 1 (5 cached buffers + gaussian ptr)`; per-study cudaMalloc/cudaFree table (0/0 on studies 2/3); nsys pair `infr02_replay_20260819_151533.{nsys-rep,sqlite,cuda_api_sum.txt}` present |
| 8 | D-22 (AMENDED): stock `cupyx.scipy.ndimage` mirror ships behind `HOLOSCAN_GPU_RESAMPLE` (default ON); custom RawKernel discarded, kept as provenance; ≥99% accuracy bar met | ✓ VERIFIED | `gpu_zoom.py:81-83` — RawKernel machinery docstring-marked "NOT WIRED — provenance only, discarded per D-22a"; `gpu_zoom.py:133` — `os.environ.get("HOLOSCAN_GPU_RESAMPLE","1") != "0"` (default ON, confirmed programmatically); `stock_gpu_zoom`/`stock_gpu_resize` wired at all 3 flag sites (`preprocess_operator.py:255,345,353`, `postresample_operator.py:139`); **`evidence/step6_per_tensor_accuracy.json`: 100.0000% elements equal, max_abs_diff 0.0, on ALL 5 tensors, `all_meet_99pct: true`** — exceeds the ≥99% bar; `evidence/gpu_resample_verdict.md` records the kernel FAIL numbers (o3 100.000000% differ, max_abs 2.588608e+00, illegal-address crash) and the D-22b gate outcome |
| 9 | 2×2 benchmark matrix: 2 scopes × 4 cells × (1 warmup + 3 measured), per-operator columns, CUDA device recorded, no NA cells | ✓ VERIFIED | `.planning/benchmarks/phase3_results.csv` — parsed with Python csv: 8 cells × 4 reps (rep 1 warmup=true, reps 2–4 measured) + 8 summary rows; `gpu=0` on every row; all `ok=true`; `cell` column present; no NA per-operator cells in bundle rows |
| 10 | Report carries the D-18 two bars with per-operator deltas vs Phase 2 | ✓ VERIFIED | `03-BENCHMARK-REPORT.md` §2/§3: fullres 49,672.9 ms = 1.244× vs 61.8 s AND 1.150× vs 57.14 s with per-stage Δ table; bundle 104,180.1 ms = 1.243× vs Phase 2 129.54 s (−19.6%) with per-stage Δ table; scope-asymmetry note present. **Minor cosmetic error found:** headline "1.639× vs 169,747 ms" — recomputation gives 169,746.7/104,180.1 = **1.629×** (the Phase 2 row 1.310× is correct). No verdict impact: the controlling same-scope bar and the vs-Phase-2 bars are arithmetically correct, and the improvement is >1.6× either way. |
| 11 | Final shipping-config gate: all rows green, pixel-identical to the Phase 2/3 baseline (pixel-exact equivalence maintained) | ✓ VERIFIED | `03-GATE-RESULTS.json` (read in full): fullres 99.99986% / 3 differing voxels (the documented Phase-2 fp16↔fp32 boundary), lowres 100%/0, cascade 100%/0, bundle 100%/0; SR delta 0.0% ×4; residency static+runtime PASS; `all_gates_pass: true`; note records all four flags at shipped defaults + 4 GiB RMM pin, device 0. Per-plan gates all green too: `gates/03-GATE-{serial,concurrent,mem003,infr02,resample-off,resample-on}.json` — all 6 `all_gates_pass: true`, 4/4 rows pass each. GPU-resample ON adds **zero** divergence vs the OFF baseline (same 3 boundary voxels) |
| 12 | Measured improvement over Phase 2 on the same hardware and corpus; serial/scipy cells reproduce Phase 2 (no regressed cell); matrix isolation consistent | ✓ VERIFIED | Cross-checked raw CSVs (`phase3_results.csv` vs `phase2_results.csv` vs `baseline_results.csv`, all GPU 0 / airway dev corpus): fullres shipping 49,672.9 vs 57,140.3 ms → **1.150×** (57140.3/49672.9 = 1.1503 ✓); bundle shipping 104,180.1 vs 129,542.9 ms → **1.243×** (1.2434 ✓, −19.6% ✓); GPU-resample isolation −6.8/−7.4 s (fullres conc/serial) and −11.0/−20.2 s (bundle) — all match CSV to the 0.1 s; concurrency −12.03 s (−9.5%) on the Phase 2 config, −2.84 s on shipping; serial-resample-off reproduces Phase 2 within −0.86% (fullres) / −1.83% (bundle); baseline reference 169,746.7 ± 7,274 ms recomputed from `baseline_results.csv` raw reps ✓ |

**Score:** 12/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `my_app/app.py` | EventBasedScheduler + `HOLOSCAN_CONCURRENT_FRAGMENTS` | ✓ VERIFIED | Lines 60, 638-642; wired at compose() tail, default ON |
| `my_app/gpu_bootstrap.py` | RMM `initial_pool_size` pin | ✓ VERIFIED | 4 GiB pin with Open-Q1 docstring |
| `my_app/operators/slidewindow_operator.py` | `release()` + torch `_ShapeCache` sites + fold-1 clone | ✓ VERIFIED | release() with DAG-order guard; cache at predicted_logits/n_predictions/workon/gaussian |
| `my_app/operators/postresample_operator.py` | `release_fn` callback + flag ON at resample site | ✓ VERIFIED | `release_fn` at :359/:394/:516-517; `stock_gpu_resize` at :139 |
| `my_app/operators/buffer_cache.py` | `_ShapeCache` (torch+cupy, no LRU) | ✓ VERIFIED | 144 lines, substantive, invariants documented |
| `my_app/operators/preprocess_operator.py` | CuPy cache sites + 2 flag sites | ✓ VERIFIED | 5 CuPy cache sites; `stock_gpu_resize`/`stock_gpu_zoom` at 3 branches |
| `my_app/operators/gpu_zoom.py` | Stock CuPy mirror + provenance kernel | ✓ VERIFIED | 734 lines; `stock_gpu_zoom`/`stock_gpu_resize`/`gpu_resample_enabled` shipped; RawKernel block docstring-marked provenance-only |
| `scripts/test_weight_release.py` | Headless hook suite | ✓ VERIFIED (ran) | Exit 0 in this verification session |
| `scripts/test_buffer_cache.py` | D-24(a) headless suite | ✓ VERIFIED (ran) | PASS in this verification session, incl. fold-1-clone regression guard |
| `scripts/test_gpu_zoom.py` + `test_gpu_zoom_verdict.py` | Arbiter + bounded verdict | ✓ VERIFIED | Both present; verdict numbers match `evidence/gpu_resample_verdict.md` |
| `.planning/scripts/phase3_benchmark.py` | 2×2 harness | ✓ VERIFIED | Present; produced the CSV (cell/gpu columns, warmup excluded) |
| `.planning/benchmarks/phase3_results.csv` | Matrix record | ✓ VERIFIED | 40 data rows parsed and cross-checked |
| `03-BENCHMARK-REPORT.md` | 8-section close-out report | ✓ VERIFIED | §1–§8 present; numbers cross-checked; one cosmetic arithmetic slip (§3 headline 1.639× → should be 1.629×) |
| `03-GATE-RESULTS.json` | Final shipping-config gate | ✓ VERIFIED | Read in full; all green, pixel-identical to baseline |
| `evidence/` (gpu_resample_verdict.md, infr02_proof.md, mem003_vram.md + CSVs/logs, step6_per_tensor_accuracy.json) | Proof records | ✓ VERIFIED | All present; internal numbers consistent with gates/CSV |
| `gates/03-GATE-*.json` (6 files) | Per-plan D-25 gates | ✓ VERIFIED | All `all_gates_pass: true` |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `app.py` compose() | `EventBasedScheduler` | `self.scheduler(EventBasedScheduler(self, worker_thread_number=5, name="concurrent"))` before run() | ✓ WIRED | app.py:641 |
| `NnUnetConfigSubgraph.compose()` | `PostResampleOperator` | `release_fn=sw.release` (single injection, aux configs only) | ✓ WIRED | Verified single injection + `if self._emit_lowres_seg` condition |
| `PostResampleOperator.compute()` tail | release callback | `self._release_fn()` after final emit | ✓ WIRED | postresample_operator.py:516-517; fires exactly once (test case c) |
| `PreprocessOperator` | `_ShapeCache` (cupy) | `self._buf_cache.get(...)` at vol/mask/vol_c/one_hot/vol2 | ✓ WIRED | 5 call sites confirmed |
| `SlideWindowOperator` | `_ShapeCache` (torch) | cache get at predicted_logits/n_predictions/workon; fold-1 clone at emit | ✓ WIRED | lines 344-347, 375, 433, 688, 697-698 |
| flag sites → `gpu_zoom` | stock mirror | `gpu_resample_enabled()` check → `stock_gpu_resize`/`stock_gpu_zoom` | ✓ WIRED | 3 sites; OFF path byte-for-byte scipy (resample-off gate = exact baseline) |
| `phase3_benchmark.py` → `phase2_results.csv` | per-operator deltas | report §2/§3 delta tables use exact CSV column names | ✓ WIRED | Cross-checked against both CSVs |
| `03-GATE-RESULTS.json` ← `phase2_gate.py --report` | gate harness | JSON schema matches phase2_gate output (pixel_diff/sr/residency sections) | ✓ WIRED | Residency static+runtime logs referenced |

### Data-Flow Trace (Level 4)

Not applicable in the web sense — this is a GPU inference pipeline, not a render layer. The analogous hollow-wiring check (does the shipped path actually execute the shipped code) was performed instead:

| Shipped path | Executed by default? | Status |
|---|---|---|
| `EventBasedScheduler` concurrency | Yes — `HOLOSCAN_CONCURRENT_FRAGMENTS` unset → "1" → ON (confirmed programmatically) | ✓ FLOWING |
| Stock CuPy resample | Yes — `HOLOSCAN_GPU_RESAMPLE` unset → "1" → ON (confirmed programmatically); ON-gate JSON is the shipping gate | ✓ FLOWING |
| `_ShapeCache` buffers | Yes — unconditional at operator setup (no flag); D-24(b) replay proves address reuse | ✓ FLOWING |
| MEM-003 release hook | Yes — unconditional for aux config; env opt-out `HOLOSCAN_KEEP_LOWRES_WEIGHTS=1` only for A/B measurement | ✓ FLOWING |
| Custom RawKernel | **Correctly NOT wired** — provenance only, per the amended D-22a gate outcome | ✓ CORRECT (discarded, not a hollow path) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| INFR-02 cache semantics (multi-call reuse, shape invalidation, fold-1 clone regression guard) | `CUDA_VISIBLE_DEVICES=0 /tmp/monai-env/.venv/bin/python scripts/test_buffer_cache.py` | `RESULT: PASS (all D-24(a) cache semantics green)`; fold-1 clone guard proves the caught aliasing bug stays fixed | ✓ PASS |
| MEM-003 release-hook semantics (exactly-once, aux-only, post-release guard) | `CUDA_VISIBLE_DEVICES=0 /tmp/monai-env/.venv/bin/python scripts/test_weight_release.py` | `RESULT: PASS`, exit 0; compute() after release raises the DAG-ordering guard | ✓ PASS |
| Shipped defaults resolve ON + modules importable | venv python: `gpu_resample_enabled()`, env default check, `_ShapeCache` import | `GPU_RESAMPLE default: True`, `CONCURRENT_FRAGMENTS default: True`, import OK | ✓ PASS |
| 2×2 matrix + two bars arithmetic | Python csv parse of `phase3_results.csv` / `phase2_results.csv` / `baseline_results.csv` | All headline numbers reproduce (1.150×, 1.243×, −19.6%, isolation deltas); one report headline off by 0.01× (1.639 vs 1.629) | ✓ PASS (with cosmetic finding) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| INFR-02 | 03 | Shape-keyed GPU buffer reuse | ✓ SATISFIED (dev corpus) | Truths 6-7; D-24(a)+(b) proof; D-26 user reference examples = blocked-on-external |
| MEM-003 | 02 | Free 3d_lowres weights after aux fragment | ✓ SATISFIED | Truths 4-5; release-once + both-level VRAM delta |
| GPUP-01 | 04 | GPU resampler | ✓ SATISFIED — met-with-documented-tolerance per the AMENDED D-22b gate | Truth 8: 100.0000% per-tensor vs scipy (≥99% bar), ON gate green; tolerance = the amended bar itself, user-directed |
| GPUP-02 | 04 | Zero CPU-GPU transfers in resample span | ✓ SATISFIED — met-with-documented-tolerance | Residual recorded in `gpu_resample_verdict.md`: numpy mean/std reductions + ~8 MB mask round trip stay CPU per D-12/D-13 (Phase 1/2 decisions) |
| TEST-01 | 01-05 | Gate suite on ≥5-CT corpus | ⚠️ SATISFIED on dev corpus; **≥5-CT half blocked-on-external (D-26)** | Single airway dev study, Phase 0/1 deviation carried; §8(1) of report |
| TEST-002 | 01-05 | SR 0.1% tolerance check | ✓ SATISFIED | Every gate run: `sr_delta_pct: 0.0`, `sr_ok: true` ×4, incl. final `03-GATE-RESULTS.json` |
| TEST-003 | 01-05 | Automated pixel-diff fails on divergence | ✓ SATISFIED | `scripts/pixel_diff.py` drives every gate JSON (`byte_identity_pct`/`differing_voxels`/`iou` + `fail_reasons`); 6/6 plan gates + final gate all green |
| TEST-006 | 05 | Benchmark script, E2E + per-operator | ✓ SATISFIED | `phase3_benchmark.py` + `phase3_results.csv` (2×2, per-operator columns, device column) |
| TEST-007 | 05 | Final gate re-run in shipping config | ✓ SATISFIED | `03-GATE-RESULTS.json` — all flags at shipped defaults, all rows green |

No orphaned requirements: all phase-mapped REQ-IDs appear in plan frontmatters and are covered above.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `03-BENCHMARK-REPORT.md` | §3 table ("speedup vs 169,747 ms: 1.639×") | Arithmetic slip — 169,746.7/104,180.1 = 1.629× | ℹ️ Info (cosmetic) | No verdict impact: the controlling same-scope bar (1.150×) and vs-Phase-2 bundle bar (1.243×) are both correct; the improvement exceeds 1.6× either way. Suggest a one-line doc fix. |

No TODO/FIXME/placeholder comments, no empty implementations, no hardcoded-empty-data stubs found in any Phase 3 modified file. The only `NOT WIRED` marker in the codebase (`gpu_zoom.py` RawKernel block) is the **intended, user-directed provenance retention** per D-22a — not an anti-pattern.

### Deferred-with-Reason (D-20/CONTEXT §7 — verified documented in report §7)

- **ACCEL-01/02/03** — deferred, blocked on ncu admin access (`ERR_NVGPUCTRPERM`, live-probed 2026-08-19); inference kernels already 91-96% GPU-busy per Phase 2 `cuda_gpu_kern_sum`; hostile compile environment documented by the D-22 kernel attempt.
- **MEM-01** — deferred, not a measured bottleneck (models load once per fresh process; bootstrap amortized).
- **MEM-02** — deferred, 8 GB target hardware-unverifiable on A100-40GB; measured data points shipped anyway (4 GiB RMM pin, ~0.8 GB pool release, 9.552 GiB measured peak).
- **pylibraft CC** — evaluated, not taken (postprocess ~1.5% of bundle E2E; CuPy path adequate under the D-20 trim).

### External Dependencies (D-26 — blocked-on-external, NON-BLOCKING; not gaps)

1. **≥5-CT corpus re-run (TEST-01 final gate)** — blocked on CT data. Phase 3 verified on the single airway dev study per the Phase 0/1 deviation. Re-opens as a gap plan if the corpus lands.
2. **ncu kernel profiling (ACCEL-01/02/03)** — blocked on admin access (`ERR_NVGPUCTRPERM`). The `cuda_gpu_kern_sum` export is the standing ceiling. Re-opens if `NVreg_RestrictProfilingToAdminUsers=0` is enabled.
3. **INFR-02 user reference examples** — did not arrive before the Plan-03 gate; INFR-02 shipped with the D-24(a)+(b) proof strategy. If the examples land, fold them into the gate oracle.

### Gaps Summary

None. All three success criteria are met and independently re-derived from raw artifacts:

1. **All optimizations motivated by Phase 2 profiling data** — GPU resample (bottleneck #1, ~28.8 s / 22.2%), concurrency (bottleneck #2, ~25 s of serial CPU spans behind GPU inference), MEM-003 and INFR-02 (churn/VRAM), all trace to Phase 2 profiles; TensorRT/torch.compile explicitly out of scope with reasons; the 2×2 matrix *predicted* the resample-collapse that materialized exactly (28.8 s → 9.65 s resample-dominated spans).
2. **Pixel-exact equivalence maintained** — the final shipping-config gate is pixel-identical to the Phase 2 baseline (fullres 99.99986%/3 = the pre-existing documented fp16↔fp32 boundary, all others 100%/0, SR 0.0%, residency PASS); each of the four optimizations passed its own D-25 gate re-run at the same baseline; the GPU-resample ON path adds **zero** new divergence.
3. **Measured improvement over Phase 2, same hardware and corpus** — bundle 129.54 s → 104.18 s (1.243×, −19.6%) and same-scope fullres 57.14 s → 49.67 s (1.150×); the serial/scipy control cell reproduces Phase 2 within 1.83%, ruling out methodology drift; GPU pinned to device 0 in both CSVs.

One cosmetic report arithmetic slip (1.639× → 1.629× in §3) is recorded as Info-only.

---

_Verified: 2026-08-20T13:41:33Z_
_Verifier: pi (gsd-verifier) — initial verification; long GPU runs cited from committed artifacts per verifier protocol, arithmetic independently re-derived from the raw CSVs/JSON_
