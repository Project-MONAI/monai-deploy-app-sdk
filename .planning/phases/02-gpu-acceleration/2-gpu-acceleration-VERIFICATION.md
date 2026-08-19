---
phase: 02-gpu-acceleration
verified: 2026-08-19T07:45:00Z
status: passed
score: 38/38 must-haves verified
---

# Phase 2: GPU Acceleration Verification Report

**Phase Goal:** Replace remaining CPU-bound operations with GPU equivalents, wire all nnUNet configurations (including cascade), and achieve measurable latency improvement over the baseline.
**Verified:** 2026-08-19 07:45 UTC
**Status:** **PASSED**
**Re-verification:** No — initial verification

All 6 plans' must-haves (38 truths, 19 artifacts, 14 key links) verified against the
actual codebase, committed gate evidence, and live re-execution of the headless
unit suites. No blockers. One bookkeeping warning (stale REQUIREMENTS.md rows for
INFR-01/02/03) and one info-level pre-existing TODO.

## Goal Achievement

### Observable Truths

Must-haves taken from each plan's `must_haves:` frontmatter (2-gpu-acceleration-0[1-6]-PLAN.md).

#### Plan 01 — CuPy preprocess port

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Transpose (PREP-01) + crop-slice (PREP-04) on GPU via CuPy | ✓ VERIFIED | `preprocess_operator.py` L663–693: `cp.array(raw, cp.float32)`, `cp.ascontiguousarray(vol[None,...].transpose(0,3,2,1))`, GPU slicer `vol_t[(slice(None),)+slicer]` |
| 2 | Per-channel normalization element-wise on GPU; mean/std reductions in numpy (PREP-02) | ✓ VERIFIED | L744–764: `np.float32(ch_np.mean()/.std())` reductions, `cp.asarray(mean)` element-wise GPU apply; `cp.clip` |
| 3 | Resampling stays on scipy/scikit-image CPU path (PREP-03, D-13) with accepted round-trip | ✓ VERIFIED | L776–784: `vol_out = np.ascontiguousarray(vol_c.get())` (D2H, D-13 accepted) → unchanged `_resample_to_shape` |
| 4 | All CuPy ops fp32 + C-contiguous; no CuPy reductions | ✓ VERIFIED | Grep: reductions only via `np.float32(ch_np[...].mean()/.std())`; `cp.ascontiguousarray` at every write boundary (L668, 693, 705–713, 772); no `cp.mean/sum` |
| 5 | fullres-only E2E pixel-exact vs testdata/ref_fullres_only | ✓ VERIFIED | `plan01-gates/pixel_diff_fullres.json`: 99.9999% byte-identity, 2 differing voxels, IoU 0.99914, `pass: true`; regression re-gated in plan03/plan04 |
| 6 | gpu_residency.py passes with deliberate D-13 allow-list (not silenced) | ✓ VERIFIED | `scripts/gpu_residency.py` L89: `preprocess_operator.py` allow-listed with explicit D-13 reason string; `plan01-gates/gpu_residency_{static,runtime}.txt` = RESULT: PASS |

#### Plan 02 — RMM + memory budget

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `import rmm` before any holoscan/monai.deploy import; hazard documented; torch backend "pluggable" (INFR-01, D-14) | ✓ VERIFIED | `app.py` first import = `from my_app import gpu_bootstrap` (L40–44) + `install_torch_allocator()`; `plan02-gates/e2e_log_excerpt.txt`: `memory_allocator_backend: pluggable`; `scripts/test_gpu_bootstrap.py` (102 lines) pins the hazard via subprocesses |
| 2 | RMM pool pre-allocation in compose (warm tensor sized by budget) | ✓ VERIFIED | `app.py` L411 `memory_budget:` JSON record + L617 `gpu_bootstrap.warm_pool(plan.total_bytes)` at end of compose; `gpu_bootstrap.py` `rmm.reinitialize(pool_allocator=True, ...)` + `change_current_allocator(rmm_torch_allocator)` |
| 3 | Budget calculator → BudgetPlan full_volume/defer_to_incremental, unit-tested with forced defer | ✓ VERIFIED | **Live re-run**: `scripts/test_mem_budget.py` → `RESULT: PASS` incl. `test_defer_forced_synthetic` (40.60 GB > 40 GB → defer) and boundary case; matches `plan02-gates/test_mem_budget.txt` |
| 4 | Defer branch reachable in code; documented unexercised on A100-40GB | ✓ VERIFIED | `ensemble_average_operator.py` L300 `defer_strategy`, L394 release-after-accumulate; docstring L37–42 states bit-identical order + real OOM unexercised; `app.py` L501 wires `defer_strategy=(plan.strategy == "defer_to_incremental")` |
| 5 | INFR-02 explicitly deferred to Phase 3 (D-17), no implementation attempted | ✓ VERIFIED | D-17 in `02-CONTEXT.md` L41 (locked deviation); `ensemble_average_operator.py` L49 deferral note; no cross-study buffer-reuse implementation present in codebase |
| 6 | fullres-only E2E exits 0 with RMM active | ✓ VERIFIED | `plan02-gates/e2e_log_excerpt.txt`: pluggable backend + complete `study_timing_summary` (n_records=8) + `End run`; `plan02-gates/pixel_diff_fullres.json` pass |

#### Plan 03 — Cascade operator support + model-list semantics

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `resolve_run_model_list` reproduces reference semantics exactly, unit-tested on the real bundle (PIPE-03) | ✓ VERIFIED | `config/__init__.py` L182–277 (default = plans.json configs-with-model-dirs in plans.json order; lowres re-inserted before cascade; ensemble = run minus 3d_lowres with self-ensemble fallback; empty-ensemble error). **Live re-run** of `scripts/test_cascade_config.py`: all model-list cases PASS against the real bundle |
| 2 | Cascade preprocess consumes GPU lowres_seg (argmax, NO CC), builds 2-channel input, zero disk I/O (PIPE-04, D-09/D-10) | ✓ VERIFIED | `preprocess_operator.py` L703–716 (seg on-GPU transpose/crop), L826–842 (D-10 integer-dtype enforcement, raw probabilities raise); `postresample_operator.py` L273–292 (argmax uint8 CUDA, no CC); no file I/O anywhere on the lowres_seg path |
| 3 | Cascade seg resample = CPU scipy replica with reference seg kwargs, unit-tested vs vendored nnunetv2 | ✓ VERIFIED | `_resample_seg_to_shape` (is_seg path, plans kwargs); **live re-run** `test_seg_resample_replica` (3 shape regimes, `np.array_equal` vs vendored nnunetv2 2.8.1) PASS |
| 4 | One-hot on GPU bit-exact vs vendored `convert_labelmap_to_one_hot`; one-hot never normalized | ✓ VERIFIED | `preprocess_operator.py` L800–810 `cp.stack([(seg3==lbl).astype(cp.float32) ...])`; **live re-run** `test_one_hot_vs_reference` (CPU + CuPy, `np.array_equal`) PASS; normalization loop (step 3) predates the seg concat |
| 5 | Config-generic instantiation (D-02); no dummy 2d model (D-01/D-03) | ✓ VERIFIED | `config/__init__.py` L255: cascade producer data-driven off plans.json `previous_stage`; grep for hard-coded config names limited to the documented reference-reorder literals; no 2d model dir in `examples/apps/cchmc_nnunet_fifteen_ckpt_app/models/` and none created |
| 6 | fullres-only E2E regression still passes (Plan 01 CuPy path behavior-untouched) | ✓ VERIFIED | `plan03-gates/task{2,3}-fullres-e2e.log` + `task{2,3}-fullres-pixel-diff.txt` (both pass) |

#### Plan 04 — Multi-fragment DAG

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Each resolved config runs as its own holoscan Fragment in one DAG; DICOM I/O / ensemble / postprocess app-level (PIPE-03) | ✓ VERIFIED | `app.py` L448–475: `nnunet_{cfg}` Fragment per `run_list`; app-level ensemble/postprocess/writers at L504–612 |
| 2 | Fragment iteration over resolved list; `HOLOSCAN_MODEL_LIST` env selection, default = reference default (D-02/PIPE-03) | ✓ VERIFIED | `app.py` L389–400: `os.environ.get("HOLOSCAN_MODEL_LIST")` → `resolve_run_model_list`; `plan04-gates/e2e-*.log` show the four expected `run_model_list`/`ensemble_model_list` pairs |
| 3 | All four model-list configs run E2E exit 0, SEG/SR/SC written; cascade crosses fragment boundary with zero disk I/O (PIPE-04) | ✓ VERIFIED | `plan04-gates/e2e-{3d_fullres,3d_lowres,3d_cascade_fullres,bundle}.log` + `verification.txt`; bundle log shows `run_model_list=['3d_fullres','3d_lowres','3d_cascade_fullres']`, SEG/SR/SC writes, no lowres_seg disk artifacts |
| 4 | GXF port discipline — declared ports wired/conditional in all 4 configs (conditional emit flags) | ✓ VERIFIED | `emit_lowres_seg=True` gated on cascade producers; `INPUT_LOWRES_SEG` declared only when `params.previous_stage` set (preprocess `setup`); 4/4 configs exiting 0 (unwired declared port would hang) + `plan04-gates/verification.txt` "every fragment's operators fired exactly once" |
| 5 | Ensemble per-config `prob_<cfg>` ports, ensemble-list-order accumulation (INF-009 met-with-deviation, D-19) | ✓ VERIFIED | `ensemble_average_operator.py` L346 `spec.input(f"prob_{cfg}")`, L366–373 list-order receive (never arrival order); `app.py` L504–505 `add_flow(subgraphs[cfg], ensemble_op, {("probabilities", f"prob_{cfg}")})` for `ensemble_list`; **live re-run** `test_ensemble_order` PASS (`torch.equal`); D-19 deviation documented at source (L53–59) |
| 6 | NVTX + timing records config-tagged; per-study aggregate keyed by TOP-LEVEL application (INFR-005, Pitfall 9) | ✓ VERIFIED | `gpu_util.py` `_root()` → `fragment.application`; `StudyTimingCollector.record` keyed by `id(_root(fragment))`; e2e bundle log: one `study_timing_summary` with n_records=14 spanning all 3 sub-Fragments; per-config `"config"` fields present |
| 7 | One CudaStreamPool per fragment (NonBlocking, reserved_size=1, per-fragment nvtx); overlap best-effort, honest note (INFR-004, D-16) | ✓ VERIFIED | `app.py` L470–475 `CudaStreamPool(..., stream_flags=1, reserved_size=1, nvtx_identifier=f"streams_{cfg}")`; honest "NOT visible / single stream id 7" observation in `02-BENCHMARK-REPORT.md` §5 |

#### Plan 05 — Per-config oracles + pixel-exact gates

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Fresh per-config oracles: testdata/ref_lowres_only, testdata/ref_cascade_only | ✓ VERIFIED | Both dirs exist with SEG/SC/SR; `gates/oracle_provenance.md` documents generation with the UNMODIFIED reference app (git-clean verified) + effective model lists + voxel counts (2404 / 2519). **All 4 oracle SEG sha256 hashes re-computed on disk: MATCH** the values in `02-GATE-RESULTS.json` |
| 2 | Per-config pixel-exact gates PASS (fullres/lowres/cascade vs their oracles; D-08 controlling level) | ✓ VERIFIED | `gates/02-GATE-RESULTS.json`: fullres 99.99986% (3 voxels, IoU 0.99871), lowres 100.00000% (0 voxels, IoU 1.0), cascade 100.00000% (0 voxels, IoU 1.0); `lists_match: true` for all rows |
| 3 | Final bundle gate: fast default run vs testdata/current_output (D-06) | ✓ VERIFIED | Gate JSON row `bundle`: 100.00000%, 0 differing voxels, IoU 1.0, 2447 voxels both sides; sanity check (bundle 382 voxels differ from fullres-only) present and ok |
| 4 | DICOM-SR airway volume within 0.1% in every gated run | ✓ VERIFIED | Gate JSON: `sr_fast == sr_oracle == 1.0`, `sr_delta_pct: 0.0`, `sr_ok: true` in all 4 rows; plan01 `sr_check.txt`: "Airway Volume: 1 mL" exact match |
| 5 | gpu_residency (static + runtime) passes in the multi-fragment BUNDLE configuration | ✓ VERIFIED | Gate JSON `residency: {static: PASS, runtime: PASS}` — runtime ran HOLOSCAN_MODEL_LIST unset (bundle) with the deliberate D-13 allow-list; postprocess exactly-once boundary confirmed |
| 6 | TEST-005 recorded met-with-deviation (2d blocked-on-model D-01/D-03; no dummy 2d model) | ✓ VERIFIED | Gate JSON `deviations[TEST-005-2d]` cites D-01/D-03/D-04; no 2d model exists in the bundle or fast app |

#### Plan 06 — Two-bar benchmark + profiling

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Benchmark script: E2E + per-operator (per-config), fresh-process reps, 1 warmup + ≥3 measured, mean±std (TEST-006) | ✓ VERIFIED | `.planning/scripts/phase2_benchmark.py` (11.7 KB); `phase2_results.csv`: 10 data rows = 2 scopes × (1 warmup + 3 measured) + 2 summary rows; report §1 documents fresh-process-per-rep method |
| 2 | BOTH latency bars reported: same-scope vs 61.8 s AND bundle vs 169.7 s (TEST-007, D-18) | ✓ VERIFIED | **Re-derived from CSV**: fullres mean 57,140.3 ± 250.4 ms → 1.082× vs 61,834 ms (Phase 1, `baseline-2026-08-18.csv`); bundle mean 129,542.9 ± 896.4 ms → 1.310× vs 169,747 ± 7,274 ms (**re-derived** from `.planning/baseline_results.csv`: mean of 178,137/165,216/165,887 = 169,746.7; inference mean 138,235.7 matches report's 138,236). Report §2/§3 tables match exactly |
| 3 | Bundle-vs-61.8 s printed with scope asymmetry documented as deviation from literal D-18 | ✓ VERIFIED | Report §4: 0.477× printed, "NOT the acceptance bar", scope note + RESEARCH.md 70–80 s estimate vs 129.5 s measured explained |
| 4 | Results in `.planning/benchmarks/phase2_results.csv`, per-config columns parseable from `timing:` JSON logs | ✓ VERIFIED | CSV parsed programmatically this verification — per-config `preprocess/inference/postresample_ms_<cfg>` columns present and non-empty in the right scopes |
| 5 | nsys trace + stats in `.planning/profiles/phase2/`; cudaMalloc/cudaFree only setup/warmup (RMM, no per-tile churn); per-config NVTX legible (INFR-005) | ✓ VERIFIED | `.nsys-rep` + `.sqlite` + 4 stat exports + README + ncu_status all present; `phase2_bundle_cuda_api_sum.txt`: **9 cudaMalloc / 1 cudaFree** in a study launching 629,929 kernels (report §5 pins the 8 post-setup calls as pool-expansion events); per-config NVTX ranges in `nvtx_sum`/`nvtx_kern_sum` |
| 6 | ncu documented UNAVAILABLE (ERR_NVGPUCTRPERM), no fake kernel metrics | ✓ VERIFIED | `ncu_status.txt`: Nsight Compute installed but blocked, verified 2026-08-19 live probe, admin requirement noted; no ncu-derived numbers anywhere in profile dir or report |
| 7 | Stream-overlap observation recorded honestly (D-16) | ✓ VERIFIED | Report §5: "NOT visible — every kernel on a single CUDA stream (id 7)", likely-cause analysis (serial fragment scheduling, not pools), listed as Phase 3 lever #2 |

**Truth score: 38/38 VERIFIED**

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `my_app/operators/preprocess_operator.py` | CuPy transpose/crop/normalize + scipy resample | ✓ VERIFIED | 886 lines; `import cupy as cp`; D-13 key link `C_CONTIGUOUS` assert present (L779) |
| `my_app/scripts/gpu_residency.py` | Updated ALLOWED_TRANSFER_FILES + D-13 justification | ✓ VERIFIED | Allow-list entry with D-13 reason string (L89) |
| `my_app/gpu_bootstrap.py` | RMM-before-holoscan bootstrap | ✓ VERIFIED | `rmm.reinitialize(pool_allocator=True` + `rmm_torch_allocator` + `warm_pool`; first import of app.py |
| `my_app/mem_budget.py` | compute_memory_budget → BudgetPlan | ✓ VERIFIED | Dataclass with `full_volume`/`defer_to_incremental`; live unit tests PASS |
| `scripts/test_mem_budget.py` | Synthetic-size tests incl. forced defer | ✓ VERIFIED | **Live re-run PASS** (6 cases incl. boundary) |
| `scripts/test_gpu_bootstrap.py` | Import-order subprocess self-test | ✓ VERIFIED | 102 lines, subprocess-based (e2e excerpt shows pluggable backend) |
| `my_app/config/__init__.py` | resolve_run_model_list + cascade params | ✓ VERIFIED | `def resolve_run_model_list` (L182); `previous_stage`, `foreground_labels` data-driven |
| `my_app/operators/postresample_operator.py` | Conditional lowres_seg output (argmax uint8, no CC) | ✓ VERIFIED | `emit_lowres_seg` flag; D-09/D-10 documented; no-CC noted |
| `scripts/test_cascade_config.py` | Model-list/seg-resample/one-hot unit tests | ✓ VERIFIED | 381 lines; **live re-run: 14/14 PASS** (matches plan04-gates/verification.txt) |
| `my_app/app.py` | Fragment factory + HOLOSCAN_MODEL_LIST + CudaStreamPool | ✓ VERIFIED | All key-link patterns present (L389–506, L470–475, L617) |
| `my_app/operators/gpu_util.py` | App-keyed StudyTimingCollector + config NVTX | ✓ VERIFIED | `_root()` → `fragment.application` |
| `my_app/operators/ensemble_average_operator.py` | `prob_<cfg>` ports + ordered accumulation + defer branch | ✓ VERIFIED | All patterns present; D-19 documented at source |
| `testdata/ref_lowres_only`, `testdata/ref_cascade_only` | Fresh per-config reference oracles | ✓ VERIFIED | SEG/SC/SR present; sha256 re-verified on disk |
| `.planning/scripts/reference_fullres_run.py` | `--config` comma-list support | ✓ VERIFIED | Present; `split(',')` in source; provenance doc cites it |
| `.planning/scripts/phase2_gate.py` | 4-config gate harness | ✓ VERIFIED | 16.4 KB; produced the combined JSON |
| `.planning/phases/.../gates/02-GATE-RESULTS.json` | Machine-readable gate evidence | ✓ VERIFIED | 4 pixel gates + 4 SR + residency + deviations + sanity; `all_gates_pass: true` |
| `.planning/scripts/phase2_benchmark.py` | Benchmark harness | ✓ VERIFIED | Present; `phase2_results` output target |
| `.planning/benchmarks/phase2_results.csv` | Roadmap-required benchmark CSV | ✓ VERIFIED | 11 lines (header + 10); numbers re-derived and reconciled |
| `.planning/profiles/phase2/` | nsys rep + sqlite + stat exports | ✓ VERIFIED | 8 files incl. `ncu_status.txt` |
| `.planning/phases/.../02-BENCHMARK-REPORT.md` | Two-bar report | ✓ VERIFIED | Two bars, §4 deviation, §5 profiling, §6 Phase 3 handoff |

### Key Link Verification

| From | To | Via | Status |
|---|---|---|---|
| preprocess_image (GPU path) | `_resample_to_shape` (scipy CPU) | `.get()` → `np.ascontiguousarray` → `C_CONTIGUOUS`/fp32 assert (L778–781) | ✓ WIRED |
| preprocess transfers | gpu_residency ALLOWED_TRANSFER_FILES | D-13 reason string (L89) | ✓ WIRED |
| app.py | gpu_bootstrap | FIRST import before `from monai.deploy...` (L40–47) | ✓ WIRED |
| app.py compose() | mem_budget | `memory_budget:` JSON record (L411) + `warm_pool(plan.total_bytes)` (L617) | ✓ WIRED |
| EnsembleAverageOperator | BudgetPlan.strategy | `defer_strategy=(plan.strategy == "defer_to_incremental")` (L501) | ✓ WIRED |
| postresample lowres_seg | cascade preprocess `lowres_seg` | `add_flow(subgraphs[aux_prev], subgraphs[cfg], {("lowres_seg","lowres_seg")})` (app.py L491) | ✓ WIRED |
| cascade preprocess | SlideWindow (2ch) | `cp.concatenate([vol_gpu, one_hot], axis=0)`; num_input_channels=2 from plans (test PASS) | ✓ WIRED |
| `_resample_seg_to_shape` | vendored nnunetv2 | live `np.array_equal` unit test PASS (3 regimes) | ✓ WIRED |
| frag postresample (probabilities) | app-level ensemble | `add_flow(subgraphs[cfg], ensemble_op, {("probabilities", f"prob_{cfg}")})` for `ensemble_list` (L504–505) | ✓ WIRED |
| sub-Fragment timing | app-level study_timing_summary | `_root()`/`fragment.application` keying; bundle summary n_records=14 in e2e log | ✓ WIRED |
| fast `timing:` logs | phase2_results.csv | regex-parsed per-config columns; CSV re-derived this verification | ✓ WIRED |
| nsys trace | cudaMalloc churn check | cuda_api_sum: 9 malloc/1 free vs 629,929 kernels | ✓ WIRED |
| HOLOSCAN_MODEL_LIST=3d_cascade_fullres | testdata/ref_cascade_only | same effective run/ensemble lists (`lists_match: true` in gate JSON) | ✓ WIRED |
| fast default bundle | testdata/current_output | 100.00000% pixel gate (gate JSON) | ✓ WIRED |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|---|---|---|---|---|
| Fast-app pipeline | image volume, logits, lowres_seg, probability maps | DICOMDataLoaderOperator → real bundle inference (`examples/apps/cchmc_nnunet_fifteen_ckpt_app/models`, MRI_NICU-Airway_TRAINv2) | Yes — real A100 E2E runs with real outputs, oracle-hashed | ✓ FLOWING |
| phase2_results.csv | per-operator ms | app `timing:` JSON logs of real subprocess runs | Yes — re-derived means match report to the digit | ✓ FLOWING |
| 02-GATE-RESULTS.json | pixel/SR/residency results | `phase2_gate.py` run against real oracles | Yes — SEG sha256s re-verified on disk | ✓ FLOWING |
| Benchmark report | latency bars | CSV above | Yes — 169,747 ms reference re-derived from baseline_results.csv | ✓ FLOWING |

### Behavioral Spot-Checks (live re-execution this verification)

| Behavior | Command | Result | Status |
|---|---|---|---|
| Budget calculator incl. forced defer branch | `venv python scripts/test_mem_budget.py` | `RESULT: PASS` (6/6, boundary + 40.6 GB defer) | ✓ PASS |
| Model-list semantics + cascade params + seg-resample replica + one-hot + ensemble order (real bundle, vendored nnunetv2) | `venv python scripts/test_cascade_config.py` (from app dir) | 14/14 PASS, `ALL PASS` | ✓ PASS |
| Benchmark CSV reconciles with report | python csv re-derivation | fullres 57,140.3 ± 250.4 ms (1.082×); bundle 129,542.9 ± 896.4 ms (1.310×) | ✓ PASS |
| Oracle SEG integrity | `shasum -a 256` × 4 vs gate JSON | 4/4 MATCH | ✓ PASS |
| 4-config E2E + residency + pixel gates | — | Committed gate evidence cited (plan01–04-gates/, gates/02-GATE-RESULTS.json); long GPU runs not re-executed | ✓ PASS (evidence) |

### Requirements Coverage

All 16 phase requirement IDs from the plan frontmatters are accounted for; the union of the 6 plans' `requirements:` fields equals the phase ID set exactly — **no orphaned requirements**.

| Requirement | Source Plan | Description (REQUIREMENTS.md) | Status | Evidence |
|---|---|---|---|---|
| PREP-01 | 01 | GPU transpose to nnUNet orientation | ✓ Met | CuPy transpose (truth P1-1) + pixel gates |
| PREP-02 | 01 | GPU per-channel normalization before resample | ✓ Met | CuPy element-wise, numpy reductions (P1-2) |
| PREP-03 | 01 | Reference scipy/skimage resampling path | ✓ Met | Unchanged `_resample_to_shape` (P1-3); D-13 |
| PREP-04 | 01 | GPU crop/pad to nnUNet input shape | ✓ Met | CuPy slicer (P1-1) |
| INFR-01 | 02 | RMM pool allocator as CUDA allocator | ✓ Met | Pluggable backend logged (P2-1); RMM pool churn check PASS (P6-5) |
| INFR-02 | 02 | Pre-allocated buffers reused across compute()/studies | ✓ **Deferred with cited decision (D-17)** — locked in 02-CONTEXT.md L41; single-study dev corpus can't prove it; Phase 3 will add multi-study corpus. No implementation attempted (P2-5). REQUIREMENTS.md row still says "Pending" — consistent with deferral |
| INFR-03 | 02 | Memory budget before full-volume logits; defer to incremental | ✓ Met (defer branch reachable, real-OOM unexercised per D-15) | BudgetPlan + live tests (P2-3); app wiring (P2-4) |
| PIPE-03 | 03, 04 | Each config = independent Fragment in one DAG | ✓ Met (2d covered via config-genericity + D-01/D-03 deviation, see TEST-005) | `nnunet_<cfg>` fragments; 4/4 E2E (P3-1/5, P4-1/2/3) |
| PIPE-04 | 03, 04 | Cascade lowres seg → fullres preprocess one-hot channel, no disk I/O | ✓ Met | Cross-fragment flow + 2-channel input (P3-2, P4-3) |
| INF-009 | 04 | In-place GPU probability averaging, no .npz I/O | ✓ **Met-with-deviation (D-19)** — Phase 1 in-place accumulation + exact CuPy final division kept for bit-exactness; deviation documented at source (P4-5) |
| INFR-004 | 04 | CudaStreamPool per operator/config | ✓ Met (overlap best-effort per D-16; honestly observed as not visible — nsys shows single stream due to serial scheduling) | app.py L470–475; report §5 (P4-7, P6-7) |
| INFR-005 | 04, 06 | NVTX markers per operator for nsys correlation | ✓ Met (per-config ranges + app-keyed aggregate) | gpu_util `_root`; nsys NVTX legible (P4-6, P6-5) |
| TEST-01 | 05 | Pixel-identical SEG on ≥5-study reference corpus | ✓ **Met-with-deviation** — 4/4 gate configs pixel-exact on the single airway dev-study corpus (Phase 0 corpus deviation carried); ≥5-CT re-run deferred, recorded in gate JSON `deviations[TEST-01-corpus]` |
| TEST-005 | 05 | Test suite covers all four nnUNet configs | ✓ **Met-with-deviation** — 3 3D configs + bundle gated; 2d blocked-on-model (D-01/D-03/D-04); recorded in gate JSON `deviations[TEST-005-2d]` |
| TEST-006 | 06 | Benchmark script: E2E + per-operator latency | ✓ Met | phase2_benchmark.py + CSV (P6-1/4) |
| TEST-007 | 06 | Benchmark comparison vs reference app, speedup + absolute latency | ✓ Met | Two-bar report, numbers re-derived (P6-2/3) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `my_app/operators/dicom_series_selector_operator.py` | 200 | `# TODO type is not json now.` | ℹ️ Info | Pre-existing (Phase 1 file, not in any Phase 2 files_modified list); not a stub — functional selection logic. No action needed this phase. |

No TODO/FIXME/placeholder in any Phase 2-modified file. No empty handlers, no hardcoded-empty props, no console.log-only implementations, no stub API returns found in the 12 Phase 2 code files. The `defer_to_incremental` branch is documented-unexercised (real-OOM path) — an honest, locked deviation (D-15), not a hidden stub: the branch is code-reachable, bit-identity-preserving, and unit-tested with synthetic sizes.

### Bookkeeping Warning (non-blocking)

`.planning/REQUIREMENTS.md` lines 48–50 and the status table (lines 144–146) still show
**INFR-01, INFR-02, INFR-03 as unchecked/`Pending`** while all other Phase 2 rows were
updated ("Last updated: 2026-08-19" note mentions only TEST-01/TEST-005). The code
evidence shows INFR-01 and INFR-03 as Done (Phase 2 Plan 02) and INFR-02 as
deferred-per-D-17. Recommend a REQUIREMENTS.md touch-up commit; does not affect the
phase gate.

### Human Verification Required

None for gate purposes. All automated checks passed. Optional follow-ups (not blockers):

1. **Admin ncu unblock** — kernel-level counter profiling requires
   `NVreg_RestrictProfilingToAdminUsers=0`/sudo (`ncu_status.txt`); Phase 3 lever #3
   needs it.
2. **≥5-CT corpus re-run** — the TEST-01 final gate on a multi-study corpus when the
   user supplies additional reference examples (also unblocks INFR-02 per D-17).
3. **2d model** — when a real 2d checkpoint exists, `HOLOSCAN_MODEL_LIST=2d` is a
   test, not a code change (D-04).

### Gaps Summary

No gaps. The phase goal is achieved:

- **CPU-bound ops replaced with GPU equivalents** — transpose, crop, normalization,
  one-hot, CC postprocess, and ensemble division on CuPy/GPU; resampling deliberately
  retained on the CPU reference path (locked D-13), with the round-trip accepted and
  measured (22.2 s of bundle preprocess, the top Phase 3 lever — handed off with
  trace citations).
- **All nnUNet configurations wired including cascade** — reference-semantic
  model-list resolution (unit-tested on the real bundle), one Fragment per config,
  zero-disk-I/O cross-fragment cascade handoff, 4/4 configurations E2E, per-config
  pixel-exact gates against fresh reference oracles (all hashes re-verified on disk),
  2d honestly blocked-on-model (D-01/D-03).
- **Measurable latency improvement over baseline** — same-scope 57.14 s vs 61.83 s
  (1.082×, −7.6%) and headline bundle 129.54 s vs 169.75 s (1.310×, −23.7%); both
  bars re-derived from the committed CSVs to the digit, scope-asymmetry deviation
  documented per D-18.

All locked deviations from 02-CONTEXT.md (D-01/D-03, D-17, D-18, ncu-permission,
single-study corpus, D-19, D-15, D-16) are reflected in code comments, the gate
JSON, and the benchmark report exactly as scoped — none converted into silent gaps.

---

_Verified: 2026-08-19 07:45 UTC_
_Verifier: Claude (gsd-verifier)_
