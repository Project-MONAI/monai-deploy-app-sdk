---
phase: 1-core-pipeline
verified: 2026-08-18T21:54:02Z
status: passed
score: 24/24 must-haves verified
verifier: gsd-verifier (adversarial, independent re-run)
notes: |
  PASS WITH DOCUMENTED DEVIATIONS. Two deliberate, project-approved deviations prevent
  literal satisfaction of two acceptance criteria; both are reasoned and evidenced below:
  (1) FP16-vs-FP32 TTA accumulation (INF-004 mandates FP32; the vendored nnUNet 2.8.1
  reference accumulates FP16 — verified in source, nnUNet/nnunetv2/inference/predict_from_raw_data.py:621-623),
  which makes literal "bit-for-bit" vs the reference unreachable by design. Gate calibrated to
  >=99.999% voxel identity vs a run-to-run-deterministic 3d_fullres-only oracle; independent
  re-run this verification: 99.99990% byte identity, 2 differing voxels, IoU 0.999142.
  (2) >=5-study final gate deferred per TEST-01 deviation (2026-08-17) — corpus does not exist yet;
  re-run required when supplied.
re_verification: false
human_verification:
  - test: "Final Phase-1 gate on >=5 CT studies (TEST-01 deviation)"
    expected: "pixel_diff.py vs a freshly regenerated 3d_fullres-only reference passes (>=99.99% identity / <=10000 diff voxels) and SR within 0.1% for each study"
    why_human: "Reference corpus is not yet available; requires studies to be supplied and full reference+fast runs"
  - test: "Visual check of SC overlay on a workstation (e.g., OHIF)"
    expected: "Airway contour overlay renders plausibly on the CT series, consistent with the reference app's SC"
    why_human: "SC is bit-identical under frame-axis transpose (byte-level verified), but rendered appearance is a visual judgment"
---

# Phase 1: Core Pipeline — Verification Report

**Phase Goal:** An end-to-end working Holoscan DAG that processes a single CT study from DICOM input to DICOM-SEG output, running all computation on GPU with CPU-equivalent preprocessing, producing pixel-exact results matching the reference app.

**Verified:** 2026-08-18T21:54:02Z
**Status:** **passed (with documented deviations — see "Deviation Judgments")**
**Re-verification:** No — initial verification. No prior VERIFICATION.md existed.

**Method:** did not trust the SUMMARYs. Every major claim below was re-derived from the
codebase, the committed git history, the committed Nsight trace, or an independent fresh
E2E re-run performed during this verification (A100, GPU 0, venv `/tmp/monai-env/.venv`).

---

## Goal Achievement

### Observable Truths (success criteria + acceptance criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Single config (3d_fullres) runs the full chain Preprocess → SlideWindow → PostResample → (EnsembleAverage) → Postprocess → DICOM-SEG | ✓ VERIFIED | Fresh re-run this verification: `python my_app -i testdata/airway_input -m <models> -o /tmp/verify_fast_out` → **exit 0**, SC/SEG/SR produced, post-CC voxel count 3655. DAG in `my_app/app.py` lines 309–349 (13 `add_flow` calls; preprocess feeds both slidewindow and postresample via `preprocessed`/`preprocessed_meta`). Ensemble in chain as no-op single-config (n_configs=1 in timing log) |
| 2 | DICOM-SEG matches reference on the reference corpus (bit-for-bit criterion) | ✓ VERIFIED **within documented deviation** | Independent `pixel_diff.py` re-run: fresh output vs `testdata/ref_fullres_only/SEG` → **99.99990% byte identity, 2/16,777,216 differing voxels, IoU 0.999142, geometry MATCH, RESULT PASS** (Plan 05 recorded 99.99986%/3 voxels — 1-voxel delta at the same z=195 tube boundary, consistent with documented argmax-boundary sensitivity). Literal bit-for-bit unreachable: INF-004 mandates FP32 accumulation while the reference source accumulates FP16 (verified: `nnUNet/nnunetv2/inference/predict_from_raw_data.py:621-623` `dtype=torch.half`). See Deviation Judgment #1. The full-bundle historical `current_output` comparison (383 voxels, IoU 0.852) confirms it is the wrong oracle, as Plan 04/05 documented |
| 3 | All intermediate tensors verified to stay on GPU between inference and postprocessing | ✓ VERIFIED **with documented allow-list** | Re-ran `scripts/gpu_residency.py --static`: **PASS** — 0 transfers in preprocess/slidewindow/ensemble/sc_overlay; 5 in `postresample_operator.py` (documented reference-CPU scipy resample path, Phase 0/1 decision ratified by Phase 2 task 2.3 "Keep resampling on reference CPU path"); **exactly 1** `.cpu()` boundary transfer at `postprocess_operator.py:509` (the authorized POST-03 transfer to the SEG writer). `assert_on_gpu` guard present in all 5 `compute()`s and executed during my fresh E2E re-run (would raise on CPU fallback — run exited 0). Note: the *runtime* hook mode of gpu_residency.py was NOT re-run by the verifier (it runs the full app a second time); the static scan + live E2E with boundary asserts is the evidence. Nuance on the literal criterion wording: "no `.cpu()` between inference output and DICOM-SEG writer" is literally violated by the documented resample transfer — see Deviation Judgment #3 |
| 4 | DICOM I/O uses existing, unchanged SDK operators | ✓ VERIFIED | `git log --since=2026-08-17 -- monai/` → **empty**. `git log 3d7d123..HEAD -- examples/apps/cchmc_nnunet_fifteen_ckpt_app/` → **empty** (reference app untouched in Phase 1). `app.py` uses SDK `DICOMDataLoaderOperator`/`DICOMSeriesToVolumeOperator`/`DICOMSegmentationWriterOperator`/`DICOMTextSRWriterOperator` directly; timed writers are app-local subclasses in `app.py` that call the unmodified base `compute`; SC/selector operators are verbatim copies of the reference app's custom (non-SDK) operators |
| 5 | DICOM-SR measurements match within 0.1% | ✓ VERIFIED | Independent pydicom extraction this verification: fresh fast SR, `ref_fullres_only` SR, and `current_output` SR all contain exactly one text item: **`Airway Volume: 1 mL`** — exact match (0.0%) |
| 6 | Pipeline runs E2E on the dev corpus without operator errors | ✓ VERIFIED | Fresh re-run exit 0, no entity failures; 8 timing records + `study_timing_summary` in `/tmp/verify_e2e.log` with real StudyInstanceUID `2.25.250447264451981637974165848369641322606`. (≥5-study final gate deferred — see Deviation Judgment #2) |
| 7 | All 5 core operators exist as Holoscan operator subclasses with setup() and compute() | ✓ VERIFIED | `grep class`: `PreprocessOperator(Operator)` (preprocess_operator.py:371), `SlideWindowOperator(Operator)` (slidewindow_operator.py:388), `PostResampleOperator(Operator)` (postresample_operator.py:262), `EnsembleAverageOperator(Operator)` (ensemble_average_operator.py:203), `PostprocessOperator(Operator)` (postprocess_operator.py:334). All five define `setup(self, spec)` and `compute(self, op_input, op_output, context)`. Substantive (300–700-line modules), not stubs |
| 8 | NVTX markers in all operators, visible in Nsight trace | ✓ VERIFIED | Queried the committed trace directly (`.planning/profiles/fast_dag_app_20260818_130000.sqlite`, NVTX_EVENTS): ranges **preprocess (10.24 s), inference (27.84 s), postresample (1.70 s), ensemble_average (0.010 s), postprocess (5.17 s)** + write_seg/write_sr/write_sc — all five operator boundaries present. Source: every `compute()` wraps in `nvtx_range(...)` (gpu_util) |
| 9 | Operator-level timing logs emitted per study | ✓ VERIFIED | Fresh re-run log: 8 JSON records `{operator, study, start, end (ISO-8601), start_ns, end_ns, duration_ms, ...}` + `study_timing_summary` aggregate (preprocess 10863 ms, inference 27492 ms, postresample 1703 ms, ensemble 8.7 ms, postprocess 2622 ms, writes 1089 ms, total 43778 ms) |
| 10 | app.py uses the new operator chain instead of NNUnetSegOperator | ✓ VERIFIED | `app.py` instantiates the 5-operator chain (lines ~215–240, "replaces NNUnetSegOperator"); `NNUnetSegOperator` appears only in comments (lines 26, 215). Fresh E2E ran the new chain |
| 11 | Baseline comparison shows latency numbers | ✓ VERIFIED | `.planning/benchmarks/baseline-2026-08-18.csv` exists: 3 warmups + 5 measured runs, measured totals 69536.4/61834.4/62154.6/61204.5/61623.9 → **median 61834 ms E2E** (matches Plan 05 claim exactly). Phase-0 reference baseline `.planning/baseline_results.csv`: 165.2–180.5 s (n=4, 169.7 s mean per ROADMAP). The two harnesses/bundles differ — the report correctly says not directly comparable; fast-app CSV is the Phase 2 target |
| 12 | Pixel diff tool fails on divergence (TEST-003) | ✓ VERIFIED | `scripts/pixel_diff.py` with `--min-identity`/`--max-diff-voxels`/`--exact`/`--json`, exit 0/1 semantics; re-run this verification: PASS exit 0 on the gate; tool previously verified to exit 1 on corruption (Plan 05) |
| 13 | GPU-residency test flags premature transfers (TEST-004) | ✓ VERIFIED | `scripts/gpu_residency.py` static re-run PASS with explicit allow-list and per-line attribution; `--self-test` (injects illegal `.cpu()`) documented in Plan 05 as PASS — self-test not re-run by verifier (needs a GPU E2E run) |
| 14 | Correct Phase-1 gate oracle exists and is gitignored | ✓ VERIFIED | `testdata/ref_fullres_only/` on disk with SEG/SR/SC (generated by `.planning/scripts/reference_fullres_run.py`, pinned `model_list=['3d_fullres']`); `.gitignore:278 testdata/ref_fullres_only/` confirmed via `git check-ignore`. `testdata/current_output` (full-bundle) correctly NOT used as the gate oracle — Plan 04 finding 6 documented the 2447-vs-3655 ensemble-scope mismatch |

**Score:** 24/24 plan must-have truths verified (counts below); the 2 caveats are documented deviations, not implementation gaps.

### Per-Plan Must-Have Truths (from PLAN frontmatter `must_haves:`)

**Plan 01 (5/5)**

| Truth | Status | Evidence |
|---|---|---|
| PreprocessOperator is a Holoscan subclass with setup()/compute() | ✓ | preprocess_operator.py:371, 413, 481; ran in fresh E2E |
| Reproduces reference nnUNet CPU path, pixel-exact | ✓ | Plan 01 V1: max abs diff 0.0 vs `run_case_npy` oracle (16,777,216 voxels); code contains the full reference replica (`preprocess_reference`, `_resample_to_shape` with pre-upcast `dtype_out`, Plan-05 C-contiguity fix `d881fe2` present); downstream evidence: final SEG 99.99990% vs reference |
| Emits MemoryData(GPU) equivalent for zero-copy handoff | ✓ | holoscan-cu13 4.2 has no MemoryData — documented substitution: `holoscan.core.Tensor` via DLPack, `device_type == kDLDeviceCUDA` asserted in `to_holoscan_gpu_tensor` (deviation #2, Plan 01). SlideWindow consumes via `torch.utils.dlpack.from_dlpack` (slidewindow_operator.py:533) |
| assert_on_gpu raises on CPU fallback, never swallows | ✓ | gpu_util.py `assert_on_gpu`/`assert_cuda_available`; static scan confirms guard in all 5 compute(); Plan 02 G1/G2 verified RuntimeError propagation |
| Per-config params from config, not hard-coded | ✓ | `config/__init__.py` `load_preprocess_params` keyed by config_name; Plan 01 V4 verified fullres vs lowres spacing differ with zero code change |

**Plan 02 (7/7)**

| Truth | Status | Evidence |
|---|---|---|
| Model load in setup(), not compute() | ✓ | `setup()` at slidewindow_operator.py:451 loads ModelBundle; load-counter evidence (Plan 02 V1a/h/i: count 1, second study 1.0 s); fresh E2E log "model loaded ONCE in setup" |
| Same patch size / overlap / Gaussian weighting as reference | ✓ | Uses nnUNet's own `compute_steps_for_sliding_window(tile_step_size=0.5)`, `compute_gaussian`, `pad_nd_image` — MONAI's own `sliding_window_inference` measured non-identical (kernel max diff 0.034) and replaced, documented deviation #1 Plan 02. Bisect: no-gaussian SW bit-exact vs reference internals |
| TTA in exact nnUNet order | ✓ | Plan 02 V3a/b: flip-combination lists equal to reference (`(2),(3),(4),(2,3),(2,4),(3,4),(2,3,4)`); `build_mirror_axis_combinations` in source |
| FP32 accumulators, sequential += | ✓ | Source: every accumulation `.float()` before `+=`; **this is the documented deviation from the reference's FP16** (INF-004 is explicit in REQUIREMENTS.md: "accumulates TTA results using FP32 accumulators" — the plan required it; the consequence, 3–4.5e-01 logit bound with identical argmax at the intra-operator level, is Plan 02's deviation note #4) |
| GPU eager, no CPU fallback, boundary assert | ✓ | Plan 02 G1–G4; `assert_on_gpu` entry/exit in compute (lines 536, 550) |
| autocast at outermost inference boundary | ✓ | Per-fold autocast with fold `load_state_dict` outside autocast (measured torch 2.13 corruption with a single loop-wide ctx, deviation #2). Autocast never crosses operator boundaries — INF-011 intent preserved |
| Config-driven, no hard-coded trainer class | ✓ | `InferenceParams`/`load_inference_params`; network via `get_network_from_plans`; checkpoint auto-order (`best_model.pt`); cascade in_ch=2 derived; Plan 02 V4 suite |

**Plan 03 (4/4)**

| Truth | Status | Evidence |
|---|---|---|
| PostResample: softmax→argmax-prep, resample to original shape, revert crop+transpose | ✓ | `postresample_reference` (bit-exact vs reference export path, Plan 03 1.9 V1: 0.0 over 10.6M values incl. the thread-scope softmax fix) + `revert_crop_and_transpose_gpu` (GPU permute/fill) |
| EnsembleAverage: in-memory element-wise mean, argmax AFTER averaging, no .npz | ✓ | `average_probabilities` (reference accumulation order; CuPy bit-exact `/= n`); `argmax_to_segmentation`; no disk I/O in source; 12/12 Plan-03 checks |
| Postprocess: GPU CC with pkl rules + keep-largest | ✓ | Custom deterministic CuPy two-pass min-seed `cc_label_gpu` (26-conn), MONAI-parity `keep_largest_component_gpu`, interpreted `postprocessing.pkl`; 14/14 blob trials + real seg voxel-identical to MONAI/nnUNet |
| GPU tensor out, exactly-once CPU transfer at boundary | ✓ | postprocess_operator.py:509 is the sole seg transfer (`assert_on_gpu(seg_gpu)` immediately before); confirmed by gpu_residency static re-run (count 1) |

**Plan 04 (4/4)**

| Truth | Status | Evidence |
|---|---|---|
| DAG assembled, NNUnetSegOperator replaced, no SDK core changes | ✓ | Verified above (Truths 4, 10) incl. empty `git log monai/` |
| E2E DICOMDIR → SEG/SR/SC without operator errors | ✓ | Fresh re-run this verification, exit 0 |
| NVTX markers in every operator's compute() | ✓ | Verified above (Truth 8), trace queried directly |
| Structured timing logs per operator per study | ✓ | Verified above (Truth 9), observed live |

**Plan 05 (4/4)**

| Truth | Status | Evidence |
|---|---|---|
| Pixel diff tool fails on divergence | ✓ | Verified above (Truth 12) |
| GPU-residency test flags premature transfers | ✓ | Verified above (Truth 3/13) |
| DICOM-SEG pixel-exact (bit-for-bit) to freshly regenerated 3d_fullres reference | ✓ **within documented deviation** | Independent re-run: 99.99990% / 2 voxels / IoU 0.999142 vs `testdata/ref_fullres_only` (Plan 05: 99.99986% / 3 voxels). Root cause of the residual is the FP16↔FP32 argmax boundary (1 solid voxel at a 1-voxel-thick tube → few contour voxels); reference proven run-to-run deterministic (two fresh reference runs: 0 differing voxels, Plan 05). See Deviation Judgment #1 |
| SR within 0.1% | ✓ | Exact string match this verification: `Airway Volume: 1 mL` on all three SRs |

### Key Link Verification

| From | To | Via | Status |
|---|---|---|---|
| PreprocessOperator | SlideWindowOperator | `add_flow(preprocess_op, slidewindow_op, ("preprocessed","preprocessed"))` (app.py:318); DLPack zero-copy | ✓ WIRED |
| PreprocessOperator | PostResampleOperator | `("preprocessed_meta","preprocessed_meta")` (app.py:319) | ✓ WIRED |
| SlideWindowOperator | PostResampleOperator | `("logits","logits")` (app.py:320) | ✓ WIRED |
| PostResampleOperator | EnsembleAverageOperator | `("probabilities","probabilities")` (app.py:321) | ✓ WIRED |
| EnsembleAverageOperator | PostprocessOperator | `("seg","seg")` uint8 (app.py:322) | ✓ WIRED |
| PostprocessOperator | SEG writer | `("seg","seg_image")` (app.py:333) | ✓ WIRED |
| PostprocessOperator | SR writer | `("result_text","text")` (app.py:341) | ✓ WIRED |
| PostprocessOperator | SC writer | `("dicom_sc_dir","dicom_sc_dir")` (app.py:349) | ✓ WIRED |
| SDK DICOM I/O | chain | loader→selector→volume (app.py:309–317); series list fan-out to writers (app.py:310–315, 328–330) | ✓ WIRED |

All 13 flow edges present in source; all exercised in the fresh E2E re-run (8 operator/writer ticks visible in my run's timing log, same entity set as the Nsight trace).

### Data-Flow Trace (Level 4)

No hollow artifacts. Every rendered/emitted value traces to real data: preprocessed volume ← DICOM pixels (loader); logits ← real 5-fold 3d_fullres checkpoint (weights bit-exact to checkpoint files, Plan 02 V4e1); probabilities ← logits; seg ← averaged probabilities; SEG pixels ← post-CC seg (3655 voxels, verified in fresh run); SR text ← voxel volume computed from the solid mask; SC ← same seg via `reference_label_to_contour`. No hardcoded empty props or static returns found in the operator chain.

### Behavioral Spot-Checks (run live by the verifier this session)

| Behavior | Command | Result | Status |
|---|---|---|---|
| Fresh E2E on dev study | `python my_app -i testdata/airway_input -m <models> -o /tmp/verify_fast_out` (venv, ulimit -s unlimited, GPU 0) | exit 0; SC/SEG/SR; label_counts {0:16773561, 1:3655}; contour_voxels 2330 | ✓ PASS |
| Pixel-exact gate vs 3d_fullres-only oracle | `scripts/pixel_diff.py /tmp/verify_fast_out/SEG testdata/ref_fullres_only/SEG` | 99.99990% byte identity, 2 diff voxels, IoU 0.999142, geometry MATCH, exit 0 | ✓ PASS |
| Non-oracle sanity (full bundle) | same vs `testdata/current_output/SEG` | 383 diff voxels, IoU 0.8516 — matches Plan 05's 382/0.852 (bundle-scope difference, as documented) | ✓ CONSISTENT |
| SR measurement parity | pydicom text extraction on 3 SRs | all three: `Airway Volume: 1 mL` | ✓ PASS |
| GPU residency static gate | `scripts/gpu_residency.py --static` | PASS; exactly 1 postprocess boundary `.cpu()`; allow-listed resample path (5 sites); guard present ×5 | ✓ PASS |
| NVTX in committed Nsight trace | sqlite query of `NVTX_EVENTS` in `fast_dag_app_20260818_130000.sqlite` | 5 operator ranges + 3 writer ranges with plausible durations | ✓ PASS |
| Structured timing | grep of fresh run log | 8 `timing: {json}` records + `study_timing_summary` (total 43778 ms) | ✓ PASS |
| Baseline CSV numbers | cat `.planning/benchmarks/baseline-2026-08-18.csv` | median of measured = 61834.4 ms; consistent with Plan 05 | ✓ PASS |
| Reference FP16 accumulation (deviation basis) | grep vendored nnUNet | `nnUNet/nnunetv2/inference/predict_from_raw_data.py:621,623` `dtype=torch.half` | ✓ CONFIRMED |
| Runtime gpu_residency hook mode / `--self-test` | not re-run (each requires a full GPU E2E; Plan 05 results stand) | — | ? NOT RE-RUN (documented) |
| `pixel_diff.py` exit-1-on-corruption | not re-run (destructive to a scratch copy; Plan 05 result stands) | — | ? NOT RE-RUN (documented) |

### Requirements Coverage

The plan frontmatters carry **no `requirements:` field** (each SUMMARY notes this explicitly); the authoritative mapping is the ROADMAP Phase-1 task table. IDs from the verification brief are all accounted for below. **Note:** `TEST-001` in the brief does not exist in REQUIREMENTS.md — the pixel-exact gate is `TEST-01`; no requirement is orphaned by this.

| Requirement | Description (abridged) | Status | Evidence |
|---|---|---|---|
| PREP-01 | Transpose to nnUNet orientation | ✓ SATISFIED (CPU per phase scope; GPU port is Phase 2 task 2.1) | `preprocess_reference` tf from plans.json; bit-exact V1 |
| PREP-02 | Normalization per plans before resample | ✓ SATISFIED (CPU per phase scope) | ZScoreNormalization replica, config-driven scheme/flags/stats |
| PREP-03 | Reference scipy/skimage resampling path | ✓ SATISFIED | `_resample_to_shape` 1:1 replica; bit-exact; explicitly deferred to GPU in Phase 2 |
| PREP-04 | Crop/pad to expected shape | ✓ SATISFIED | bbox crop replica; bbox/shapes match reference exactly (V1b/c/d) |
| PREP-05 | Emit MemoryData(GPU) zero-copy | ✓ SATISFIED (with documented API substitution) | `holoscan.core.Tensor` DLPack on kDLDeviceCUDA — 4.2 API has no MemoryData; zero-copy buffer retention measured (67.1 MB alloc delta) |
| INF-001 | Eager GPU inference, no CPU fallback | ✓ SATISFIED | torch.no_grad eager; assert guards; G1/G2 checks |
| INF-002 | MONAI sliding_window_inference, same patch/overlap/Gaussian | ⚠ SATISFIED IN SUBSTANCE, LITERAL DEVIATION | nnUNet's own steps/gaussian used because MONAI 1.3.0's kernel/steps measurably differ (max kernel diff 0.034); same MONAI-style loop, reference-exact weighting; no-gaussian path bit-exact vs reference. Deviation #1 Plan 02, justified by the must-have's controlling intent ("same … as the reference nnUNet predictor") |
| INF-003 | TTA in reference order | ✓ SATISFIED | flip lists equal to reference, V3a/b |
| INF-004 | FP32 TTA accumulators, sequential += | ✓ SATISFIED (and is itself the documented numeric deviation vs the FP16 reference) | source-verified; consequence bounded (≤4.54e-01 logits, identical argmax in-operator; 2–3 final voxels E2E) |
| INF-005 | device assert at every boundary, never swallow | ✓ SATISFIED | guards in all 5 computes (static scan); RuntimeError propagation verified (G2) |
| INF-006 | All 4 configs config-driven | ✓ SATISFIED for the operator | fullres exercised E2E; lowres params verified; cascade in_ch=2 derived (V4b). Full 4-config E2E is Phase 2 scope (PIPE-03/TEST-005) — Phase-1 acceptance is single-config by design |
| INF-007 | Custom trainer via checkpoint path | ✓ SATISFIED | checkpoint auto-order + explicit-name resolution (FileNotFoundError lists available); no trainer-class hardcoding |
| INF-008 | Load in setup()/on_insert() | ✓ SATISFIED | load count 1 at graph-build; second study 1.0 s |
| INF-009 | In-memory ensemble averaging, no .npz | ✓ SATISFIED | no disk I/O in operator; reference accumulation order bit-exact |
| INF-010 | argmax after averaging | ✓ SATISFIED | `argmax_to_segmentation` post-average; 100% vs reference conversion |
| INF-011 | autocast not split across operators | ✓ SATISFIED | per-fold autocast inside SlideWindow only (deviation #2 Plan 02, measured corruption otherwise) |
| POST-01 | GPU CC with pkl size rules | ✓ SATISFIED | `cc_label_gpu` + interpreted pkl rules; 14/14 parity trials |
| POST-02 | Revert crop/transpose to DICOM orientation | ✓ SATISFIED | GPU `permute(0,*dims)` + crop fill; bit-exact revert |
| POST-03 | Exactly-once CPU transfer at boundary | ✓ SATISFIED | single `.cpu()` at postprocess_operator.py:509 (static re-verified) |
| PIPE-01 | DAG without SDK core changes | ✓ SATISFIED | empty `git log monai/` for the phase; writers subclassed in app.py only |
| PIPE-02 | Replace NNUnetSegOperator in app.py | ✓ SATISFIED | chain instantiated; name appears only in comments |
| PIPE-05 | E2E DICOMDIR → DICOM-SEG without operator errors | ✓ SATISFIED (dev corpus) | fresh re-run exit 0 |
| INFR-005 | NVTX markers visible in Nsight | ✓ SATISFIED | trace queried directly: 5 named ranges |
| INFR-006 | Structured operator timing logs | ✓ SATISFIED | live-verified in fresh run |
| TEST-01 | Pixel-identical DICOM-SEG on ≥5 CT studies | ⚠ SATISFIED AT DEV-LEVEL, FINAL GATE DEFERRED | Dev corpus gate passes (99.99990%, documented FP16/FP32 residual). ≥5-study re-run pending corpus supply per the TEST-01 deviation note in REQUIREMENTS.md (2026-08-17) — an approved scope deferral, tracked in human_verification |
| TEST-002 | SR within 0.1% | ✓ SATISFIED | exact string match, independent re-check |
| TEST-003 | Pixel diff tool, fails on divergence | ✓ SATISFIED | pixel_diff.py; exit-code semantics; gate re-run exit 0 |
| TEST-004 | GPU-residency test flags premature transfers | ✓ SATISFIED | gpu_residency.py static re-run PASS; runtime/self-test per Plan 05 (not re-run by verifier) |
| (brief's "TEST-001") | — | n/a | No such ID in REQUIREMENTS.md; assumed alias for TEST-01, covered above |

Out of scope for Phase 1 (not gaps): TEST-005 (4-config suite, Phase 2), PIPE-03/04 (Phase 2), INFR-01/02/03/004 (Phase 0/2). TEST-006/007 already closed in Phase 0.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `my_app/operators/dicom_series_selector_operator.py` | 200 | `# TODO type is not json now.` | ℹ️ Info | Verbatim copy of the reference app's custom operator (pre-existing comment, not introduced by this phase); all-selected rules path is exercised and working in E2E |

No stubs, no empty handlers, no hardcoded empty props, no placeholder returns in any of the 5 operators or app.py. One deliberate empty string: `Sample_Rules_Text = ""` (reference app's select-all convention, documented in Plan 04 self-check).

---

## Deviation Judgments (adversarial assessment)

### 1. "Bit-for-bit identical" vs 99.99990% — **justified, not a gap**
The requirement pair is internally contradictory: INF-004 *mandates* FP32 TTA accumulation, while the reference (vendored nnUNet 2.8.1) *accumulates FP16* (independently confirmed at `nnUNet/nnunetv2/inference/predict_from_raw_data.py:621-623`). Bit-for-bit logits vs that oracle are therefore unreachable without violating INF-004 — no implementation choice could satisfy both. The verification chain shows the deviation was handled rigorously: (a) the gate oracle was recalibrated to a 3d_fullres-only reference (the full-bundle `current_output` was proven a wrong oracle by Plan 04 finding 6 and my 383-voxel/IoU-0.852 re-check); (b) the reference was proven run-to-run deterministic (two fresh runs, 0 differing voxels), so the residual is attributable to precision, not nondeterminism; (c) the residual is 1 solid argmax voxel on a 1-voxel-thick tube (≤3 contour voxels; my re-run: 2), i.e. a sub-0.0001% effect with IoU 0.999+; (d) SR is exact and SC is bit-identical under frame-axis transpose; (e) the choice is documented in Plan 02 deviation note #4, Plan 05 "Documented tolerance", the ROADMAP Phase-1 status, and this report. **Judgment: the goal's "pixel-exact" intent (same algorithm, same pipeline, divergence only at the FP16/FP32 decision boundary that the requirements themselves mandate) is met. Literal byte-equality is impossible under the project's own requirements and is not treated as a failure.**
Minor observation: Plan 05 recorded 3 differing voxels (fast 3656 vs ref 3655); my independent re-run produced 3655 (count-identical to the reference) with 2 differing contour voxels at the same z-slice. This 1-voxel fluctuation is consistent with the documented argmax-boundary sensitivity and stays far inside tolerance, but it does mean the final-voxel output is not fully run-to-run frozen on the fast side — worth a one-line note for Phase 2's determinism work.

### 2. ≥5-study final gate — **approved deferral, not an implementation gap**
The dev corpus is a single MR study by explicit TEST-01 deviation (2026-08-17, recorded in REQUIREMENTS.md and the ROADMAP acceptance line: "final gate must re-run on ≥5 studies once supplied"). The studies do not exist in the repo; nothing in the code prevents the re-run (`pixel_diff.py` + `reference_fullres_run.py` are corpus-agnostic). Tracked as the human/final-gate item in frontmatter. The verification brief's own framing ("dev corpus: 1 MR study per TEST-01 deviation") confirms this is in-scope for this phase's acceptance.

### 3. Literal GPU-residency wording vs the resample CPU transfer — **criterion ambiguity, project-ratified**
Acceptance criterion 4 ("no `.cpu()` or `.numpy()` between inference output and DICOM-SEG writer") is literally violated by the postresample logits→CPU hop — *if read literally*. However: the Phase-1 goal itself specifies "CPU-equivalent preprocessing"; the Phase-1 risk assessment says "Don't port resampling to GPU in this phase"; and Phase 2 task 2.3 explicitly keeps resampling on the reference CPU path (deferring GPU resampling to v2). The project's own plan of record ratifies this transfer; gpu_residency.py encodes it as an explicit allow-list with per-line justification, and the *intent* of the criterion (no silent CPU fallback of the GPU computation path; exactly one boundary transfer for output) is verified to hold. Reported here for transparency rather than as a gap.

---

## Issues Encountered (by the verifier)

1. `pixel_diff`/app input path: initial E2E attempt used a wrong relative path to `testdata` (repo root, not parent of `examples`) — harness issue only; app args validated correctly (exit 2 with clear error). Second run: exit 0.
2. nsys sqlite format changed (no `nvtx_pushpop_range` table; ranges live in `NVTX_EVENTS` with eventType 59/60 + StringIds) — queried the new schema instead; trace contents fully verifiable.
3. First SR text walker crashed on a pydicom iteration quirk (specific-character-set element); replaced with a keyword-based walk — all three SR texts extracted cleanly.
4. 4 of 8 A100s were occupied by another workload at re-run time; pinned `CUDA_VISIBLE_DEVICES=0` (free) — no interference observed; timings consistent with Plan 04/05 numbers (inference 27.49 s vs 27.5 s).

## What was NOT re-run (and why)

- `gpu_residency.py` runtime hook mode and `--self-test` (each = full GPU E2E; Plan 05 results stand; the live E2E with per-boundary `assert_on_gpu` guards provides equivalent live evidence that no boundary tensor fell off-GPU).
- `pixel_diff.py --exact` failure path / corruption negative test (Plan 05 verified exit 1 on corruption).
- ≥5-study gate (no corpus available — see Deviation Judgment #2).
- SC overlay visual rendering (byte-level identity verified under frame-axis transpose by Plan 05; visual check is the human item in frontmatter).

## Next-Phase Readiness (Phase 2: GPU Acceleration)

- **Baseline target:** fast-app in-study pipeline 41.3–43.8 s (preprocess ~10.9 s cold is the largest Phase-2 target; inference 27.5 s). `.planning/benchmarks/baseline-2026-08-18.csv` is the correct comparison artifact, not the Phase-0 full-bundle CSV.
- **Carry-over note:** final-voxel output shows ≤1-voxel run-to-run fluctuation on the fast side at the FP16/FP32 boundary — if Phase 2 introduces any nondeterministic GPU op (e.g., non-deterministic CuPy kernels), the 2–3-voxel tolerance is the tripwire; keep `pixel_diff.py` (with `--json`) as a CI gate.
- **Multi-config plumbing exists:** `EnsembleAverageOperator` already accepts N stacked probability tensors; `load_inference_params` resolves lowres/cascade; cascade one-hot input channel reserved in the preprocess emit path.
- **Known hazards** (documented across plan summaries): `my_app` editable-install name collision (run from app root with app root + `my_app` on sys.path), 32 MB stack (`ulimit -s unlimited`), DLPack ownership (clone before `cp.from_dlpack`), GXF every-transmitter-needs-a-receiver (opt-in outputs).
- **Final Phase-1 gate obligation** survives into close-out: re-run `reference_fullres_run.py` + fast app + `pixel_diff.py` on ≥5 studies once supplied, and update REQUIREMENTS.md TEST-01 / ROADMAP checkboxes (all still marked Pending).

---

_Verified: 2026-08-18T21:54:02Z_
_Verifier: gsd-verifier (independent re-run, adversarial)_
