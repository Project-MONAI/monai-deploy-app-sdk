---
phase: 1-core-pipeline
plan: 04
subsystem: dag-assembly
tags: [holoscan-cu13, monai-deploy, nnunetv2, torch, nvtx, nsys, timing, dicom-seg, dicom-sr, dicom-sc, gxf]

# Dependency graph
requires:
  - phase: 1-core-pipeline
    plan: 02
    provides: "SlideWindowOperator (preprocessed GPU tensor in -> fold-averaged logits out) + PreprocessOperator's preprocessed_meta"
  - phase: 1-core-pipeline
    plan: 03
    provides: "PostResample/EnsembleAverage/Postprocess operators with declared I/O + gpu_util (nvtx_range, GpuTiming, device guards)"
provides:
  - "Runnable cchmc-nnunet-fast app: DICOMDataLoader -> DICOMSeriesSelector -> DICOMSeriesToVolume -> Preprocess -> SlideWindow -> PostResample -> EnsembleAverage -> Postprocess -> SEG/SR/SC writers (NNUnetSegOperator replaced, no SDK core edits)"
  - "PostprocessOperator dicom_sc_dir side output: reference-parity SC overlay (LabelToContour + jet + alpha 0.7 + SaveImaged .dcm) via operators/sc_overlay.py"
  - "EnsembleAverageOperator uint8 seg output (argmax-after-average) for the DAG"
  - "NVTX-verified operator boundaries in an Nsight Systems trace of the assembled DAG"
  - "Structured per-operator timing records {operator, study, start, end, duration_ms} + per-study study_timing_summary aggregate incl. write stages"
  - "Study identity propagation: per-fragment study registry (StudyInstanceUID from the DICOM Image metadata)"
affects: [validation-tools, pixel-exact-gate, multi-config, latency-benchmark]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "GXF graph rule: every declared operator output needs a downstream receiver — a declared-but-unconnected output makes the scheduler reject the entity ('No receiver connected to transmitter ... The entity will never tick'); make such outputs opt-in (emit flag) rather than wiring dummy consumers"
    - "Study identity without context user data: holoscan 4.2 Python exposes no compute-context user data, so the first Image-seeing operator registers the DICOM StudyInstanceUID per-fragment (gpu_util.set_study_id); downstream tensor-only operators and the writer subclasses read it back (get_study_id) — valid for this app's single-study-per-run scope"
    - "Writer observability without SDK core edits: subclass the SDK SEG/SR writers (and the custom SC writer) and wrap compute() with nvtx_range + GpuTiming; the unmodified base compute is invoked explicitly"
    - "End-to-end run command: from the app root, `python my_app -i <dicom> -m <bundle> -o <out>` with the uv venv python and ulimit -s unlimited (my_app name-collision hazard + 32 MB stack requirement)"

key-files:
  created:
    - examples/apps/cchmc-nnunet-fast/my_app/operators/sc_overlay.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/dicom_series_selector_operator.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/dicom_sc_writer_operator.py
    - .planning/profiles/fast_dag_app_20260818_130000.nsys-rep
    - .planning/profiles/fast_dag_app_20260818_130000.sqlite
  modified:
    - examples/apps/cchmc-nnunet-fast/my_app/app.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/__init__.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/postprocess_operator.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/gpu_util.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py

key-decisions:
  - "EnsembleAverageOperator gains a uint8 'seg' output (argmax-after-average, the exact plan-03 argmax_to_segmentation on the averaged probabilities) because the plan's DAG wires ensemble -> postprocess on a seg and no other operator in the chain computes the argmax; the verified 'averaged_probabilities' output is preserved (now opt-in)"
  - "averaged_probabilities is opt-in (emit_averaged_probabilities, default True): a declared output with no downstream receiver makes the GXF scheduler reject the entity, so the DAG constructs it with the flag False"
  - "SC overlay is reproduced (not imported) in operators/sc_overlay.py behind PostprocessOperator's new dicom_sc_dir output: per-label MONAI LabelToContour on the 3-D mask, jet(256) colormap, alpha=0.7 blend, SaveImaged(output_ext='.dcm', output_dtype=int16) into output_folder/'temp' — the exact reference LabelToContourd/OverlayImageLabeld/SaveImaged math; the copied reference DICOMSCWriterOperator consumes and removes the temp dir"
  - "Study uid propagates via a per-fragment registry (not via new DAG I/O): PreprocessOperator reads StudyInstanceUID from the Image metadata and registers it; slidewindow/postresample/ensemble read it back. Adding optional meta inputs to the tensor-only operators was rejected as a larger contract change for observability only"
  - "The slidewindow operator's NVTX range keeps the plan-02 name 'inference' (reference-app parity, Phase 0 demo trace used preprocess/inference/postprocess) rather than the plan text's 'slidewindow'"
  - "Timed writer subclasses live in app.py (not SDK core): the SDK SEG writer does not define self._logger, so the wrapper falls back to a module logger"

patterns-established:
  - "CCHMCNNUnetFastApp.compose(): 15 add_flow edges — loader->selector->volume->{preprocess, postprocess(image)}; preprocess->{slidewindow(preprocessed), postresample(preprocessed_meta)}; slidewindow->postresample(logits); postresample->ensemble(probabilities); ensemble->postprocess(seg); postprocess->{seg_writer(seg_image), sr_writer(text), sc_writer(dicom_sc_dir)}; selector->{all 3 writers(study_selected_series_list)}"
  - "Structured timing record: GpuTiming.stop() -> {operator, label, study, start(ISO-8601), end(ISO-8601), start_ns, end_ns, duration_ms}; every operator logs 'timing: <json>' and records into StudyTimingCollector; app.run() logs 'study_timing_summary: {study, operators{...}, total_ms, n_records}' after the run"

requirements-completed: []  # PLAN frontmatter has no `requirements:` field; success criteria cover PIPE-01/02/05 + INFR-005/006

# Metrics
duration: ~40min
completed: 2026-08-18
---

# Phase 1 Plan 04: DAG Assembly + Observability Summary

**The five GPU operators are now a real Holoscan DAG in app.py (NNUnetSegOperator gone, SDK core untouched): one airway study runs DICOMDIR in -> SEG/SR/SC out with exit 0, the Nsight trace shows all five named operator boundaries, and every operator + writer emits a {operator, study, start, end, duration_ms} record with a per-study latency aggregate — the assembled pipeline reproduces the Plan 03 verified chain exactly (3655 post-CC voxels, 'Airway Volume: 1 mL')**

## Performance

- **Duration:** ~40 min
- **Started:** 2026-08-18T16:34:35Z
- **Completed:** 2026-08-18T17:14:00Z
- **Tasks:** 3 (1.13, 1.14, 1.15)
- **Files modified:** 8 (5 created, 11 touched across commits incl. copies and profiles)

## Accomplishments

- **Task 1.13 — DAG assembly (5b8fe30):** `app.py` composes `DICOMDataLoaderOperator -> DICOMSeriesSelectorOperator -> DICOMSeriesToVolumeOperator -> PreprocessOperator -> SlideWindowOperator -> PostResampleOperator -> EnsembleAverageOperator -> PostprocessOperator -> {DICOMSegmentationWriterOperator, DICOMTextSRWriterOperator, DICOMSCWriterOperator}` (15 flows). No SDK core edits — the SEG/SR/SC writers are subclassed in app code only for observability. The reference's custom `DICOMSeriesSelectorOperator` + `DICOMSCWriterOperator` were copied into `my_app/operators/`. `PostprocessOperator` gained the `dicom_sc_dir` side output via new `sc_overlay.py` (reference-parity LabelToContour + jet + alpha + `SaveImaged(.dcm)` int16 into `output_folder/temp`). `EnsembleAverageOperator` gained the uint8 `seg` output (argmax-after-average) the DAG needs. **E2E on the airway study: exit 0, SC/SEG/SR under the output dir, zero operator errors; 3655 post-CC voxels and SR text 'Airway Volume: 1 mL' — identical to the Plan 03 in-harness verified chain.**
- **Task 1.14 — NVTX verification (e6f7dd6):** all five operators already wrapped `compute()` in `nvtx_range` (Plans 01–03); verified in an `nsys profile --trace=cuda,nvtx` run of the composed app: `nvtx_pushpop_sum` shows **preprocess (10.2 s), inference (27.8 s), postresample (1.7 s), ensemble_average (10 ms), postprocess (5.2 s)** — five named operator boundaries (plus write_seg/write_sr/write_sc from the timed writer subclasses). Trace committed to `.planning/profiles/fast_dag_app_20260818_130000.*`.
- **Task 1.15 — structured timing (123226f):** `GpuTiming` records now carry `{operator, study, start, end (ISO-8601), start_ns, end_ns, duration_ms}`; each operator logs `timing: <json>` per study; study identity = the DICOM StudyInstanceUID registered per-fragment by PreprocessOperator; `StudyTimingCollector` + `app.run()` emit `study_timing_summary` with per-operator latencies (preprocess/inference/postresample/ensemble_average/postprocess + write_seg/write_sr/write_sc) and total_ms. Verified in a live run: 8 records + aggregate, all carrying the real study UID.

## Task Commits

1. **Task 1.13: Assemble Holoscan DAG, replace NNUnetSegOperator** - `5b8fe30` (feat)
2. **Task 1.14: NVTX markers in all operators** - `e6f7dd6` (test — markers existed from Plans 01–03; commit is the Nsight trace evidence)
3. **Task 1.15: Structured operator-level timing logs** - `123226f` (feat)

## Files Created/Modified

- `my_app/app.py` (rewritten) — full DAG compose(), timed writer subclasses, per-study aggregate logging.
- `my_app/operators/sc_overlay.py` (created) — `generate_contour`, `create_overlay`, `save_sc_dicom`, `write_sc_overlay` (reference-parity SC math).
- `my_app/operators/dicom_series_selector_operator.py`, `my_app/operators/dicom_sc_writer_operator.py` (created; verbatim copies of the reference app's custom operators, no app-internal imports).
- `my_app/operators/postprocess_operator.py` — `output_folder` param, `dicom_sc_dir` output, `_write_sc_output` (affine + shape checked against the Image).
- `my_app/operators/ensemble_average_operator.py` — `seg` output (argmax-after-average); `averaged_probabilities` opt-in via `emit_averaged_probabilities`.
- `my_app/operators/gpu_util.py` — GpuTiming ISO/structured fields, `set_study_id`/`get_study_id` per-fragment registry, `StudyTimingCollector`.
- `my_app/operators/{preprocess,slidewindow,postresample,ensemble_average,postprocess}_operator.py` — structured `timing: <json>` log lines + collector records + study wiring.
- `my_app/operators/__init__.py` — exports the two copied DICOM operators, sc_overlay helpers, and the new gpu_util API.
- `.planning/profiles/fast_dag_app_20260818_130000.{nsys-rep,sqlite}` — Nsight trace of the assembled DAG (INFR-005 evidence).

## Verification Results

Full app runs from `examples/apps/cchmc-nnunet-fast` with `/tmp/monai-env/.venv/bin/python my_app -i testdata/airway_input -m <ref bundle models> -o <out>`, `ulimit -s unlimited`. GPU: A100-SXM4-40GB, real 3d_fullres 5-fold bundle (no synthetic stand-in).

| Check | Expected | Actual | Status |
|---|---|---|---|
| App imports, fragment composes | no errors | import + compose OK (compose loads model once in setup, 1.43 s) | PASS |
| E2E single airway study, exit 0 | SC/SEG/SR under output dir | exit 0; SEG (256 frames), SR, SC produced | PASS |
| No operator errors | clean run | no entity failures, no compute exceptions | PASS |
| New chain, not NNUnetSegOperator | no reference-operator import | app.py references NNUnetSegOperator only in comments | PASS |
| Final seg vs Plan 03 verified chain | 3655 post-CC voxels | 3655 (label_counts {0: 16773561, 1: 3655}) | PASS |
| SR text | 'Airway Volume: 1 mL' | exact match (fresh reference run: identical string) | PASS |
| SC structure | multi-frame true-color, reference-parity | 256 frames 256×256 RGB, OT modality, same pixel-data layout/length as fresh reference SC | PASS |
| Nsight trace NVTX (INFR-005) | 5 named operator ranges | preprocess / inference / postresample / ensemble_average / postprocess all present (+ write_seg/write_sr/write_sc) | PASS |
| Timing records (INFR-006) | per-operator {operator, study, start, end, duration_ms} | 8 JSON records (5 operators + 3 writers) with real StudyInstanceUID | PASS |
| Per-study aggregate | preprocess/inference/postprocess/write latencies | study_timing_summary: preprocess 10598 ms, inference 27493 ms, postresample 1680 ms, ensemble_average 9 ms, postprocess 526 ms, write_seg 834 ms, write_sr 5 ms, write_sc 167 ms, total 41312 ms | PASS |
| SEG vs fresh reference run (full-bundle) | see note | 99.88432% byte-identical; voxels 2447 (ref) vs 3655 (ours) — ensemble-scope difference, see Deviation note 6 | DOCUMENTED |

## Decisions Made

- **Ensemble emits the seg** (see key-decisions): the plan's key link `PostResample -> EnsembleAverage -> Postprocess` requires a uint8 seg at the postprocess input; the plan-03 contract left argmax to the caller, but no other DAG node performs it, so the operator now also emits `seg` (the exact verified `argmax_to_segmentation` on the averaged probabilities).
- **Opt-in unused outputs** instead of dummy consumers: keeps the verified `averaged_probabilities` contract for direct/harness use while satisfying GXF's every-transmitter-needs-a-receiver rule.
- **Study registry over DAG I/O changes**: observability-only concern; adding optional `preprocessed_meta` inputs to tensor-only operators would have widened their verified contracts for a log field.
- **'inference' stays the slidewindow NVTX name** (plan-02 reference-parity decision; the plan text's 'slidewindow' naming not applied — see note 5).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] EnsembleAverageOperator had no seg output for the DAG**
- **Found during:** Task 1.13 (DAG wiring — the plan's key link `EnsembleAverage -> Postprocess` has no source of a uint8 seg)
- **Issue:** plan 03's operator contract emitted only `averaged_probabilities` and left argmax "to the caller"; the assembled DAG has no caller node — postprocess requires a seg input.
- **Fix:** added the `seg` output = `argmax_to_segmentation(averaged)` (the exact plan-03 reference-parity function; argmax-after-average), emitted as a zero-copy CUDA uint8 tensor. `averaged_probabilities` preserved (opt-in).
- **Files modified:** examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py
- **Verification:** E2E final seg 3655 post-CC voxels — identical to the Plan 03 verified chain; SR text exact.
- **Committed in:** 5b8fe30

**2. [Rule 1 - Bug] A declared output with no downstream receiver makes the GXF scheduler reject the entity**
- **Found during:** Task 1.13 (first E2E: "No receiver connected to transmitter of DownstreamReceptiveSchedulingTerm ... of entity 'ensemble_average_op'. The entity will never tick." — graph stopped after postresample)
- **Issue:** holoscan 4.2/GXF requires every transmitter to have a receiver; `averaged_probabilities` is not consumed by the DAG, so the entity never ticked and the run produced no outputs.
- **Fix:** `emit_averaged_probabilities` constructor flag (default True, preserving plan-03 behavior); the output is declared/emitted only when set; app.py passes False.
- **Files modified:** examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py, examples/apps/cchmc-nnunet-fast/my_app/app.py
- **Verification:** E2E runs to completion, exit 0, SC/SEG/SR produced.
- **Committed in:** 5b8fe30

**3. [Rule 1 - Bug] SDK DICOMSegmentationWriterOperator defines no `_logger`**
- **Found during:** Task 1.13 (E2E: `AttributeError: 'TimedDICOMSegmentationWriterOperator' object has no attribute '_logger'` in the timed wrapper)
- **Issue:** the SR/SC writers set `self._logger`, but the SDK SEG writer does not, so the shared timing wrapper crashed on its log line after the write had succeeded.
- **Fix:** wrapper uses `getattr(operator, "_logger", None) or logging.getLogger(f"timed_{type(operator).__name__}")`.
- **Files modified:** examples/apps/cchmc-nnunet-fast/my_app/app.py
- **Verification:** E2E exit 0; all three writer timing records logged.
- **Committed in:** 5b8fe30

**4. [Rule 1 - Bug] `import datetime` (module) broke the new ISO timestamps**
- **Found during:** Task 1.15 (first E2E: `AttributeError: module 'datetime' has no attribute 'fromtimestamp'` in GpuTiming.stop)
- **Issue:** gpu_util does a module-level `import datetime`; the new `_iso` helper called `datetime.fromtimestamp` on the module instead of the class.
- **Fix:** `from datetime import datetime, timezone`.
- **Files modified:** examples/apps/cchmc-nnunet-fast/my_app/operators/gpu_util.py
- **Verification:** E2E exit 0; all records carry ISO-8601 `start`/`end`.
- **Committed in:** 123226f

### Plan-interpretation notes (not code deviations)

**5. Slidewindow NVTX range named 'inference', not 'slidewindow'.**
The plan's verify step lists the five range names as "(preprocess, slidewindow, postresample, ensemble_average, postprocess)". The operator's range was named `inference` in Plan 02 (reference-app parity: the reference/Phase 0 demo trace used preprocess/inference/postprocess, and the NVTX marker requirement INFR-005 is about *boundaries*, not specific strings). Kept as `inference`; both names are documented here and in the trace.

**6. Gate-calibration finding for Plan 05 (ensemble scope): `testdata/current_output` is NOT a 3d_fullres-only reference.**
A fresh reference-app run (this plan, for comparison) logs: `Derived model_list from plans.json: ['3d_lowres', '3d_fullres', '3d_cascade_fullres']` and `All files to average: ['tmp/3d_fullres/Img_in_context.npz', 'tmp/3d_cascade_fullres/Img_in_context.npz']` — i.e. the reference app runs a **2-config cascade ensemble** (3d_lowres is the cascade auxiliary), producing **2447** post-CC voxels (matching historical `testdata/current_output`: 99.88432% byte-identical to our SEG, all differences in the airway band). The Phase 1 fast app is single-config **3d_fullres** (explicit plan scope: "Config default 3d_fullres"; Phase 2 adds cascade), producing **3655** voxels — which is exactly what Plan 03's in-harness 3d_fullres-only reference chain produced (B2b 100% identical, B5 = 3655). **Consequence:** Plan 05's pixel-exact gate must compare against a 3d_fullres-only reference run (as Plan 03 did in-harness), or the reference app must be run with `model_list=['3d_fullres']`; comparing against the raw full-bundle `testdata/current_output` will show ~1200 airway-voxel differences by construction, not by pipeline error.

**7. `files_modified` list extended.** The plan lists only app.py + operators/__init__.py, but the DAG requires (per the plan's own Context): the two copied reference DICOM operators, the SC side-output (new sc_overlay.py + PostprocessOperator change), the ensemble seg output (EnsembleAverageOperator change), and the gpu_util timing/study infrastructure. All additions are app-scoped; no SDK core files were touched.

---

**Total deviations:** 4 auto-fixed (3 bug, 1 missing-critical) + 3 interpretation notes
**Impact on plan:** all must-haves and success criteria met; deviation 1 was required to make the plan's DAG key link realizable; deviation 6 is carried to Plan 05 as the controlling gate-calibration input.

## Issues Encountered

- **GXF no-receiver rejection (Deviation 2)** is a non-obvious holoscan 4.2 graph rule — the error names the *destination* entity while the cause is an *unconsumed output*; future multi-output operators should audit every declared output for a flow edge at compose time.
- **Reference app model list:** the reference derives its model list from `plans.json` ∩ present config dirs (3 configs), not from an app pin — any "same-environment" reference regeneration for Plan 05 must pin `model_list=['3d_fullres']` to match Phase 1 scope.
- **my_app name-collision + 32 MB stack hazards** (known from Phase 0) avoided as before: runs from the app root, `ulimit -s unlimited`, venv python.

## User Setup Required

None — bundle at `examples/apps/cchmc_nnunet_fifteen_ckpt_app/models` (3d_fullres) as before; all verification ran on the real bundle on the A100.

## Next Phase Readiness

- **Plan 05 (validation/pixel-exact gate):** app runs E2E via `python my_app -i/-m/-o`; outputs land in `<out>/{SEG,SR,SC}`; SC temp dir is consumed+removed by the SC writer (no residue). Gate must use a 3d_fullres-only reference (Deviation note 6). Timing instrumentation is in place for latency comparison vs the 169.7 ± 7.3 s/study baseline (measured pipeline: preprocess 10.6 s + inference 27.5 s + postresample 1.7 s + ensemble 0.009 s + postprocess 0.53 s + writes ~1.0 s ≈ 41.3 s aggregate for the warm single-study run, excluding app start/model load).
- **Phase 2 (multi-config):** `EnsembleAverageOperator` already accepts stacked `(N, C, Z, Y, X)` probabilities; the DAG change is a second `PostResample` branch per config feeding the ensemble (its `averaged_probabilities`/`seg` outputs are config-agnostic).
- **Known residual (unchanged from Plan 03):** E2E seg variability vs a same-scope reference is the upstream fp16-vs-fp32 argmax boundary + reference run-to-run nondeterminism; post-inference math is bit-exact per stage.

---
*Phase: 1-core-pipeline, Plan: 04*
*Completed: 2026-08-18*

## Self-Check: PASSED

- All 5 key created files + SUMMARY present on disk (verified with [ -f ]).
- All 3 task commits on nnunet-fast: 5b8fe30, e6f7dd6, 123226f (verified in git log).
- No stubs: all emitted values are data-driven (DICOM-derived study UID, measured
  timings, real 5-fold logits, affine-derived volumes, data-driven SC overlay);
  Sample_Rules_Text = "" is the reference app's select-all convention, not a placeholder.
