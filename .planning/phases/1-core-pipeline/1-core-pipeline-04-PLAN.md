---
phase: "1-core-pipeline"
plan: "04"
type: "execute"
wave: 3
depends_on: ["02", "03"]
files_modified:
  - "examples/apps/cchmc-nnunet-fast/my_app/app.py"
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/__init__.py"
autonomous: true
user_setup: []
must_haves:
  truths:
    - "app.py assembles a Holoscan DAG connecting the existing DICOM I/O operators to the five new GPU operators, replacing NNUnetSegOperator (no SDK core changes)"
    - "The DAG runs end-to-end from DICOMDIR input to DICOM-SEG/SR/SC output on a single study without operator errors"
    - "Every operator's compute() has NVTX markers (torch.cuda.nvtx) at entry/exit"
    - "Structured operator-level timing logs (start, end, duration_ms) are emitted per operator per study"
  artifacts:
    - "examples/apps/cchmc-nnunet-fast/my_app/app.py"
  key_links:
    - "DICOMSeriesToVolumeOperator -> Preprocess -> SlideWindow -> PostResample -> EnsembleAverage -> Postprocess -> DICOMSegmentationWriterOperator"
    - "series_selector + Postprocess -> DICOMTextSRWriterOperator (SR) and DICOMSCWriterOperator (SC)"
    - "app.py new chain replaces NNUnetSegOperator (PIPE-02)"
---

# Phase 1 Plan 04: DAG Assembly + Observability

## Objective
- **What:** Wire the five operators into a Holoscan DAG in `app.py`, replacing the monolithic
  `NNUnetSegOperator`, and add NVTX markers + structured timing logs to every operator.
- **Why:** This makes the new pipeline runnable end-to-end and measurable, and is the integration
  point the pixel-exact gate (plan 05) exercises.
- **Output:** A runnable `cchmc-nnunet-fast` app producing SC/SEG/SR from a DICOM study, with
  operator boundaries visible in an Nsight trace and per-operator timing logs.

## Context
@.planning/PROJECT.md
@.planning/STATE.md

Reference DAG (`cchmc_nnunet_fifteen_ckpt_app/my_app/app.py`): `DICOMDataLoaderOperator →
DICOMSeriesSelectorOperator → DICOMSeriesToVolumeOperator → NNUnetSegOperator → {DICOMSegmentationWriterOperator,
DICOMTextSRWriterOperator, DICOMSCWriterOperator}`. Phase 1 replaces `NNUnetSegOperator` with the
chain `Preprocess → SlideWindow → PostResample → EnsembleAverage → Postprocess`, keeping the
unchanged SDK DICOM I/O operators. Config default `3d_fullres`.

Reuse the custom `DICOMSeriesSelectorOperator` and `DICOMSCWriterOperator` from the reference app
(copy into the new app). `output_labels=[1]`; SC overlay generation (LabelToContour + jet colormap
+ alpha blend, `SaveImaged(.dcm)`) is reproduced in the PostprocessOperator's SC side-output.

## Tasks

<task type="code">
  <name>1.13 Assemble Holoscan DAG, replace NNUnetSegOperator</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/app.py, examples/apps/cchmc-nnunet-fast/my_app/operators/__init__.py</files>
  <action>Build the Application: DICOMDataLoaderOperator -> DICOMSeriesSelectorOperator -> DICOMSeriesToVolumeOperator -> PreprocessOperator -> SlideWindowOperator -> PostResampleOperator -> EnsembleAverageOperator -> PostprocessOperator -> DICOMSegmentationWriterOperator; plus series_selector -> DICOMTextSRWriterOperator (SR text from Postprocess) and DICOMSCWriterOperator (SC dir from Postprocess). Wire all add_flow connections. No SDK core edits. Export the five operators from operators/__init__.py.</action>
  <verify>The app imports, the fragment composes without errors, and a single airway study produces SC/SEG/SR under the output dir with exit 0.</verify>
  <done>app.py uses the new operator chain, not NNUnetSegOperator (PIPE-01, PIPE-02, PIPE-05).</done>
</task>

<task type="code">
  <name>1.14 NVTX markers in all operators</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/*.py</files>
  <action>Wrap each operator's compute() body in a torch.cuda.nvtx range named after the operator (using gpu_util.nvtx_range). Verify the markers appear in an Nsight Systems trace.</action>
  <verify>An nsys trace of one study shows five named NVTX ranges (preprocess, slidewindow, postresample, ensemble_average, postprocess).</verify>
  <done>Operator boundaries visible in Nsight (INFR-005).</done>
</task>

<task type="code">
  <name>1.15 Structured operator-level timing logs</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/*.py, examples/apps/cchmc-nnunet-fast/my_app/app.py</files>
  <action>Use gpu_util timing helper to log, per operator per study, a JSON-structured record {operator, study, start, end, duration_ms}. Aggregate per-study latency (preprocess/inference/postprocess/write). Emit to stdout/log file.</action>
  <verify>One study run yields a structured timing record for each of the five operators with start/end/duration_ms, and a per-study aggregate.</verify>
  <done>Per-operator timing logs emitted for each study (INFR-006).</done>
</task>

## Verification
- `cchmc-nnunet-fast` runs end-to-end on the airway study (config `3d_fullres`) producing SC/SEG/SR.
- An Nsight trace shows all five operator NVTX boundaries.
- Structured per-operator timing logs are emitted for the study.

## Success Criteria
- [ ] app.py assembles the DAG from DICOM I/O to the five new GPU operators without SDK core changes (PIPE-01)
- [ ] app.py uses the new operator chain instead of NNUnetSegOperator (PIPE-02)
- [ ] End-to-end DICOMDIR → DICOM-SEG/SR/SC on a single study without operator errors (PIPE-05)
- [ ] NVTX markers present in all operators; visible in an Nsight trace (INFR-005)
- [ ] Structured per-operator timing logs (start/end/duration_ms) emitted per study (INFR-006)
