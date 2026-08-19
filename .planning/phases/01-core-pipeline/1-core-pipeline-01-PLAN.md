---
phase: "1-core-pipeline"
plan: "01"
type: "execute"
wave: 1
depends_on: []
files_modified:
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/__init__.py"
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py"
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/gpu_util.py"
  - "examples/apps/cchmc-nnunet-fast/my_app/config/__init__.py"
autonomous: true
user_setup:
  - "Run reference app to produce a fresh reference output at testdata/current_output (see .planning/scripts/REFERENCE_RUN_GUIDE.md) so the pixel-exact gate has a target."
must_haves:
  truths:
    - "PreprocessOperator exists as a Holoscan operator subclass with setup() and compute()"
    - "Preprocessing reproduces the reference nnUNet CPU path (transpose -> crop_to_nonzero -> normalize -> resample) using scipy/scikit-image, pixel-exact"
    - "PreprocessOperator emits a MemoryData with DeviceType::GPU for zero-copy handoff"
    - "A helper asserts tensor.device.type == 'cuda' at the operator boundary and raises (never swallows) on CPU fallback"
    - "Per-config parameters (spacing, transpose order, crop) are loaded from config, not hard-coded"
  artifacts:
    - "examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py"
    - "examples/apps/cchmc-nnunet-fast/my_app/operators/gpu_util.py"
  key_links:
    - "PreprocessOperator.compute() -> MemoryData(DeviceType::GPU) -> SlideWindowOperator (plan 02)"
    - "DefaultPreprocessor.run_case_npy reference path -> PreprocessOperator steps"
---

# Phase 1 Plan 01: PreprocessOperator + GPU Handoff

## Objective
- **What:** Build the first operator — `PreprocessOperator` — that takes the in-memory
  `Image` from `DICOMSeriesToVolumeOperator` and produces the nnUNet-preprocessed volume on GPU.
- **Why:** This is the head of the new operator chain and establishes the zero-copy GPU handoff
  contract (`MemoryData`/`DeviceType::GPU`) every downstream operator relies on.
- **Output:** A working `PreprocessOperator` + a shared `gpu_util.py` (device assertions, NVTX,
  timing helpers) and config loading.

## Context
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/01-core-pipeline/1-core-pipeline-VERIFICATION.md (if exists)

Reference ground truth (from analysis of `cchmc_nnunet_fifteen_ckpt_app`): the reference
`ModelnnUNetWrapper.forward` force-converts input to CPU numpy and `DefaultPreprocessor.run_case_npy`
runs, in order: `astype(float32)` → **transpose** (`transpose_forward`, nnUNet x,y,z order) →
**crop** `crop_to_nonzero` (records `bbox_used_for_cropping` + `shape_before_cropping`) →
**normalize** (per-channel mean/std from training plans; must precede resample) → **resample**
(`resampling_fn_data` to `configuration_manager.spacing`). All CPU/numpy. Phase 1 MUST keep
this CPU reference path (PREP-03) for pixel-exactness; only the final tensor is placed on GPU.

Scope: single config `3D_fullres` first, but the operator is **config-driven** (spacing,
transpose order, crop) so Phase 2 adds 2D / 3D_lowres / 3D_cascade_fullres without redesign.

## Tasks

<task type="code">
  <name>1.1 PreprocessOperator (reference CPU path)</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py</files>
  <action>Implement a Holoscan operator subclass. setup(spec): spec.input("image"), spec.output("preprocessed") (MemoryData GPU). compute(): read the in-memory Image, run the nnUNet reference preprocessing order on the CPU reference path — transpose (0,3,2,1 → nnUNet x,y,z) then crop_to_nonzero (record bbox + pre-crop shape) then normalize (per-channel mean/std from jsonpkls/plans.json) then resample via scipy/scikit-image reference path to the config spacing. Return a dict/tensor bundle (preprocessed volume + recorded bbox/shape + spacing/transpose metadata) to PostResample/Ensemble for revert.</action>
  <verify>On the airway study, the operator's cropped+resampled+normalized volume matches the reference `DefaultPreprocessor.run_case_npy` output for 3D_fullres (max abs diff 0 on the same input).</verify>
  <done>Operator runs on the airway study; intermediate matches reference CPU path exactly.</done>
</task>

<task type="code">
  <name>1.2 Emit MemoryData with DeviceType::GPU</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py</files>
  <action>After the CPU reference preprocessing, move the final float32 tensor to CUDA and emit it as a MemoryData annotated DeviceType::GPU (zero-copy handoff contract). Preserve channel count (C=1 for 3D_fullres; C=2 one-hot reserved for cascade in Phase 2).</action>
  <verify>Downstream consumer sees MemoryData.device_type() == DeviceType::GPU and a CUDA tensor.</verify>
  <done>GPU-residency of the emitted buffer is observable by the next operator.</done>
</task>

<task type="code">
  <name>1.6a Device assertions + shared gpu_util</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/gpu_util.py</files>
  <action>Create gpu_util.py with: assert_on_gpu(tensor) raising RuntimeError (never swallow) if tensor.device.type != 'cuda'; an nvtx_range context manager (torch.cuda.nvtx); and a structured timing helper (start/end/duration_ms, JSON-serializable). Use assert_on_gpu at the PreprocessOperator entry (validate incoming) and exit (validate outgoing).</action>
  <verify>A unit check: passing a CPU tensor to assert_on_gpu raises; a CUDA tensor passes. NVTX + timing helpers are importable and emit structured fields.</verify>
  <done>gpu_util.py provides device assertion, NVTX, and timing primitives used by all operators.</done>
</task>

<task type="code">
  <name>1.7a Config-driven parameters</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/config/__init__.py, examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py</files>
  <action>Load per-config parameters (spacing, transpose/permute order, crop, normalization mean/std) from the model bundle jsonpkls/plans.json + dataset.json keyed by config name; the operator takes a config_name (default 3d_fullres) and reads parameters from it — no hard-coded values.</action>
  <verify>Instantiating PreprocessOperator with config_name='3d_fullres' loads spacing/mean/std from plans.json; changing config_name changes the loaded params.</verify>
  <done>Preprocessing is configuration-driven, not hard-coded.</done>
</task>

## Verification
- `PreprocessOperator` runs end-to-end on the airway study (config `3D_fullres`) and its
  preprocessed volume is bit-exact vs the reference CPU path.
- The emitted buffer is a `MemoryData` with `DeviceType::GPU` and a CUDA tensor.
- `assert_on_gpu` raises on a CPU tensor (silent-fallback guard proven).
- Parameters come from `plans.json`, not constants.

## Success Criteria
- [ ] PreprocessOperator is a Holoscan operator subclass with setup() and compute()
- [ ] Preprocessing uses the reference scipy/scikit-image CPU path (PREP-03) and is pixel-exact vs reference
- [ ] Emits MemoryData(DeviceType::GPU) for zero-copy handoff (PREP-05)
- [ ] assert_on_gpu at entry/exit never swallows RuntimeError (INF-005)
- [ ] Config-driven (spacing/crop/normalize from plans.json) (PREP-01..04)
- [ ] transpose + crop match nnUNet training orientation (PREP-01, PREP-04)
- [ ] normalization per-channel mean/std applied before resample (PREP-02)
