---
phase: 1-core-pipeline
plan: 01
subsystem: pipeline
tags: [holoscan-cu13, nnunetv2, monai-deploy, torch, scipy, scikit-image, dlpack, nvtx, rmm]

# Dependency graph
requires:
  - phase: 0-foundation
    provides: "venv at /tmp/monai-env/.venv (holoscan-cu13 4.2.0, monai 1.3.0, vendored nnunetv2 2.8.1), airway dev corpus (testdata/airway_input), regenerated reference output (testdata/current_output), app scaffold (examples/apps/cchmc-nnunet-fast)"
provides:
  - "PreprocessOperator: in-memory Image -> bit-exact nnUNet-preprocessed float32 volume (CPU reference path) -> zero-copy GPU tensor + revert metadata"
  - "gpu_util: assert_on_gpu / assert_cuda_available (never-swallow CPU-fallback guard), nvtx_range, GpuTiming structured records"
  - "config.load_preprocess_params: per-config params from jsonpkls/plans.json + dataset.json (config-driven, no hard-coded values)"
  - "GPU handoff contract for downstream operators: holoscan.core.Tensor via DLPack with device_type kDLDeviceCUDA (cu13 4.2 equivalent of MemoryData(DeviceType::GPU))"
affects: [SlideWindowOperator, inference-core, PostResample, EnsembleAverage, DAG-assembly, validation-tools]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Reference-replica preprocessing: numpy/scipy/scikit-image primitives copied 1:1 from nnUNet DefaultPreprocessor.run_case_npy for bit-exactness (PREP-03)"
    - "GPU handoff via DLPack: torch CUDA tensor -> holoscan.core.Tensor.from_dlpack (zero-copy), device asserted before emit"
    - "Config-driven operators: all per-config values loaded from bundle plans.json keyed by config_name"
    - "Device guards at every operator boundary (entry: assert_cuda_available; exit: assert_on_gpu)"

key-files:
  created:
    - examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/gpu_util.py
  modified:
    - examples/apps/cchmc-nnunet-fast/my_app/config/__init__.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/__init__.py

key-decisions:
  - "GPU handoff primitive is holoscan.core.Tensor (DLPack) — MemoryData does not exist in the holoscan-cu13 4.2 Python API; Tensor.device.device_type==kDLDeviceCUDA is the 4.2 equivalent of MemoryData(DeviceType::GPU)"
  - "Normalization is config-driven per plans.json (ZScoreNormalization, use_mask_for_norm=false): for this bundle the reference computes z-score from the cropped image, not from plans.json intensity stats — plans.json drives the scheme/flags/stats, ground truth stays run_case_npy"
  - "acvl_utils get_bbox_from_mask and nnUNet resampling are replicated inline (numpy/scipy/skimage) so the app has no runtime dependency on nnunetv2 for preprocessing; nnunetv2 is used only as the verification oracle"
  - "ANISO_THRESHOLD=3 kept as a documented module constant (framework constant, absent from plans.json)"
  - "Commit order followed dependencies (gpu_util -> config -> operator) rather than plan listing order, to keep every commit consistent"

patterns-established:
  - "PreprocessOperator.compute(): nvtx_range('preprocess') + GpuTiming -> assert_cuda_available (entry) -> reference CPU path -> .cuda() -> assert_on_gpu (exit) -> emit holoscan Tensor + meta dict"
  - "preprocess_reference(data, spacing_xyz, params) returns (volume, properties) with bbox_used_for_cropping, shape_before_cropping, shape_after_cropping_and_before_resampling, new_shape, original_spacing, target_spacing, transpose_forward for PostResample/Ensemble revert"

requirements-completed: []  # PLAN frontmatter has no `requirements:` field; success criteria cover PREP-01..05, INF-005, INFR-006

# Metrics
duration: ~50min
completed: 2026-08-18
---

# Phase 1 Plan 01: PreprocessOperator + GPU Handoff Summary

**Bit-exact nnUNet reference preprocessing (transpose → crop → normalize → resample) on CPU with a zero-copy DLPack GPU handoff, config-driven from plans.json, guarded by never-swallow device assertions**

## Performance

- **Duration:** ~50 min
- **Started:** 2026-08-18T02:45:00Z (approx., after Phase-0 close-out at 02:39Z)
- **Completed:** 2026-08-18T03:34:37Z
- **Tasks:** 4 (1.1, 1.2, 1.6a, 1.7a)
- **Files modified:** 4

## Accomplishments

- `PreprocessOperator` reproduces `DefaultPreprocessor.run_case_npy` **bit-exactly** on the airway study (3D_fullres): max abs diff **0.0** vs the reference vendored-nnunetv2 oracle (shape (1, 255, 256, 255), crop bbox and pre-crop shape match exactly).
- Final float32 volume is emitted as a **zero-copy GPU tensor** (`holoscan.core.Tensor` via DLPack, `device_type == kDLDeviceCUDA`); the CUDA buffer provably survives `compute()` return (allocator delta 67,108,864 B ≥ 66,585,600 B tensor).
- `assert_on_gpu` raises `RuntimeError` on a CPU tensor and passes on a CUDA tensor (unit-checked); `assert_cuda_available` guards operator entry — silent CPU fallback is impossible (INF-005).
- All parameters (spacing, transpose order, crop, normalization scheme/flags/stats, resampling kwargs) load from `jsonpkls/plans.json`/`dataset.json` keyed by `config_name`; switching `3d_fullres` → `3d_lowres` changes the loaded spacing ([0.703125, 0.70310, 0.703125] → [0.89070, 0.89067, 0.89070]) with zero code change (PREP-01..04).
- NVTX `preprocess` range + structured JSON-serializable timing records (`label/start_ns/end_ns/duration_ms`) wired into `compute()`.

## Task Commits

Each task was committed atomically (dependency order — see Deviations):

1. **Task 1.6a: Device assertions + shared gpu_util** - `ed7ec81` (feat)
2. **Task 1.7a: Config-driven parameters** - `b7f9132` (feat)
3. **Tasks 1.1 + 1.2: PreprocessOperator (reference CPU path) + MemoryData/GPU emit** - `8d78d63` (feat) — same file, single atomic commit

**Plan metadata:** not committed (orchestrator handles SUMMARY/STATE/ROADMAP commits).

## Files Created/Modified

- `examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py` — reference-replica preprocessing (`_create_nonzero_mask`, `_get_bbox_from_mask`, `_compute_new_shape`, `_determine_do_sep_z_and_axis`, `_normalize_channel`, `_resample_to_shape`, `preprocess_reference`), `to_holoscan_gpu_tensor` (DLPack wrap + CUDA device check), and `PreprocessOperator` (setup: image → preprocessed + preprocessed_meta; compute: guards, NVTX, timing, GPU move, emit).
- `examples/apps/cchmc-nnunet-fast/my_app/operators/gpu_util.py` — `assert_on_gpu`, `assert_cuda_available`, `nvtx_range`, `GpuTiming`.
- `examples/apps/cchmc-nnunet-fast/my_app/config/__init__.py` — `PreprocessParams` dataclass, `load_preprocess_params`, `find_jsonpkls_dir`.
- `examples/apps/cchmc-nnunet-fast/my_app/operators/__init__.py` — package exports.

## Verification Results

| Check | Expected | Actual | Status |
|---|---|---|---|
| V1 bit-exact vs reference `run_case_npy` (airway, 3d_fullres) | max abs diff 0 | **0.0**, shape (1,255,256,255) both sides | PASS |
| V1b bbox / V1c pre-crop shape / V1d resampled shape match | equal | [[0,256]×3] / (256,256,256) / (255,256,255) | PASS |
| V2a assert_on_gpu on CPU tensor | raises RuntimeError | raised ("tensor is on cpu … contract violated") | PASS |
| V2b assert_on_gpu on CUDA tensor | passes | passed | PASS |
| V3 NVTX + timing helpers | importable, structured JSON | record `{label, start_ns, end_ns, duration_ms}` JSON-serializable | PASS |
| V4a 3d_fullres params from plans.json | spacing matches | [0.703125, 0.7031021118164062, 0.703125] | PASS |
| V4b changing config_name changes params | lowres ≠ fullres | [0.89070, 0.89067, 0.89070] vs [0.70313, 0.70310, 0.70313] | PASS |
| V4c transpose/normalize from plans | [1,0,2] / ZScore / mask=False | match | PASS |
| V5a compute() end-to-end (fake I/O) | no error | ran, emitted both outputs | PASS |
| V5b emitted buffer | GPU-backed holoscan Tensor | `Tensor`, `device_type == DLCUDA` | PASS |
| V5c shape/nbytes | (1,255,256,255), 66,585,600 B | match | PASS |
| V5e zero-copy buffer retention after compute | alloc delta ≥ tensor bytes | 67,108,864 B ≥ 66,585,600 B | PASS |
| V5d meta dict for revert | bbox/shape/spacing fields | present | PASS |

GPU: A100-SXM4-40GB, CUDA available (checked before verification — no fake verification needed).

## Decisions Made

- **GPU handoff = `holoscan.core.Tensor` (DLPack), not `MemoryData`:** the holoscan-cu13 4.2 Python API has no `MemoryData`/`DeviceType` symbols (`holoscan.core` exposes `Tensor`/`DLDeviceType`). `Tensor.from_dlpack(cuda_tensor)` is the zero-copy GPU buffer primitive; `device.device_type == kDLDeviceCUDA` (2) is asserted before emit. The plan's contract (zero-copy GPU handoff observable by the next operator) is preserved — plan 02's SlideWindowOperator should consume this `Tensor` (e.g. back to torch via DLPack).
- **Normalization ground truth stays `run_case_npy`:** the plan text says "per-channel mean/std from plans.json", but for this bundle the reference uses `ZScoreNormalization` with `use_mask_for_norm=false` — mean/std computed from the cropped image (plans.json `foreground_intensity_properties_per_channel` is not used for z-score). The operator is config-driven (scheme/flags/stats all from plans.json) and bit-exact, which is the controlling gate. CT/No normalization schemes are also supported for other bundles.
- **No runtime nnunetv2 dependency in the app:** the reference algorithms (bbox, mask, resampling, z-score) are short and pure numpy/scipy/skimage, so they are replicated 1:1; the vendored nnunetv2 is used only as the verification oracle. This keeps the app's dependency surface small for the Phase 3 optimization work.
- **Commit order (1.6a → 1.7a → 1.1/1.2)** follows module dependencies so every commit is consistent; plan listed 1.1 first but its file imports the later tasks' modules.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Resampling output dtype: float64 instead of float32**
- **Found during:** Task 1.1 verification (V1 bit-exact gate failed with max abs diff 2.38e-07, 16.6M voxels off by ~1 ulp)
- **Issue:** `_resample_to_shape` computed `reshaped_final`'s dtype from `data` *after* the `astype(float)` (float64) upcast; the reference captures `dtype_out = data.dtype` (float32) *before* upcasting and casts the resize result back to float32.
- **Fix:** capture `dtype_out = data.dtype` before `data = data.astype(float, copy=False)` in `_resample_to_shape`.
- **Files modified:** examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py
- **Verification:** re-ran the full verification script — V1 max abs diff became 0.0 (bit-exact), all 17 checks PASS.
- **Committed in:** 8d78d63 (task 1.1/1.2 commit)

**2. [Rule 3 - Blocking] Plan references `MemoryData(DeviceType::GPU)`, absent from holoscan-cu13 4.2**
- **Found during:** Task 1.2 (GPU handoff)
- **Issue:** `from holoscan.core import MemoryData` → ImportError; the 4.2 Python API's zero-copy GPU primitive is `holoscan.core.Tensor` (DLPack-backed, `device.device_type`), see Decisions.
- **Fix:** `to_holoscan_gpu_tensor()` wraps the contiguous CUDA tensor via `Tensor.from_dlpack` and asserts `device.device_type == kDLDeviceCUDA`; documented in module docstring + SUMMARY.
- **Files modified:** examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py
- **Verification:** V5b/V5c/V5e — emitted object is a `Tensor` on DLCUDA with correct shape/nbytes and the buffer survives compute() return.
- **Committed in:** 8d78d63 (task 1.1/1.2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both were necessary to meet the plan's controlling gates (bit-exactness; observable GPU residency). No scope creep.

## Issues Encountered

- **`DICOMSeries` construction in the verification harness:** the loader passes raw pydicom Datasets to `series.add_sop_instance(ds)` — wrapping them in `DICOMSOPInstance` first broke `get_pixel_array`. Fixed in the (uncommitted) verification script by passing Datasets directly, matching `DICOMDataLoaderOperator`.
- **`my_app` package collision** (known Phase-0 hazard) avoided by running verification with the app root + `my_app` dir on `sys.path` and never importing the reference app's `my_app`; the operator uses a `try my_app.* / except flat` import shim so it works in either resolution mode.
- **Pre-existing dirty file:** `.planning/STATE.md` showed as modified before execution started (orchestrator-managed); left untouched by this plan's commits per instructions.

## User Setup Required

None - no external service configuration required. (The plan's `user_setup` item — fresh reference run at `testdata/current_output` — was already completed in Phase 0 close-out on 2026-08-17.)

## Next Phase Readiness

- Plan 02 (SlideWindowOperator / inference core) can consume the `preprocessed` output: a zero-copy `holoscan.core.Tensor` on CUDA plus the `preprocessed_meta` dict (bbox, pre-crop shape, new_shape, spacing, transpose) needed for post-resampling revert.
- Config plumbing for `3d_lowres` / `3d_cascade_fullres` is already in place (`load_preprocess_params` verified against lowres); cascade's C=2 one-hot input channel is reserved in the emit path.
- Remaining Phase-1 gate (plan 05): pixel-exact E2E vs `testdata/current_output` — preprocessing head is now proven bit-exact against the same reference path the reference app uses.

---
*Phase: 1-core-pipeline, Plan: 01*
*Completed: 2026-08-18*

## Self-Check: PASSED

- All 4 key files present (preprocess_operator.py, gpu_util.py, config/__init__.py, operators/__init__.py) + this SUMMARY.
- All task commits present on `nnunet-fast`: ed7ec81, b7f9132, 8d78d63.
- No stubs introduced: every emitted field is data-driven (params from plans.json, volume from the reference path, bbox/shape from the actual crop).
