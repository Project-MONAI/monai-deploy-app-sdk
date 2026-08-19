---
phase: 1-core-pipeline
plan: 02
subsystem: inference
tags: [holoscan-cu13, nnunetv2, monai-deploy, torch, autocast, tta, sliding-window, fp32-accumulator, dlpack, nvtx]

# Dependency graph
requires:
  - phase: 1-core-pipeline
    plan: 01
    provides: "PreprocessOperator GPU tensor (holoscan.core.Tensor via DLPack, device kDLDeviceCUDA) + preprocessed_meta; gpu_util (assert_on_gpu/assert_cuda_available, nvtx_range, GpuTiming); config.find_jsonpkls_dir + load_preprocess_params"
provides:
  - "SlideWindowOperator: preprocessed GPU tensor -> fold-averaged logits (heads, H, D, W) FP32 on CUDA, zero-copy holoscan Tensor out"
  - "ModelBundle + load_model_bundle: setup-time one-shot model load (architecture from plans.json, all fold weights from the resolved checkpoint path, CUDA-resident)"
  - "config.InferenceParams + load_inference_params: per-config inference params (patch size, mirror axes, checkpoint auto-order, folds, channels/heads, network architecture)"
  - "Reference-replica inference numerics: TTA in exact nnUNet order with FP32 sequential +=, reference steps/gaussian, per-fold sequential accumulation on GPU"
affects: [PostResample, EnsembleAverage, Postprocess, DAG-assembly, validation-tools]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Setup-time model loading: holoscan 4.2 Operator.__init__ invokes setup(spec) at graph-build time — model + all fold weights load once there; compute() is inference-only (no per-study cold start)"
    - "Reference-parity autocast boundary: one torch.autocast('cuda') per fold's sliding-window inference, fold weight loads OUTSIDE any active autocast (loading inside an active autocast measurably shifts the following forward on torch 2.13)"
    - "FP32 accumulator discipline: every accumulation (TTA sum, sliding window, visit counts, fold sum) in FP32; FP16 only inside each network forward under autocast"
    - "Never-swallow error contract: no OOM-to-CPU handler in the inference path (unlike nnUNet's); RuntimeError/OOM propagates; device asserted at entry and exit"
    - "Config- and checkpoint-driven loading: InferenceParams from plans.json + nnunet_checkpoint.pth; network via get_network_from_plans (no hard-coded trainer class)"

key-files:
  created:
    - examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py
  modified:
    - examples/apps/cchmc-nnunet-fast/my_app/config/__init__.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/__init__.py

key-decisions:
  - "MONAI sliding_window_inference cannot be used as-is: its step generator (fixed int(roi*(1-overlap)) interval) and analytic Gaussian kernel differ from the reference nnUNet 2.8.1 (verified: normalized-kernel max abs diff 0.034 on the 128^3 patch; different step sets on non-dev shapes) — the operator runs the same MONAI-style loop using nnUNet's own pure utilities (compute_steps_for_sliding_window, compute_gaussian, pad_nd_image)"
  - "Autocast boundary = per-fold (reference parity), not a single context around the whole fold loop: mid-loop load_state_dict inside an active autocast corrupts the following forward (reproduced on a minimal Conv3d+InstanceNorm net, torch 2.13); the per-fold scope still satisfies 'autocast never split across operator boundaries'"
  - "FP32 accumulation is a deliberate departure from nnUNet 2.8.1's FP16 accumulators (plan INF-004); vs the fp16 reference the logits differ by max 4.54e-01 / mean 3.7e-03 on the airway study, and segmentation (softmax+argmax) is 100.00000% voxel-identical"
  - "Inference I/O reuses the Plan 01 GPU handoff contract: incoming holoscan.core.Tensor consumed via torch.utils.dlpack.from_dlpack; logits emitted via to_holoscan_gpu_tensor (zero-copy, CUDA-asserted)"
  - "cudnn.benchmark=True enabled at model load for nnUNetPredictor parity; eager mode (no torch.compile) asserted"
  - "tile_step_size=0.5 / use_gaussian=True remain constructor parameters with the reference defaults (they are reference predictor constants, not bundle fields); everything bundle-specific is config-driven"
  - "Commit order followed dependencies (numeric core -> device guards -> config wiring) rather than plan task order, so every commit is consistent (Plan 01 precedent); tasks 1.3+1.4+1.5 landed together as one inseparable numeric core"

patterns-established:
  - "SlideWindowOperator.compute(): nvtx_range('inference') + GpuTiming -> assert_cuda_available -> receive + from_dlpack -> assert_on_gpu (entry) -> no_grad: per-fold [load_state_dict (outside autocast) -> autocast: sliding_window_predict (pad -> TTA mirror passes fp32 += -> gaussian-weighted fp32 accumulate -> /= visit counts -> inf check raises)] -> sequential fp32 fold sum -> /= n_folds -> assert_on_gpu (exit) -> emit holoscan Tensor"
  - "TTA: normal pass first, then itertools.combinations of the +2-shifted allowed mirroring axes (sizes 1..N) — exact reference order; each forward output cast .float() BEFORE accumulating"

requirements-completed: []  # PLAN frontmatter has no `requirements:` field; success criteria cover INF-001..008, INF-011

# Metrics
duration: ~105min
completed: 2026-08-18
---

# Phase 1 Plan 02: SlideWindowOperator (Inference Core) Summary

**GPU-resident nnUNet sliding-window inference: model + all folds loaded once in setup (1.2 s), TTA in exact nnUNet order with FP32 sequential accumulation, reference-parity steps/gaussian/autocast boundaries — 100.00000% voxel-identical segmentation vs the reference predictor on the airway study, 27.1 s/study inference with zero cold start on subsequent studies**

## Performance

- **Duration:** ~105 min
- **Started:** 2026-08-18T04:00:00Z (approx., after Plan 01 close-out at 03:50Z)
- **Completed:** 2026-08-18T05:44:00Z
- **Tasks:** 4 (1.3, 1.4+1.5, 1.6b+1.10-device, 1.7+1.10-checkpoint)
- **Files modified:** 3

## Accomplishments

- `SlideWindowOperator` loads the network (built from the bundle `plans.json` architecture entry — no hard-coded trainer class) and **all 5 folds' weights once in `setup()`** (graph-build time; 1.2 s, CUDA-resident); `compute()` is inference-only — the second study's `compute()` cost 1.0 s for a 128³ volume with the load counter still at 1 (INF-008).
- Inference replicates the reference `nnUNetPredictor` 1:1 in its numerically-relevant choices: `pad_nd_image`, `compute_steps_for_sliding_window(tile_step_size=0.5)`, `compute_gaussian(sigma_scale=1/8, value_scaling_factor=10)`, gaussian-weighted accumulation with `logits /= visit_counts`, TTA = normal + all mirror permutations of the checkpoint's `allowed_mirroring_axes` in the exact reference combination order, per-fold sequential accumulation then divide (INF-002/003/004).
- **Headline numeric gate:** on the airway study (255×256×255, 27 patches × 8 TTA passes × 5 folds), our fold-averaged logits vs the reference `predict_logits_from_preprocessed_data`: **max abs diff 4.539e-01, mean 3.715e-03** (the reference accumulates in FP16; plan INF-004 pins FP32), and **segmentation after softmax/argmax is 100.00000% voxel-identical (16,646,400/16,646,400)**.
- Device invariant enforced: forced-CPU input raises `RuntimeError` at entry (`assert_on_gpu`); a simulated forward `RuntimeError` propagates out of `compute()` (nothing is swallowed, no nnUNet-style OOM-to-CPU re-run exists in the path); exit logits asserted CUDA FP32 (INF-001/005).
- Config-driven (INF-006/007): `load_inference_params` resolves patch size (128³), mirror axes ((0,1,2) from checkpoint metadata), checkpoint auto-order (`best_model.pt` selected — no `final_model.pt` in this bundle), folds (0–4), channels/heads, and the network architecture per `config_name`; `3d_cascade_fullres` correctly derives 2 input channels; explicit `checkpoint_name='final_model.pt'` raises `FileNotFoundError` listing available files; built network weights are bit-exact vs the checkpoint file (max diff 0.0).
- Inference timing: 27.1 s/study for the full 5-fold TTA pass on the airway volume (reference: 26.8 s in-process; the 138 s baseline figure includes the reference app's per-study cold start + CPU round-trip, which this operator eliminates).

## Task Commits

Task 1.3+1.4+1.5 are one inseparable numeric core (setup-load + TTA order + FP32 accumulation must be verified together against the reference), so they landed as one atomic commit — dependency-ordered, Plan 01 precedent:

1. **Tasks 1.3 + 1.4 + 1.5: SlideWindowOperator core (setup-time model load, reference steps/gaussian, TTA in nnUNet order, FP32 accumulation)** - `46884b2` (feat)
2. **Task 1.6b + 1.10 (device part): device assertions + never-swallow error handling** - `22df6a2` (feat)
3. **Task 1.7 + 1.10 (checkpoint part): config- and checkpoint-driven via InferenceParams** - `b8a5266` (feat)

**Plan metadata:** committed separately by the orchestrator step.

## Files Created/Modified

- `examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py` (created) — `ModelBundle`, `build_mirror_axis_combinations`, `build_network_from_params`, `load_model_bundle`, `mirror_and_predict`, `sliding_window_predict`, `predict_logits`, `SlideWindowOperator`; re-exports `resolve_checkpoint_name`/`detect_available_folds` from config.
- `examples/apps/cchmc-nnunet-fast/my_app/config/__init__.py` — `InferenceParams` dataclass, `load_inference_params`, `resolve_checkpoint_name`, `detect_available_folds`, `_resolve_config_dir_and_jsonpkls`, `NNUNET_CHECKPOINT_FILENAME`, `DEFAULT_CHECKPOINT_ORDER`.
- `examples/apps/cchmc-nnunet-fast/my_app/operators/__init__.py` — exports `SlideWindowOperator`.

## Verification Results

Verification harnesses ran from the app root with the venv python (`/tmp/monai-env/.venv/bin/python`, `ulimit -s unlimited`), app root + `my_app` dir on `sys.path` (avoids the `my_app` editable-install collision). Reference oracle = the reference app's own `nnunet_bundle.py` loaded by file path (no `my_app` import).

| Check | Expected | Actual | Status |
|---|---|---|---|
| V1a model loaded exactly once in setup | count 1 | 1 (1.2 s, at graph-build) | PASS |
| V1b network CUDA + eval | cuda:0, training False | yes | PASS |
| V1c eager mode | not an OptimizedModule | confirmed | PASS |
| V1d config-driven values | patch (128,128,128), tta (0,1,2), 5 folds | match, trainer=nnUNetTrainer | PASS |
| V1e incoming tensor | CUDA | cuda:0 (1,255,256,255) | PASS |
| V1f logits shape | (2,255,256,255) | (2,255,256,255) | PASS |
| V1g logits device/dtype | CUDA FP32 | cuda:0 float32 | PASS |
| V1h/i no load in compute / 2nd study | count stays 1 | 1; second study 1.0 s | PASS |
| V1j second-study shape (128³ vol) | (2,128,128,128) | match | PASS |
| V2a logits vs reference | ≪ reference scale | max 4.539e-01, mean 3.715e-03 (fp16-ref vs fp32-ours) | PASS (see Deviations #4) |
| V2b segmentation after softmax/argmax | identical | **100.00000%** (16,646,400/16,646,400) | PASS |
| V3a/b TTA order vs reference expression | equal lists | equal, n=7 order [(2,),(3,),(4,),(2,3),(2,4),(3,4),(2,3,4)] | PASS |
| G0 construction load | once | count 1 | PASS |
| G1 forced-CPU input | RuntimeError at entry | "assert_on_gpu: tensor is on cpu … contract violated" | PASS |
| G2 forward RuntimeError | propagates (not swallowed) | "simulated OOM/forward failure" escaped compute() | PASS |
| G3 compute() glue inert | compute == direct predict_logits | max diff 0.0 | PASS |
| G4 emitted tensor | CUDA FP32 (2,128,128,128) | match | PASS |
| V4a1–a8 fullres InferenceParams | patch/ckpt/folds/axes/channels/arch/trainer/paths | all match (ckpt auto-order → best_model.pt) | PASS |
| V4b config_name drives values | cascade in_ch=2 vs fullres 1 | match | PASS |
| V4c1/c2 explicit checkpoint | missing → FileNotFoundError; explicit ok | both | PASS |
| V4d config-dir-as-model_path | resolves | yes | PASS |
| V4e1 weights == checkpoint file | bit-exact | max diff 0.0 | PASS |
| V4e2/e3 network cuda+eval, 5 folds resident | yes | yes | PASS |
| V4f1 operator bundle config-driven | matches params | yes | PASS |
| V4f2/f3 glue inert + emitted type | 0.0 / cuda fp32 | match | PASS |
| Post-refactor numeric re-run | unchanged | max abs diff 4.539061e-01 (identical digits), seg 100% | PASS |

Bisect evidence (same process, same weights — verified bit-exact to the checkpoint file): raw forward vs reference internals **bit-exact** (0.0); SW no-gaussian **bit-exact**; SW+gaussian max 5.0e-01; SW+gaussian+TTA max 5.1e-01 — i.e. the only residual vs-reference difference is the deliberate FP16-vs-FP32 accumulation.

GPU: A100-SXM4-40GB, torch 2.13.0+cu130, real model run (no synthetic stand-in).

## Decisions Made

- **nnUNet's own pure utilities instead of MONAI `sliding_window_inference`** (deviation #1): MONAI 1.3.0's step generator and Gaussian kernel differ from the reference (measured), so calling it as-is could not satisfy "same overlap and Gaussian weighting as the reference nnUNet predictor". The operator runs the identical MONAI-style loop (extract patch → TTA predictor → gaussian-weighted accumulate → divide) with `compute_steps_for_sliding_window`, `compute_gaussian`, `pad_nd_image` from the vendored nnunetv2.
- **Per-fold autocast boundaries** (deviation #2): the plan's "autocast at the outermost inference boundary only" was first implemented as one context around the whole fold loop; this measurably corrupted fold 2+ outputs (13.2 max diff) because torch 2.13 shifts the following forward when `load_state_dict` runs inside an active autocast. The reference itself loads each fold between separate per-fold autocast scopes; we now replicate exactly that. Autocast still never crosses operator boundaries (INF-011 intent preserved).
- **FP32 vs reference FP16 accumulation:** the plan (INF-004) deliberately pins FP32; the practical consequence is a bounded logits diff (≤4.54e-01) with pixel-identical segmentation. The plan's "~1e-6 max abs diff" tolerance assumes an FP32 reference; the vendored nnUNet 2.8.1 reference accumulates in FP16 (verified in source: `torch.zeros(..., dtype=torch.half)` for both the sliding-window logits and visit counts, FP16 TTA sum under autocast), so that tolerance is unreachable against it — the controlling gate (identical segmentation) passes at 100%.
- **Setup timing semantics:** holoscan 4.2 `Operator.__init__` calls `self.setup(spec)` at construction, so "load in setup()" concretely means "at graph-build/fragment-construction time, exactly once" — verified by load counter + second-study timing.
- **cudnn.benchmark=True** set at model load (nnUNetPredictor does the same on cuda).
- **`tile_step_size`/`use_gaussian` stay constructor params** with reference defaults — they are predictor hyperparameters absent from `plans.json` in this bundle; all bundle-specific values are config-driven.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] MONAI `sliding_window_inference` cannot reproduce the reference weighting**
- **Found during:** Task 1.3 (implementation + source/measurement comparison of monai 1.3.0 vs vendored nnunetv2 2.8.1)
- **Issue:** MONAI's built-in step generator (fixed `int(roi*(1-overlap))` interval) and analytic Gaussian kernel differ from the reference: normalized-kernel max abs diff 0.034 (mean 0.0018) on the 128³ patch and different step sets on non-dev shapes (e.g. (300,250,180)). Using it as-is would violate must-have #2 ("same … Gaussian weighting as the reference nnUNet predictor") and endanger the plan-05 pixel-exact gate.
- **Fix:** the operator runs the same MONAI-style sliding-window loop with nnUNet's own pure utilities (`compute_steps_for_sliding_window`, `compute_gaussian`, `pad_nd_image`); documented in the module docstring.
- **Files modified:** examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py
- **Verification:** bisect showed raw-forward and no-gaussian SW **bit-exact** vs reference internals on identical weights; gaussian/TTA differences reduced to FP16-vs-FP32 accumulation noise (≤5.1e-01 on synthetic, ≤4.54e-01 real data); end-to-end segmentation 100% identical.
- **Committed in:** 46884b2

**2. [Rule 1 - Bug] Single autocast around the whole fold loop corrupts fold 2+ outputs**
- **Found during:** Task 1.3 verification (logits max abs diff 13.24 vs reference with 100% seg agreement failing at 99.977%)
- **Issue:** with one `torch.autocast` wrapping the entire per-fold loop (the literal "outermost boundary" reading), each mid-loop `network.load_state_dict(...)` runs inside an active autocast. On torch 2.13 this measurably shifts the *following* forward (reproduced on a minimal Conv3d+InstanceNorm net: fold-2 output differs between shared-ctx and separate-ctx runs; thread-count ruled out). The reference loads each fold's weights between separate per-fold autocast scopes, which is why the reference is stable.
- **Fix:** `predict_logits` now opens one `torch.autocast("cuda")` per fold's sliding-window inference with the fold's `load_state_dict` outside any active autocast — exact reference parity, still never split across operators.
- **Files modified:** examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py
- **Verification:** after the fix, per-fold outputs match reference internals at ≤0.51 (FP16-accumulation noise only), and the full 5-fold comparison dropped from 13.24 → 4.539e-01 max with **100.00000%** segmentation agreement.
- **Committed in:** 46884b2

**3. [Rule 3 - Blocking] holoscan 4.2 calls `setup(spec)` from `Operator.__init__`**
- **Found during:** Task 1.3 (first harness run: `AttributeError: 'SlideWindowOperator' object has no attribute '_bundle'`)
- **Issue:** `holoscan/core/__init__.py:743` — `Operator.__init__` invokes `self.setup(spec)` before the subclass constructor body finishes; setup loads the model, so constructor state must exist first.
- **Fix:** all state touched by `setup`/`_load_model` is initialized **before** `super().__init__(...)`; `_load_model` is idempotent so explicit `setup` calls in tests are harmless. Concretely, "model load in setup()" = load at graph-build/fragment-construction time, once.
- **Files modified:** examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py
- **Verification:** V1a (load count 1), V1h/i (no load in compute; second study 1.0 s), G0.
- **Committed in:** 46884b2

### Plan-interpretation notes (not code deviations)

**4. V2 tolerance "~1e-6" is unreachable against the actual reference (documented, gate met at seg level).**
The vendored nnUNet 2.8.1 reference accumulates in FP16 (source-verified); the plan's INF-004 deliberately switches us to FP32, so logits differ by ≤4.54e-01/mean 3.7e-03. The plan's own controlling criterion — "identical after argmax/softmax to the reference seg" — passes at 100.00000% (16,646,400/16,646,400 voxels). Final pixel-exactness vs the DICOM-SEG gate is plan 05's.

**5. Task→commit mapping:** 4 tasks → 3 commits (1.3+1.4+1.5 combined as the inseparable numeric core; 1.6b+1.10-device; 1.7+1.10-checkpoint), dependency-ordered per the Plan 01 precedent so every commit is self-consistent.

---

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking) + 2 interpretation notes
**Impact on plan:** all must-haves met; deviations #1–#2 were required to meet the numeric-equivalence must-haves, not scope creep.

## Issues Encountered

- **Reference wrapper pre-loads no weights:** `ModelnnUNetWrapper.__init__` builds the network with random init; per-fold weights load inside `predict_logits_from_preprocessed_data`. Early bisect runs that called `predictor.network` directly compared against a random-init network (max diff ~37) — harness bug, not an app bug. Also, the reference network only moves to CUDA lazily inside `predict_sliding_window_return_logits`.
- **Verification-harness stale-tensor bug:** a fold-weight "mismatch" flag was caused by capturing `state_dict()` *before* the `load_state_dict` call; direct comparison proved in-memory and network weights bit-exact to the checkpoint files (0.0).
- **InstanceNorm spatial-1 error:** a 32³ probe input is invalid for the 6-stage UNet (`_verify_spatial_size` — the "when training" message is misleading); probes use 128³.
- **`torch.device('cuda') == torch.device('cuda:0')` is False:** the data-device assert compares `device.type` + index instead of tensor equality.
- **Plan-01-known hazards avoided:** `my_app` name collision (harness runs with app root + `my_app` dir on `sys.path`; reference module loaded by file path); 32 MB stack (`ulimit -s unlimited` for all verification runs).

## User Setup Required

None. The plan's `user_setup` item (models at `examples/apps/cchmc_nnunet_fifteen_ckpt_app/models`, 3d_fullres available) was already satisfied; all verification ran against the real 3d_fullres 5-fold bundle on the A100.

## Next Phase Readiness

- **Plan 03 (PostResample/EnsembleAverage/Postprocess)** can consume the `logits` output: a zero-copy `holoscan.core.Tensor` on CUDA, FP32, shape `(num_heads=2, 255, 256, 255)` for the airway study (head count is config-driven: 1 foreground + 1, from the label manager). `preprocessed_meta` (bbox, pre-crop shape, new_shape, spacing, transpose) remains on the PreprocessOperator's output — in the DAG (plan 04) it can be wired directly to PostResample (one output may feed multiple operators); SlideWindowOperator intentionally declares only `preprocessed` in / `logits` out per the plan's I/O spec.
- **Plan 04 (DAG assembly):** instantiate `SlideWindowOperator(fragment, model_path=..., config_name='3d_fullres')` between the preprocess and postprocess operators; NVTX `inference` range + JSON timing log are already wired.
- **Performance:** inference core measured 27.1 s/study (5 folds, TTA on, A100) with model load fully amortized at setup — the reference app's per-study cold start and CPU round-trip are eliminated; the 169.7 s baseline's inference share (~138 s) is the target for the plan-05 end-to-end comparison.
- **Multi-config (Phase 2):** `3d_lowres` and `3d_cascade_fullres` are both present in the bundle and resolve through `load_inference_params` (cascade verified to derive 2 input channels); the cascade's previous-stage one-hot stacking is reserved for plan 03/Phase 2.
- **Runtime nnunetv2 dependency for inference:** the operator imports `nnunetv2` utilities (network construction, steps/gaussian, `determine_num_input_channels`) at load/inference time — the vendored editable copy in the venv. Preprocessing (plan 01) remains nnunetv2-free.

---
*Phase: 1-core-pipeline, Plan: 02*
*Completed: 2026-08-18*

## Self-Check: PASSED

- All 4 key files present (slidewindow_operator.py, config/__init__.py, operators/__init__.py + this SUMMARY).
- All task commits present on `nnunet-fast`: 46884b2, 22df6a2, b8a5266.
- No stubs introduced: every emitted value is data-driven (params from plans.json/nnunet_checkpoint.pth, logits from the real 5-fold model, checkpoint/folds resolved from disk).
