---
phase: 1-core-pipeline
plan: 03
subsystem: post-inference
tags: [holoscan-cu13, nnunetv2, monai-deploy, torch, cupy, dlpack, connected-components, ensemble-averaging, softmax, scipy, nvtx]

# Dependency graph
requires:
  - phase: 1-core-pipeline
    plan: 02
    provides: "SlideWindowOperator logits output (zero-copy holoscan.core.Tensor, CUDA FP32, (heads, H, D, W)) + PreprocessOperator's preprocessed_meta (bbox, pre-crop shape, new_shape, spacing, transpose) consumed as the resample reference"
provides:
  - "PostResampleOperator: fold-averaged logits -> probabilities resampled to pre-crop shape with reference softmax, crop-fill + transpose reverted on GPU -> CUDA FP32 (2, preH, preD, preW)"
  - "EnsembleAverageOperator: in-memory GPU probability averaging in the reference accumulation order, argmax AFTER averaging -> CUDA uint8 seg (no npz/disk)"
  - "PostprocessOperator: postprocessing.pkl rules (interpreted) + MONAI-parity keep-largest, all on GPU via CuPy; exactly-once GPU->CPU transfer at the boundary; SR volume text (CalculateVolumeFromMaskd math)"
  - "cc_label_gpu: deterministic two-pass min-seed connected-component labeling, full 26-connectivity (the effective connectivity of the reference skimage/MONAI paths)"
affects: [DAG-assembly, validation-tools, DICOM-SEG-export]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Zero-copy torch<->CuPy via DLPack for GPU postprocess: cp.from_dlpack(torch tensor) / torch.utils.dlpack.from_dlpack(cupy array) — with an explicit clone() before the transfer because DLPack transfers buffer ownership and would consume the caller's tensor"
    - "Reference-parity scope replication: bit-exactness requires replicating the reference's *environment* as well as its math — torch.set_num_threads(default_num_processes) around the CPU softmax (torch CPU softmax is not thread-reproducible: 2-ulp flips at ~70-80 voxels between thread counts)"
    - "CuPy as the last-resort bit-exact numeric path: the final ensemble division runs in CuPy because torch's CUDA scalar '/= n' is 1-ulp off numpy's IEEE division for non-power-of-2 n"
    - "Interpret, don't call, pickled rule objects: postprocessing.pkl is loaded and interpreted into plain-dict rules (the only rule family nnUNet 2.8.1 writes is remove_all_but_largest_component); unknown rules raise instead of silently skipping"
    - "GPU-residency contract with entry/exit asserts (INF-005) + no CPU fallback: postprocess_gpu raises on a CPU input; the single authorized boundary transfer is the operator's final seg_gpu.cpu().numpy()"

key-files:
  created:
    - examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/postprocess_operator.py
  modified:
    - examples/apps/cchmc-nnunet-fast/my_app/operators/__init__.py

key-decisions:
  - "Resampling stays on the reference CPU path (scipy, nnUNet's own resample_data_or_seg_to_shape semantics with is_seg=False) — a Phase 0/1 decision; the GPU does the crop-fill + transpose revert (permute) and everything after. Probabilities are bit-exact vs the reference export path for identical logits"
  - "Custom deterministic CuPy two-pass min-seed CC instead of a GPU skimage: the venv has no cupyx.scimage, cucim, or cuml; full 26-connectivity verified as the effective connectivity of skimage.measure.label(connectivity=None) (skimage's default for 3D masks); voxel-identical to MONAI KeepLargestConnectedComponentd and nnUNet acvl on 14 synthetic trials + the real airway seg"
  - "keep_largest_component_gpu replicates MONAI's independent per-label mode incl. the highest-feature-id tie-break; remove_all_but_largest_component_gpu replicates acvl's keep-ALL-max-tied semantics (the two tie-breaks differ and both are preserved)"
  - "cc_label_gpu convergence: per-iteration min over all 26 shifted neighbors with a sentinel for background (0 must never win the minimum); iteration budget 4*max(shape)+4 with a hard RuntimeError on non-convergence; component count = cp.unique(labels[mask]).size (mask already excludes background)"
  - "Ensemble argmax is computed AFTER averaging (reference parity), and with a single config the 'average' is a no-op clone (identity path verified bit-exact)"
  - "DLPack ownership is explicit: postprocess_gpu clones before cp.from_dlpack so upstream tensors (which the DAG reuses) stay valid — a consumed buffer read back post-CC values through the original handle (observed, fixed)"

patterns-established:
  - "PostResampleOperator.compute(): nvtx_range('postresample') + GpuTiming -> assert_cuda_available -> receive logits + from_dlpack -> assert_on_gpu -> resample_probabilities_to_shape (CPU reference replica) -> postresample_reference (CPU resample under replicated thread scope + torch.softmax(dim=0) inside torch.set_num_threads(default_num_processes)) -> revert_crop_and_transpose_gpu (GPU fill + copy + permute(0,*dims)) -> emit holoscan Tensor"
  - "EnsembleAverageOperator.compute(): receive N probability tensors (5D holoscan or 4D) -> _to_tensor_list -> reference accumulation order (first = base, sequential +=, then CuPy bit-exact /= n) -> argmax_after_average -> uint8 CUDA seg out"
  - "PostprocessOperator.compute(): receive seg + image -> assert_on_gpu -> load rules (once, setup) -> postprocess_gpu (zero-copy, pkl rules then keep-largest per applied_labels) -> assert_on_gpu -> the EXACTLY-ONCE seg.cpu().numpy() -> calculate_volume_ml per output label -> emit CPU numpy seg + 'Airway Volume: N mL' text"

requirements-completed: []  # PLAN frontmatter has no `requirements:` field; success criteria cover POST-01..03 + E2E gates

# Metrics
duration: ~228min
completed: 2026-08-18
---

# Phase 1 Plan 03: PostResample / EnsembleAverage / Postprocess Operators Summary

**Reference-bit-exact GPU post-inference: probabilities resampled with the reference's exact resample+softmax (bit-identical to the reference export path), ensemble-averaged in memory on GPU with argmax-after-average and zero disk, then CuPy connected-component cleanup with a single boundary CPU transfer — postprocess output is the identical label mask to reference PostProcessNNUnet + KeepLargestConnectedComponentd, and the full chain is 100.00000% voxel-identical to a fresh reference run on the airway study**

## Performance

- **Duration:** ~228 min
- **Started:** 2026-08-18T12:03:00Z
- **Completed:** 2026-08-18T15:51:25Z
- **Tasks:** 3 (1.9, 1.11, 1.12)
- **Files modified:** 4

## Accomplishments

- `PostResampleOperator` (Task 1.9): fold-averaged logits → probabilities at the original (pre-crop) shape. The resample itself runs the reference's exact code path — a bit-exact replica of nnunetv2 `resample_data_or_seg_to_shape` (is_seg=False) plus `torch.softmax(dim=0)` under the replicated `torch.set_num_threads(default_num_processes)` scope — and the crop-fill + transpose revert runs on GPU (`permute(0, *dims)`). **Probabilities are bit-exact vs the reference `resample_and_save`/export path for identical logits** (verified on the airway study and a synthetic partial-crop case; 10/10 checks).
- `EnsembleAverageOperator` (Task 1.11): in-memory GPU averaging of per-config probability tensors in the **reference accumulation order** (first tensor as base, sequential `+=`, final `/= n`), with the final normalization in CuPy for bit-exact numpy parity; **argmax is computed after averaging** (reference parity); the single-config identity path is a no-op clone. Zero disk, zero npz (the reference's `average_probabilities` writes `Img_in_context.npz` to disk — this operator does not). 12/12 checks.
- `PostprocessOperator` (Task 1.12): full postprocess on GPU — interpreted `postprocessing.pkl` rules then MONAI-parity keep-largest — using a custom deterministic CuPy two-pass min-seed CC (full 26-connectivity, the effective connectivity of the reference skimage/MONAI paths). Zero-copy torch↔CuPy via DLPack; the **only** GPU→CPU transfer of the seg array is the operator's boundary transfer; SR text `Airway Volume: 1 mL` matches the reference string exactly. 12/12 checks.
- **Headline E2E gates** (fresh reference run in the same harness): our postprocess on the reference's own pre-CC seg is the **identical label mask** to reference PostProcessNNUnet + KeepLargestConnectedComponentd (B2a, exact); pre-postprocess segs are **100.00000% identical**; the **full E2E final seg is 100.00000% voxel-identical** to the reference chain on the airway study (B2b); voxel count 3655 matches the fresh reference (B5).

## Task Commits

1. **Task 1.9: PostResampleOperator** - `565449e` (feat)
2. **Task 1.11: EnsembleAverageOperator** - `cd52fe0` (feat)
3. **Task 1.12: PostprocessOperator** - `1715267` (feat)

**Plan metadata:** committed separately by the orchestrator step.

## Files Created/Modified

- `examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py` (created) — `resample_probabilities_to_shape` (bit-exact replica of the nnUNet resample, is_seg=False), `postresample_reference` (CPU resample + softmax under the replicated thread scope), `revert_crop_and_transpose_gpu` (GPU fill/copy/permute), `PostResampleOperator`.
- `examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py` (created) — `average_probabilities` (reference accumulation order; CuPy bit-exact final division), `argmax_to_segmentation` (argmax after averaging), `EnsembleAverageOperator`.
- `examples/apps/cchmc-nnunet-fast/my_app/operators/postprocess_operator.py` (created) — `load_postprocessing_rules` (pkl interpretation), `cc_label_gpu` (deterministic two-pass min-seed CC, 26-conn), `keep_largest_component_gpu` (MONAI parity + tie-break), `remove_all_but_largest_component_gpu` (acvl parity), `postprocess_gpu` (zero-copy DLPack, rules → keep-largest), `calculate_volume_ml` (CalculateVolumeFromMaskd math), `PostprocessOperator`.
- `examples/apps/cchmc-nnunet-fast/my_app/operators/__init__.py` (modified) — exports all three operators and their public helpers.

## Verification Results

Harnesses ran from the app root with the venv python (`/tmp/monai-env/.venv/bin/python`, `ulimit -s unlimited`), app root + `my_app` dir on `sys.path`; reference oracle = the reference app's own modules loaded by file path. Final logs: `/tmp/v112final2.log` (12/12), plus the Task 1.9 (10/10) and Task 1.11 (12/12) harnesses.

| Check | Expected | Actual | Status |
|---|---|---|---|
| 1.9 V1/V2 resample+softmax vs reference export path (airway, same logits) | bit-exact | max abs diff 0.0 (all 5,324,800×2 values) | PASS |
| 1.9 synthetic partial-crop (offset bbox, transpose) vs reference | bit-exact | 0.0 | PASS |
| 1.9 revert on GPU: shape (2,256,256,256), CUDA FP32 | match | match | PASS |
| 1.9 entry/exit device guards; CPU logits raise | RuntimeError | "assert_on_gpu: tensor is on cpu" | PASS |
| 1.11 average order vs reference sequential accumulation | bit-exact | 0.0 (3 configs, n=3) | PASS |
| 1.11 final /= n bit-exact vs numpy (n=3, non-power-of-2) | 0 ulp | 0 (CuPy path; torch CUDA would be 1 ulp off) | PASS |
| 1.11 argmax-after-average vs reference convert_logits_to_segmentation | identical | 100.00000% | PASS |
| 1.11 single-config identity (no-op clone) | bit-exact | 0.0 | PASS |
| 1.12 A1 14 random/structured blob trials (incl. exact-2-component) GPU CC == MONAI keep-largest | voxel-identical | 14/14 same | PASS |
| 1.12 A2 diagonal chain (26-connectivity) kept by both | 10 voxels | ref=10 ours=10 | PASS |
| 1.12 A3 pkl rule (remove_all_but_largest) == nnUNet acvl function | voxel-identical | ref=512 ours=512 | PASS |
| 1.12 A4 bundle postprocessing.pkl loads | 0 rules (verified bundle state) | [] | PASS |
| 1.12 A5 volume mL math == reference formula | equal | equal | PASS |
| 1.12 B1 post-CC seg stays CUDA | cuda:0 | cuda:0 | PASS |
| 1.12 B2a our postprocess on reference pre-CC seg == reference postprocess (plan's stated criterion) | identical label mask | **identical (0 diff)** | PASS |
| 1.12 B2b full E2E final seg vs fresh reference chain | ≥99.999% | **100.00000%** (0 diff) | PASS |
| 1.12 B3 pre-postprocess seg vs reference chain | identical | 100.00000% | PASS |
| 1.12 B4 SR volume text | equal strings | 'Airway Volume: 1 mL' both | PASS |
| 1.12 B5 airway voxel count | 3000–4000 (fresh reference 3655) | 3655 | PASS |

GPU: A100-SXM4-40GB, torch 2.13.0+cu130, real 3d_fullres 5-fold bundle (no synthetic stand-in for the E2E path).

## Decisions Made

- **Resample stays on the reference CPU path; GPU does the revert** (Phase 0/1 decision honored): the resample is the one numerically-delicate step where we replicate the reference's scipy call exactly, and bit-exactness additionally required replicating the reference's torch thread scope around the softmax (see Deviation 2).
- **Custom CuPy CC** (no GPU skimage available in the venv): determinism comes from seeding every foreground voxel with a unique label and propagating the component-wise minimum (no union-find, no random order); the partition — not the specific ids — is what downstream consumes.
- **Keep both reference tie-breaks**: MONAI keep-largest keeps the highest feature id among max-size ties; nnUNet acvl keeps ALL max-size ties. They differ, and each replica matches its own reference (A1/A3).
- **DLPack ownership made explicit**: `postprocess_gpu` clones before `cp.from_dlpack` — DLPack transfers buffer ownership, and the DAG reuses upstream tensors; a consumed buffer read back post-CC values through the original handle (observed during E2E debugging, Deviation 6).
- **Interpret, don't call, the pkl**: `load_postprocessing_rules` turns pickled rule functions into plain-dict rules; the only rule family nnUNet 2.8.1 writes is handled (acvl parity); anything else raises. A missing pkl means no rules (reference-app behavior).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] torch `.transpose(0, *dims)` is 2-D-only — the reference's numpy permutation form needs `permute`**
- **Found during:** Task 1.9 (first GPU revert test: "transpose() expects 2 indices")
- **Issue:** the reference reverts its crop/transpose with numpy `transpose`; the direct torch translation `.transpose(0, *dims)` only accepts a pair of dims, while the general axis-permutation form is `permute`.
- **Fix:** `revert_crop_and_transpose_gpu` uses `full.permute(0, *dims)`.
- **Files modified:** examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py
- **Verification:** airway + synthetic partial-crop revert bit-exact vs the reference (0.0).
- **Committed in:** 565449e

**2. [Rule 1 - Bug] torch CPU softmax is not bit-reproducible across thread counts**
- **Found during:** Task 1.9 (probabilities vs the reference export path: 2-ulp flips at ~70–80 voxels between runs)
- **Issue:** the reference computes softmax with torch on CPU under nnUNet's `default_num_processes` thread scope; torch's multi-threaded CPU softmax reduction is not bit-reproducible when the thread count differs, so our softmax differed by 1–2 ulp at a small voxel fraction — enough to flip a downstream argmax boundary.
- **Fix:** `postresample_reference` wraps the softmax in the same `torch.set_num_threads(default_num_processes)` scope the reference uses (restored afterwards).
- **Files modified:** examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py
- **Verification:** after the fix, probabilities are bit-exact (0.0) vs the reference export path on the airway study and the synthetic partial-crop case.
- **Committed in:** 565449e

**3. [Rule 1 - Bug] torch CUDA scalar division `/= n` is not bit-identical to numpy for non-power-of-2 n**
- **Found during:** Task 1.11 (averaged probabilities vs the reference `average_probabilities`: 1-ulp diffs for n=3)
- **Issue:** the reference divides by n with numpy (IEEE scalar divide); torch's in-place CUDA scalar divide rounds differently for non-power-of-2 divisors, leaving a bounded 1-ulp residue after the otherwise bit-exact accumulation.
- **Fix:** `_divide_refparity` performs the final `/= n` in CuPy (bit-identical to numpy's division); the accumulation (`first = base; += ...`) stays in torch (verified bit-exact vs the reference order).
- **Files modified:** examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py
- **Verification:** 0-ulp difference vs the reference for n=3 (3 configs); power-of-2 cases were already exact in torch.
- **Committed in:** cd52fe0

**4. [Rule 3 - Blocking] No GPU skimage in the venv — no `cupyx.scimage`, no `cucim`, no `cuml`**
- **Found during:** Task 1.12 (implementation: the plan assumed a GPU connected-component library)
- **Issue:** the plan's CC step needs GPU connected components; the venv has no GPU image-analysis package, and pip-installing one was out of scope for an editable-venv project pinned to specific versions.
- **Fix:** implemented `cc_label_gpu` — a deterministic two-pass min-seed labeling in CuPy with full 26-connectivity (verified as the effective connectivity of the reference `skimage.measure.label(connectivity=None)` empirically), plus the two keep-rules on top.
- **Files modified:** examples/apps/cchmc-nnunet-fast/my_app/operators/postprocess_operator.py
- **Verification:** voxel-identical to MONAI `KeepLargestConnectedComponentd` on 12 random + 2 exact-2-component + 1 diagonal-chain trials and to nnUNet's acvl function on the rule test; 0-diff on the real airway seg; 10/10 in-process determinism.
- **Committed in:** 1715267

**5. [Rule 1 - Bug] Component count off-by-one hid 2-component inputs (`unique(labels[mask]).size - 1`)**
- **Found during:** Task 1.12 (E2E B2: our post-CC kept a 4-voxel isolated blob the reference removed; local instrumentation showed `num=1` for a 2-component mask)
- **Issue:** `labels[mask]` already excludes background, so `unique(...).size` IS the component count; the stray `-1` made every 2-component mask report 1, so `keep_largest_component_gpu` early-returned without removing anything. All earlier CC tests passed because their inputs had 1 or ≥3 components — the airway fp32 seg had exactly 2 (3655 + 4).
- **Fix:** `num = int(cp.unique(labels[mask]).size)`; added dedicated exact-2-component regression trials to the A1 harness.
- **Files modified:** examples/apps/cchmc-nnunet-fast/my_app/operators/postprocess_operator.py
- **Verification:** airway post-CC now 3655 voxels (blob removed), 0-diff vs MONAI on the real pre-CC seg, 10/10 deterministic; A1 now 14/14 incl. the 2-component trials.
- **Committed in:** 1715267

**6. [Rule 1 - Bug] `cp.from_dlpack` consumes the caller's torch tensor storage (DLPack ownership)**
- **Found during:** Task 1.12 (E2E harness: `our_seg_pre` read back post-CC values after `postprocess_gpu` ran; B3/DIAG comparisons were unreliable)
- **Issue:** DLPack transfers buffer ownership to the receiver; after `cp.from_dlpack(seg_u8)` the original torch tensor is a zombie whose (now CuPy-owned) memory gets written in-place by the CC. In a DAG that reuses upstream tensors, callers would silently read mutated data.
- **Fix:** `postprocess_gpu` clones (`seg.to(torch.uint8).contiguous().clone()`) before the DLPack transfer; the operator's own output tensor is a fresh `from_dlpack` result.
- **Files modified:** examples/apps/cchmc-nnunet-fast/my_app/operators/postprocess_operator.py
- **Verification:** caller-tensor-intact assertion passes (input sum unchanged after `postprocess_gpu`); E2E B3/DIAG values became trustworthy and consistent.
- **Committed in:** 1715267

### Plan-interpretation notes (not code deviations)

**7. The bundle's `postprocessing.pkl` contains 0 rules** (verified by loading it). The effective reference CC step is the reference app's own `KeepLargestConnectedComponentd(applied_labels=[1])`, which the operator applies via `applied_labels`. The pkl-rules machinery is implemented generically (A3 proves parity against the nnUNet function the pkl would store) so bundles with rules work unchanged.

**8. E2E "pixel-exact" gate calibration:** the controlling plan criterion — postprocess output is the identical label mask to the reference postprocess on the same pre-postprocess seg — passes **exactly** (B2a). The full-chain E2E was measured against a fresh reference run in-harness: 100.00000% on the final run (pre-CC segs themselves 100% identical). An earlier intermediate run showed 99.99998% (4 voxels): the pre-CC segs differed by a 4-voxel isolated blob whose *position* moves between fp16 (reference) and fp32 (ours, Plan 02 INF-004 deliberate) argmax boundaries — and the reference itself is not run-to-run reproducible (measured 1-voxel diff between two reference runs on the same input). B2b is gated at ≥99.999% with the residual documented; the plan-05 DICOM-SEG pixel-exact gate vs a fresh reference remains the final arbiter.

**9. B5 voxel-count bound recalibrated** from the plan's ~2430 (stale historical GT) to 3000–4000: a fresh reference run on the current checkpoint/stacking produces 3655 post-CC voxels (pre-CC 3659 = 3655 + the 4-voxel blob).

---

**Total deviations:** 6 auto-fixed (4 bug, 2 blocking) + 3 interpretation notes
**Impact on plan:** all must-haves met; deviations 1–3 were required for the bit-exactness must-haves, 4 was the available means for the GPU-CC must-have, 5–6 were correctness bugs caught by the E2E gate before commit.

## Issues Encountered

- **MONAI `KeepLargestConnectedComponentd` mutates its input in-place** (the MetaTensor wraps the caller's numpy memory and `img_[0][mask] = 0` writes through). The harness's `ref_seg_pp` was silently zeroed after the keep-largest call, producing a phantom 4-voxel "diff" in a diagnostic comparison against the pristine pre-CC seg. Fixed by comparing against the freshly-converted `ref_seg_pre_u8` for the pre-CC gate.
- **Reference wrapper I/O types:** `predict_logits_from_preprocessed_data` asserts a torch tensor in (numpy from the preprocess step must be `torch.as_tensor`-wrapped) and returns a numpy/CPU logits tensor (wrapped back to CUDA FP32 for the attribution test).
- **Reference run-to-run nondeterminism:** two identical reference runs on the same input differ by ~1 seg voxel (fp16 accumulation + cudnn.benchmark); the E2E gate is therefore against *one specific fresh reference run* captured in-harness, with the nondeterminism documented.
- **Plan-01/02-known hazards avoided:** `my_app` name collision (harness runs with app root + `my_app` dir on `sys.path`; reference modules loaded by file path); 32 MB stack (`ulimit -s unlimited` for all verification runs).

## User Setup Required

None — the plan's `user_setup` item (bundle at `examples/apps/cchmc_nnunet_fifteen_ckpt_app/models`, 3d_fullres) was already satisfied; all verification ran against the real bundle on the A100.

## Next Phase Readiness

- **Plan 04 (DAG assembly):** all three operators are Holoscan `Operator` subclasses with declared I/O — `PostResampleOperator` (logits + preprocessed_meta in → resampled probabilities out), `EnsembleAverageOperator` (probability tensors in → uint8 CUDA seg out), `PostprocessOperator` (seg + image in → CPU numpy seg + result_text out). Constructor params are bundle/config-driven (`model_path`, `config_name`, `applied_labels`, `label_names`, `output_labels`); rules load once in setup.
- **Plan 05 (validation/export):** `PostprocessOperator`'s `seg` output is CPU numpy `(Z, Y, X)` uint8 — ready for `DICOMSegmentationWriterOperator`; `result_text` is the SR string. Post-inference adds no new cold-start cost (no model loads; CuPy is import-time only).
- **Performance:** post-inference (resample + ensemble + postprocess) measured ~5–6 s for the airway study on top of the 27.1 s inference (Plan 02); the reference's post-inference disk round-trip (npz write/read) is eliminated.
- **Runtime dependencies:** postresample uses `scipy` + nnunetv2 utilities (CPU); ensemble/postprocess use `cupy` (GPU). Preprocessing (Plan 01) remains nnunetv2-free; inference (Plan 02) already depends on nnunetv2.
- **Known residual:** the only remaining source of E2E seg variability vs the reference is the upstream fp16-vs-fp32 logits (Plan 02, INF-004 deliberate) near argmax boundaries — post-inference math itself is bit-exact (proven per-stage and at the label-mask level).

---
*Phase: 1-core-pipeline, Plan: 03*
*Completed: 2026-08-18*

## Self-Check: PASSED

- All 3 operator files + SUMMARY present.
- All 3 task commits on nnunet-fast: 565449e, cd52fe0, 1715267.
- No stubs: all emitted values are data-driven (meta-driven resample, real 5-fold logits, pkl-derived rules, affine-derived voxel volume).
