---
phase: 2-gpu-acceleration
plan: 03
subsystem: cascade-plumbing
tags: [cascade, model-list, resolve-run-model-list, seg-resample-replica, one-hot, lowres-seg, d-02, d-09, d-10, d-13, pipe-03, pipe-04]

requires:
  - phase: 2-gpu-acceleration (plan 01)
    provides: CuPy preprocess flow (the 1-channel path this plan keeps byte-for-byte), _determine_do_sep_z_and_axis / to_holoscan_gpu_tensor / gpu_util helpers
  - phase: 02-gpu-acceleration (plan 02)
    provides: fullres-only E2E running clean under the RMM pluggable allocator (the regression gate this plan re-runs)
  - vendored nnunetv2 2.8.1 (./nnUNet) + batchgenerators
    provides: the bit-exact reference primitives (resample_data_or_seg, resize_segmentation, convert_labelmap_to_one_hot) the replicas are unit-tested against

provides:
  - "resolve_run_model_list(model_list_arg, plans, model_root) -> (run, ensemble) with the reference app's exact model-list semantics (nnunet_seg_operator.py:91-99) plus data-driven previous-stage auto-insertion (PIPE-03) — unit-tested on the real bundle"
  - "PreprocessParams.previous_stage / resample_seg_order / resample_seg_order_z / resample_seg_force_separate_z / foreground_labels, loaded from the PlansManager-resolved configuration (the raw cascade entry inherits everything)"
  - "Cascade-capable PreprocessOperator: optional lowres_seg input (declared ONLY for cascade configs), GPU argmax-seg crop with the image-derived bbox, CPU seg-resample replica (bit-exact vs vendored), GPU one-hot (bit-exact vs vendored), 2-channel (image, one-hot) fp32 C-contiguous output, zero disk I/O (PIPE-04 operator level, D-09/D-10)"
  - "PostResampleOperator conditional lowres_seg output (emit_lowres_seg/emit_probabilities flags): post-softmax argmax uint8 CUDA, original DICOM orientation, NO CC, via the new revert_crop_gpu helper (Plan 04 wires the cross-fragment flow)"
  - "scripts/test_cascade_config.py — 12 unit cases against the real bundle + vendored reference (6 model-list semantics, cascade params, num_input_channels=2, seg-resample replica np.array_equal, one-hot np.array_equal incl. the CuPy op, revert_crop_gpu)"
  - "Gate evidence: .planning/phases/02-gpu-acceleration/plan03-gates/ (Task 2 + Task 3 fullres E2E logs + pixel-diff outputs, both 99.99986% / 3 boundary voxels)"

affects: [phase 2 plan 04 — Plan 04 instantiates one fragment per resolve_run_model_list(config_name) entry: lowres fragment = PostResampleOperator(emit_lowres_seg=True, emit_probabilities=False), cascade fragment = PreprocessOperator(config_name='3d_cascade_fullres') whose lowres_seg input receives the lowres fragment's lowres_seg output; the port contract (uint8 CUDA, original DICOM orientation, same array order as image.asnumpy()) is fixed by this plan]

tech-stack:
  added: []
  patterns:
    - "Reference-semantics helpers cite the reference source in the docstring (nnunet_seg_operator.py:91-99) and are unit-tested against the REAL bundle, not fixtures"
    - "Bit-exact CPU replicas of vendored nnunetv2 primitives, np.array_equal-tested against the vendored function (the D-11 final-gate-only rule applies to the CuPy port as a whole, not to CPU-reference replicas)"
    - "Optional ports declared conditionally in setup() off bundle data (params.previous_stage) — never declared-and-left-unwired (RESEARCH Pitfall 7)"
    - "holoscan 4.2 quirk: constructor state initialized BEFORE super().__init__ in both operators (setup() runs during Operator.__init__)"

key-files:
  created:
    - examples/apps/cchmc-nnunet-fast/scripts/test_cascade_config.py
    - .planning/phases/02-gpu-acceleration/plan03-gates/task2-fullres-e2e.log
    - .planning/phases/02-gpu-acceleration/plan03-gates/task2-fullres-pixel-diff.txt
    - .planning/phases/02-gpu-acceleration/plan03-gates/task3-fullres-e2e.log
    - .planning/phases/02-gpu-acceleration/plan03-gates/task3-fullres-pixel-diff.txt
  modified:
    - examples/apps/cchmc-nnunet-fast/my_app/config/__init__.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py

key-decisions:
  - "Reference semantics control over the plan's inconsistent test bullet: resolve_run_model_list(['3d_lowres']) raises the reference's exact ValueError (empty ensemble) — the plan's task-1 case list sketched standalone lowres as ensemblable, contradicting both the replicated reference (nnunet_seg_operator.py:96-98) and the plan's own must-have 'ensemble = run list minus 3d_lowres; error when ensemble empty'; the ValueError is now unit-tested directly"
  - "load_preprocess_params reads ALL per-config fields from the PlansManager-resolved configuration: the raw cascade entry carries only inherits_from + previous_stage (spacing/normalization/resampling kwargs are inherited), so the raw cfg['spacing'] access crashed on 3d_cascade_fullres (Rule 3); the resolved configuration is the single venv-verified source (previous_stage is present there too)"
  - "Auto-insertion is data-driven off the raw plans 'previous_stage' field (D-02): the only string literals in the new code path are the reference's own '3d_lowres'/'3d_cascade_fullres' in the reference-reorder step, each commented to nnunet_seg_operator.py:92-95"
  - "Post-softmax argmax == reference argmax-of-resampled-logits (softmax is monotone per voxel) — documented at the torch.argmax site with the D-09 citation; NO connected-component cleanup (the reference cascade input is pre-CC; KeepLargestCC runs only on the final output)"
  - "One-hot channel count is config-generic (one channel per foreground_labels, foreground_labels order == convert_labelmap_to_one_hot order); the airway bundle yields the 2-channel input; the one-hot channel is never normalized (concat after image-channel normalize)"
  - "revert_crop_gpu returns the ORIGINAL (pre-transpose) array order — the same as the 4D probability revert — so the output SHAPE is shape_before_cropping permuted by transpose_backward (the plan's 'output shape == shape_before_cropping' test expectation is only true for identity transpose_forward); the test asserts np.array_equal against a hand-rolled numpy reference, which is the controlling check"

patterns-established:
  - "Orientation contract: every lowres_seg tensor in the system is uint8 CUDA in the same array order as image.asnumpy() — PostResample emits it (revert_crop_gpu), Preprocess consumes it (identical layout chain to the image raw upload, so the image-derived bbox applies verbatim)"
  - "Structured timing records extended: preprocess gains lowres_seg/lowres_seg_shape when the port is active; postresample gains lowres_seg_shape/lowres_seg_dtype (parseable by Plan 06's benchmarking)"

requirements-completed: [PIPE-03, PIPE-04]
requirements-deferred: []  # PIPE-04's cross-fragment FLOW wiring is Plan 04's job; this plan delivered the operator-level plumbing + port contract (the plan's scope)

deviations:
  - "Rule 1 (bug): revert_crop_gpu used the 4D-with-channel permutation form permute(*[i + 1 for i in tb]) on a 3D tensor (IndexError) — the 3D reference is .transpose(transpose_backward) as-is (no +1); fixed"
  - "Rule 1 (test expectation): test_revert_crop_gpu's 'output shape == shape_before_cropping' assert is wrong for non-identity transpose_forward (the reference revert returns the pre-transpose array order, so the shape is tb-permuted) — the implementation matches the reference; the test now asserts np.array_equal against the hand-rolled numpy reference (shape + values)"
  - "Rule 3 (blocker): load_preprocess_params crashed with KeyError 'spacing' on the cascade config (raw entry inherits the field) — switched to the PlansManager-resolved configuration for all fields (also where resampling_fn_seg_kwargs resolves for the inherited entry, as the plan prescribed)"
  - "Plan-text fix: the task-1 case list's 'standalone lowres is ensemblable' bullet contradicts the reference being replicated and the plan's own must-haves — implemented + tested the reference behavior (ValueError) instead; see key-decisions"

metrics:
  duration: ~31min
  tasks: 3/3
  commits: 3
  unit-tests: "12 cases (9 model-list/params + 3 bit-exact replica tests), all exit 0"
  e2e-regression: "Task 2 + Task 3 fullres-only E2E exit 0; pixel_diff 99.99986% byte-identity vs testdata/ref_fullres_only (3 documented fp16<->fp32 boundary voxels, same coordinates as Phase 1/Plan 02 gates); SR exact"

completed: 2026-08-19
---

# Phase 2 Plan 03: Operator-Level Cascade Support + Reference Model-List Semantics Summary

**`resolve_run_model_list` now reproduces the reference app's model-list semantics exactly (plus the data-driven previous-stage auto-insertion the in-memory cascade requires), `PreprocessOperator` consumes a GPU argmax `lowres_seg`, resamples it on the bit-exact CPU reference seg path, one-hots it on GPU, and emits the 2-channel (image, one-hot) fp32 input with zero disk I/O, and `PostResampleOperator` can conditionally emit the cascade input (argmax uint8, original DICOM orientation, no CC) — every new primitive unit-tested `np.array_equal` against the vendored nnunetv2 2.8.1 reference, with the 1-channel fullres-only E2E staying exit 0 + 99.99986% pixel-exact (3 documented boundary voxels).**

## What was built

### Task 1 — config: model-list semantics + cascade PreprocessParams (commit 0b265e1)

- `resolve_run_model_list(model_list_arg, plans, model_root) -> (run, ensemble)` in
  `my_app/config/__init__.py`, docstring cites `nnunet_seg_operator.py:91-99`:
  - default = plans.json `configurations` order filtered to existing model dirs
    (2d filtered out on this bundle); explicit arg used as given;
  - **auto-insertion (documented fast-app divergence):** for each config whose raw
    plans entry has `previous_stage` = `p` (not in the list, has a model dir), `p`
    is inserted immediately before it — `['3d_cascade_fullres']` →
    `['3d_lowres', '3d_cascade_fullres']`. The reference CRASHES on cascade-only
    (missing `tmp/3d_lowres/Img_in_context.nii.gz`); the in-memory cascade needs
    the stage present. Fully data-driven (no config-name literals in this step —
    D-02; a future 2d stage needs zero code changes).
  - reference reorder (lines 92-95): both `3d_lowres` + `3d_cascade_fullres`
    present → lowres moved immediately before cascade;
  - `ensemble = run minus '3d_lowres'`, `ValueError` with the reference's exact
    message when empty (now unit-tested directly: `['3d_lowres']` raises).
- `PreprocessParams` gained `previous_stage`, `resample_seg_order` (1),
  `resample_seg_order_z` (0), `resample_seg_force_separate_z` (None),
  `foreground_labels` (`(1,)` for the airway bundle) — all loaded from the
  PlansManager-resolved configuration (Rule 3: the raw cascade entry inherits
  `spacing` + `resampling_fn_seg_kwargs`, so raw-key access crashed).
- 9 unit cases against the real bundle, all PASS.

### Task 2 — cascade-capable PreprocessOperator (commit c930280)

- `_resize_segmentation`: bit-exact replica of vendored
  `batchgenerators.augmentations.utils.resize_segmentation` (per-label multihot
  skimage resize, mode='edge', clip=True, anti_aliasing=False, ≥0.5 threshold).
- `_resample_seg_to_shape(data, new_shape, current_spacing, new_spacing, params)`:
  bit-exact replica of the vendored `resample_data_or_seg_to_shape` SEGMENTATION
  path (`is_seg=True`): `resize_segmentation` (NOT the data path's plain
  skimage resize), cascade seg kwargs from `PreprocessParams` (order=1,
  order_z=0, force_separate_z=None), float intermediate + integer output dtype,
  separate-z via `map_coordinates` (order_z=0 direct path + the not-exercised
  unique-label branch kept for fidelity), same equal-shape short-circuit.
  **Unit-tested `np.array_equal` vs the vendored function** on 3 random uint8
  regimes: short-circuit / plain resize / separate-z + map_coordinates
  (anisotropy 4 > 3, axis 2, shape[axis] changes).
- `INPUT_LOWRES_SEG = "lowres_seg"` port, declared in `setup()` ONLY when
  `params.previous_stage is not None` (params loaded eagerly; non-cascade
  configs have no such port — Pitfall 7 discipline).
- `preprocess_image(image, lowres_seg=None)`: the seg takes the EXACT layout
  chain of the image raw upload (uint8 G2G via `cp.array`, same tf view), is
  cropped with the IMAGE-derived bbox (GPU, layout-only), round-trips D2H
  (~16 MB uint8 — D-13), is resampled with `_resample_seg_to_shape` to the
  image's target shape/spacings, one-hotted on GPU
  (`(seg == lbl).astype(cp.float32)` per `foreground_labels`), and
  `cp.concatenate([vol_gpu, one_hot], axis=0)` → `(2, *new_shape)` fp32
  C-contiguous (image FIRST, one-hot LAST — reference `np.vstack((data,
  seg_onehot))`; one-hot NEVER normalized). Zero disk I/O.
- `compute()`: receives the port only when declared, `assert_on_gpu`,
  **D-10 integer-dtype guard** (float `lowres_seg` → `ValueError`,
  probabilities forbidden); timing record gains `lowres_seg`/`lowres_seg_shape`.
- 1-channel path byte-for-byte the Plan 01 flow; `__init__` state moved before
  `super()` (holoscan 4.2 setup-during-init).
- Gates: unit tests exit 0; **fullres-only E2E exit 0 + pixel_diff 99.99986%**
  (3 boundary voxels, IoU 0.998714) vs `testdata/ref_fullres_only`.

### Task 3 — PostResampleOperator conditional lowres_seg output (commit 1d5d340)

- `emit_lowres_seg=False` / `emit_probabilities=True` constructor flags,
  initialized BEFORE `super().__init__` (same pattern as
  `EnsembleAverageOperator`); both gate the `spec.output` declarations in
  `setup()` (Plan 04 will pass `emit_lowres_seg=True, emit_probabilities=False`
  to the lowres fragment without this file being re-edited).
- `revert_crop_gpu(seg_crop, meta)`: background-0 fill at the pre-crop shape +
  insert at `bbox_used_for_cropping` + `transpose_backward` permute → 3D uint8
  CUDA in original DICOM orientation (Rule 1 fix: 3D permutation is `tb` as-is,
  no channel `+1`). Unit-tested `np.array_equal` against a hand-rolled numpy
  reference (non-identity tf `(1,2,0)`, interior bbox): outside-bbox == 0,
  inside == inverse-permuted input.
- `postresample()` with `emit_lowres_seg`: computes the seg BEFORE revert on the
  pre-revert resampled probabilities —
  `torch.argmax(torch.from_numpy(probabilities_cpu), dim=0).to(torch.uint8)` —
  with the D-09 equivalence comment (post-softmax argmax == reference
  argmax-of-resampled-logits; softmax is monotone per voxel). **No
  connected-component cleanup** (the reference cascade input is pre-CC).
- `compute()`: `assert_on_gpu` + emit on `OUTPUT_LOWRES_SEG`; timing record
  gains `lowres_seg_shape` + `lowres_seg_dtype: "uint8"`.
- Gates: unit tests exit 0; **fullres-only E2E exit 0 + pixel_diff 99.99986%**
  (identical 3 boundary voxels) under the default flags — Phase 1 behavior
  intact.

## Gate results

| Gate | Result |
|---|---|
| `scripts/test_cascade_config.py` (12 cases) | exit 0, all PASS |
| Task 2 fullres-only E2E (`/tmp/p2p3_e2e`) | exit 0 (SEG/SC/SR) |
| Task 2 pixel_diff vs `testdata/ref_fullres_only` | 99.99986% byte-identity, 3 boundary voxels, IoU 0.998714, PASS |
| Task 3 fullres-only E2E (`/tmp/p2p3_e2e_b`) | exit 0 (SEG/SC/SR) |
| Task 3 pixel_diff vs `testdata/ref_fullres_only` | 99.99986% byte-identity, same 3 boundary voxels, IoU 0.998714, PASS |

Evidence: `.planning/phases/02-gpu-acceleration/plan03-gates/`.

## Success criteria (plan)

- [x] PIPE-03: `resolve_run_model_list` replicates reference semantics, config-generic (D-02), unit-tested on the real bundle
- [x] PIPE-04 (operator level): GPU argmax-seg → CPU seg-resample → GPU one-hot → 2-channel input, zero disk I/O (D-09/D-10), bit-exact primitives vs vendored reference
- [x] 2d stays blocked-on-model (D-01/D-03): no dummy model, no 2d-specific code (auto-insertion is plans-driven; `2d` is filtered out of the default list for lacking a model dir, exactly as the reference does)
- [x] Regression: fullres-only pixel-exact gate still passes (2× E2E + pixel_diff)
- [x] Commits made (small, imperative): 0b265e1, c930280, 1d5d340

## Known stubs

None. All new wiring is either unit-tested against the real bundle / vendored
reference or exercised by the E2E regression. The only not-yet-wired artifact is
the cross-fragment FLOW (lowres fragment's `lowres_seg` output → cascade
fragment's `lowres_seg` input), which is explicitly Plan 04's scope ("Plan 04
wires the cross-fragment flow" — plan key_links); the port contract it wires is
fully defined and tested by this plan.

## Self-Check: PASSED

All key files verified present (test_cascade_config.py, config/__init__.py,
preprocess_operator.py, postresample_operator.py, plan03-gates evidence) and
all three task commits (0b265e1, c930280, 1d5d340) found in git history.
