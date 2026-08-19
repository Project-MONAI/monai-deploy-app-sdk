---
phase: "2-gpu-acceleration"
plan: "03"
type: "execute"
wave: 2
depends_on: ["01"]
files_modified:
  - "examples/apps/cchmc-nnunet-fast/my_app/config/__init__.py"
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py"
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py"
  - "examples/apps/cchmc-nnunet-fast/scripts/test_cascade_config.py"
autonomous: true
requirements: [PIPE-03, PIPE-04]
must_haves:
  truths:
    - "A single `resolve_run_model_list` reproduces the reference app's model-list semantics exactly (default = plans.json configs with model dirs, plans.json order; lowres moved immediately before cascade_fullres when both present; ensemble = run list minus 3d_lowres; error when ensemble empty) — verified by unit tests against the real bundle (PIPE-03)"
    - "Cascade preprocess consumes a GPU `lowres_seg` tensor (post-softmax argmax, NO connected-component cleanup — the reference cascade input is pre-CC seg) and builds the 2-channel input = preprocessed image + one-hot float32 channel, zero disk I/O (PIPE-04, D-09; D-10: raw probabilities are FORBIDDEN)"
    - "The cascade seg resample is the CPU scipy path with the reference seg kwargs (is_seg=True, order=1, order_z=0, force_separate_z=None) — a replica unit-tested against the vendored nnunetv2 2.8.1 function on random arrays (PREP-03/D-13)"
    - "One-hot on GPU: `(seg == 1).astype(cp.float32)`, bit-exact vs the vendored `convert_labelmap_to_one_hot` (unit-tested); the one-hot channel is NEVER normalized"
    - "Fragment instantiation stays config-generic (D-02): no config name hard-coded in the new wiring beyond the reference-semantic literals ('3d_lowres'/'3d_cascade_fullres' are data-driven off plans.json `previous_stage`); 2d would drop in with zero code changes (D-01/D-03: 2d is blocked-on-model — NO dummy 2d model is created)"
    - "The fullres-only E2E regression still passes (Plan 01 CuPy path untouched in behavior)"
  artifacts:
    - path: "examples/apps/cchmc-nnunet-fast/my_app/config/__init__.py"
      provides: "resolve_run_model_list + PreprocessParams.previous_stage / seg-resample kwargs / foreground_labels"
      contains: "def resolve_run_model_list"
    - path: "examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py"
      provides: "optional lowres_seg input port; _resample_seg_to_shape CPU replica; GPU one-hot + 2-channel concat"
      contains: "lowres_seg"
    - path: "examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py"
      provides: "conditional lowres_seg output (argmax uint8 CUDA, original DICOM orientation, no CC)"
      contains: "lowres_seg"
    - path: "examples/apps/cchmc-nnunet-fast/scripts/test_cascade_config.py"
      provides: "unit tests: model-list semantics, seg-resample replica vs vendored nnunetv2, one-hot vs vendored convert_labelmap_to_one_hot"
  key_links:
    - from: "postresample lowres_seg (uint8 CUDA, original orientation)"
      to: "cascade preprocess `lowres_seg` input"
      via: "holoscan core Tensor / DLPack zero-copy (Plan 04 wires the cross-fragment flow)"
      pattern: "lowres_seg"
    - from: "cascade preprocess"
      to: "SlideWindow (num_input_channels=2)"
      via: "2-channel fp32 C-contiguous tensor emitted on the existing `preprocessed` port; load_inference_params already resolves num_input_channels=2 for 3d_cascade_fullres"
      pattern: "concatenate"
    - from: "_resample_seg_to_shape"
      to: "vendored nnunetv2 resample_data_or_seg_to_shape (is_seg=True)"
      via: "unit test np.array_equal on random uint8 volumes"
      pattern: "resample_data_or_seg_to_shape"
---

# Phase 2 Plan 03: Operator-Level Cascade Support + Reference Model-List Semantics

## Objective
- **What:** Build the operator-level building blocks for the multi-fragment DAG:
  (1) `resolve_run_model_list` with the reference app's exact list semantics, (2)
  cascade-capable `PreprocessOperator` (optional `lowres_seg` input → seg resample on the
  CPU reference path → GPU one-hot → 2-channel input), (3) conditional `lowres_seg`
  output on `PostResampleOperator` (post-softmax argmax, no CC, original orientation).
- **Why:** PIPE-03/PIPE-04 need config-generic, zero-disk-I/O cascade plumbing (D-09/D-10)
  BEFORE the DAG is assembled in Plan 04. Interface-first: the ports and contracts are
  defined and unit-tested here; Plan 04 only wires fragments.
- **Output:** config helpers + cascade operator extensions + `scripts/test_cascade_config.py`
  unit tests + a clean fullres-only E2E regression.

## Execution Environment

- Python: `/tmp/monai-env/.venv/bin/python`. Vendored nnunetv2 2.8.1 at `./nnUNet/` (do
  NOT install from PyPI).
- Real bundle: `/users/srv-mde/projects/monai-deploy-app-sdk/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models`
  (config dirs: `3d_fullres`, `3d_lowres`, `3d_cascade_fullres`; `2d` absent — D-01).
- Fullres-only E2E regression command (app is still the Phase 1/Plan-01 single-config
  DAG until Plan 04):
  ```bash
  cd examples/apps/cchmc-nnunet-fast && ulimit -s unlimited
  /tmp/monai-env/.venv/bin/python my_app -i /users/srv-mde/projects/monai-deploy-app-sdk/testdata/airway_input \
    -m /users/srv-mde/projects/monai-deploy-app-sdk/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models -o <scratch>
  ```
  then `pixel_diff.py <out>/SEG testdata/ref_fullres_only/SEG`.
- Commit after each major change.

## Context

@.planning/phases/2-gpu-acceleration/02-CONTEXT.md
@.planning/phases/2-gpu-acceleration/02-RESEARCH.md
@examples/apps/cchmc-nnunet-fast/my_app/config/__init__.py
@examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py
@examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py
@examples/apps/cchmc_nnunet_fifteen_ckpt_app/my_app/nnunet_seg_operator.py
@nnUNet/nnunetv2/preprocessing/resampling/default_resampling.py
@nnUNet/nnunetv2/inference/data_iterators.py
@nnUNet/nnunetv2/utilities/label_handling/label_handling.py

**Verified facts (2026-08-19, live — trust these over RESEARCH.md where they differ):**
- plans.json `configurations` dict order: `['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres']`;
  `2d` has no model dir. Resolved via `PlansManager` (venv probe):
  `3d_cascade_fullres` → `previous_stage_name = '3d_lowres'`,
  `resampling_fn_seg_kwargs = {'is_seg': True, 'order': 1, 'order_z': 0, 'force_separate_z': None}`
  (resolves through `inherits_from: 3d_fullres`), `num_input_channels = 2`,
  `num_segmentation_heads = 2`; all other configs → `previous_stage_name = None`.
  Raw plans.json: cascade entry is `{'inherits_from': '3d_fullres', 'previous_stage': '3d_lowres'}`
  (the `previous_stage` field lives on the raw config). dataset.json labels:
  `{'airway': 1, 'background': 0}`.
- Reference list logic (`nnunet_seg_operator.py` lines 91–99, read verbatim for the
  replication): default list = plans.json config order filtered to existing model dirs;
  if BOTH `3d_lowres` and `3d_cascade_fullres` are present: remove lowres, re-insert
  immediately before cascade; `ensemble_model_list = [m for m in run if m != "3d_lowres"]`;
  `ValueError` if the ensemble is empty.
- ⚠ **RESEARCH.md correction (live-probed 2026-08-19):** running the reference with
  `model_list=['3d_cascade_fullres']` ALONE CRASHES —
  `RuntimeError: ... tmp/3d_lowres/Img_in_context.nii.gz does not exist` (the reference
  cascade reads the previous stage's exported .nii.gz, which only exists if lowres
  actually ran; the list logic only REORDERS, it does not auto-insert). Consequence:
  `resolve_run_model_list` in the fast app must AUTO-INSERT the previous stage when a
  config with a `previous_stage` is requested without it (data-driven off plans.json
  `previous_stage`, not hard-coded to '3d_lowres') — the in-memory cascade path requires
  it, and this is the documented divergence from the reference's crash-on-cascade-only
  behavior.
- Reference cascade chain (vendored nnunetv2 2.8.1, verified in RESEARCH Pattern 4 and
  re-checked here): lowres seg = argmax of resampled logits (no CC — the reference's
  KeepLargestCC runs only on the final output, never on the cascade input); cascade
  preprocess transposes the seg with the same `transpose_forward`, crops with the
  IMAGE-derived bbox, runs `resampling_fn_seg` (CPU scipy, is_seg=True order=1 order_z=0),
  then `convert_labelmap_to_one_hot(seg[0], foreground_labels=[1], data.dtype=float32)`
  → `np.vstack((data, seg_onehot))` → 2 channels (image first, one-hot last), one-hot
  never normalized.
- Post-softmax argmax == reference argmax-of-resampled-logits (softmax is monotone per
  voxel ⇒ identical argmax; the fast PostResample resamples LOGITS before softmax, same
  order as the reference). D-10 locked: do NOT feed raw probabilities into the cascade.

## Tasks

<task type="auto">
  <name>Task 1: config — resolve_run_model_list + cascade-aware PreprocessParams + unit tests</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/config/__init__.py, examples/apps/cchmc-nnunet-fast/scripts/test_cascade_config.py</files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/my_app/config/__init__.py (existing `PreprocessParams`, `load_preprocess_params`, `load_inference_params`, `_resolve_config_dir_and_jsonpkls` — extend, don't refactor)
    - examples/apps/cchmc_nnunet_fifteen_ckpt_app/my_app/nnunet_seg_operator.py (lines 91–99 and `_get_model_list_from_plans` at line ~145 — the EXACT semantics to replicate)
    - examples/apps/cchmc_nnunet_fifteen_ckpt_app/models/jsonpkls/plans.json (real config order + cascade entry)
  </read_first>
  <action>
1. In `my_app/config/__init__.py` add:
   ```python
   def resolve_run_model_list(
       model_list_arg: Optional[Sequence[str]],
       plans: Mapping[str, Any],
       model_root: ModelPath,
   ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
   ```
   Replicating the reference semantics EXACTLY (cite `nnunet_seg_operator.py:91-99` in
   the docstring):
   - `model_list_arg is None` → plans.json `configurations` dict order, filtered to
     configs with an existing model dir under `model_root` (mirror
     `_get_model_list_from_plans`; raise FileNotFoundError if none).
   - explicit arg → use as given (no filtering).
   - Auto-insertion (fast-app extension, data-driven, see Context correction): for each
     config `c` in the list whose raw plans entry has `previous_stage` = `p` and `p` is
     NOT already in the list and `p` HAS a model dir under `model_root`: remove `c`,
     insert `p` immediately before `c`. (With the real bundle this makes
     `['3d_cascade_fullres']` → `['3d_lowres', '3d_cascade_fullres']`; a future 2d or
     other cascade stage needs no code change — D-02.)
   - Then the reference reorder: if both `3d_lowres` and `3d_cascade_fullres` present →
     remove lowres, insert immediately before cascade (these two literals are the
     reference app's own semantics — keep them, with a comment quoting lines 92–95).
   - `ensemble = tuple(m for m in run if m != "3d_lowres")`; raise
     `ValueError("At least one non-auxiliary model configuration is required for ensemble inference.")`
     (reference's exact message) if `ensemble` is empty.
   - Return `(run, ensemble)`.
   Add to `__all__`.
2. Extend `PreprocessParams` with FIELDS (all with defaults so existing constructions
   keep working) and load them in `load_preprocess_params`:
   - `previous_stage: Optional[str] = None` — from the RAW plans config entry
     `cfg.get("previous_stage")`.
   - `resample_seg_order: int = 1`, `resample_seg_order_z: int = 0`,
     `resample_seg_force_separate_z: Optional[bool] = None` — from
     `cfg.get("resampling_fn_seg_kwargs", {})` (the raw entry lacks it for the cascade
     config because it inherits — resolve via
     `PlansManager(plans).get_configuration(config_name).configuration["resampling_fn_seg_kwargs"]`,
     the venv-verified source; fall back to raw `cfg` if the key is absent there too).
   - `foreground_labels: Tuple[int, ...] = ()` — `tuple(sorted(v for v in labels.values() if v != 0))`
     (dataset.json; `(1,)` for the airway bundle) — mirrors the reference's
     `label_manager.foreground_labels` used by `convert_labelmap_to_one_hot`.
3. Create `scripts/test_cascade_config.py` (plain asserts; run with
   `/tmp/monai-env/.venv/bin/python scripts/test_cascade_config.py`; model root = the
   real bundle path from Execution Environment; print PASS per case, exit non-zero on
   failure). AT LEAST these cases:
   - `resolve_run_model_list(None, plans, root)` ==
     `(('3d_fullres', '3d_lowres', '3d_cascade_fullres'), ('3d_fullres', '3d_cascade_fullres'))`
     (2d filtered out; lowres reordered before cascade; ensemble = minus lowres).
   - `resolve_run_model_list(['3d_cascade_fullres'], plans, root)` ==
     `(('3d_lowres', '3d_cascade_fullres'), ('3d_cascade_fullres',))`.
   - `resolve_run_model_list(['3d_lowres', '3d_cascade_fullres'], plans, root)` ==
     `(('3d_lowres', '3d_cascade_fullres'), ('3d_cascade_fullres',))`.
   - `resolve_run_model_list(['3d_lowres'], plans, root)` ==
     `(('3d_lowres',), ('3d_lowres',))` (standalone lowres is ensemblable).
   - `resolve_run_model_list(['3d_fullres'], plans, root)` ==
     `(('3d_fullres',), ('3d_fullres',))`.
   - `resolve_run_model_list(['3d_lowres'], ...)` is the only lowres-only ensemble case;
     and `ValueError` raised when the ensemble would be empty is NOT reachable with
     this bundle — instead assert the reorder case
     `['3d_cascade_fullres', '3d_lowres']` → `(('3d_lowres', '3d_cascade_fullres'), ...)`.
   - `load_preprocess_params(root, '3d_cascade_fullres')` → `previous_stage == '3d_lowres'`,
     `resample_seg_order == 1`, `resample_seg_order_z == 0`,
     `resample_seg_force_separate_z is None`, `foreground_labels == (1,)`.
   - `load_preprocess_params(root, '3d_fullres')` → `previous_stage is None`.
   - `load_inference_params(root, '3d_cascade_fullres').num_input_channels == 2`.
  </action>
  <acceptance_criteria>
    - `grep -n "def resolve_run_model_list" my_app/config/__init__.py` returns 1 line; `grep -n "nnunet_seg_operator.py:91" my_app/config/__init__.py` returns >= 1 line (docstring citation of the reference semantics).
    - `grep -n "previous_stage\|resample_seg_order\|foreground_labels" my_app/config/__init__.py` shows all five new PreprocessParams fields defined and populated in `load_preprocess_params`.
    - `grep -c "assert" scripts/test_cascade_config.py` >= 10.
    - `/tmp/monai-env/.venv/bin/python scripts/test_cascade_config.py` exits 0 and prints a PASS line per case.
    - No config name appears in the NEW auto-insertion code path except via `cfg.get("previous_stage")` (grep the function body: the only string literals are '3d_lowres' and '3d_cascade_fullres' inside the reference-reorder step, each with a comment quoting the reference lines) — D-02 config-genericity.
  </acceptance_criteria>
  <verify>cd examples/apps/cchmc-nnunet-fast && /tmp/monai-env/.venv/bin/python scripts/test_cascade_config.py</verify>
  <done>Model-list resolution reproduces the reference semantics (plus the data-driven previous-stage auto-insertion the in-memory cascade requires) and PreprocessParams carries everything the cascade needs — all unit-tested against the real bundle.</done>
</task>

<task type="auto">
  <name>Task 2: cascade-capable PreprocessOperator — lowres_seg input, CPU seg-resample replica, GPU one-hot, 2-channel output</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py, examples/apps/cchmc-nnunet-fast/scripts/test_cascade_config.py</files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py (post-Plan-01 state: the CuPy `preprocess_image` flow, `_resample_to_shape`, `to_holoscan_gpu_tensor`, `setup()` port declaration, `compute()`; the NOTE that holoscan 4.2 calls setup() during Operator.__init__)
    - nnUNet/nnunetv2/preprocessing/resampling/default_resampling.py (line 89 `resample_data_or_seg_to_shape` — the SEG path to replicate: is_seg semantics, order/order_z, separate-z, no anti-aliasing; also `resample_data_or_seg`)
    - nnUNet/nnunetv2/inference/data_iterators.py (`PreprocessAdapterFromNpy.generate_train_batch` — one-hot + `np.vstack((data, seg_onehot))` order)
    - nnUNet/nnunetv2/utilities/label_handling/label_handling.py (`convert_labelmap_to_one_hot`, line 259)
    - nnUNet/nnunetv2/preprocessing/preprocessors/default_preprocessor.py (`run_case_npy` — seg transpose with same transpose_forward, crop with image-derived bbox via crop_to_nonzero, then resampling_fn_seg)
    - .planning/phases/2-gpu-acceleration/02-RESEARCH.md (Pattern 4 "Cascade wiring" implementation steps + Code Examples "Cascade reference chain")
  </read_first>
  <action>
1. New CPU helper `_resample_seg_to_shape(data, new_shape, current_spacing, new_spacing, params)`
   in preprocess_operator.py — a bit-exact replica of the vendored
   `resample_data_or_seg_to_shape` for the SEGMENTATION path (is_seg=True), using the
   same primitives the existing `_resample_to_shape` uses (`_determine_do_sep_z_and_axis`,
   `skimage.transform.resize`, `scipy.ndimage.map_coordinates`) with the SEG semantics
   from `default_resampling.py`: `anti_aliasing` disabled for is_seg (verify the exact
   flag in the vendored source — `resize(..., anti_aliasing=not is_seg)` style),
   `order = params.resample_seg_order` (1), `order_z = params.resample_seg_order_z` (0),
   `force_separate_z = params.resample_seg_force_separate_z` (None), dtype preserved
   (uint8/int stays integer — do NOT float-cast seg data where the reference doesn't),
   and the same "no resampling necessary → return input unchanged" short-circuit.
   READ THE VENDORED FUNCTION FIRST and mirror its exact branch structure; where the
   data-path replica `_resample_to_shape` differs, follow the vendored seg path, not the
   data replica.
2. Unit-test the replica in `scripts/test_cascade_config.py` (extend the file):
   `test_seg_resample_replica`: generate 3 random uint8 (1, Z, Y, X) volumes with mixed
   values {0, 1} (incl. a near-isotropic shape that triggers the separate-z branch and a
   shape where `new_shape == shape` for the short-circuit branch), call BOTH
   `_resample_seg_to_shape(...)` (our replica, with the cascade PreprocessParams loaded
   from the real bundle) AND the vendored
   `resample_data_or_seg_to_shape(seg, new_shape, current_spacing, target_spacing,
   order=1, order_z=0, force_separate_z=None, is_seg=True,
   force_separate_z=..., is_seg=True)` — import from
   `nnunetv2.preprocessing.resampling.default_resampling` — and assert
   `np.array_equal` (exact, not allclose). Also add `test_one_hot_vs_reference`: random
   uint8 seg with values {0, 1}, compare
   `((seg == 1).astype(np.float32))` (the GPU op's CPU twin — and, for the GPU op
   itself, run the same expression in CuPy and `.get()`) against the vendored
   `convert_labelmap_to_one_hot(seg, [1], np.float32)` → `np.array_equal`.
   (These unit tests validate CPU primitives against the VENDORED REFERENCE — permitted
   and required; D-11's final-gate-only rule applies to the CuPy port as a whole, not to
   CPU-reference replicas.)
3. `PreprocessOperator` changes:
   - Port constant `INPUT_LOWRES_SEG = "lowres_seg"`.
   - `setup()`: load params eagerly when `model_path` is set (call `_load_params()`) and
     declare the optional input ONLY for cascade configs:
     `if self._params.previous_stage is not None: spec.input(self.INPUT_LOWRES_SEG)`.
     (A declared input with no flow = silent hang; a declared output with no receiver =
     GXF rejection — RESEARCH Pitfall 7. For non-cascade configs the port simply does
     not exist.)
   - In `preprocess_image`'s new CuPy flow, when `lowres_seg` is provided (pass it in
     as an optional arg), AFTER the image is cropped on GPU but BEFORE normalization:
     (a) receive the seg as a CUDA uint8 3D tensor in the SAME array order as
         `image.asnumpy()` (the orientation contract set by Task 3);
     (b) `seg4 = cp.array(torch_tensor)` (uint8 H2D-free G2G copy via
         `__cuda_array_interface__`; do NOT use `cp.from_dlpack` — ownership pitfall),
         then apply the SAME layout chain as the image raw upload:
         `seg4 = cp.ascontiguousarray(seg4.transpose(2, 1, 0))[None, ...]` then
         `seg4 = cp.ascontiguousarray(seg4.transpose(0, 3, 2, 1))` then the tf view
         `seg4 = seg4.transpose(0, *[i + 1 for i in tf])` — mirroring the image
         EXACTLY (same orientation, so the image-derived bbox applies);
     (c) crop with the IMAGE-derived bbox (computed from the image — the reference
         crops seg with the image's `crop_to_nonzero` box; identical image ⇒ identical
         box): `seg_c = cp.ascontiguousarray(seg4[(slice(None),) + slicer])` (GPU,
         layout-only, bit-exact on 0/1);
     (d) D2H (~16 MB uint8 — D-13 accepted): `seg_np = seg_c.get()`;
     (e) `seg_r = _resample_seg_to_shape(seg_np, new_shape, original_spacing,
         params.spacing, params)` — the SAME `new_shape`/spacings as the image (the
         reference resamples seg to the cascade target shape inside cascade preprocess);
     (f) one-hot on GPU (bit-exact, verified):
         `oh = (cp.array(seg_r) == 1).astype(cp.float32)` → shape (1, *new_shape);
     (g) CONCATENATE with the normalized image volume (still on GPU, (1, *new_shape)
         fp32) on the channel axis, image FIRST, one-hot LAST:
         `vol2 = cp.ascontiguousarray(cp.concatenate([vol_normed, oh], axis=0))` →
         (2, *new_shape) fp32 C-contiguous. The one-hot channel is NEVER normalized
         (concat happens after the per-channel normalize of the IMAGE channels only).
     (h) emit `vol2` as the `preprocessed` output (SlideWindow is already
         config-driven: `load_inference_params` resolves `num_input_channels=2` for the
         cascade config — no SlideWindow changes).
   - Keep the 1-channel (non-cascade) path byte-for-byte the Plan 01 flow: when no
     `lowres_seg` input is declared, behavior is identical to Plan 01.
   - `compute()`: when the port exists,
     `seg_ht = op_input.receive(self.INPUT_LOWRES_SEG)`;
     `seg_tensor = torch.utils.dlpack.from_dlpack(seg_ht)`; `assert_on_gpu(seg_tensor)`;
     pass to `preprocess_image`. Record `"lowres_seg": true` and the seg shape in the
     timing record.
   - D-10 enforcement: raise `ValueError` if a float-valued (non-integer-dtype)
     `lowres_seg` arrives — the cascade input must be the argmax seg, never
     probabilities.
4. Regression: run `scripts/test_cascade_config.py` (all cases incl. the new ones) →
   exit 0; fullres-only E2E (Execution Environment) → exit 0 +
   `pixel_diff.py <out>/SEG testdata/ref_fullres_only/SEG` exit 0 (the 1-channel path is
   untouched in behavior).
  </action>
  <acceptance_criteria>
    - `grep -n "def _resample_seg_to_shape" my_app/operators/preprocess_operator.py` returns 1 line; the replica's docstring cites `default_resampling.py` / `resample_data_or_seg_to_shape`.
    - `grep -n "INPUT_LOWRES_SEG\|lowres_seg" my_app/operators/preprocess_operator.py` shows: the port constant, the conditional `spec.input` guarded by `self._params.previous_stage is not None`, the receive + `assert_on_gpu`, and the D-10 integer-dtype guard (grep "D-10").
    - `grep -n "cp.concatenate" my_app/operators/preprocess_operator.py` returns 1 line — the channel concat with the image volume FIRST (read the expression to confirm axis=0 and order).
    - `grep -n "convert_labelmap_to_one_hot\|resample_data_or_seg_to_shape" scripts/test_cascade_config.py` both return >= 1 line (tests call the VENDORED functions); the tests use `np.array_equal` (grep shows zero `allclose` in the seg-resample/one-hot tests).
    - `/tmp/monai-env/.venv/bin/python scripts/test_cascade_config.py` exits 0.
    - Fullres-only E2E exit 0 + pixel_diff exit 0 (1-channel regression intact).
  </acceptance_criteria>
  <verify>cd examples/apps/cchmc-nnunet-fast && /tmp/monai-env/.venv/bin/python scripts/test_cascade_config.py && ulimit -s unlimited && /tmp/monai-env/.venv/bin/python my_app -i /users/srv-mde/projects/monai-deploy-app-sdk/testdata/airway_input -m /users/srv-mde/projects/monai-deploy-app-sdk/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models -o /tmp/p2p3_e2e && /tmp/monai-env/.venv/bin/python scripts/pixel_diff.py /tmp/p2p3_e2e/SEG testdata/ref_fullres_only/SEG; echo "PIXEL_DIFF_EXIT=$?"</verify>
  <done>Cascade preprocess consumes a GPU argmax seg, resamples it on the CPU reference path with the exact seg kwargs, one-hots on GPU bit-exactly, and emits the 2-channel (image, one-hot) fp32 C-contiguous input with zero disk I/O — unit-tested against the vendored nnunetv2 reference; 1-channel behavior unchanged.</done>
</task>

<task type="auto">
  <name>Task 3: PostResampleOperator conditional lowres_seg output (argmax uint8, no CC, original orientation)</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py, examples/apps/cchmc-nnunet-fast/scripts/test_cascade_config.py</files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py (full file: `postresample_reference` (resample logits then softmax, pre-revert), `revert_crop_and_transpose_gpu` (zeros + bg=1 fill + inverse-permute), `postresample()`, `compute()`, and the NOTE that holoscan 4.2 calls setup() during Operator.__init__ — flags must be initialized BEFORE super().__init__, as ensemble_average_operator.py does with emit_averaged_probabilities)
    - .planning/phases/2-gpu-acceleration/02-RESEARCH.md (Pattern 4 implementation bullets: "Lowres fragment emits an extra port lowres_seg = argmax(...) as uint8 CUDA tensor (original DICOM orientation)... NO connected-component cleanup")
  </read_first>
  <action>
1. `PostResampleOperator` constructor: add `emit_lowres_seg: bool = False` and
   `emit_probabilities: bool = True` (the latter is consumed by Plan 04's conditional
   wiring — declare it now so Plan 04 doesn't re-edit this file's init pattern; the
   `emit_probabilities=False` case simply omits the existing `probabilities` output
   declaration). Initialize both as attributes BEFORE `super().__init__(*args, **kwargs)`
   (holoscan 4.2 init quirk — same pattern as `EnsembleAverageOperator`).
2. `setup()`: `spec.output(self.OUTPUT_PROBABILITIES)` only when
   `self._emit_probabilities`; `spec.output(self.OUTPUT_LOWRES_SEG)` (new constant
   `OUTPUT_LOWRES_SEG = "lowres_seg"`) only when `self._emit_lowres_seg`.
3. New helper `revert_crop_gpu(seg_crop, meta, device="cuda")` — like
   `revert_crop_and_transpose_gpu` but for an INTEGER seg: fill the original-shape
   volume with 0 (background), insert the cropped seg at `bbox_used_for_cropping`,
   inverse-permute with the same `transpose_backward` — returns a 3D uint8 CUDA tensor
   in original DICOM orientation (the same array order as `image.asnumpy()` — this is
   the orientation contract Task 2's preprocess consumes).
4. In `postresample()`, when `self._emit_lowres_seg`: after
   `postresample_reference` produces the pre-revert resampled probabilities numpy array,
   compute the seg BEFORE revert, on the same pre-revert data:
   `seg_crop = torch.argmax(torch.from_numpy(probabilities_cpu), dim=0).to(torch.uint8)`
   (post-softmax argmax == reference argmax-of-resampled-logits — softmax is monotone
   per voxel; document this equivalence in a comment with the D-09 citation).
   NO connected-component cleanup (the reference cascade input is pre-CC — verified).
   `seg_full = revert_crop_gpu(seg_crop, meta)`.
   Return `(probabilities_gpu, seg_full)` when emitting (otherwise keep the existing
   single-return behavior for non-lowres fragments).
5. `compute()`: when `self._emit_lowres_seg`: `assert_on_gpu(seg_full)`;
   `op_output.emit(to_holoscan_gpu_tensor(seg_full), self.OUTPUT_LOWRES_SEG)`; add
   `"lowres_seg_shape"` + `"lowres_seg_dtype": "uint8"` to the timing record.
6. Unit test in `scripts/test_cascade_config.py`: `test_revert_crop_gpu` — random
   uint8 cropped seg + a synthetic meta dict (bbox not touching borders, non-identity
   transpose_forward) → assert: output shape == shape_before_cropping, values outside
   the bbox slice == 0, values inside == the (inverse-permuted) input seg,
   `np.array_equal` against a hand-rolled numpy reference of the same fill/insert/permute
   steps.
7. Regression: `scripts/test_cascade_config.py` exit 0; fullres-only E2E exit 0 +
   pixel_diff exit 0 (default flags: emit_lowres_seg=False, emit_probabilities=True →
   byte-for-byte Phase 1 behavior).
  </action>
  <acceptance_criteria>
    - `grep -n "emit_lowres_seg\|emit_probabilities" my_app/operators/postresample_operator.py` shows both flags initialized BEFORE the `super().__init__` line (line-number check) and both gating the `spec.output` calls in setup().
    - `grep -n "def revert_crop_gpu" my_app/operators/postresample_operator.py` returns 1 line; `grep -n "argmax" my_app/operators/postresample_operator.py` shows the `torch.argmax(..., dim=0).to(torch.uint8)` expression with a comment citing D-09 / "no CC".
    - NO connected-component or `keep_largest` call anywhere in the lowres_seg path (grep `label\|connected\|keep_largest` in the new code — zero matches outside pre-existing modules).
    - `test_revert_crop_gpu` present in `scripts/test_cascade_config.py` using `np.array_equal`; `/tmp/monai-env/.venv/bin/python scripts/test_cascade_config.py` exits 0.
    - Fullres-only E2E exit 0 + `pixel_diff.py` exit 0 vs `testdata/ref_fullres_only` (Phase 1 behavior intact under default flags).
  </acceptance_criteria>
  <verify>cd examples/apps/cchmc-nnunet-fast && /tmp/monai-env/.venv/bin/python scripts/test_cascade_config.py && ulimit -s unlimited && /tmp/monai-env/.venv/bin/python my_app -i /users/srv-mde/projects/monai-deploy-app-sdk/testdata/airway_input -m /users/srv-mde/projects/monai-deploy-app-sdk/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models -o /tmp/p2p3_e2e_b && /tmp/monai-env/.venv/bin/python scripts/pixel_diff.py /tmp/p2p3_e2e_b/SEG testdata/ref_fullres_only/SEG; echo "PIXEL_DIFF_EXIT=$?"</verify>
  <done>The lowres fragment can emit the cascade input (argmax uint8, original orientation, zero disk I/O, no CC) while every non-cascade path behaves exactly as Phase 1/Plan 01.</done>
</task>

## Verification
- `scripts/test_cascade_config.py` exit 0: model-list semantics (6+ cases), seg-resample
  replica `np.array_equal` vs vendored nnunetv2, one-hot `np.array_equal` vs vendored
  `convert_labelmap_to_one_hot`, `revert_crop_gpu` numpy-reference test.
- Fullres-only E2E + pixel_diff exit 0 (regression).
- Port discipline: the `lowres_seg` input exists ONLY on cascade-config preprocess
  operators (asserted in tests via `load_preprocess_params(...).previous_stage`); no
  declared port can be left unwired by the Plan 04 flow table (Plan 04 re-verifies end
  to end).

## Success Criteria
- [ ] PIPE-03: `resolve_run_model_list` replicates reference semantics, config-generic (D-02), unit-tested on the real bundle
- [ ] PIPE-04 (operator level): GPU argmax-seg → CPU seg-resample → GPU one-hot → 2-channel input, zero disk I/O (D-09/D-10), bit-exact primitives vs vendored reference
- [ ] 2d stays blocked-on-model (D-01/D-03): no dummy model, no 2d-specific code (wiring is plans-driven)
- [ ] Regression: fullres-only pixel-exact gate still passes
- [ ] Commits made (small, imperative)
