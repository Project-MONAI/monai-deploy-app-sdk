---
phase: "2-gpu-acceleration"
plan: "01"
type: "execute"
wave: 1
depends_on: []
files_modified:
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py"
  - "examples/apps/cchmc-nnunet-fast/scripts/gpu_residency.py"
autonomous: true
requirements: [PREP-01, PREP-02, PREP-03, PREP-04]
must_haves:
  truths:
    - "Transpose (PREP-01) and crop-slice (PREP-04) of the input volume run on GPU via CuPy"
    - "Per-channel normalization element-wise ops (PREP-02) run on GPU via CuPy, with mean/std reductions computed by the existing numpy code"
    - "Resampling stays on the reference scipy/scikit-image CPU path (PREP-03, D-13) with the accepted GPU<->CPU round-trip"
    - "All CuPy ops stay fp32 and C-contiguous (D-12); no CuPy reductions anywhere"
    - "fullres-only E2E on the airway study remains pixel-exact vs testdata/ref_fullres_only (final-gate-only correctness per D-11)"
    - "gpu_residency.py passes with preprocess_operator.py added to the allow-list deliberately (not silenced)"
  artifacts:
    - path: "examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py"
      provides: "CuPy transpose/crop/normalize + unchanged scipy resample"
      contains: "import cupy as cp"
    - path: "examples/apps/cchmc-nnunet-fast/scripts/gpu_residency.py"
      provides: "updated ALLOWED_TRANSFER_FILES with D-13 justification"
      contains: "preprocess_operator.py"
  key_links:
    - from: "preprocess_image (GPU path)"
      to: "_resample_to_shape (scipy CPU)"
      via: ".get() -> np.ascontiguousarray -> assert C_CONTIGUOUS fp32 before the resample call"
      pattern: "C_CONTIGUOUS"
    - from: "preprocess_operator.py transfers"
      to: "gpu_residency.py ALLOWED_TRANSFER_FILES"
      via: "file-name key with a documented D-13 reason string"
      pattern: "D-13"
---

# Phase 2 Plan 01: CuPy Port of Preprocessing (transpose / crop / normalize)

## Objective
- **What:** Port the element-wise and layout preprocessing ops (transpose, crop-slice,
  per-channel normalization) from numpy to CuPy in `PreprocessOperator`, keeping the
  scipy/skimage resample and ALL reductions on the verified numpy/CPU reference path.
- **Why:** Phase 2's CPU-bound preprocessing goes to GPU (tasks 2.1–2.3) while preserving
  the Phase 1 pixel-exact invariant. Correctness strategy is final-gate-only (D-11): no
  per-op byte-identity checks — the port is proven by the existing fullres pixel-exact
  gate and the residency gate.
- **Output:** CuPy-backed `preprocess_image`; deliberately updated residency allow-list;
  a passing fullres-only E2E + pixel-exact + residency verification.

## Execution Environment (from AGENTS.md / STATE.md — applies to every task)

- Python: `/tmp/monai-env/.venv/bin/python` only. No new venvs. No PyPI nnunetv2 (vendored
  editable at `./nnUNet/`).
- Run the fast app from its app root (the `my_app` editable-install name collision is live):
  ```bash
  cd examples/apps/cchmc-nnunet-fast
  ulimit -s unlimited
  /tmp/monai-env/.venv/bin/python my_app \
    -i /users/srv-mde/projects/monai-deploy-app-sdk/testdata/airway_input \
    -m /users/srv-mde/projects/monai-deploy-app-sdk/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models \
    -o <scratch-out-dir>
  ```
- Pixel gate tool: `/tmp/monai-env/.venv/bin/python scripts/pixel_diff.py <new_SEG> <ref_SEG>`
  (default `--min-identity 99.9 --max-diff-voxels 10000`; Phase 1 measured 99.99986%).
- Commit after each major change (short imperative messages).

## Context

@.planning/PROJECT.md
@.planning/STATE.md
@.planning/phases/02-gpu-acceleration/02-CONTEXT.md
@.planning/phases/02-gpu-acceleration/02-RESEARCH.md
@examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py
@examples/apps/cchmc-nnunet-fast/scripts/gpu_residency.py

Locked decisions in force: D-11 (final-gate-only correctness), D-12 (fp32 + C-contiguous
throughout; Phase 1 proved a single contiguity slip → 1-ulp divergence at ~16M voxels),
D-13 (resampling stays on the reference scipy/CPU path; the resulting GPU→CPU→GPU
round-trip is expected and accepted, ~64 MB fp32 per study). RESEARCH.md Pitfall 4:
`cp.mean`/`cp.std` are NOT bit-identical to numpy (reduction order) — reductions must stay
numpy. RESEARCH.md "bit-exact rules" (measured 2026-08-19): element-wise `-`, `/`, clip,
comparisons, `(x==1).astype(float32)`, transpose+ascontiguousarray ARE bit-identical to
numpy for fp32.

The Phase 1 app is still the single-config `3d_fullres` DAG (app.py unchanged by this
plan) — the fullres-only E2E run here is just that DAG with the new CuPy preprocess.

## Tasks

<task type="auto">
  <name>Task 1: Port transpose / crop / normalize to CuPy in PreprocessOperator</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py</files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py (current CPU path; the functions `_create_nonzero_mask`, `_get_bbox_from_mask`, `_compute_new_shape`, `_determine_do_sep_z_and_axis`, `_normalize_channel`, `_resample_to_shape`, `preprocess_reference`, `preprocess_image`, `compute` — keep their public signatures and the `properties` dict keys)
    - .planning/phases/02-gpu-acceleration/02-RESEARCH.md (Pattern 2 "CuPort of transpose/crop/normalize" — the exact data flow; "bit-exact rules" 1–4; Code Examples "Verified bit-exact CuPy element-wise ops")
  </read_first>
  <action>
Rewrite the compute path of `preprocess_image` (and the helpers it needs) so the data flow
becomes, per study, keeping the EXACT transformation sequence of today (same transpose
axes, same crop, same normalization order, same resample call):

1. ONE H2D of the raw volume (new): `raw = np.ascontiguousarray(arr.T, dtype=arr.dtype)`
   then `vol = cp.array(raw, dtype=cp.float32)` then
   `vol = cp.ascontiguousarray(vol[None, ...].transpose(0, 3, 2, 1))`
   (mirrors today's `np.ascontiguousarray(arr.T)[None, ...].transpose(0, 3, 2, 1)`;
   `cp.array` with `dtype` does the integer→fp32 cast on-device; a plain copy via
   `__cuda_array_interface__`-style H2D — do NOT use `cp.from_dlpack` here to avoid the
   DLPack-ownership pitfall).
2. Apply the reference `tf` transpose as a VIEW (as today's `preprocess_reference` does):
   `vol_t = vol.transpose(0, *[i + 1 for i in tf])` — do not materialize yet.
3. Mask + bbox (PREP-04; the mask MATH stays scipy-CPU because the reference's
   `binary_fill_holes` is scipy): on GPU compute the per-channel OR exactly as
   `_create_nonzero_mask` does (`mask = vol_t[0] != 0; for c in 1..: mask |= vol_t[c] != 0`)
   as uint8, `mask_np = mask.get()`, then `binary_fill_holes(mask_np)` (scipy, unchanged)
   and `bbox = _get_bbox_from_mask(...)` (unchanged CPU function). Record
   `properties["shape_before_cropping"]`, `properties["bbox_used_for_cropping"]` exactly
   as today (from the post-tf-transpose, pre-crop shape).
4. Crop on GPU (PREP-04):
   `vol_c = cp.ascontiguousarray(vol_t[(slice(None),) + slicer])`
   (materialize — CuPy transpose/slice return views; D-12). Record
   `shape_after_cropping_and_before_resampling`, `new_shape` (via the unchanged
   `_compute_new_shape`), `original_spacing`, `target_spacing`, `transpose_forward`
   exactly as today.
5. Normalize on GPU (PREP-02), reductions on numpy (Pitfall 4, MANDATORY):
   `vol_np = vol_c.get()` (D2H fp32, ~64 MB — D-13 accepted). Compute the stats with the
   SAME numpy expressions as the current `_normalize_channel`:
   - ZScoreNormalization (the active scheme for this bundle): `mean = vol_np.mean()`,
     `std = vol_np.std()`, `eps = 1e-8`. For masked norm (`use_mask_for_norm` channel):
     `mean = vol_np[m].mean()`, `std = vol_np[m].std()` with `m` = the sliced numpy
     nonzero mask broadcast to the channel.
   Then the element-wise ops on GPU in fp32:
   `vol_c = (vol_c - cp.asarray(np.float32(mean))) / cp.asarray(np.float32(max(std, eps)))`
   (build the scalars as `np.float32` exactly as the numpy path produces them — the
   reference does `image -= mean; image /= (max(std, eps))` in float32).
   Keep the full scheme dispatch (`ZScoreNormalization` / `CTNormalization` with
   `np.clip` + plan props / `NoNormalization`) so the operator stays config-generic; the
   CTNormalization clip/sub/divide are element-wise CuPy ops with the plan constants.
   The one-hot cascade channel (Phase 2, later plan) must NEVER be normalized — this
   stays true because normalization runs before the channel concat.
6. Resample (PREP-03, UNCHANGED function): `vol_out = np.ascontiguousarray(vol_c.get())`
   then BEFORE the call assert
   `assert vol_out.dtype == np.float32 and vol_out.flags['C_CONTIGUOUS']`
   then `vol_out = _resample_to_shape(vol_out, new_shape, original_spacing, params.spacing, params)`.
   Do NOT modify `_resample_to_shape` / `resample_probabilities_to_shape` (scipy
   `map_coordinates` + `skimage.transform.resize` must stay byte-for-byte the Phase 1 code).
7. `compute()` is unchanged: `torch.as_tensor(vol_out, dtype=torch.float32).to("cuda")`
   → `to_holoscan_gpu_tensor` → emit `preprocessed` + `preprocessed_meta`.

Keep `preprocess_reference` (the pure-CPU function) in the module untouched — it is the
documented CPU path and a fallback. All CuPy arrays in the new path are fp32 (uint8 for
the mask only). Every `.get()` result that enters the resample or leaves the operator
must be C-contiguous (D-12). Add an explanatory comment at the H2D/D2H sites citing D-13.
  </action>
  <acceptance_criteria> (grep commands run from `examples/apps/cchmc-nnunet-fast`)
    - `grep -n "import cupy as cp" my_app/operators/preprocess_operator.py` returns 1 line.
    - `grep -c "cp.ascontiguousarray" my_app/operators/preprocess_operator.py` >= 3 (raw-volume transpose materialize, tf/crop materialize, no other requirement).
    - NO CuPy reductions: `grep -nE "vol[a-z_]*\.mean\(\)|vol[a-z_]*\.std\(\)" my_app/operators/preprocess_operator.py` returns only matches on `vol_np` (numpy) — zero matches on a CuPy array (any `cp.` array).
    - `_resample_to_shape` body is byte-identical to the Phase 1 version (`git diff --no-index` or `git diff HEAD~0` inspection: the function definition block is unchanged; it still calls `resize(..., mode="edge", anti_aliasing=False)` and `map_coordinates(...)`).
    - `grep -n "C_CONTIGUOUS" my_app/operators/preprocess_operator.py` shows the new assert immediately before the `_resample_to_shape(` call.
    - `properties` dict emitted by the new path contains exactly the same keys as before: `shape_before_cropping`, `bbox_used_for_cropping`, `shape_after_cropping_and_before_resampling`, `new_shape`, `original_spacing`, `target_spacing`, `transpose_forward` (verified by a fullres-only run logging the timing record that embeds bbox/shape).
    - Fullres-only E2E run (command in Execution Environment) exits 0 and writes SEG/SR/SC.
  </acceptance_criteria>
  <verify>/tmp/monai-env/.venv/bin/python -c "import sys; sys.path.insert(0, 'examples/apps/cchmc-nnunet-fast'); import my_app.operators.preprocess_operator" && cd examples/apps/cchmc-nnunet-fast && ulimit -s unlimited && /tmp/monai-env/.venv/bin/python my_app -i /users/srv-mde/projects/monai-deploy-app-sdk/testdata/airway_input -m /users/srv-mde/projects/monai-deploy-app-sdk/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models -o /tmp/p2p1_e2e_fullres; echo "EXIT=$?"; ls /tmp/p2p1_e2e_fullres/SEG /tmp/p2p1_e2e_fullres/SR /tmp/p2p1_e2e_fullres/SC</verify>
  <done>Transpose, crop-slice, and element-wise normalization run on CuPy (fp32, C-contiguous); Z-score mean/std are computed by the numpy code path; the scipy resample is untouched; fullres-only E2E exits 0.</done>
</task>

<task type="auto">
  <name>Task 2: Update gpu_residency.py allow-list deliberately + run the pixel-exact and residency gates</name>
  <files>examples/apps/cchmc-nnunet-fast/scripts/gpu_residency.py, .planning/phases/02-gpu-acceleration/ (gate evidence)</files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/scripts/gpu_residency.py (the `ALLOWED_TRANSFER_FILES` dict at line ~81, `run_static`, `run_runtime`, `run_self_test`, and `main` — use the script's own CLI as written there)
    - .planning/phases/02-gpu-acceleration/02-CONTEXT.md (code_context: "the test's allow-list must be updated deliberately, not silenced")
  </read_first>
  <action>
1. Add ONE entry to `ALLOWED_TRANSFER_FILES` in gpu_residency.py:
   `"preprocess_operator.py": "D-13 accepted GPU<->CPU round-trip: one H2D of the raw "
                               "volume, D2H of the uint8 nonzero mask, D2H fp32 for the "
                               "Z-score reductions (numpy reference path) and for the "
                               "scipy CPU resample; returns to CUDA in-operator"`
   Do NOT remove or weaken any existing entry; do NOT add wildcard or regex handling;
   the exactly-once boundary transfer in `postprocess_operator.py` remains the only
   final-stage `.cpu()`.
2. Run the static scan: `/tmp/monai-env/.venv/bin/python scripts/gpu_residency.py --static`
   (from the app root; exact flags per the script's `main`) — `preprocess_operator.py`
   must report ALLOWED, everything else as before, exit 0.
3. Run the runtime/self-test path per the script's CLI (`--runtime` / `--self-test`
   with `--input testdata/airway_input --model <models dir> --output /tmp/p2p1_residency`).
4. Run the pixel-exact gate on the Task 1 output:
   `/tmp/monai-env/.venv/bin/python scripts/pixel_diff.py /tmp/p2p1_e2e_fullres/SEG testdata/ref_fullres_only/SEG`
   (run from the app root; default tolerances).
5. Compare SR: the airway-volume SR text must still read "Airway Volume: 1 mL" (0.1%
   tolerance per TEST-002).
6. Save the gate evidence (pixel_diff JSON via `--json`, residency output, SR value) into
   `.planning/phases/02-gpu-acceleration/plan01-gates/` and commit with an imperative
   message.
  </action>
  <acceptance_criteria>
    - `grep -n "preprocess_operator.py" scripts/gpu_residency.py` matches inside `ALLOWED_TRANSFER_FILES` with a comment containing the string `D-13`.
    - The three pre-existing allow-list keys `postresample_operator.py`, `postprocess_operator.py`, `sc_overlay.py` are still present, unmodified (`grep -n` each).
    - `--static` run exits 0 and its output classifies `preprocess_operator.py` as `ALLOWED` (not `VIOLATION`).
    - `pixel_diff.py` exits 0: reported identity >= 99.99% (Phase 1 measured 99.99986%; differing voxels only in the documented fp16↔fp32 argmax-boundary class, <= a handful of voxels, far under `--max-diff-voxels 10000`).
    - SR airway volume matches the reference within 0.1% ("Airway Volume: 1 mL").
    - Evidence files exist under `.planning/phases/02-gpu-acceleration/plan01-gates/` and the commit exists in `git log --oneline -3`.
  </acceptance_criteria>
  <verify>cd examples/apps/cchmc-nnunet-fast && /tmp/monai-env/.venv/bin/python scripts/gpu_residency.py --static && /tmp/monai-env/.venv/bin/python scripts/pixel_diff.py /tmp/p2p1_e2e_fullres/SEG testdata/ref_fullres_only/SEG; echo "PIXEL_DIFF_EXIT=$?"</verify>
  <done>Residency allow-list deliberately extended for D-13; static + runtime residency pass; fullres SEG is pixel-exact vs the existing fullres-only reference (D-11 final gate) — the CuPy port is proven without per-op checks.</done>
</task>

## Verification
- `grep`-verifiable CuPy presence + numpy-only reductions (Task 1 criteria).
- fullres-only E2E exit 0 + `pixel_diff.py` exit 0 vs `testdata/ref_fullres_only`.
- `gpu_residency.py --static` and runtime modes exit 0 with the new ALLOWED entry.
- SR within 0.1%.

## Success Criteria
- [ ] PREP-01 transpose on CuPy, PREP-04 crop-slice on CuPy, PREP-02 element-wise normalize on CuPy
- [ ] PREP-03 resample unchanged on the scipy/CPU reference path with the D-13 round-trip documented at the transfer sites
- [ ] fp32 + C-contiguous invariant asserted (D-12); zero CuPy reductions
- [ ] Pixel-exact fullres gate passes (D-11 strategy: final gate only)
- [ ] Residency gate passes with a deliberate, documented allow-list edit
- [ ] Commits made (small, imperative)
