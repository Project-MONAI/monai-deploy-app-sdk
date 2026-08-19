---
phase: 2-gpu-acceleration
plan: 01
subsystem: preprocessing
tags: [cupy, gpu-preprocess, transpose, crop, normalization, pixel-diff, gpu-residency, d-13]

requires:
  - phase: 1-core-pipeline (plans 01-05)
    provides: cchmc-nnunet-fast 5-operator GPU pipeline, pixel_diff.py, gpu_residency.py, ref_fullres_only gate oracle
provides:
  - "CuPy-backed preprocess_image: raw-volume transpose (PREP-01), crop-slice (PREP-04), element-wise per-channel normalize (PREP-02) on GPU in fp32 C-contiguous (D-12)"
  - "Z-score/CT mean-std reductions kept on the numpy reference path (CuPy reductions not bit-identical — RESEARCH Pitfall 4)"
  - "scipy/skimage resample unchanged on the CPU reference path (PREP-03, D-13) with C_CONTIGUOUS fp32 assert at the GPU->CPU boundary"
  - "Deliberately extended gpu_residency ALLOWED_TRANSFER_FILES (preprocess_operator.py, D-13 documented)"
  - "Gate evidence: .planning/phases/02-gpu-acceleration/plan01-gates/ (pixel_diff JSON, static/runtime/self-test residency, SR check)"
affects: [phase 2 plans 02-06 — cascade fragment will reuse this config-generic preprocess (2-channel input; one-hot concat happens after normalization)]

tech-stack:
  added: []
  patterns:
    - "One H2D of the raw volume via cp.array (deliberately NOT cp.from_dlpack — DLPack transfers buffer ownership)"
    - "CuPy transpose/slice return views — materialize with cp.ascontiguousarray at every boundary (D-12)"
    - "Reductions on numpy, element-wise math on CuPy (bit-identical for fp32 — RESEARCH verified rules)"
    - "np.float32(x) scalar construction matches numpy's weak-scalar casts exactly (Z-score + CTNormalization)"

key-files:
  created:
    - .planning/phases/02-gpu-acceleration/plan01-gates/pixel_diff_fullres.json
    - .planning/phases/02-gpu-acceleration/plan01-gates/gpu_residency_static.txt
    - .planning/phases/02-gpu-acceleration/plan01-gates/gpu_residency_runtime.txt
    - .planning/phases/02-gpu-acceleration/plan01-gates/gpu_residency_selftest.txt
    - .planning/phases/02-gpu-acceleration/plan01-gates/sr_check.txt
  modified:
    - examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py
    - examples/apps/cchmc-nnunet-fast/scripts/gpu_residency.py

key-decisions:
  - "CuPy port scope: transpose/crop/element-wise-normalize only; mask binary_fill_holes stays scipy (reference math), uint8 mask round-trips GPU<->CPU"
  - "Masked Z-score normalization keeps the reference's masked-assignment semantics on GPU (boolean-indexed element-wise op — bit-identical); inactive for this bundle (use_mask_for_norm=[False]) but config-generic"
  - "Residency allow-list extended deliberately with a D-13 reason string, no existing entry weakened (per 02-CONTEXT: 'updated deliberately, not silenced')"

patterns-established:
  - "Final-gate-only correctness (D-11): no per-op byte-identity checks — the CuPy port is proven by the fullres pixel-exact gate + residency gate"
  - "Every .get() entering _resample_to_shape is np.ascontiguousarray + explicit dtype/C_CONTIGUOUS assert (D-12, Phase 1's 1-ulp lesson)"

requirements-completed: [PREP-01, PREP-02, PREP-03, PREP-04]

duration: ~40min
completed: 2026-08-19
---

# Phase 2 Plan 01: CuPy Port of Preprocessing (transpose / crop / normalize) Summary

**PreprocessOperator's layout + element-wise stages (raw-volume transpose, crop-slice, per-channel Z-score/CT normalization) now run on GPU via CuPy in fp32 C-contiguous, with all reductions and the scipy resample deliberately kept on the numpy/scipy CPU reference path — the fullres-only E2E stays pixel-exact vs testdata/ref_fullres_only (99.99990% SEG byte-identity, 2 documented fp16↔fp32 boundary voxels, SR exact) and the residency gate passes with a deliberately documented D-13 allow-list entry.**

## Performance

- **Duration:** ~40 min (2 GPU E2E runs for Task 1 + 2 for Task 2 evidence dominate)
- **Tasks:** 2
- **Files modified:** 2 (preprocess_operator.py, gpu_residency.py) + 5 evidence files

## Gate Results (must-haves)

| must-have | Result |
|---|---|
| Transpose (PREP-01) + crop-slice (PREP-04) on GPU via CuPy | **PASS** — single H2D of the raw volume, on-device int→fp32 cast, `cp.ascontiguousarray` materialization at the transpose and crop; grep-verified (`import cupy as cp`, 3× `cp.ascontiguousarray`) |
| Element-wise normalize (PREP-02) on GPU, reductions on numpy | **PASS** — Z-score `mean`/`std` from the numpy array; element-wise `-`/`/` (and CT clip/sub/div) on CuPy in fp32; grep-verified: zero `.mean()`/`.std()` on any CuPy array |
| Resample stays on reference scipy/skimage CPU path (PREP-03, D-13) | **PASS** — `_resample_to_shape` byte-identical to Phase 1 (git diff: no changes inside the function); `np.ascontiguousarray(vol_c.get())` + `assert dtype==float32 and C_CONTIGUOUS` immediately before the call; D-13 comments at both transfer sites |
| fp32 + C-contiguous, no CuPy reductions (D-12) | **PASS** — all CuPy arrays fp32 (uint8 mask only); materialized via `cp.ascontiguousarray` at every boundary |
| Fullres-only E2E pixel-exact vs testdata/ref_fullres_only (D-11 final gate) | **PASS** — E2E exit 0, SEG/SR/SC written; `pixel_diff.py`: **99.99990% byte-identity (2,097,150/2,097,152 bytes), 2 differing voxels / 16,777,216, IoU 0.999142** — the same documented fp16↔fp32 argmax-boundary class as Phase 1 (Phase 1 measured 99.99986%/3 voxels; reference itself is run-to-run deterministic). SR exact: `Airway Volume: 1 mL` in both (0.1% tolerance, TEST-002) |
| gpu_residency.py passes with preprocess_operator.py deliberately allow-listed | **PASS** — static: `preprocess_operator.py … [ALLOWED]`, RESULT PASS, exit 0; runtime E2E: PASS, exit 0 (1 postprocess boundary `.cpu()`, 3 resample-path transfers, 0 illegal); self-test: PASS (injected illegal `.cpu()` flagged). Entry added with the D-13 reason string; the three pre-existing entries untouched |

## Accomplishments

- **Task 1** (`215b66a`): rewrote `preprocess_image`'s compute path to the CuPy flow — (1) ONE H2D: `np.ascontiguousarray(arr.T)` → `cp.array(raw, dtype=cp.float32)` (plain CUDA-array-interface copy, deliberately NOT `cp.from_dlpack`) → `cp.ascontiguousarray(vol[None, ...].transpose(0, 3, 2, 1))`; (2) reference `tf` transpose as a view; (3) per-channel nonzero OR on GPU as uint8, D2H, scipy `binary_fill_holes` + unchanged `_get_bbox_from_mask`; (4) crop on GPU, materialized; (5) per-channel normalize: numpy reductions (incl. masked path with reference's masked-assignment semantics via GPU boolean indexing) + element-wise CuPy sub/div/clip with `np.float32` scalars; (6) unchanged scipy `_resample_to_shape` after `C_CONTIGUOUS` fp32 assert; (7) `compute()` unchanged (torch → DLPack holoscan Tensor). All `properties` keys identical (verified via the timing record: bbox `[[0,256],[0,256],[0,256]]`, `shape_before_cropping [256,256,256]`, `new_shape [255,256,255]`). `preprocess_reference` (pure CPU) kept untouched as the documented fallback.
- **Task 2** (`303a248`): `gpu_residency.py` `ALLOWED_TRANSFER_FILES` gained exactly one entry — `"preprocess_operator.py": "D-13 accepted GPU<->CPU round-trip: one H2D of the raw volume, D2H of the uint8 nonzero mask, D2H fp32 for the Z-score reductions (numpy reference path) and for the scipy CPU resample; returns to CUDA in-operator"` — no existing entry removed or weakened, no wildcard handling. All three residency modes + the pixel-exact gate + SR check ran and passed; evidence committed under `.planning/phases/02-gpu-acceleration/plan01-gates/`.

## Task Commits

1. **Task 1: CuPy port of transpose/crop/normalize** — `215b66a` (feat)
2. **Task 2: residency allow-list (D-13) + gate evidence** — `303a248` (test)

**Plan metadata:** (docs commit at the end of this plan)

## Files Created/Modified

- `examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py` — CuPy `preprocess_image` (H2D/transpose/crop/normalize on GPU; numpy reductions; D-13 comments; C_CONTIGUOUS assert); module docstring updated; `preprocess_reference`/`_resample_to_shape`/`_normalize_channel` untouched
- `examples/apps/cchmc-nnunet-fast/scripts/gpu_residency.py` — one new `ALLOWED_TRANSFER_FILES` entry with D-13 justification
- `.planning/phases/02-gpu-acceleration/plan01-gates/` — pixel_diff JSON, static/runtime/self-test residency transcripts, SR check

## Deviations from Plan

None — plan executed exactly as written. (One plan typo noted, not a deviation: Task 2's `pixel_diff.py` path `testdata/ref_fullres_only/SEG` is relative to the repo root, not the app root — ran with the absolute path as the plan's intent implies.)

## Self-Check: PASSED

All 8 created/modified files exist; commits `215b66a` and `303a248` are in git history.
