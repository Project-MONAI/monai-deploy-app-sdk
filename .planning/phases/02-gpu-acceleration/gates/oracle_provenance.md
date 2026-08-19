# Per-Config Reference Oracle Provenance (Phase 2 Plan 05, D-05)

Generated 2026-08-19 with `.planning/scripts/reference_fullres_run.py`
(reference app `cchmc_nnunet_fifteen_ckpt_app` UNMODIFIED — verified via
`git status examples/apps/cchmc_nnunet_fifteen_ckpt_app` = clean), venv
`/tmp/monai-env/.venv`, `ulimit -s unlimited`, A100-SXM4-40GB, input
`testdata/airway_input` (256-slice airway MR study), model bundle
`examples/apps/cchmc_nnunet_fifteen_ckpt_app/models`
(`MRI_NICU-Airway_TRAINv2`).

| Oracle | `--config` (pinned model_list) | run_model_list (effective) | ensemble_model_list | wall time | post-CC segment voxels | SEG/SR/SC |
|---|---|---|---|---|---|---|
| `testdata/ref_lowres_only` | `3d_lowres` | `['3d_lowres']` | `['3d_lowres']` (self-ensemble — see below) | 74 s | 2404 | yes |
| `testdata/ref_cascade_only` | `3d_lowres,3d_cascade_fullres` | `['3d_lowres', '3d_cascade_fullres']` | `['3d_cascade_fullres']` | 131 s | 2519 | yes |

Both runs exit 0. Voxel counts decoded with the REFERENCE_RUN_GUIDE
"Quick SEG parity check" snippet (1-bit little-endian pixel data).

## Sanity expectations (per the plan)

- lowres-only (2404) ≠ cascade-only (2519) ≠ fullres-only (2330, existing
  `testdata/ref_fullres_only`) — different configurations, as expected.
- cascade-only (2519) is in the same ballpark as the reference full-bundle
  `testdata/current_output` (2447) — as expected (the cascade probability
  map is the bundle's dominant ensembled member).

## Deviation: the lowres-only oracle uses the documented self-ensemble fallback

The reference app RAISES on `model_list=['3d_lowres']`
(`ValueError: At least one non-auxiliary model configuration is required
for ensemble inference` — `nnunet_seg_operator.py:96-98`): with lowres as
the only config the ensemble would be empty. The fast app documents a
self-ensemble fallback for exactly this case (Phase 2 Plan 04 decision:
lowres-only is runnable, ensemble=run, both unit-tested). For the gate to
compare like-for-like, the reference oracle must use the SAME semantics.

`reference_fullres_run.py` therefore instantiates a harness-side SUBCLASS
of the reference operator (the example app itself is unmodified): it pins
`model_list` as before, and for auxiliary-only lists it catches the
reference ValueError and replays the plain tail of the original `__init__`
with `ensemble_model_list = run_model_list` (self-ensemble). Everything
downstream — model loading, inference, `EnsembleProbabilitiesToSegmentation`,
postprocessing.pkl, `KeepLargestConnectedComponentd`, and the SEG/SR/SC
writing — is the reference's own untouched code path.

Two harness fixes were needed beyond the plan text (the plan only
anticipated the cascade-only crash, not this ValueError):

1. The replay initially forwarded the full kwargs dict to the holoscan
   `Operator` parent constructor; the original `__init__` consumes the
   named params (app_context, model_path, output_folder, output_labels,
   model_list, model_name, save_probabilities, save_files) before its own
   `super()` call, so the parent must receive only the leftovers (e.g.
   `name=`) or pybind fails with "python object could not be converted to
   Arg". Fixed by popping the named params before the parent call.

## Cascade-only oracle semantics (D-07, live-probed 2026-08-19)

`--config 3d_cascade_fullres` alone CRASHES the reference
(`RuntimeError: ... tmp/3d_lowres/Img_in_context.nii.gz does not exist`) —
the reference cascade reads the previous stage's exported `.nii.gz`, which
exists only if lowres ACTUALLY RAN; the reference list logic reorders but
does not auto-insert. Pinning `['3d_lowres', '3d_cascade_fullres']` runs
lowres (auxiliary, non-ensembled), cascade consumes its export, and the
reference's own `ensemble_model_list` excludes lowres → the SEG is
cascade-only, exactly matching the fast app's
`HOLOSCAN_MODEL_LIST=3d_cascade_fullres` run (auto-insert +
`ensemble=['3d_cascade_fullres']`).

## Tracking convention

Per the repo's existing convention, the oracle bytes are NOT committed —
`testdata/current_output/` and `testdata/ref_fullres_only/` are gitignored
(lines 277-278 of `.gitignore`); `testdata/ref_lowres_only/` and
`testdata/ref_cascade_only/` were added to `.gitignore` to match.
Provenance (this file) + the voxel counts/checksums in
`02-GATE-RESULTS.json` record the oracles; regenerate any time via
`REFERENCE_RUN_GUIDE.md` + the two commands above.
