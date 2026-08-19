---
phase: 2-gpu-acceleration
plan: 05
subsystem: pixel-exact-gates
tags: [per-config-oracles, pixel-exact, bundle-gate, sr-compare, gpu-residency, d-05, d-06, d-07, d-08, d-13, d-01, d-03, test-01, test-005]

requires:
  - phase: 2-gpu-acceleration (plan 04)
    provides: the 4 runnable HOLOSCAN_MODEL_LIST configurations (fullres / lowres / cascade / bundle), run_model_list= / ensemble_model_list= log lines, the multi-fragment DAG the residency runtime check exercises
  - phase: 01-core-pipeline (plan 05)
    provides: pixel_diff.py + gpu_residency.py gate tools, the ref_fullres_only oracle pattern, the documented fp16↔fp32 3-voxel boundary class
  - phase: 01-core-pipeline (plan 04)
    provides: the gate-calibration finding that testdata/current_output is the reference FULL-BUNDLE run (2447 voxels) — the D-06 bundle-gate target

provides:
  - "testdata/ref_lowres_only + testdata/ref_cascade_only: the two missing per-config reference oracles (D-05), generated with the reference app UNMODIFIED; provenance + sha256 in gates/oracle_provenance.md and 02-GATE-RESULTS.json"
  - "phase2_gate.py: deterministic 4-row gate runner (fast-app E2E + pixel_diff + SR volume + model-list assertion) + residency static/runtime + bundle-ensembles sanity, writing the combined JSON"
  - "gates/02-GATE-RESULTS.json: machine-readable evidence — all 4 gates PASS (fullres 99.99986% / 3 documented boundary voxels; lowres, cascade, bundle 100.00000% / 0 differing voxels), SR 0.0% delta everywhere, residency static+runtime PASS, TEST-005-2d + TEST-01-corpus deviations recorded for VERIFICATION.md"

affects: [phase 2 plan 06 — benchmark/profiling plan; VERIFICATION.md — consumes 02-GATE-RESULTS.json deviations (TEST-005 met-with-deviation, TEST-01 dev-corpus deviation)]

tech-stack:
  added: []
  patterns:
    - "The reference harness can reproduce the fast app's documented self-ensemble fallback (ensemble = run) for auxiliary-only model lists via a harness-side subclass: catch the reference ValueError mid-__init__, replay the plain tail of the original __init__ (the raise fires AFTER run_model_list is set), pop the named params before the holoscan parent call — everything downstream (inference, ensemble, CC, SEG/SR/SC writing) is the reference's untouched code"

key-files:
  created:
    - .planning/scripts/phase2_gate.py
    - .planning/phases/02-gpu-acceleration/gates/02-GATE-RESULTS.json
    - .planning/phases/02-gpu-acceleration/gates/oracle_provenance.md
    - testdata/ref_lowres_only (gitignored per testdata oracle convention)
    - testdata/ref_cascade_only (gitignored per testdata oracle convention)
  modified:
    - .planning/scripts/reference_fullres_run.py
    - .gitignore

key-decisions:
  - "The lowres-only reference oracle uses the fast app's documented self-ensemble semantics (ensemble=run), not the reference's ValueError: the reference app raises on model_list=['3d_lowres'] (empty ensemble) and cannot produce a lowres-only SEG at all; the gate compares like-for-like only if both sides use the same semantics. Implemented as a harness-side subclass — the example app stays unmodified (Rule 3, plan assumed only the cascade-only crash)"
  - "Oracle bytes follow the repo's existing testdata convention: NOT committed (testdata/current_output/ and testdata/ref_fullres_only/ are gitignored); ref_lowres_only/ + ref_cascade_only/ added to .gitignore; provenance (model_lists, wall times, voxel counts) recorded in gates/oracle_provenance.md and per-oracle SEG sha256 in 02-GATE-RESULTS.json"

requirements-completed: [TEST-01, TEST-005]
requirements-deferred:
  - "TEST-005 2d config: blocked-on-model (D-01/D-03) — recorded as met-with-deviation in the gate JSON for VERIFICATION.md; a real 2d model is a test, not a code change (D-04)"
  - "TEST-01 >=5-CT-study re-run: deferred final gate (single airway dev-corpus deviation, carried from Phase 0/1)"

deviations:
  - "Rule 3 (blocking, plan-level gap): the plan's correction anticipated the cascade-only reference crash but NOT that the reference also raises on lowres-only ('At least one non-auxiliary model configuration is required for ensemble inference' — nnunet_seg_operator.py:96-98). Generated the lowres-only oracle with a harness-side subclass that reproduces the fast app's documented self-ensemble fallback (Phase 2 Plan 04 decision); reference app unmodified. Sub-finding inside that fix: the original __init__ consumes its named kwargs before its super() call, so the replay must pop app_context/model_path/output_folder/output_labels/model_list/model_name/save_probabilities/save_files before the holoscan Operator parent call or pybind fails ('python object could not be converted to Arg')"
  - "Rule 1 (bug, gate-script-internal): the refactored phase2_gate.py left the main() tail (sanity + residency + JSON + summary) unreachable after an early return — the first two end-to-end runs exited 0 having done nothing past the last gate row; caught by the missing JSON/report, fixed by restructure"
  - "Oracle data not committed: the plan said 'follow what testdata/ref_fullres_only and testdata/current_output are currently tracked as' — they are gitignored, so the new oracles are gitignored too; checksum + provenance recorded instead (see key-decisions)"

metrics:
  duration: ~65min
  tasks: 2/2
  commits: 9
  gates: "4/4 pixel gates PASS — fullres_only 99.99986% byte-identity / 3 differing voxels (the documented fp16↔fp32 boundary class, IoU 0.998714, fast 2331 vs oracle 2330 voxels); lowres_only 100.00000% / 0 (2404=2404); cascade_only 100.00000% / 0 (2519=2519); bundle 100.00000% / 0 (2447=2447 vs testdata/current_output, D-06). SR airway volume 0.0% delta (1.0 mL = 1.0 mL) on all 4 rows. Residency static + runtime PASS (multi-fragment bundle config). Sanity: bundle SEG differs from fullres-only by 382 voxels (the bundle actually ensembles)"

completed: 2026-08-19
---

# Phase 2 Plan 05: Per-Config Reference Oracles + Pixel-Exact Gates (incl. Bundle Gate) Summary

**Both missing per-config reference oracles now exist (generated with the reference app unmodified), and all four pixel-exact gates PASS — fullres-only at the documented 99.99986% (3 fp16↔fp32 boundary voxels) and lowres-only, cascade-only, and the FINAL BUNDLE gate vs `testdata/current_output` at 100.00000% byte-identity with zero differing voxels — with SR airway volume exact (0.0% delta, well inside the 0.1% bar) on every row and GPU residency green (static + runtime) in the multi-fragment bundle configuration; TEST-01 and TEST-005 (met-with-deviation) are satisfied on the dev corpus with the 2d/corpus deviations recorded in the gate JSON for VERIFICATION.md.**

## What was built

### Task 1 — model-list reference harness + both per-config oracles (commits 8adeef8, b423800, 4a956e4)

- `reference_fullres_run.py` `--config` now accepts a COMMA-SEPARATED list (`model_list = [c.strip() for c in args.config.split(',') if c.strip()]`); default stays `3d_fullres` so the Phase 1 gate command is unchanged; docstring documents both Phase 2 invocations incl. WHY lowres must be in the cascade-only list (the reference cascade reads the previous stage's exported `.nii.gz` — pinning `['3d_cascade_fullres']` alone crashes, live-probed).
- **Lowres-only oracle (`testdata/ref_lowres_only`, 2404 post-CC voxels, 74 s):** the reference app RAISES on `model_list=['3d_lowres']` (empty ensemble — the same ValueError Plan 03 replicated and Plan 04 worked around in the fast app with the documented self-ensemble fallback). The harness now instantiates a SUBCLASS of the reference operator: it pins `model_list` as before and, for auxiliary-only lists, catches the ValueError and replays the plain tail of the original `__init__` with `ensemble_model_list = run_model_list` (ensemble=run — exactly the fast-app semantics the gate compares against). Model loading, inference, `EnsembleProbabilitiesToSegmentation`, `KeepLargestConnectedComponentd`, and the SEG/SR/SC writing are the reference's own untouched code. `git status examples/apps/cchmc_nnunet_fifteen_ckpt_app` = clean.
- **Cascade-only oracle (`testdata/ref_cascade_only`, 2519 post-CC voxels, 131 s):** `--config 3d_lowres,3d_cascade_fullres` — lowres runs (auxiliary, non-ensembled), cascade consumes its export, the reference's own `ensemble_model_list` excludes lowres → cascade-only SEG, matching D-07 and the fast app's `HOLOSCAN_MODEL_LIST=3d_cascade_fullres` run by construction.
- Provenance in `gates/oracle_provenance.md`: model_lists, wall times, decoded post-CC voxel counts (lowres 2404 ≠ cascade 2519 ≠ fullres 2330 — different configs, as expected; cascade in the same ballpark as the bundle's 2447, as expected), the self-ensemble deviation, and the gitignore convention note.
- Oracle bytes follow the repo convention: `testdata/ref_lowres_only/` + `testdata/ref_cascade_only/` added to `.gitignore` (matching `current_output/`, `ref_fullres_only/`); sha256 + voxel counts recorded in the gate JSON.

### Task 2 — phase2_gate.py + 02-GATE-RESULTS.json (commits bf0c39f, 1f2aa69, e53e026, c5cc593, 07fde68, 257fdb4)

- `.planning/scripts/phase2_gate.py`: deterministic gate runner. Per row: (1) fast-app E2E subprocess (venv python, 32 MB stack rlimit per `baseline_benchmark.py`, `HOLOSCAN_MODEL_LIST` per row, unset = bundle) into `/tmp/phase2_gate/<row>`; (2) `pixel_diff.py <fast> <oracle> --json` (D-08 defaults 99.9% / 10000 voxels); (3) SR "Airway Volume: N mL" parsed from both SR outputs, relative delta ≤ 0.1%; (4) logged `run_model_list=` / `ensemble_model_list=` asserted against the expected pair (Python-repr, `ast.literal_eval`). After the rows: bundle-vs-fullres "actually ensembles" sanity (382 differing voxels ✓), `gpu_residency.py --static` + `--runtime` (runtime runs the multi-fragment BUNDLE config — `preprocess_operator.py` ALLOWED via the deliberate Plan 01 D-13 entry, `postprocess_operator.py` the only exactly-once boundary), combined JSON, summary table, exit non-zero iff any gate fails. Per-row crash isolation: a crashed row is a FAIL, never an aborted run.

## Gate results (machine-readable: `gates/02-GATE-RESULTS.json`, `all_gates_pass: true`)

| Gate | HOLOSCAN_MODEL_LIST | oracle | byte-identity | diff voxels | fast/oracle voxels | SR delta | result |
|---|---|---|---|---|---|---|---|
| fullres_only | `3d_fullres` | `testdata/ref_fullres_only` | 99.99986% | 3 (documented fp16↔fp32 boundary class, IoU 0.998714) | 2331 / 2330 | 0.0% | PASS |
| lowres_only | `3d_lowres` | `testdata/ref_lowres_only` | 100.00000% | 0 | 2404 / 2404 | 0.0% | PASS |
| cascade_only | `3d_cascade_fullres` | `testdata/ref_cascade_only` | 100.00000% | 0 | 2519 / 2519 | 0.0% | PASS |
| bundle (final, D-06) | unset (default) | `testdata/current_output` | 100.00000% | 0 | 2447 / 2447 | 0.0% | PASS |

- Residency: **static PASS + runtime PASS** in the multi-fragment bundle configuration (D-13 boundary deliberate).
- Model-list assertions: every row's logged run/ensemble pair matches the Plan 04 table exactly (incl. the cascade auto-insert `[3d_lowres, 3d_cascade_fullres]` → ensemble `[3d_cascade_fullres]`).
- Sanity: bundle SEG differs from fullres-only SEG by 382 voxels (2447 vs 2331) — the bundle genuinely ensembles fullres + cascade_fullres probability maps as the reference bundles them.
- Note: lowres/cascade/bundle are **fully byte-identical** (stronger than the D-08 segmentation-level controlling bar); only fullres-only carries the documented 3-voxel fp16↔fp32 argmax boundary — same class, same magnitude, as Phase 1 and Plans 01–04 regressions.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Reference app raises on lowres-only (plan-level gap)**
- **Found during:** Task 1 (first `--config 3d_lowres` run: `ValueError: At least one non-auxiliary model configuration is required for ensemble inference`)
- **Issue:** The plan's live-probed correction covered the cascade-only crash (`tmp/3d_lowres/...nii.gz does not exist`) but not that `model_list=['3d_lowres']` also cannot run the reference — the ensemble would be empty. D-05 requires a lowres-only reference oracle; the reference app cannot produce one as-is.
- **Fix:** harness-side subclass in `reference_fullres_run.py` reproduces the fast app's DOCUMENTED self-ensemble fallback (Phase 2 Plan 04 decision, unit-tested there): catch the ValueError mid-`__init__`, replay the plain tail with `ensemble_model_list = run_model_list`. Includes a sub-fix discovered on the first attempt: the replay must pop the original `__init__`'s named kwargs (app_context, model_path, output_folder, output_labels, model_list, model_name, save_probabilities, save_files) before the holoscan `Operator` parent call — the original consumes them before its own `super()` call, and forwarding them to pybind fails with "python object could not be converted to Arg".
- **Files modified:** `.planning/scripts/reference_fullres_run.py`
- **Commit:** b423800 + 4a956e4
- **Why not Rule 4:** no architecture change — the example app is untouched, the fallback semantics already exist (and are unit-tested) in the fast app, and the oracle must match them for the gate to be meaningful.

**2. [Rule 1 - Bug] phase2_gate.py main() tail made unreachable by the fault-tolerance refactor**
- **Found during:** Task 2 (two end-to-end runs exited 0 with no JSON, no residency, no summary — missing report file)
- **Issue:** restructuring the per-row loop into `_run_gate_row` left the post-loop tail (sanity + residency + JSON + summary) after the function's `return r` — valid but dead code; `main()` then fell off its end and returned `None` → exit 0.
- **Fix:** moved the tail back into `main()` after the row loop; verified end-to-end (the run that produced the PASS results).
- **Files modified:** `.planning/scripts/phase2_gate.py`
- **Commit:** 07fde68

### Convention note (not a fix)

The plan left the oracle tracking convention to the repo: `testdata/current_output/` and `testdata/ref_fullres_only/` are gitignored, so `ref_lowres_only/` + `ref_cascade_only/` are gitignored too; provenance + sha256 recorded in `oracle_provenance.md` + `02-GATE-RESULTS.json` instead of committing the bytes.

## Known Stubs

None. Every JSON field is populated from actual gate runs; both deviations (TEST-005-2d blocked-on-model, TEST-01-corpus single dev study) are recorded verbatim in the gate JSON for VERIFICATION.md.

## Success criteria (plan)

- [x] TEST-005: all three 3D configs validated with a per-config oracle (fresh `ref_lowres_only` + `ref_cascade_only` + existing `ref_fullres_only`); 2d documented blocked-on-model (met-with-deviation, D-01/D-03) — in the gate JSON
- [x] TEST-01: pixel-exact on the dev corpus across all configs + the final bundle gate vs `testdata/current_output` (D-06) — 100.00000% / 0 differing voxels
- [x] SR within 0.1% in every gated configuration (0.0% — exact — on all 4 rows)
- [x] GPU residency green in the multi-fragment DAG (D-13 boundary deliberate; static + runtime PASS)
- [x] Gate evidence machine-readable in `02-GATE-RESULTS.json` + committed

## Self-Check: PASSED

All claimed files verified present (phase2_gate.py, 02-GATE-RESULTS.json, oracle_provenance.md, both oracle SEG dirs) and all nine task commits (8adeef8, b423800, 4a956e4, bf0c39f, 1f2aa69, e53e026, c5cc593, 07fde68, 257fdb4) found in git history.
