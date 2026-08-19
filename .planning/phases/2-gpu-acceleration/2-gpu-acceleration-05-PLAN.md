---
phase: "2-gpu-acceleration"
plan: "05"
type: "execute"
wave: 4
depends_on: ["04"]
files_modified:
  - ".planning/scripts/reference_fullres_run.py"
  - ".planning/scripts/phase2_gate.py"
  - "testdata/ref_lowres_only"
  - "testdata/ref_cascade_only"
  - ".planning/phases/2-gpu-acceleration/gates/"
autonomous: true
requirements: [TEST-01, TEST-005]
must_haves:
  truths:
    - "Fresh per-config reference oracles exist: testdata/ref_lowres_only (reference run model_list=['3d_lowres']) and testdata/ref_cascade_only (reference run model_list=['3d_lowres','3d_cascade_fullres'] — cascade-only SEG, D-05/D-07)"
    - "Per-config pixel-exact gates PASS: fast fullres-only vs ref_fullres_only; fast lowres-only vs ref_lowres_only; fast cascade-only vs ref_cascade_only — segmentation-level identity, the D-08 controlling level (fp16-ref vs fp32-ours logits make zero-voxel-diff unreachable by design; Phase 1 measured 99.99986%)"
    - "The FINAL BUNDLE GATE passes: fast app default run (fullres + cascade_fullres probability maps averaged exactly as the reference bundles them) vs testdata/current_output (D-06)"
    - "DICOM-SR airway volume matches within 0.1% in every gated run"
    - "gpu_residency.py (static + runtime) passes in the multi-fragment bundle configuration with the deliberate D-13 allow-list from Plan 01"
    - "TEST-005 is recorded as met-with-deviation: 2d is blocked-on-model (D-01/D-03/D-04) — no dummy 2d model exists or is created; the deviation is written into the gate JSON and ready for VERIFICATION.md"
  artifacts:
    - path: "testdata/ref_lowres_only"
      provides: "reference SEG/SC/SR for the standalone lowres gate"
    - path: "testdata/ref_cascade_only"
      provides: "reference SEG/SC/SR for the cascade-only gate"
    - path: ".planning/scripts/reference_fullres_run.py"
      provides: "--config now accepts a comma-separated model list (needed because cascade-only oracles require lowres to actually run — live-proven)"
      contains: "split(',')"
    - path: ".planning/scripts/phase2_gate.py"
      provides: "runs all 4 fast-app gate configurations + pixel_diff + SR compare + residency, writes a combined JSON"
    - path: ".planning/phases/2-gpu-acceleration/gates/02-GATE-RESULTS.json"
      provides: "machine-readable gate evidence (4 pixel gates, 4 SR checks, residency, deviations)"
  key_links:
    - from: "fast HOLOSCAN_MODEL_LIST=3d_cascade_fullres"
      to: "testdata/ref_cascade_only"
      via: "both run lowres (auxiliary, non-ensembled) + cascade with ensemble=[3d_cascade_fullres] — same reference semantics"
      pattern: "3d_cascade_fullres"
    - from: "fast default (bundle) run"
      to: "testdata/current_output"
      via: "pixel_diff.py SEG compare; ensemble = fullres + cascade_fullres probability maps (D-06)"
      pattern: "current_output"
---

# Phase 2 Plan 05: Per-Config Reference Oracles + Pixel-Exact Gates (incl. Bundle Gate)

## Objective
- **What:** Generate the two missing per-config reference oracles (D-05), then run the
  four pixel-exact gates — fullres-only, lowres-only, cascade-only, and the final bundle
  vs `testdata/current_output` (D-06) — plus SR and residency checks, writing a combined
  gate JSON.
- **Why:** TEST-01/TEST-005 are the phase's correctness gates: every config must be
  pixel-exact (segmentation-level, D-08) and the bundle must match the reference's
  full-bundle output.
- **Output:** `testdata/ref_lowres_only`, `testdata/ref_cascade_only`,
  `02-GATE-RESULTS.json`, all gates PASS (or a precisely characterized FAIL to fix).

## Execution Environment

- Python: `/tmp/monai-env/.venv/bin/python`; `ulimit -s unlimited` for every app run.
- Fast app runs (app root; env var per Plan 04):
  ```bash
  cd examples/apps/cchmc-nnunet-fast && ulimit -s unlimited
  HOLOSCAN_MODEL_LIST=<cfg> /tmp/monai-env/.venv/bin/python my_app \
    -i /users/srv-mde/projects/monai-deploy-app-sdk/testdata/airway_input \
    -m /users/srv-mde/projects/monai-deploy-app-sdk/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models \
    -o <scratch>
  ```
- pixel_diff: `/tmp/monai-env/.venv/bin/python examples/apps/cchmc-nnunet-fast/scripts/pixel_diff.py <new> <ref> [--json out.json]`
  (defaults `--min-identity 99.9 --max-diff-voxels 10000`).
- Residency: `/tmp/monai-env/.venv/bin/python examples/apps/cchmc-nnunet-fast/scripts/gpu_residency.py --static`
  and the runtime/self-test modes per the script's own `main()` (run from the app root).
- Reference runs take ~124 s each (A100). Reference oracles must be generated with the
  reference app UNMODIFIED — only `.planning/scripts/reference_fullres_run.py` may be
  extended (example-app modification is out of scope).
- Commit after each major change (oracles are generated data — record their voxel counts
  and a checksum line in the gate JSON, not the bytes themselves, unless the repo
  convention for testdata/ says to commit them; follow what testdata/ref_fullres_only and
  testdata/current_output are currently tracked as and match that).

## Context

@.planning/phases/2-gpu-acceleration/02-CONTEXT.md (D-05..D-08, D-01/D-03)
@.planning/phases/2-gpu-acceleration/02-RESEARCH.md (Pattern 5 — with the correction below)
@.planning/scripts/REFERENCE_RUN_GUIDE.md
@.planning/scripts/reference_fullres_run.py

**⚠ RESEARCH.md correction (live-probed 2026-08-19, supersedes Pattern 5):**
`reference_fullres_run.py --config 3d_cascade_fullres` (model_list pinned to
`['3d_cascade_fullres']`) **CRASHES**:
`RuntimeError: ... tmp/3d_lowres/Img_in_context.nii.gz does not exist` — the reference
cascade reads the previous stage's exported `.nii.gz`, which exists only if lowres
ACTUALLY RAN; the reference list logic reorders but does not auto-insert. Therefore the
cascade-only oracle must pin `model_list=['3d_lowres', '3d_cascade_fullres']` — lowres
runs (auxiliary), cascade consumes its export, and the reference's own
`ensemble_model_list` excludes lowres → the SEG is cascade-only, exactly matching
D-07 and the fast app's `HOLOSCAN_MODEL_LIST=3d_cascade_fullres` semantics. This is why
Task 1 extends the harness to accept a comma-separated `--config` list.

Gate matrix (fast-app side → reference side):

| Gate | fast run (HOLOSCAN_MODEL_LIST) | reference oracle |
|---|---|---|
| 1 fullres-only | `3d_fullres` | `testdata/ref_fullres_only` (existing) |
| 2 lowres-only | `3d_lowres` | `testdata/ref_lowres_only` (Task 1) |
| 3 cascade-only | `3d_cascade_fullres` | `testdata/ref_cascade_only` (Task 1) |
| 4 bundle (final) | unset (default) | `testdata/current_output` (existing, reference full-bundle) |

Tolerances (D-08, Phase 1 precedent): segmentation-level identity is controlling —
`pixel_diff.py` default tolerances; Phase 1 measured 99.99986% byte-identity (3
differing voxels = the documented fp16↔fp32 argmax boundary; the reference is
run-to-run deterministic). SR: airway volume within 0.1% ("Airway Volume: 1 mL" in
Phase 1).

## Tasks

<task type="auto">
  <name>Task 1: Extend reference_fullres_run.py to a model-list harness + generate both per-config oracles</name>
  <files>.planning/scripts/reference_fullres_run.py, testdata/ref_lowres_only, testdata/ref_cascade_only</files>
  <read_first>
    - .planning/scripts/reference_fullres_run.py (current `--config` single-value pin: `kwargs["model_list"] = [args.config]` — extend, keep the 3d_fullres default behavior identical)
    - .planning/scripts/REFERENCE_RUN_GUIDE.md (run prerequisites, .env, my_app name-collision rule, ~124 s runtime)
    - examples/apps/cchmc_nnunet_fifteen_ckpt_app/my_app/nnunet_seg_operator.py (lines 91–99 — why `['3d_lowres','3d_cascade_fullres']` yields a cascade-only ensemble SEG)
  </read_first>
  <action>
1. In `reference_fullres_run.py`: change `--config` to accept a COMMA-SEPARATED list —
   `model_list = [c.strip() for c in args.config.split(",") if c.strip()]`
   (default stays `"3d_fullres"` so the Phase 1 gate command is unchanged); update the
   usage docstring with the two Phase 2 invocations. No other behavior changes (the
   monkey-pin of `NNUnetSegOperator` with `kwargs["model_list"] = model_list` stays).
2. Generate the oracles (each ~124 s; run from the repo root — the script resolves
   REPO_ROOT itself; the script cd's into the reference app's namespace internally, so
   the my_app name-collision rule is handled by it, as in Phase 1):
   ```bash
   ulimit -s unlimited
   /tmp/monai-env/.venv/bin/python .planning/scripts/reference_fullres_run.py --config 3d_lowres --output testdata/ref_lowres_only
   /tmp/monai-env/.venv/bin/python .planning/scripts/reference_fullres_run.py --config 3d_lowres,3d_cascade_fullres --output testdata/ref_cascade_only
   ```
   Each must exit 0 and write `SEG/`, `SR/`, `SC/` under the output dir.
3. Record provenance: for each new oracle, the model_list used, wall time, and the
   post-CC segment voxel count (decode the SEG 1-bit pixels with the REFERENCE_RUN_GUIDE
   "Quick SEG parity check" snippet), written into
   `.planning/phases/2-gpu-acceleration/gates/oracle_provenance.md`.
   Sanity expectations (characterize any surprise, don't force-fit): lowres-only and
   cascade-only voxel counts will DIFFER from each other and from fullres (different
   configurations); cascade-only should be in the same ballpark as the bundle's 2447.
4. Commit the harness change (and the testdata dirs per the repo's existing
   testdata tracking convention).
  </action>
  <acceptance_criteria>
    - `grep -n "split(',')" .planning/scripts/reference_fullres_run.py` returns 1 line; the default value of `--config` is still `"3d_fullres"` (grep `default="3d_fullres"`).
    - `ls testdata/ref_lowres_only/{SEG,SR,SC}` and `ls testdata/ref_cascade_only/{SEG,SR,SC}` each list at least one file.
    - `grep -rn "3d_lowres,3d_cascade_fullres" .planning/scripts/reference_fullres_run.py` shows the cascade-only invocation documented in the docstring (with a comment explaining WHY lowres must be in the list — the crash from the Context correction).
    - `.planning/phases/2-gpu-acceleration/gates/oracle_provenance.md` exists with both model_lists, wall times, and voxel counts.
    - The reference app itself is UNMODIFIED: `git status examples/apps/cchmc_nnunet_fifteen_ckpt_app` shows no changes.
  </acceptance_criteria>
  <verify>cd /users/srv-mde/projects/monai-deploy-app-sdk && ls testdata/ref_lowres_only/SEG testdata/ref_cascade_only/SEG && grep -n "split(',')" .planning/scripts/reference_fullres_run.py</verify>
  <done>Both per-config oracles exist, generated with the reference app unmodified (D-05), with the cascade-only semantics correct by construction (lowres auxiliary + cascade-only ensemble, D-07).</done>
</task>

<task type="auto">
  <name>Task 2: phase2_gate.py — run all 4 gates + SR + residency, write 02-GATE-RESULTS.json</name>
  <files>.planning/scripts/phase2_gate.py, .planning/phases/2-gpu-acceleration/gates/02-GATE-RESULTS.json</files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/scripts/pixel_diff.py (CLI: two positional SEG dirs + `--json`; exit codes; the identity/voxel-count JSON fields it emits)
    - examples/apps/cchmc-nnunet-fast/scripts/gpu_residency.py (static/runtime/self-test modes and their exit codes; `ALLOWED_TRANSFER_FILES` — must already contain the Plan 01 `preprocess_operator.py` D-13 entry)
    - .planning/phases/2-gpu-acceleration/2-gpu-acceleration-04-PLAN.md (the four HOLOSCAN_MODEL_LIST configurations + expected run/ensemble lists)
    - .planning/phases/2-gpu-acceleration/02-CONTEXT.md (D-06 bundle gate; D-01/D-03 2d deviation)
  </read_first>
  <action>
Create `.planning/scripts/phase2_gate.py` — a deterministic gate runner (subprocesses,
venv python, `ulimit -s unlimited` via `resource.setrlimit` or a bash wrapper per
`baseline_benchmark.py`'s existing pattern):

For each of the 4 gate rows in the Context table:
 1. Run the fast app (`HOLOSCAN_MODEL_LIST` set per row; bundle row leaves it unset)
    into a scratch dir under `/tmp/phase2_gate/<row>`; require exit 0.
 2. `pixel_diff.py <fast>/SEG <oracle>` with `--json` → capture identity %, differing
    voxel count, exit code. Gate: exit 0 (defaults 99.9% / 10000 voxels).
 3. SR compare: extract the airway-volume number from the fast SR vs the oracle SR
    (the SR text carries "Airway Volume: N mL" — parse both) and require relative delta
    <= 0.001 (0.1%).
 4. Log the fast run's `run_model_list=` / `ensemble_model_list=` lines and assert they
    match the expected pair for the row (Plan 04 table) — a wrong list would make the
    comparison meaningless.

After the 4 rows:
 5. Residency: `gpu_residency.py --static` (exit 0) + the runtime/self-test mode per its
    CLI against the bundle configuration (the multi-fragment DAG); capture status.
    `preprocess_operator.py` must appear ALLOWED (Plan 01's deliberate D-13 entry) and
    `postprocess_operator.py` remains the only exactly-once final boundary.
 6. Write `.planning/phases/2-gpu-acceleration/gates/02-GATE-RESULTS.json` with:
    per row: {row, model_list, fast_voxels, oracle_voxels, byte_identity_pct,
    diff_voxels, sr_fast, sr_oracle, sr_delta_pct, pass}; plus:
    residency: {static: "PASS", runtime: "PASS"};
    deviations: [
      {"id": "TEST-005-2d", "text": "2d config blocked-on-model (D-01/D-03): the bundle
       has no 2d model; TEST-005 counts as met-with-deviation, same pattern as the
       Phase 0 corpus deviation. Fragment wiring is config-generic (D-02) — a real 2d
       model is a test, not a code change (D-04)."},
      {"id": "TEST-01-corpus", "text": "Single airway dev-study corpus (TEST-01
       deviation, carried from Phase 0/1); the >=5-CT-study re-run remains a deferred
       final gate."}
    ];
    top-level "all_gates_pass": bool.
 7. Print a human summary table; exit non-zero iff any gate fails.
Run it end-to-end. If a gate FAILS: diagnose (the usual suspects, in order: ensemble
order, cascade one-hot orientation/bbox, seg-resample kwargs, contiguity) and fix the
FAST APP (never the oracle, never by loosening tolerances below the Phase 1 precedent),
re-run, and note the fix in the JSON's "fixes" array.
  </action>
  <acceptance_criteria>
    - `.planning/scripts/phase2_gate.py` exists; `grep -n "ref_lowres_only\|ref_cascade_only\|ref_fullres_only\|current_output" .planning/scripts/phase2_gate.py` shows all four oracle targets; `grep -n "HOLOSCAN_MODEL_LIST" .planning/scripts/phase2_gate.py` >= 1 line.
    - `/tmp/monai-env/.venv/bin/python .planning/scripts/phase2_gate.py` exits 0 and prints a summary table with 4 PASS rows.
    - `02-GATE-RESULTS.json` exists, is valid JSON (`/tmp/monai-env/.venv/bin/python -c "import json; json.load(open('.planning/phases/2-gpu-acceleration/gates/02-GATE-RESULTS.json'))"`), has `all_gates_pass: true`, per-row `byte_identity_pct` >= 99.9 (expected ~99.999%+), `sr_delta_pct` <= 0.1 for every row, and both deviation entries (grep `2d` and `TEST-01-corpus`).
    - Residency static + runtime PASS for the bundle configuration is recorded in the JSON; `grep -n "preprocess_operator.py" examples/apps/cchmc-nnunet-fast/scripts/gpu_residency.py` still shows the D-13 ALLOWED entry (untouched by this plan).
    - The bundle row's fast SEG differs from the fullres-only SEG (different configs — sanity that the bundle actually ensembles; voxel count in the JSON) while matching `testdata/current_output` within tolerance (D-06: ~2447-voxel class).
    - Commits made for the gate script, the results JSON, and any fast-app fixes found by the gates.
  </acceptance_criteria>
  <verify>cd /users/srv-mde/projects/monai-deploy-app-sdk && /tmp/monai-env/.venv/bin/python .planning/scripts/phase2_gate.py; echo "GATE_EXIT=$?"; /tmp/monai-env/.venv/bin/python -c "import json; d=json.load(open('.planning/phases/2-gpu-acceleration/gates/02-GATE-RESULTS.json')); print(d['all_gates_pass'])"</verify>
  <done>All four pixel-exact gates pass at the D-08 controlling level (per-config + bundle vs current_output, D-06), SR within 0.1% everywhere, residency green in the multi-fragment DAG, and the 2d/corpus deviations are recorded for VERIFICATION.md — TEST-01 and TEST-005 (met-with-deviation) satisfied on the dev corpus.</done>
</task>

## Verification
- `phase2_gate.py` exit 0; `02-GATE-RESULTS.json` `all_gates_pass == true`.
- Four SEG gates vs the four oracles (identity ~99.999% class); four SR checks <= 0.1%.
- Residency static + runtime PASS (bundle config).
- Oracles committed/recorded with provenance; reference app unmodified.

## Success Criteria
- [ ] TEST-005: all three 3D configs validated with a per-config oracle; 2d documented blocked-on-model (met-with-deviation, D-01/D-03)
- [ ] TEST-01: pixel-exact on the dev corpus across all configs + the final bundle gate vs `testdata/current_output` (D-06)
- [ ] SR within 0.1% in every gated configuration
- [ ] GPU residency green in the multi-fragment DAG (D-13 boundary deliberate)
- [ ] Gate evidence machine-readable in 02-GATE-RESULTS.json + committed
