---
phase: "0-foundation"
plan: "01"
type: "execute"
wave: 1
depends_on: []
files_modified: []
autonomous: true
user_setup: []
must_haves:
  truths:
    - "cu13 dependency set resolves in /tmp/monai-env/.venv (holoscan-cu13, cupy-cuda13x, rmm-cu13, torch 2.13+cu130, monai 1.3.0, nnunetv2 editable)"
    - "Reference app runs end-to-end on the airway corpus and emits SC/SEG/SR"
    - "Baseline latency is recorded with per-stage breakdown to .planning/baseline_results.csv"
    - "Nsight harness produces a valid trace with NVTX ranges"
    - "RMM pool allocator is verified active"
  artifacts:
    - "examples/apps/cchmc-nnunet-fast/ (app scaffold)"
    - ".planning/baseline_results.csv"
    - ".planning/scripts/baseline_benchmark.py"
    - ".planning/scripts/nsight_profile.sh"
    - ".planning/profiles/nsight_demo_target_*.nsys-rep"
    - ".planning/scripts/test_rmm.py"
  key_links:
    - "baseline_benchmark.py -> baseline_results.csv"
    - "nsight_profile.sh -> .planning/profiles/*.nsys-rep"
---

# Phase 0 Plan 01: Foundation (backfilled)

> Backfilled 2026-08-17 to close Phase 0 in the GSD cycle. Phase 0 was executed ad-hoc
> (outside plan/execute); this plan records what was actually built and verified so the
> phase is credit-verified and its requirements can be tracked by the milestone audit.

## Objective
- **What:** Establish scaffolding, cu13 dependencies, a reference corpus, a baseline benchmark,
  an Nsight profiling harness, and RMM verification.
- **Why:** Every later phase must prove improvement with hard numbers; this phase is the
  "before" baseline and the correctness reference.
- **Output:** Scaffolded `cchmc-nnunet-fast` app, baseline CSV, working Nsight + RMM tooling.

## Tasks

<task type="code">
  <name>0.1 Scaffold cchmc-nnunet-fast app</name>
  <files>examples/apps/cchmc-nnunet-fast/</files>
  <action>Mirror SDK app structure: app.py, my_app/operators/, my_app/config/, pyproject.toml.</action>
  <verify>Directory exists with app.py skeleton and standard MAP layout.</verify>
  <done>App scaffold present; editable-install registered.</done>
</task>

<task type="code">
  <name>0.2 cu13 dependency pins</name>
  <files>examples/apps/cchmc-nnunet-fast/pyproject.toml, .planning/STATE.md</files>
  <action>Pin holoscan-cu13, cupy-cuda13x, rmm-cu13, monai, torch, pydicom, highdicom; remove cupy-cuda12x.</action>
  <verify>Import chain resolves in /tmp/monai-env/.venv.</verify>
  <done>All cu13 deps resolve; no cupy-cuda12x present.</done>
</task>

<task type="code">
  <name>0.3 Reference corpus</name>
  <files>testdata/airway_input, testdata/airway_output</files>
  <action>Assemble reference study with ground-truth SC/SEG/SR.</action>
  <verify>Study + SC/SEG/SR present on disk.</verify>
  <done>Single airway MR series (256 slices) with SC/SEG/SR. Deviation: ≥5-CT deferred to Phase 1 gate (TEST-01).</done>
</task>

<task type="code">
  <name>0.4 Baseline benchmark script</name>
  <files>.planning/scripts/baseline_benchmark.py</files>
  <action>Run reference app on corpus; record E2E + per-stage timing to CSV.</action>
  <verify>Script produces CSV with study/rep/total_ms/per-stage columns.</verify>
  <done>baseline_benchmark.py runs and emits baseline_results.csv.</done>
</task>

<task type="code">
  <name>0.5 Baseline results</name>
  <files>.planning/baseline_results.csv</files>
  <action>Run 3 reps (warmup excluded), store mean ± std.</action>
  <verify>CSV present with numbers.</verify>
  <done>169,747 ± 7,274 ms/study (n=3); setup ~12.8 s / inference ~138 s / postprocess 9–23 s / write ~1.2 s.</done>
</task>

<task type="code">
  <name>0.6 Nsight profiling harness</name>
  <files>.planning/scripts/nsight_profile.sh, .planning/scripts/nvtx_markers.py, .planning/profiles/</files>
  <action>nsys CLI wrapper; verify nsys in PATH; generate demo trace with NVTX ranges.</action>
  <verify>Demo .nsys-rep + .sqlite exist; preprocess/inference/postprocess NVTX ranges visible.</verify>
  <done>nsight_demo_target_20260817_111555.nsys-rep produced and verified.</done>
</task>

<task type="code">
  <name>0.7 RMM verification</name>
  <files>.planning/scripts/test_rmm.py</files>
  <action>Smoke test setting the RMM pool allocator and confirming it is active.</action>
  <verify>test_rmm.py PASSED; allocator backend active (pluggable/cudaAsync).</verify>
  <done>Driver 610.57.04 / CUDA 13.3, A100-SXM4-40GB; RMM active.</done>
</task>

## Verification
All 7 tasks complete; 5/5 acceptance criteria met (criterion 2 with documented TEST-01
corpus deviation). Reference app re-run reproduces historical ground truth (see VERIFICATION.md).

## Success Criteria
- [x] pyproject.toml + cu13 deps resolve in venv
- [x] Reference corpus with SC/SEG/SR ground truth (single airway MR series; deviation documented)
- [x] Baseline benchmark CSV at .planning/baseline_results.csv
- [x] Nsight harness produces a valid trace
- [x] RMM pool allocator verified active
