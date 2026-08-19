---
phase: "1-core-pipeline"
plan: "05"
type: "execute"
wave: 4
depends_on: ["04"]
files_modified:
  - "examples/apps/cchmc-nnunet-fast/scripts/pixel_diff.py"
  - "examples/apps/cchmc-nnunet-fast/scripts/gpu_residency.py"
autonomous: true
user_setup:
  - "Freshly regenerated reference output present at testdata/current_output (run REFERENCE_RUN_GUIDE.md). Historical GT at testdata/airway_output."
must_haves:
  truths:
    - "A pixel-level diff tool compares the new app's DICOM-SEG to the reference and fails on any divergence"
    - "A GPU-residency test verifies all intermediate tensors stay on CUDA and flags any .cpu()/.numpy() call before the final output stage"
    - "The new app's DICOM-SEG is pixel-exact (bit-for-bit) to the freshly regenerated reference on the airway study (3D_fullres)"
    - "DICOM-SR measurements (airway volume) match the reference within 0.1%"
  artifacts:
    - "examples/apps/cchmc-nnunet-fast/scripts/pixel_diff.py"
    - "examples/apps/cchmc-nnunet-fast/scripts/gpu_residency.py"
  key_links:
    - "pixel_diff.py: new SEG vs testdata/current_output/SEG (primary) and testdata/airway_output/SEG (secondary)"
    - "gpu_residency.py: scans operator chain for premature CPU transfers"
---

# Phase 1 Plan 05: Validation Tools + Pixel-Exact E2E Gate

## Objective
- **What:** Build the automated validation tooling (pixel-diff, GPU-residency) and run the
  end-to-end correctness gate on the airway study.
- **Why:** Correctness is the Phase 1 gate — the app must be pixel-exact to the reference and
  keep everything on GPU. These tools make that provable and CI-able.
- **Output:** Two validation scripts + a passing pixel-exact E2E run + SR measurement comparison.

## Context
@.planning/PROJECT.md
@.planning/STATE.md
@.planning/scripts/REFERENCE_RUN_GUIDE.md

Gate target: the **freshly regenerated reference** `testdata/current_output` (confirmed 99.902%
byte-identical to historical GT `testdata/airway_output`, 2026-08-17). Compare segment **pixel
data + geometry** (SOP UIDs differ between runs). TEST-01 deviation: single airway MR study now;
the ≥5-CT-study corpus is the **final Phase 1 acceptance gate** (blocked on CT data — carry as a
known blocker, not a plan task). Phase 3 speedup numbers are NOT the gate — correctness is.

## Tasks

<task type="code">
  <name>1.16 Pixel-level diff tool</name>
  <files>examples/apps/cchmc-nnunet-fast/scripts/pixel_diff.py</files>
  <action>Write an automated tool: read two DICOM-SEG dirs (new app vs reference), decode the binary segment (1-bit) for each, compare geometry (rows/cols/frames/spacing) and segment pixel data; report byte-identity %, differing voxel count, IoU, and per-voxel diffs; exit non-zero on any divergence beyond a documented tolerance. Reusable in CI.</action>
  <verify>Running it on testdata/current_output vs testdata/airway_output reports ~99.9% byte-identical (matches the 2026-08-17 finding) and exits 0 within tolerance; a deliberately corrupted input causes a non-zero exit.</verify>
  <done>Automated pixel-diff tool that fails on divergence (TEST-003).</done>
</task>

<task type="code">
  <name>1.17 GPU-residency test</name>
  <files>examples/apps/cchmc-nnunet-fast/scripts/gpu_residency.py</files>
  <action>Write a test that instruments/scans the operator chain for intermediate tensors: assert device == 'cuda' at every operator boundary and flag any .cpu()/.numpy() call before the final DICOM-SEG write stage (the single allowed CPU transfer in PostprocessOperator).</action>
  <verify>On a 3D_fullres run, the test passes (all intermediates on CUDA; exactly one boundary CPU transfer in Postprocess); injecting a .cpu() call in an intermediate operator causes it to fail.</verify>
  <done>GPU-residency test flags premature CPU transfers (TEST-004).</done>
</task>

<task type="code">
  <name>1.18 E2E single-config pixel-exact gate</name>
  <files>examples/apps/cchmc-nnunet-fast/ (run), .planning/benchmarks/</files>
  <action>Run cchmc-nnunet-fast end-to-end on the airway study (3D_fullres), capture the new SEG/SR/SC, run pixel_diff.py against testdata/current_output (primary) and testdata/airway_output (secondary), and record the result. Gate: pixel-exact (bit-for-bit segment data, or documented within 1-voxel registration) and SR volume within 0.1%. Save a benchmark/latency comparison vs the 169.7 s baseline (informational).</action>
  <verify>pixel_diff.py passes vs the fresh reference; SR airway volume within 0.1% of reference; GPU-residency test passes. Record PASS/FAIL with evidence.</verify>
  <done>First correctness gate passed for 3D_fullres (TEST-01 dev-study portion, TEST-002).</done>
</task>

<task type="code">
  <name>1.19 DICOM-SR measurement comparison</name>
  <files>examples/apps/cchmc-nnunet-fast/scripts/pixel_diff.py (or scripts/sr_compare.py)</files>
  <action>Compare the airway-volume SR measurement from the new app vs the reference within 0.1% tolerance; record both values and the delta.</action>
  <verify>SR airway volume delta <= 0.1%.</verify>
  <done>SR measurements match within 0.1% (TEST-002).</done>
</task>

## Verification
- pixel_diff.py passes for the new app vs the freshly regenerated reference (3D_fullres, airway study).
- gpu_residency.py passes (all intermediates on CUDA, single boundary CPU transfer).
- SR airway volume within 0.1% of reference.
- The ≥5-CT-study portion of TEST-01 is explicitly recorded as a carried blocker (CT data pending).

## Success Criteria
- [ ] Single-config (3D_fullres) pipeline runs end-to-end without errors (TEST-01 dev-study)
- [ ] DICOM-SEG is bit-for-bit identical to reference (pixel-diff passes) (TEST-01, TEST-003)
- [ ] DICOM-SR measurements match within 0.1% (TEST-002)
- [ ] No .cpu()/.numpy() between inference output and DICOM-SEG writer (GPU-residency passes) (TEST-004)
- [ ] All 5 core operators exist as Holoscan subclasses with setup() and compute()
- [ ] NVTX markers in all operators; Nsight trace shows operator boundaries
- [ ] Operator-level timing logs emitted per study
- [ ] app.py uses the new operator chain (PIPE-02)
- [ ] Baseline comparison latency numbers recorded (correctness is the gate; speedup is Phase 2/3)
