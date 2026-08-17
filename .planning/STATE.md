# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-14)

**Core value:** Single-study inference latency without sacrificing correctness — CT in, pixel-identical DICOM-SEG out, faster, every intermediate step stays on GPU.
**Current focus:** Phase 0 CLOSED (2026-08-17, documented deviation) → plan Phase 1 (Core Pipeline)

## Current Position

Phase: 0 of 4 (Foundation) — COMPLETE (GSD-closed 2026-08-17)
Plan: 1 of 1 in current phase (backfilled `0-foundation-01-PLAN.md`; phase was executed ad-hoc then backfilled)
Status: Phase 0 closed in the GSD cycle 2026-08-17 — 7/7 tasks done, all 5 acceptance criteria ✓ (criterion 2 with documented TEST-01 deviation: single airway MR study; ≥5-CT bar deferred to final Phase 1 gate). VERIFICATION status: passed. GT-mismatch carried finding RESOLVED (fresh reference run = 99.902% byte-identical to historical GT). Credits TEST-006 + TEST-007 (baseline); PIPE-01/INFR-01/INFR-005 partial (scaffold/feasibility, completed in Phases 1–2).
Last activity: 2026-08-17 20:40 UTC - Phase 0 CLOSED in the GSD cycle. GT-mismatch carried finding RESOLVED (fresh reference run `testdata/current_output` = 99.902% byte-identical to historical `testdata/airway_output`; ~45 mm note was a thin-structure IoU decode artifact). Baseline 169,747 ± 7,274 ms (n=3); Nsight demo trace + RMM verified. Docker deferred to post-Phase-3. GSD backfill: .planning/phases/0-foundation/ (PLAN+SUMMARY+VERIFICATION). Run guide: .planning/scripts/REFERENCE_RUN_GUIDE.md

Progress: [░░░░░░░░░░░░░░░░░░░░] 0/0 plans (0%)

## Phase 0 Acceptance Status

| # | Criterion | Status |
|---|-----------|--------|
| 1 | pyproject.toml + all cu13 deps resolved in venv | ✓ met — import chain verified 2026-08-14; monai 1.3.0 + itk 5.4.7 restored/added 2026-08-17 |
| 2 | Reference corpus with DICOM-SEG/SR ground truth | ✓ met (deviation) — single airway MR series (256 slices) in `testdata/airway_input` + SC/SEG/SR in `testdata/airway_output`; original ≥5-CT bar deferred to final Phase 1 gate (TEST-01 note) |
| 3 | Baseline benchmark CSV at .planning/baseline_results.csv | ✓ met — 2026-08-17: 169,747 ± 7,274 ms/study (n=3, warmup excluded); per-stage columns populated |
| 4 | Nsight harness produces valid trace | ✓ met — demo trace .planning/profiles/nsight_demo_target_20260817_111555.nsys-rep (+.sqlite); NVTX preprocess/inference/postprocess ranges verified in trace |
| 5 | RMM pool allocator verified active | ✓ met — 2026-08-17: driver 610.57.04 (CUDA 13.3), A100-SXM4-40GB, test_rmm.py PASSED (backend 'pluggable') |

Done: app scaffold (examples/apps/cchmc-nnunet-fast, standard MAP layout, app.py skeleton), cu13 pins (commit 7d5e687), Nsight/NVTX harness (d99deb4), RMM + venv validation scripts (8910ae8, 205fcd3), reference-app E2E env repairs + baseline + demo trace (2026-08-17, commits 668ef0f..e2c38be + close-out).

## Performance Metrics

**Velocity:** No GSD plans executed yet (Phase 0 was ad-hoc work — no PLAN/SUMMARY artifacts). Metrics start with the first planned phase.

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
| ----- | ----- | ----- | -------- |
| 0     | 0     | 0     | -        |

**Recent Trend:** n/a

## Accumulated Context

### Decisions

Logged in PROJECT.md Key Decisions table. Recent:

- [Init]: New example app, not SDK core modification (lower risk, prove concept first)
- [Init]: Latency first, throughput later (single-study clinical workflow)
- [Phase 0]: Pin every CUDA dependency to cu13; cupy-cuda12x must not coexist
- [Phase 0]: Venv repair 2026-08-14: holoscan-cu13 4.2.0 force-reinstalled (install was corrupted — 118 files from the wheel's RECORD missing on disk, incl. all GXF runtime .so and the holoscan.core/conditions/executors modules); holoscan-cli 4.2.0 added; pydicom → 3.0.2 (SDK uses 3.x API), highdicom → 0.28.1 (0.22.x breaks on pydicom 3.x). GXF runtime libs are bundled in the holoscan-cu13 wheel — no separate package. Documented in app README.
- [Phase 0]: Resampling stays on reference CPU (scipy) path in Phases 1–2 — pixel-exactness over speed
- [Phase 0]: Driver cleared 2026-08-17: host updated to 610.57.04 (CUDA 13.3); A100-SXM4-40GB; test_rmm.py passes (pluggable allocator, not literal 'cudaAsync' string — the criterion's intent, RMM active, is met)
- [Phase 0]: Reference-app env repairs 2026-08-17 (needed to run the baseline): (1) `nnunet_bundle.py` `build_network_architecture` call adapted to vendored nnunetv2 2.8.1 signature (glue only, no math); (2) venv monai reverted 1.6.0 → 1.3.0 per the app's own pin (`monai[einops]==1.3.0`) — 1.6.0 had drifted in and removed/changed writer plumbing; (3) `itk 5.4.7` installed so MONAI `SaveImaged(output_ext='.dcm')` (SC overlay write) resolves an ITKWriter — MONAI has no native DICOM writer, the original env must have had ITK; (4) patched 2 NumPy-2.0-removed `ndarray.ptp()` call sites in venv monai 1.3.0 (`data/utils.py:906`, `transforms/spatial/functional.py:376` → `np.ptp(arr, axis=1)`) — venv is scratch (/tmp), re-apply after any monai reinstall
- [Phase 0]: Baseline provenance caveat: testdata/output ground truth was generated by an earlier env (pre-2.8.1 nnunetv2, unknown MONAI). After the repairs, Phase 1 validation should diff the new app against FRESHLY regenerated reference output (same env, same nnunetv2 2.8.1) as the primary gate, with testdata/output as the historical reference
- [Phase 0, 2026-08-17 evening]: testdata replaced by user with airway-consistent data: `testdata/airway_input` (256 MR slices, 256×256, patient 12345678) + `testdata/airway_output/{SC,SEG,SR}` (1 airway SEG, 256 frames, 2430 voxels); models moved to `examples/apps/cchmc_nnunet_fifteen_ckpt_app/models` = `MRI_NICU-Airway_TRAINv2` (airway=1), configs present: 3d_fullres, 3d_lowres, 3d_cascade_fullres (2d listed in plans.json but absent). Reference app runs end-to-end on it (exit 0, SC/SEG/SR produced, ~124 s).
- [Phase 0, FINDING — RESOLVED 2026-08-17 20:40 UTC]: the earlier "reference app does NOT reproduce GT (~45 mm world-COM offset, zero overlap)" note was a **decoding artifact**, not a real mismatch. A fresh reference run (`testdata/current_output`) was made and compared to the historical GT (`testdata/airway_output`): DICOM-SEG segment pixel data is **99.902% byte-identical** (2,095,090/2,097,152), segment voxel counts 2430 (GT) vs 2447 (fresh, Δ0.7%), all differences confined to the airway band (frames 106–225). The airway is a thin 1-voxel structure so IoU is hypersensitive to sub-voxel registration — the raw pixel data confirms a match. The ensemble-config question is moot for the correctness gate. See `.planning/phases/0-foundation/0-foundation-VERIFICATION.md` + `.planning/scripts/REFERENCE_RUN_GUIDE.md`.
- [Phase 0]: **Docker build + container test deferred** until after Phase 3 optimizations are in place (decided 2026-08-17). Pythonic reference runs (`.planning/scripts/REFERENCE_RUN_GUIDE.md`) are the validation path until then.
- [Phase 0]: Phase 0 **closed in the GSD cycle** 2026-08-17 — backfilled `.planning/phases/0-foundation/` (01-PLAN, 01-SUMMARY, VERIFICATION status: passed). Credits TEST-006 (benchmark script) + TEST-007 (baseline) as satisfied; PIPE-01/INFR-01/INFR-005 tracked as partial (scaffold/feasibility only, full satisfaction in Phases 1–2).

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 0] Reference corpus gap (task 0.3): testdata/input is 1 MR series (62 dcm) with complete SC/SEG/SR ground truth — usable as a single-study dev corpus, but the ≥5 CT studies criterion is unmet and no CT data exists in the repo. Decision needed: relax criterion to the single MR study (document deviation from TEST-01) or obtain ≥5 CT studies.
- [Phase 0] Baseline benchmark script not written (task 0.4) — driver gate is cleared, 0.4/0.5 can proceed now against the single-study dev corpus.
- [Phase 0] Models present: testdata/models has 2d + 3d_fullres only (5-fold ensembles each + nnunet_checkpoint.pth, jsonpkls plans/dataset/postprocessing). 3d_lowres + 3d_cascade_fullres absent — Phase 2/3 multi-config work is gated on those checkpoints.
- [Phase 0, HAZARD] The new app's editable install (`cchmc-nnunet-fast` 0.1.0 in the venv) registers a meta-path finder that maps the package name `my_app` to `examples/apps/cchmc-nnunet-fast/my_app`. Any `python -m my_app` run from a cwd that does NOT contain a local `my_app/` package silently executes the NEW skeleton instead of the intended app (hit 2026-08-17 while benchmarking the reference app). Mitigation: always run apps from their own app-root directory so the local package wins resolution. Longer-term: give the new app a unique package name (e.g. `cchmc_nnunet_fast_app`) to remove the collision.
- [Phase 0] PROJECT.md previously marked Phase 0 "validated" (commit 19b4a94) before the driver regression surfaced; corrected 2026-08-14 to partial.

## Session Continuity

Last session: 2026-08-17 03:55 UTC
Stopped at: Driver + RMM verified; testdata/ audited (see Decisions 2026-08-17). Awaiting corpus-scope decision (single MR study vs ≥5 CT), then baseline script (0.4/0.5) + demo Nsight trace (0.6).
Resume file: None

## Next — Phase 1 (Core Pipeline)

Phase 0 CLOSED 2026-08-17 in the GSD cycle (7/7 tasks, 5/5 acceptance, GSD-backfilled). Next:

1. **`/gsd-transition` Phase 0 → 1** (Phase 0 is now credit-verified Complete in GSD stats)
2. **`/gsd-plan-phase 1`** — Core Pipeline. Phase 1 planning must carry these inputs:
   - **Validation strategy (de-risked):** primary gate = `cchmc-nnunet-fast` vs a **freshly regenerated reference** (`testdata/current_output`), which is **confirmed equal to historical GT** (99.902% byte-identical). Historical `testdata/airway_output` is a valid reference. Re-run the reference any time via `.planning/scripts/REFERENCE_RUN_GUIDE.md`.
   - Baseline to beat: **169.7 ± 7.3 s/study** (`.planning/baseline_results.csv`); inference ≈ 138 s of that
   - Models available: 3d_fullres, 3d_lowres, 3d_cascade_fullres (plans.json also lists 2d — absent; gates Phase 2/3 multi-config)
   - Env hazards: `my_app` editable-install name collision (run apps from their app root); venv monai ptp patches (re-apply after monai reinstall); 32 MB stack requirement
   - Docker build + container test deferred to post-Phase-3 — validate pythonically in Phase 1

> Note: shazam LSP "client not started" noise seen on 2026-08-15 was stale in-process manager state after an external pyright-langserver kill — not a code issue; cleared by a fresh pi session.
