# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-14)

**Core value:** Single-study inference latency without sacrificing correctness — CT in, pixel-identical DICOM-SEG out, faster, every intermediate step stays on GPU.
**Current focus:** Phase 0: Foundation — closing remaining acceptance criteria

## Current Position

Phase: 0 of 4 (Foundation)
Plan: 0 of 0 in current phase (Phase 0 executed ad-hoc, outside the GSD plan/execute cycle)
Status: Phase 0 partial — tasks 0.1, 0.2, 0.7 ✓; 0.6 ◐ (harness ready, no demo trace); 0.3 ◐ (1 MR study w/ ground truth, ≥5 CT corpus still missing); 0.4, 0.5 ✗ (driver no longer gates these)
Last activity: 2026-08-17 - Driver update verified (610.57.04 / CUDA 13.3, A100-SXM4-40GB); test_rmm.py PASSED (pluggable RMM allocator active) — env blocker CLEARED. testdata/ audit: input = 1 MR series (62 dcm, patient 01153813), models = 2d + 3d_fullres 5-fold ensembles + jsonpkls, output = SC(1)+SEG(2)+SR(1) matching input series UID

Progress: [░░░░░░░░░░░░░░░░░░░░] 0/0 plans (0%)

## Phase 0 Acceptance Status

| # | Criterion | Status |
|---|-----------|--------|
| 1 | pyproject.toml + all cu13 deps resolved in venv | ✓ met — full import chain verified 2026-08-14 (holoscan-cu13 4.2.0 repaired, holoscan-cli 4.2.0, pydicom 3.0.2, highdicom 0.28.1; monai.deploy.core + all example-app operator imports OK) |
| 2 | Reference corpus ≥5 CT studies with DICOM-SEG/SR ground truth | ✗ not met — testdata/input holds 1 MR series (62 dcm, patient 01153813) with matching SC+SEG(2)+SR ground truth in testdata/output (SEG SourceSeriesUID == input series UID, verified 2026-08-17); ≥5 CT studies still missing |
| 3 | Baseline benchmark CSV at .planning/baseline_results.csv | ✗ not met — script (0.4) and CSV both missing |
| 4 | Nsight harness produces valid trace | ◐ partial — .planning/scripts/nsight_profile.sh + nvtx_markers.py ready, nsys 2025.6.3 in PATH; demo trace not yet generated |
| 5 | RMM pool allocator verified active | ✓ met — 2026-08-17: driver 610.57.04 (CUDA 13.3), torch.cuda.is_available()=True on A100-SXM4-40GB, test_rmm.py PASSED (rmm.allocators.torch active, backend 'pluggable') |

Done: app scaffold (examples/apps/cchmc-nnunet-fast, standard MAP layout, app.py skeleton), cu13 pins (commit 7d5e687), Nsight/NVTX harness (d99deb4), RMM + venv validation scripts (8910ae8, 205fcd3).

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

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 0] Reference corpus gap (task 0.3): testdata/input is 1 MR series (62 dcm) with complete SC/SEG/SR ground truth — usable as a single-study dev corpus, but the ≥5 CT studies criterion is unmet and no CT data exists in the repo. Decision needed: relax criterion to the single MR study (document deviation from TEST-01) or obtain ≥5 CT studies.
- [Phase 0] Baseline benchmark script not written (task 0.4) — driver gate is cleared, 0.4/0.5 can proceed now against the single-study dev corpus.
- [Phase 0] Models present: testdata/models has 2d + 3d_fullres only (5-fold ensembles each + nnunet_checkpoint.pth, jsonpkls plans/dataset/postprocessing). 3d_lowres + 3d_cascade_fullres absent — Phase 2/3 multi-config work is gated on those checkpoints.
- [Phase 0] PROJECT.md previously marked Phase 0 "validated" (commit 19b4a94) before the driver regression surfaced; corrected 2026-08-14 to partial.

## Session Continuity

Last session: 2026-08-17 03:55 UTC
Stopped at: Driver + RMM verified; testdata/ audited (see Decisions 2026-08-17). Awaiting corpus-scope decision (single MR study vs ≥5 CT), then baseline script (0.4/0.5) + demo Nsight trace (0.6).
Resume file: None

## Resume — remaining Phase 0 work (post-driver-clear, 2026-08-17)

Done: driver 610.57.04/CUDA 13.3 confirmed, A100 visible to torch, test_rmm.py PASSED, testdata audited (input=1 MR series; models=2d+3d_fullres; output=SC/SEG/SR matching input UID).

1. **Corpus decision (task 0.3):** single MR study in testdata/input with full ground truth vs ≥5 CT studies. If relaxing: document TEST-01 deviation in REQUIREMENTS.md/ROADMAP.md.
2. Task 0.4 — write baseline benchmark script (CSV: study, total ms, per-stage ms)
3. Task 0.5 — run it on `cchmc_nnunet_fifteen_ckpt_app` → `.planning/baseline_results.csv`
4. Task 0.6 — generate one demo Nsight trace to prove the harness end-to-end
5. **Phase gate:** all 5 acceptance criteria ✓ (or documented deviation) → update ROADMAP traceability + PROJECT.md Validated, then `/gsd-transition` → Phase 1 (Core Pipeline).

> Note: shazam LSP "client not started" noise seen on 2026-08-15 was stale in-process manager state after an external pyright-langserver kill — not a code issue; cleared by a fresh pi session.
