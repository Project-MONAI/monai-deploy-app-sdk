# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-14)

**Core value:** Single-study inference latency without sacrificing correctness — CT in, pixel-identical DICOM-SEG out, faster, every intermediate step stays on GPU.
**Current focus:** Phase 0: Foundation — closing remaining acceptance criteria

## Current Position

Phase: 0 of 4 (Foundation)
Plan: 0 of 0 in current phase (Phase 0 executed ad-hoc, outside the GSD plan/execute cycle)
Status: Phase 0 partial — 5/7 tasks done; venv stack repaired & importable (2026-08-14); remaining gate is GPU driver
Last activity: 2026-08-15 - Venv repaired + static analysis pointed at venv (commits b2420c0, aa3f9ea); user stepped away to update CUDA driver 12.8→13; resume instructions in "Resume after driver update"

Progress: [░░░░░░░░░░░░░░░░░░░░] 0/0 plans (0%)

## Phase 0 Acceptance Status

| # | Criterion | Status |
|---|-----------|--------|
| 1 | pyproject.toml + all cu13 deps resolved in venv | ✓ met — full import chain verified 2026-08-14 (holoscan-cu13 4.2.0 repaired, holoscan-cli 4.2.0, pydicom 3.0.2, highdicom 0.28.1; monai.deploy.core + all example-app operator imports OK) |
| 2 | Reference corpus ≥5 CT studies with DICOM-SEG/SR ground truth | ✗ not met — testdata/ holds 1 MR series (62 dcm) with 2 SEG + 1 SR outputs |
| 3 | Baseline benchmark CSV at .planning/baseline_results.csv | ✗ not met — script (0.4) and CSV both missing |
| 4 | Nsight harness produces valid trace | ◐ partial — .planning/scripts/nsight_profile.sh + nvtx_markers.py ready, nsys 2025.6.3 in PATH; demo trace not yet generated |
| 5 | RMM pool allocator verified active | ✗ not met — test_rmm.py import/config path now works; GPU allocation step SKIPs (driver-only issue) |

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

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 0, BLOCKER] Host driver mismatch: driver 570.211.01 (CUDA 12.8) is too old for torch 2.13.0+cu130, cupy-cuda13x 13.6, rmm-cu13 26.2. `torch.cuda` init fails; test_rmm.py exits SKIP at the GPU allocation step. Nothing GPU-resident runs on this host until driver ≥ 580.x (CUDA 13) or work moves to a cu13-capable host/container. This gates tasks 0.5 (baseline run), 0.7 (RMM verify), and all of Phase 1 execution. **This is now the only environment blocker** — the former holoscan import blocker was a corrupted install, repaired 2026-08-14 (see Decisions).
- [Phase 0] Reference corpus missing: need ≥5 CT studies from cchmc_nnunet_fifteen_ckpt_app with ground-truth DICOM-SEG + DICOM-SR, checksummed (task 0.3).
- [Phase 0] Baseline benchmark script not written (task 0.4) — .planning/baseline_results.csv cannot exist until 0.3 + driver are resolved.
- [Phase 0] PROJECT.md previously marked Phase 0 "validated" (commit 19b4a94) before the driver regression surfaced; corrected 2026-08-14 to partial.

## Session Continuity

Last session: 2026-08-15 18:04
Stopped at: Paused by user to update CUDA driver (12.8 → 13). All code + docs committed on `nnunet-fast`. On return: (1) verify driver + RMM, (2) resume Phase 0 tasks 0.3/0.4/0.5/0.6. See "Resume after driver update" below.
Resume file: None

## Resume after driver update

1. **Confirm driver:** `nvidia-smi | head -4` → expect `CUDA Version: 13.x` (R580+ driver). A reboot may be required after the driver install.
2. **Re-run the GPU smoke test** — the single gate for the env blocker:
   `/tmp/monai-env/.venv/bin/python .planning/scripts/test_rmm.py`
   - prints `cudaAsync` / passes → driver blocker **cleared**. Update the Blockers section + acceptance row #5 here to ✓, commit.
   - still `SKIP` / `cudaErrorInsufficientDriver` → driver not effective yet (reboot? wrong driver package?). Fix before continuing.
3. **Resume Phase 0** — remaining acceptance work:
   - Task 0.3 — assemble reference corpus (≥5 CT studies from `cchmc_nnunet_fifteen_ckpt_app` + ground-truth DICOM-SEG/SR, checksummed)
   - Task 0.4 — write baseline benchmark script (CSV: study, total ms, per-stage ms)
   - Task 0.5 — run it on the reference app → `.planning/baseline_results.csv`
   - Task 0.6 — generate one demo Nsight trace to prove the harness end-to-end
4. **Phase gate:** all 5 acceptance criteria ✓ → update ROADMAP traceability + PROJECT.md Validated, then `/gsd-transition` → Phase 1 (Core Pipeline).

> Note: shazam LSP "client not started" noise seen on 2026-08-15 was stale in-process manager state after an external pyright-langserver kill — not a code issue; cleared by a fresh pi session.
