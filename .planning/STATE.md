# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-14)

**Core value:** Single-study inference latency without sacrificing correctness — CT in, pixel-identical DICOM-SEG out, faster, every intermediate step stays on GPU.
**Current focus:** Phase 0: Foundation — closing remaining acceptance criteria

## Current Position

Phase: 0 of 4 (Foundation)
Plan: 0 of 0 in current phase (Phase 0 executed ad-hoc, outside the GSD plan/execute cycle)
Status: Phase 0 partial — 5/7 tasks done; 1/5 acceptance criteria met; GPU driver blocker on current host
Last activity: 2026-08-14 - Progress review; STATE.md initialized; driver/CUDA-13 mismatch + holoscan 4.x import failures discovered

Progress: [░░░░░░░░░░░░░░░░░░░░] 0/0 plans (0%)

## Phase 0 Acceptance Status

| # | Criterion | Status |
|---|-----------|--------|
| 1 | pyproject.toml + all cu13 deps resolved in venv | ✓ met at pip level (holoscan-cu13 4.2.0, cupy-cuda13x 13.6.0, rmm-cu13 26.2.0, torch 2.13.0+cu130, monai 1.3.0, nnunetv2 editable) — ⚠ but stack not importable, see Blockers |
| 2 | Reference corpus ≥5 CT studies with DICOM-SEG/SR ground truth | ✗ not met — testdata/ holds 1 MR series (62 dcm) with 2 SEG + 1 SR outputs |
| 3 | Baseline benchmark CSV at .planning/baseline_results.csv | ✗ not met — script (0.4) and CSV both missing |
| 4 | Nsight harness produces valid trace | ◐ partial — .planning/scripts/nsight_profile.sh + nvtx_markers.py ready, nsys 2025.6.3 in PATH; demo trace not yet generated |
| 5 | RMM pool allocator verified active | ✗ not met — test_rmm.py now SKIPs (see blocker) |

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
- [Phase 0]: Resampling stays on reference CPU (scipy) path in Phases 1–2 — pixel-exactness over speed

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 0, BLOCKER] Holoscan 4.x stack is not importable in the venv — three verified layers (2026-08-14): (1) `import monai.deploy` fails: SDK package-level code still uses the pre-4.0 API (`holoscan.core`, `holoscan.conditions`, `holoscan.executors`, `holoscan.logger`, `holoscan.resources`), which does not exist in holoscan-cu13 4.2.0; partial 4.x compat exists in `monai/deploy/graphs/` + `utils/importutil.py` but is unreachable because `monai/deploy/__init__.py` crashes first. (2) `import holoscan.flow_graphs` fails: `libgxf_rmm.so` (+ `libgxf_std/app/ucx/serialization/core/cuda`, `libucs.so.0`) not found — the GXF 4.x runtime libraries are not installed; the wheel only bundles `holoscan/lib/` (gxf_pubsub, holoscan_core 4.2.0, etc.). Source of the missing GXF runtime (package vs container) is an open question. (3) Consequence: no example app (including the new scaffold) can run yet, independent of the driver issue below.
- [Phase 0, BLOCKER] Host driver mismatch: driver 570.211.01 (CUDA 12.8) is too old for torch 2.13.0+cu130, cupy-cuda13x 13.6, rmm-cu13 26.2. `torch.cuda` init fails; test_rmm.py exits SKIP. Nothing GPU-resident runs on this host until driver ≥ 580.x (CUDA 13) or work moves to a cu13-capable host/container. This gates tasks 0.5 (baseline run), 0.7 (RMM verify), and all of Phase 1 execution.
- [Phase 0] Reference corpus missing: need ≥5 CT studies from cchmc_nnunet_fifteen_ckpt_app with ground-truth DICOM-SEG + DICOM-SR, checksummed (task 0.3).
- [Phase 0] Baseline benchmark script not written (task 0.4) — .planning/baseline_results.csv cannot exist until 0.3 + driver are resolved.
- [Phase 0] PROJECT.md previously marked Phase 0 "validated" (commit 19b4a94) before the driver regression surfaced; corrected 2026-08-14 to partial.

## Session Continuity

Last session: 2026-08-14 18:10
Stopped at: Pre-commit LSP flag chased — confirmed holoscan 4.x stack non-importable (old-API imports in SDK + missing GXF runtime libs); STATE.md blockers updated
Resume file: None
