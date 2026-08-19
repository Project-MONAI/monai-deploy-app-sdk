# Open Q1: RMM initial-pool reservation — live re-verification (Phase 3, Plan 01, Task 1)

**Date:** 2026-08-19 · **Device:** NVIDIA A100-SXM4-40GB, **CUDA_VISIBLE_DEVICES=0**
(pinned per Pitfall 7; device 0 was free — 0 MiB used — at run time; devices 4–7
were tenant-occupied) · **Driver:** 610.57.04 / CUDA 13.3 · **rmm:** 26.02.00 (live venv `/tmp/monai-env/.venv`)

## Decision

`initial_pool_size: pinned 4 GiB`

## Measurement (method: research Open Q1 protocol)

Probe: `.planning/scripts/probes-phase3/probe_rmm_openq1.py` — fresh subprocess
from `examples/apps/cchmc-nnunet-fast`, `ulimit -s unlimited`, `CUDA_VISIBLE_DEVICES=0`;
pynvml `nvmlDeviceGetMemoryInfo` on device 0 BEFORE and immediately AFTER
`import my_app.gpu_bootstrap` (which runs `rmm.reinitialize(...)` at import —
no torch allocation yet).

| | used (GiB) |
|---|---|
| before import | 0.485 |
| after import (before pin) | 20.450 |
| **post-reinit reservation S (before pin)** | **19.97** |
| after import (after pin) | 4.900 (`rmm_probe_after_pin.json`) |
| post-reinit reservation S (after pin) | 4.41 |

## Finding

The 2026-08-19 research measurement holds against the LIVE venv: rmm 26.2.0's
default `initial_pool_size` (½ total GPU memory) reserves **S = 19.97 GiB
immediately at reinitialize**, before any torch allocation. This is
inconsistent with the Phase 2 trace's ~1.98 GB *total* cudaMalloc — the Phase 2
trace was taken before the scratch venv drifted to this rmm default
(Pitfall 8 environment drift confirmed for RMM; Phase 2 evidence stands as the
history it recorded, unchanged).

Decision rule from the plan: S ≥ 10 GiB → pin. S = 19.97 GiB → **pinned**.

## The pin

`gpu_bootstrap.py` now passes `initial_pool_size=4 * 1024**3` to
`rmm.reinitialize` (plus a module-level `logging.info("rmm initial_pool_size: …")`
at the reinitialize site; the import-order invariant — rmm before any
holoscan/torch CUDA init — is untouched).

Why 4 GiB: a fresh bundle run's log records
`memory_budget: {"total_bytes": 1038502513, "free_vram_bytes": …, "strategy": "full_volume"}`
(i.e. the airway bundle budget total is ~0.97 GiB — 1,038,502,513 bytes).
4 GiB = ~4.1× that total; `warm_pool(plan.total_bytes)` at compose end still
grows the pool to the per-bundle budget per D-14, so the pin only removes the
wasteful 20 GiB default reservation, never the warm-up. After the pin, the
same run logs `free_vram_bytes: 37659410432` (was 20,976,566,272) and
`memory_allocator_backend: pluggable`, exit 0.

## Churn baseline (INFR-02 baseline for Plan 03) — full-process nsys, bundle rep, device 0

| artifact | cudaMalloc | cudaFree | cudaLaunchKernel (+ExC) |
|---|---|---|---|
| before pin — `rmm_q1_baseline_20260819_101503.{nsys-rep,sqlite}` + `_cuda_api_sum.txt` | 9 | 1 | 540,700 (+87,480 ExC) |
| after pin — `rmm_q1_pinned_20260819_102101.{nsys-rep,sqlite}` + `_cuda_api_sum.txt` | 10 | 1 | 540,700 (+87,480 ExC) |

Kernel-launch counts are identical before/after the pin (inference untouched).
The +1 cudaMalloc after pinning is a pool-expansion event (all mallocs < 2 ms;
none per-tile) — the correct INFR-02 baseline for Plan 03's "cudaMalloc flat
across studies" proof is now the PINNED run: **10 cudaMalloc / 1 cudaFree for
one full bundle study**. The no-pin run is retained as the before-evidence
(the 9 mallocs there include the single 20 GiB initial-pool reservation).
