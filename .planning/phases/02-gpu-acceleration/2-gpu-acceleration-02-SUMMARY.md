---
phase: 2-gpu-acceleration
plan: 02
subsystem: gpu-infrastructure
tags: [rmm, pool-allocator, import-order, memory-budget, defer-strategy, warm-pool, d-14, d-15, d-17]

requires:
  - phase: 2-gpu-acceleration (plan 01)
    provides: CuPy-ported pipeline (parallel-safe — disjoint files), fullres-only E2E running clean
  - phase: 0-foundation
    provides: rmm-cu13 26.02.00 in the venv, test_rmm.py verified rmm→torch sequence

provides:
  - "RMM as torch's active CUDA allocator (backend 'pluggable') in every app process, imported before holoscan (INFR-01/D-14)"
  - "Import-order hazard pinned by a subprocess self-test (rmm-first -> pluggable; holoscan-first -> undefined symbol __cxa_call_terminate)"
  - "RMM pool pre-allocation in compose(): warm_pool(budget.total_bytes) (INFR-01/D-14)"
  - "compute_memory_budget(...) -> BudgetPlan (full_volume | defer_to_incremental), unit-tested headless with synthetic sizes forcing the defer branch (INFR-03/D-15)"
  - "EnsembleAverageOperator defer_strategy branch (code-reachable; real OOM documented unexercised)"
  - "Gate evidence: .planning/phases/02-gpu-acceleration/plan02-gates/ (pixel_diff JSON, E2E log excerpt, mem_budget test output)"

affects: [phase 2 plans 03-06 — the gpu_bootstrap first-import and the compose() budget block must survive Plan 04's compose() rewrite (Plan 04 is instructed to keep them); Plan 06's nsys cudaMalloc churn check relies on the warmed pool]

tech-stack:
  added: []
  patterns:
    - "gpu_bootstrap module = FIRST import in app.py (before json/pydicom/monai.deploy); only rmm (+ lazy torch) inside it"
    - "import-order hazard is a per-process fact — pinned by subprocesses, not in-process asserts"
    - "Pure-arithmetic calculator with an injectable free_vram_bytes so unit tests never touch the GPU"
    - "Defer branch = identical accumulation order + exact final division with earlier input release (bit-identical to full_volume path, D-19)"

key-files:
  created:
    - examples/apps/cchmc-nnunet-fast/my_app/gpu_bootstrap.py
    - examples/apps/cchmc-nnunet-fast/my_app/mem_budget.py
    - examples/apps/cchmc-nnunet-fast/scripts/test_gpu_bootstrap.py
    - examples/apps/cchmc-nnunet-fast/scripts/test_mem_budget.py
    - .planning/phases/02-gpu-acceleration/plan02-gates/pixel_diff_fullres.json
    - .planning/phases/02-gpu-acceleration/plan02-gates/e2e_log_excerpt.txt
    - .planning/phases/02-gpu-acceleration/plan02-gates/test_mem_budget.txt
  modified:
    - examples/apps/cchmc-nnunet-fast/my_app/app.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py

key-decisions:
  - "cudnn.benchmark disabled when the RMM pluggable allocator is active (torch 2.13 benchmark search calls the pluggable allocator's unsupported cacheInfo and raises on the first conv, size-independent) — RMM (INFR-01/D-14) wins over the reference's benchmark=True parity; pixel-exactness preserved (99.99986% SEG, 3 documented boundary voxels)"
  - "Budget estimate uses plans.json median_image_size_in_voxels as the preprocessed shape (study volume size is unknown at compose time); crop bounded by the resampled volume (crop ⊆ image)"
  - "Defer branch frees per-config tensors as consumed instead of changing the math — a true running mean would break D-19 bit-exactness"
  - "INFR-02 (cross-study/cross-compute buffer reuse) explicitly DEFERRED to Phase 3 per D-17 — documented in code, not implemented"
  - "Shipped allocator: RMM primary (pluggable backend logged per run); PYTORCH_CUDA_ALLOC_CONF=expandable_segments NOT set anywhere (Pitfall 6)"

patterns-established:
  - "Allocator/backend assertions in compose() fail fast at startup with an actionable message instead of mid-inference OOM noise"
  - "Structured setup-time records: memory_allocator_backend: <s> and memory_budget: {JSON} log lines (parseable by Plan 06's benchmarking)"

requirements-completed: [INFR-01, INFR-03]
requirements-deferred: [INFR-02]  # D-17 — Phase 3; documented in code, not implemented

deviations:
  - "Rule 1 (bug): test_gpu_bootstrap.py `if failures:` on a list of booleans always truthy — fixed to any(failures)"
  - "Rule 1/3 (bug/blocker): RMM pluggable allocator + torch.backends.cudnn.benchmark=True is broken in torch 2.13 (RuntimeError 'cacheInfo' on first conv3d, reproduced size-independently) — benchmark disabled under pluggable; see key-decisions"
  - "Plan-text fix: the defer test sketch (3 configs of (4,600,512,512), 40 GB free) totals ~24.4 GB under the plan's own formula — not defer; the test uses 5 configs of the same shapes (40.6 GB > 40 GB) to force the branch (D-15 intent; the required synthetic shape is unchanged)"

duration: ~40min
completed: 2026-08-19
---

# Phase 2 Plan 02: RMM Pool Pre-allocation + Memory Budget Calculator Summary

**RMM is now torch's active CUDA allocator (pluggable backend) in every app process — imported before holoscan with the undefined-symbol hazard pinned by a subprocess self-test — and a headless-unit-tested memory-budget calculator logs a `memory_budget: {...}` plan in compose(), pre-warms the RMM pool, and drives a code-reachable (documented-unexercised) one-config-at-a-time defer branch in the ensemble; fullres-only E2E stays exit 0 with `strategy: full_volume` and 99.99986% SEG byte-identity vs ref_fullres_only.**

## Performance

- **Duration:** ~40 min (2 fullres E2E runs + 1 pixel-diff gate + extensive RMM/cudnn-benchmark root-cause bisect dominate)
- **Tasks:** 2 (both auto; no checkpoints)
- **Files:** 7 created + 3 modified (+3 evidence files)

## What Was Built

### Task 1 — `gpu_bootstrap.py` + app.py wiring + import-order self-test (commit 1a27e45)

- `my_app/gpu_bootstrap.py`: `import rmm` + `rmm.reinitialize(pool_allocator=True, managed_memory=False)` +
  `rmm_torch_allocator` at module import (rmm 26.x API — no `PoolAllocator`);
  `install_torch_allocator()` (returns the backend; "pluggable" when RMM is
  active); `warm_pool(num_bytes)` (allocate-then-release so the pool retains
  the memory). No holoscan/MONAI Deploy imports anywhere in the module.
- `my_app/app.py`: the bootstrap is the FIRST import (before `import json`,
  pydicom, and every `monai.deploy` line); `compose()` logs
  `memory_allocator_backend: <s>` and asserts `backend == "pluggable"`.
- `scripts/test_gpu_bootstrap.py`: spawns two subprocesses with the venv
  python — (a) rmm-first: import gpu_bootstrap → `monai.deploy.core` →
  prints `pluggable` (exit 0); (b) holoscan-first: `monai.deploy.core` →
  `import rmm` → non-zero exit with `undefined symbol` in stderr (the live
  hazard, reported explicitly if the failure mode ever changes). Both PASS.

### Task 2 — `mem_budget.py` + tests + defer wiring + pool warm-up (commit 88773fa)

- `my_app/mem_budget.py`: `BudgetPlan` (frozen dataclass: per_config_mb,
  total_bytes, free_vram_bytes, safety_factor, strategy) +
  `compute_memory_budget(cfgs, free_vram_bytes=None, safety_factor=1.15)` —
  per-config fp32 estimate = preprocessed volume + post-crop logits +
  full-size probabilities; `total = sum × safety_factor`; `full_volume` when
  `total <= free` (inclusive), else `defer_to_incremental`. Pure arithmetic;
  torch only for the optional default free-VRAM probe → headless-testable.
- `scripts/test_mem_budget.py`: 4 tests, all PASS, GPU-free (explicit
  `free_vram_bytes`): airway-like full_volume; synthetic forced defer
  ((4, 600, 512, 512) set, ~2.5 GB preprocessed per config); inclusive
  boundary (total == free → full_volume, total−1 → defer); per_config_mb
  keys.
- `my_app/app.py` compose(): `_compute_budget_plan(model_path)` builds the
  cfg entry via `load_inference_params` + plans.json
  `median_image_size_in_voxels` (fallback: original median shape × spacing
  ratio for inherited configs); logs `memory_budget: {JSON}` (includes
  strategy/total_bytes/free_vram_bytes); passes
  `defer_strategy=(plan.strategy == "defer_to_incremental")` to
  `EnsembleAverageOperator`; calls `gpu_bootstrap.warm_pool(plan.total_bytes)`
  at the END of compose().
- `EnsembleAverageOperator`: `defer_strategy: bool = False` kwarg
  (initialized before `super().__init__`, same pattern as
  `emit_averaged_probabilities`). Defer branch =
  `_average_probabilities_defer`: identical accumulation order (first
  volume = base, sequential in-place `+=`) and the same CuPy
  `_divide_refparity` exact final division, with each consumed per-config
  tensor released as it accumulates (D-19 bit-exactness preserved; a running
  mean would break it). Module docstring + inline comments document
  INFR-02 deferral (D-17) and the UNEXERCISED real OOM path (D-15).
  `defer_strategy=False` (the default, and the airway-study runtime value)
  is byte-for-byte the Phase 1 code path.

## Gate Evidence (`.planning/phases/02-gpu-acceleration/plan02-gates/`)

- `test_gpu_bootstrap.py` exit 0: case (a) stdout `allocator_backend: pluggable`;
  case (b) non-zero exit, stderr `undefined symbol`.
- `test_mem_budget.py` exit 0: 4 PASS lines.
- Fullres-only E2E exit 0: log contains
  `memory_allocator_backend: pluggable` and
  `memory_budget: {"per_config_mb": {"3d_fullres": 335.54}, "total_bytes": 385875968, ..., "strategy": "full_volume"}`;
  SEG/SR/SC produced.
- Precautionary pixel gate (cudnn-benchmark change): vs `testdata/ref_fullres_only`
  → **99.99986% SEG byte-identity, 3 differing voxels** (the documented
  fp16↔fp32 argmax-boundary class; Phase 1 measured 3, Plan 01 measured 2),
  IoU 0.998714 — PASS.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Self-test always reported FAIL**
- **Found during:** Task 1 verification
- **Issue:** `failures` was a list of booleans; `if failures:` is truthy even
  when every entry is False, so an all-PASS run exited 1.
- **Fix:** `if any(failures):`.
- **Files modified:** `scripts/test_gpu_bootstrap.py`
- **Commit:** 1a27e45

**2. [Rule 1/3 - Bug/Blocker] RMM pluggable allocator breaks cudnn benchmark mode (torch 2.13)**
- **Found during:** Task 1 E2E regression (exit 1, `RuntimeError: CUDAPluggableAllocator does not yet support cacheInfo` in the first conv of `slidewindow_op`).
- **Issue:** Reproduced outside holoscan, bisected to
  `torch.backends.cudnn.benchmark = True` (set for nnUNet-reference parity):
  the benchmark's algorithm search unconditionally calls the pluggable
  allocator's unsupported `cacheInfo()` — fails on ANY conv size (32³
  included), so it is not a memory issue. RMM (INFR-01/D-14, locked) and
  benchmark-mode parity are mutually exclusive on torch 2.13.
- **Fix:** `slidewindow_operator.load_model_bundle` keeps
  `cudnn.benchmark = True` for native-allocator processes but flips it to
  False (with a comment) when
  `torch.cuda.memory.get_allocator_backend() == "pluggable"`. RMM wins.
  Verified: E2E exit 0 and the pixel gate above (99.99986%, 3 boundary
  voxels — within the documented Phase 1 tolerance class).
- **Files modified:** `my_app/operators/slidewindow_operator.py`
- **Commit:** 1a27e45

**3. [Plan-text inconsistency] Defer test cannot force defer with 3 configs**
- **Found during:** Task 2 (deriving the unit-test numbers)
- **Issue:** The plan's `test_defer_forced_synthetic` sketch (3 configs,
  preprocessed (4, 600, 512, 512), cropped (4, 550, 480, 480), heads=4,
  40 GB free) totals ~24.4 GB under the plan's own formula (per config
  2.52 GB preprocessed + 2.03 GB logits + 2.52 GB probabilities = 7.06 GB;
  ×3 ×1.15 = 24.4 GB < 40 GB) — it would select `full_volume`.
- **Fix:** The test uses **5 configs of the identical shapes** (5 × 7.06 ×
  1.15 = 40.6 GB > 40 GB → `defer_to_incremental`), keeping the required
  synthetic shape `(4, 600, 512, 512)` and the acceptance assertion; the
  rationale is documented in the test docstring. D-15's intent (synthetic
  sizes that force the defer branch) is met.
- **Files modified:** `scripts/test_mem_budget.py`
- **Commit:** 88773fa

### Notes (not deviations)

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was **not** set anywhere
  (Pitfall 6): RMM is the shipped strategy; expandable_segments remains the
  documented fallback only if instability appears (none observed).
- INFR-02 was deliberately NOT implemented (D-17) — code comments only.

## Known Stubs

None.

## Self-Check: PASSED

- All 7 created files exist (verified 2026-08-19).
- Commits 1a27e45 and 88773fa present in git log.
