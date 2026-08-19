---
phase: "2-gpu-acceleration"
plan: "02"
type: "execute"
wave: 1
depends_on: []
files_modified:
  - "examples/apps/cchmc-nnunet-fast/my_app/gpu_bootstrap.py"
  - "examples/apps/cchmc-nnunet-fast/my_app/mem_budget.py"
  - "examples/apps/cchmc-nnunet-fast/my_app/app.py"
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py"
  - "examples/apps/cchmc-nnunet-fast/scripts/test_gpu_bootstrap.py"
  - "examples/apps/cchmc-nnunet-fast/scripts/test_mem_budget.py"
autonomous: true
requirements: [INFR-01, INFR-02, INFR-03]
must_haves:
  truths:
    - "`import rmm` happens before any holoscan/monai.deploy import (the undefined-symbol hazard is live-reproduced and documented), so torch's allocator backend is \"pluggable\" (INFR-01, D-14)"
    - "RMM pool pre-allocation runs in setup/compose: a warm tensor sized by the budget calculator is allocated and released so the pool holds the memory before study 1 (INFR-01)"
    - "A pure-Python memory-budget calculator returns a BudgetPlan with strategy \"full_volume\" | \"defer_to_incremental\" and is unit-tested with synthetic large-volume sizes that force the defer branch (INFR-03, D-15)"
    - "The defer branch is reachable in code (ensemble frees each consumed probability tensor after accumulating) even though the real OOM path never triggers on the A100-40GB airway study — documented as unexercised, not faked"
    - "INFR-02 (pre-allocated buffers reused across compute() calls / cross-study reuse) is EXPLICITLY DEFERRED to Phase 3 per D-17 — documented in code comments and the plan, with NO implementation attempted"
    - "fullres-only E2E still exits 0 with RMM active (allocator backend logged)"
  artifacts:
    - path: "examples/apps/cchmc-nnunet-fast/my_app/gpu_bootstrap.py"
      provides: "RMM-before-holoscan bootstrap: reinitialize(pool_allocator=True) + rmm_torch_allocator + warm_pool"
      contains: "rmm.reinitialize(pool_allocator=True"
    - path: "examples/apps/cchmc-nnunet-fast/my_app/mem_budget.py"
      provides: "compute_memory_budget(...) -> BudgetPlan (dataclass, strategy field)"
      contains: "defer_to_incremental"
    - path: "examples/apps/cchmc-nnunet-fast/scripts/test_mem_budget.py"
      provides: "synthetic-size unit tests incl. a forced defer-to-incremental case"
    - path: "examples/apps/cchmc-nnunet-fast/scripts/test_gpu_bootstrap.py"
      provides: "import-order self-test (subprocesses) — rmm-first OK/pluggable, holoscan-first fails with undefined symbol"
  key_links:
    - from: "my_app/app.py"
      to: "my_app/gpu_bootstrap.py"
      via: "the FIRST import in app.py (before `from monai.deploy...`)"
      pattern: "gpu_bootstrap"
    - from: "app.py compose()"
      to: "mem_budget.compute_memory_budget"
      via: "BudgetPlan logged as a `memory_budget: {...}` JSON record; warm_pool(total_bytes) called at end of compose"
      pattern: "memory_budget:"
    - from: "EnsembleAverageOperator"
      to: "BudgetPlan.strategy"
      via: "constructor flag `defer_strategy` (default False = Phase 1 behavior)"
      pattern: "defer_strategy"
---

# Phase 2 Plan 02: RMM Pool Pre-allocation + Memory Budget Calculator

## Objective
- **What:** Wire RMM as the CUDA allocator (import-order-correct, pool pre-allocated in
  setup/compose per INFR-01/D-14) and add a unit-tested memory-budget calculator with a
  reachable defer-to-incremental branch (INFR-03/D-15). Document INFR-02 as deferred to
  Phase 3 (D-17).
- **Why:** RMM removes per-tile `cudaMalloc` churn (Phase 2 acceptance criterion); the
  budget calculator is the safety net for full-volume logits/probability allocations;
  both are prerequisites the DAG plans build on.
- **Output:** `gpu_bootstrap.py`, `mem_budget.py`, two test scripts, app.py/ensemble
  wiring, passing tests + fullres-only E2E regression.

## Execution Environment

- Python: `/tmp/monai-env/.venv/bin/python` only. GPU: A100-SXM4-40GB, driver 610.57.04.
- Run the fast app from its app root:
  ```bash
  cd examples/apps/cchmc-nnunet-fast && ulimit -s unlimited
  /tmp/monai-env/.venv/bin/python my_app -i /users/srv-mde/projects/monai-deploy-app-sdk/testdata/airway_input \
    -m /users/srv-mde/projects/monai-deploy-app-sdk/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models -o <scratch>
  ```
- rmm-cu13 26.02.00 is already in the venv. **API note (verified 2026-08-19):**
  `rmm.mr.PoolAllocator` does NOT exist in 26.x — use
  `rmm.reinitialize(pool_allocator=True)` / `rmm.mr.PoolMemoryResource`.
- Commit after each major change.

## Context

@.planning/STATE.md
@.planning/phases/02-gpu-acceleration/02-CONTEXT.md
@.planning/phases/02-gpu-acceleration/02-RESEARCH.md
@examples/apps/cchmc-nnunet-fast/my_app/app.py
@examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py
@.planning/scripts/test_rmm.py

**The single most likely first-day blocker (RESEARCH Pitfall 1, live-reproduced 2026-08-19):**
`import rmm` AFTER `import holoscan` raises
`ImportError: undefined symbol: __cxa_call_terminate`. `import rmm` must therefore be the
first thing that happens in the process (the Phase 0 smoke test only passed because it
never imported holoscan). Also Pitfall 6: do NOT also set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — RMM and expandable_segments are
alternative torch allocation strategies; RMM is primary, expandable_segments is only the
documented fallback if instability appears (record which one shipped in the SUMMARY).

Locked: D-14 (RMM required), D-15 (budget calculator unit-tested with synthetic sizes;
real OOM path documented unexercised), D-17 (INFR-02 → Phase 3; do NOT implement
cross-study buffer reuse).

This plan is parallel-safe with Plan 01 (disjoint files; Plan 01 touches
`preprocess_operator.py` + `gpu_residency.py` only). app.py changes here are additive
(first import + compose additions) and Plan 04 later rewrites `compose()` — the RMM
bootstrap import and the budget call must survive that rewrite (Plan 04 is instructed to
keep them).

## Tasks

<task type="auto">
  <name>Task 1: gpu_bootstrap.py (RMM before holoscan) + app.py wiring + import-order self-test</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/gpu_bootstrap.py, examples/apps/cchmc-nnunet-fast/my_app/app.py, examples/apps/cchmc-nnunet-fast/scripts/test_gpu_bootstrap.py</files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/my_app/app.py (current import block — `from monai.deploy...` lines; where `compose()` and `run()` are; the existing `from my_app.operators import (...)` / flat-import fallback pattern to mirror)
    - .planning/scripts/test_rmm.py (Phase 0 RMM smoke test — the verified rmm→torch sequence)
    - .planning/phases/02-gpu-acceleration/02-RESEARCH.md (Pattern 6 "RMM wiring that works in this venv" code block; Pitfalls 1 and 6)
  </read_first>
  <action>
1. Create `my_app/gpu_bootstrap.py` (no holoscan/monai.deploy imports anywhere in it):
   ```python
   """RMM bootstrap — MUST be imported before any holoscan/monai.deploy import.

   Verified hazard (2026-08-19, live-reproduced): `import rmm` after `import holoscan`
   raises ImportError: undefined symbol: __cxa_call_terminate. Importing this module
   first installs RMM as torch's CUDA allocator (pool allocator, INFR-01/D-14).
   rmm 26.x API: reinitialize(pool_allocator=True) — PoolAllocator does not exist.
   """
   import rmm  # noqa: F401  (must precede any holoscan import)

   rmm.reinitialize(pool_allocator=True, managed_memory=False)
   from rmm.allocators.torch import rmm_torch_allocator

   def install_torch_allocator() -> str:
       import torch
       if torch.cuda.is_available():
           torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
       return torch.cuda.memory.get_allocator_backend()

   def warm_pool(num_bytes: int) -> None:
       """Pre-allocate then release so the RMM pool retains the memory before
       study 1's tiles start (pool pre-allocation in setup, INFR-01/D-14)."""
       import torch
       n = max(1, int(num_bytes) // 4)
       buf = torch.empty(n, dtype=torch.float32, device="cuda")
       del buf
   ```
2. In `my_app/app.py`, make the bootstrap the FIRST import (directly after the module
   docstring, BEFORE `import json`/`from pydicom...`/`from monai.deploy...`), mirroring
   the existing dual-import style:
   ```python
   try:
       from my_app import gpu_bootstrap
   except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
       import gpu_bootstrap
   gpu_bootstrap.install_torch_allocator()
   ```
   with a comment: `# INFR-01/D-14: RMM must be imported before holoscan (undefined-symbol hazard)`.
3. In `compose()` (after `app_context`/`model_path` resolution, before operators are
   built) add:
   ```python
   backend = torch.cuda.memory.get_allocator_backend()
   self._logger.info("memory_allocator_backend: %s", backend)
   assert backend == "pluggable", (
       f"RMM torch allocator not active (backend={backend!r}); "
       "gpu_bootstrap must be imported before holoscan (INFR-01)"
   )
   ```
   (`import torch` at top of app.py is fine — torch before holoscan does not trip the
   hazard; only rmm-after-holoscan does.)
4. Create `scripts/test_gpu_bootstrap.py` — a self-contained test that spawns two
   SUBPROCESSES with `/tmp/monai-env/.venv/bin/python`:
   (a) rmm-first: a snippet that imports `gpu_bootstrap` (sys.path = app root), then
       `import monai.deploy.core`, then prints
       `torch.cuda.memory.get_allocator_backend()` → expect exit 0 and stdout
       `pluggable`;
   (b) holoscan-first: a snippet that imports `monai.deploy.core` first, then
       `import rmm` → expect non-zero exit and stderr containing `undefined symbol`
       (documents the live hazard; if the failure mode ever changes, the test reports
       it explicitly rather than silently passing).
   Print PASS/FAIL per case; exit non-zero if (a) fails.
5. Regression: fullres-only E2E (Execution Environment) exits 0; the log line
   `memory_allocator_backend: pluggable` appears in stdout.
  </action>
  <acceptance_criteria>
    - `head -40 my_app/app.py` shows the `gpu_bootstrap` import block appearing before the first `monai.deploy` import line (verified with `grep -n "gpu_bootstrap\|monai.deploy" my_app/app.py | head` — the bootstrap line number is smaller).
    - `grep -n "rmm.reinitialize(pool_allocator=True" my_app/gpu_bootstrap.py` returns 1 line; `grep -rn "PoolAllocator" my_app/gpu_bootstrap.py` returns nothing (rmm 26.x API).
    - `grep -n "import rmm\|import holoscan\|monai.deploy" my_app/gpu_bootstrap.py` shows ONLY `import rmm` (no holoscan/monai.deploy import in the module).
    - `/tmp/monai-env/.venv/bin/python scripts/test_gpu_bootstrap.py` exits 0; its output contains `pluggable` for case (a) and `undefined symbol` for case (b).
    - `grep -n "assert backend" my_app/app.py` returns 1 line (the pluggable assertion in compose()).
    - Fullres-only E2E exit 0 with a log line matching `memory_allocator_backend: pluggable`.
  </acceptance_criteria>
  <verify>cd examples/apps/cchmc-nnunet-fast && /tmp/monai-env/.venv/bin/python scripts/test_gpu_bootstrap.py && ulimit -s unlimited && /tmp/monai-env/.venv/bin/python my_app -i /users/srv-mde/projects/monai-deploy-app-sdk/testdata/airway_input -m /users/srv-mde/projects/monai-deploy-app-sdk/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models -o /tmp/p2p2_e2e 2>&1 | grep -E "memory_allocator_backend|End my_app" ; ls /tmp/p2p2_e2e/SEG</verify>
  <done>RMM is the active torch allocator (backend "pluggable") in every app process; the import-order hazard is pinned by a subprocess self-test; fullres-only E2E regresses clean.</done>
</task>

<task type="auto">
  <name>Task 2: mem_budget.py calculator + synthetic-size unit tests + defer-strategy wiring + pool warm-up</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/mem_budget.py, examples/apps/cchmc-nnunet-fast/scripts/test_mem_budget.py, examples/apps/cchmc-nnunet-fast/my_app/app.py, examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py</files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py (the Phase 1 `average_probabilities` in-place `+=` accumulation + `_divide_refparity` CuPy final division — MUST stay byte-for-byte for the full_volume path; constructor + `setup()`/`compute()` to see where the defer flag plugs in)
    - examples/apps/cchmc-nnunet-fast/my_app/config/__init__.py (`InferenceParams` — num_input_channels / num_segmentation_heads / patch_size; `load_inference_params` used to build the budget inputs)
    - .planning/phases/02-gpu-acceleration/02-RESEARCH.md (Pattern 7 memory-budget design; D-15)
    - examples/apps/cchmc-nnunet-fast/my_app/app.py (post-Task-1 state: compose() with the backend assertion)
  </read_first>
  <action>
1. Create `my_app/mem_budget.py`:
   ```python
   @dataclass(frozen=True)
   class BudgetPlan:
       per_config_mb: Dict[str, float]
       total_bytes: int
       free_vram_bytes: int
       safety_factor: float
       strategy: str  # "full_volume" | "defer_to_incremental"

   def compute_memory_budget(
       cfgs: Sequence[Mapping[str, Any]],
       free_vram_bytes: Optional[int] = None,
       safety_factor: float = 1.15,
   ) -> BudgetPlan:
   ```
   where each `cfg` mapping has: `config_name` (str), `num_input_channels` (int),
   `num_segmentation_heads` (int), `preprocessed_shape` (tuple, C×H×D×W after resample),
   `cropped_shape` (tuple, C×H×D×W pre-resample). Per-config estimate (bytes, fp32 = 4B/element):
   - preprocessed volume: prod(preprocessed_shape) × 4
   - logits (post-crop): heads × prod(cropped_shape[1:]) × 4
   - probabilities (original): heads × prod(preprocessed_shape[1:]) × 4
   `total = sum(per-config) × safety_factor`. `free_vram_bytes` defaults to
   `torch.cuda.mem_get_info()[0]`. `strategy = "full_volume" if total <= free_vram_bytes
   else "defer_to_incremental"`. Pure arithmetic — the only import for the calculator
   itself is optional torch (for the default free-VRAM probe); the function must work
   with an explicit `free_vram_bytes` argument so unit tests never need a GPU.
2. Create `scripts/test_mem_budget.py` (plain asserts, runnable headless — no GPU needed
   because `free_vram_bytes` is passed explicitly), with AT LEAST these cases (agent may
   add more; sizes are deliberately chosen to force the branches):
   - `test_full_volume_airway_like`: 3 configs, preprocessed (1, 240, 240, 240) fp32,
     cropped (1, 216, 216, 216), heads=2, heads_cascade=2, `free_vram_bytes = 40 GB`
     → `strategy == "full_volume"`.
   - `test_defer_forced_synthetic`: 3 configs, preprocessed (4, 600, 512, 512), cropped
     (4, 550, 480, 480), heads=4, `free_vram_bytes = 40 GB` → per-config preprocessed
     alone ≈ 2.5 GB → `strategy == "defer_to_incremental"`.
   - `test_boundary`: `free_vram_bytes` exactly equal to `total_bytes` →
     `"full_volume"` (document the inclusive `<=` semantics in the docstring).
   - `test_per_config_mb_keys`: `per_config_mb` has one key per input config.
   Print a PASS line per test; exit non-zero on any failure.
3. app.py wiring (in `compose()`, after the backend assertion): build the `cfgs` list for
   the current `CONFIG_NAME` via `load_inference_params(model_path, CONFIG_NAME)` + the
   preprocessed/cropped shapes from a quick plans lookup (or `PreprocessParams`);
   `plan = compute_memory_budget(cfgs)`; log
   `self._logger.info("memory_budget: %s", json.dumps(asdict-ish dict))` (must include
   `strategy`, `total_bytes`, `free_vram_bytes`); then at the END of `compose()`:
   `gpu_bootstrap.warm_pool(plan.total_bytes)` (pool pre-allocation — D-14 "pre-allocate
   GPU buffers in setup()": this is the setup-time warm allocation that lets the pool
   hold the memory before study 1's tiles).
   Pass `defer_strategy=(plan.strategy == "defer_to_incremental")` to the
   `EnsembleAverageOperator` construction.
4. `EnsembleAverageOperator`: add constructor kwarg `defer_strategy: bool = False`
   (initialize BEFORE `super().__init__` — holoscan 4.2 calls `setup()` during init;
   same pattern as the existing `emit_averaged_probabilities` flag). In `compute()`,
   after the Phase 1 in-place accumulation loop, if `defer_strategy` is True: `del` each
   consumed per-config probability tensor as it is accumulated (the loop already streams
   one input at a time) and log `ensemble_average: defer_strategy active (one-config-at-a-time
   accumulation)`; if False: EXACTLY the Phase 1 code path (unchanged). The accumulation
   ORDER and `_divide_refparity` final division are UNCHANGED in both branches (D-19 —
   bit-exactness is non-negotiable; a true running mean would not be bit-identical to
   the reference's sum/n).
   Add a module docstring note:
   "INFR-02 (pre-allocated buffers reused ACROSS compute() calls / across studies) is
   DEFERRED to Phase 3 per D-17 — the single-study dev corpus cannot prove cross-study
   reuse. The RMM pool retains allocated memory process-wide, but no explicit buffer
   reuse is implemented or claimed here."
   and a comment documenting that the real OOM path is UNEXERCISED on the A100-40GB
   airway study (D-15) — the defer branch is reachable in code but never triggered here.
5. Run tests + regression:
   `/tmp/monai-env/.venv/bin/python scripts/test_mem_budget.py` → exit 0;
   fullres-only E2E → exit 0, log contains `memory_budget:` with `"strategy": "full_volume"`
   for the airway study.
  </action>
  <acceptance_criteria>
    - `grep -n "defer_to_incremental" my_app/mem_budget.py` matches (the strategy literal); `grep -n "def compute_memory_budget" my_app/mem_budget.py` returns 1 line.
    - `grep -c "assert" scripts/test_mem_budget.py` >= 4 (one cluster per test case above); the test file contains the synthetic shape `(4, 600, 512, 512)` and asserts `strategy == "defer_to_incremental"` for it, and `free_vram_bytes` is passed explicitly (no `torch.cuda.mem_get_info()` call in the test file — headless-safe).
    - `/tmp/monai-env/.venv/bin/python scripts/test_mem_budget.py` exits 0 and prints a PASS line per case.
    - `grep -n "defer_strategy" my_app/operators/ensemble_average_operator.py my_app/app.py` matches in BOTH files; the Phase 1 `_divide_refparity` and accumulation order are unchanged (`git diff` shows the full_volume path intact).
    - `grep -n "INFR-02" my_app/operators/ensemble_average_operator.py` returns >= 1 line (the D-17 deferral docstring); `grep -n "UNEXERCISED\|unexercised" my_app/operators/ensemble_average_operator.py my_app/app.py` >= 1 line (D-15 honesty note).
    - `grep -n "warm_pool" my_app/app.py` returns 1 line (called at the end of compose()); `grep -n "memory_budget:" my_app/app.py` returns 1 line (the structured log record).
    - Fullres-only E2E exit 0; its log contains `"strategy": "full_volume"`.
  </acceptance_criteria>
  <verify>cd examples/apps/cchmc-nnunet-fast && /tmp/monai-env/.venv/bin/python scripts/test_mem_budget.py && ulimit -s unlimited && /tmp/monai-env/.venv/bin/python my_app -i /users/srv-mde/projects/monai-deploy-app-sdk/testdata/airway_input -m /users/srv-mde/projects/monai-deploy-app-sdk/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models -o /tmp/p2p2_e2e_b 2>&1 | grep -E "memory_budget|memory_allocator_backend"</verify>
  <done>BudgetPlan computed, logged, and driving the ensemble defer flag; synthetic-size unit tests force the defer branch; pool warm-up sized by the budget; INFR-02 deferral and unexercised-OOM honesty documented; fullres-only E2E regresses clean.</done>
</task>

## Verification
- `scripts/test_gpu_bootstrap.py` exits 0 (rmm-first → `pluggable`; holoscan-first → `undefined symbol`).
- `scripts/test_mem_budget.py` exits 0 (full_volume, forced defer, boundary, keys).
- Fullres-only E2E exit 0 with `memory_allocator_backend: pluggable` and `memory_budget: ... "strategy": "full_volume"`.
- git log shows the atomic commits.

## Success Criteria
- [ ] INFR-01: RMM pool allocator active (pluggable backend), imported before holoscan, pool pre-allocated in setup/compose (D-14)
- [ ] INFR-03: budget calculator present, unit-tested with synthetic sizes forcing the defer branch; defer branch reachable in code; real OOM documented unexercised (D-15)
- [ ] INFR-02: explicitly documented as DEFERRED to Phase 3 (D-17) — not implemented, not claimed
- [ ] Fullres-only E2E regression passes (no allocator-induced breakage)
- [ ] No `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` shipped silently (RMM primary; fallback only on instability, recorded if used)
