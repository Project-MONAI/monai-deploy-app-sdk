---
phase: 3-optimization
plan: "02"
subsystem: memory-lifecycle
tags: [mem-003, d-23, weight-release, rmm, open-q2, pynvml, driver-vs-pool, pixel-exact-gate]

requires:
  - phase: 3-optimization (plan 01)
    provides: shipping state (concurrent-fragments default ON + RMM initial_pool_size pinned 4 GiB) that the D-25 gate and the measurement runs replicate
provides:
  - "MEM-003 release hook: SlideWindowOperator.release() + PostResampleOperator release_fn callback, wired for exactly the aux (lowres_seg-emitting) config in NnUnetConfigSubgraph.compose() — 3d_lowres weights (~0.8 GB) freed exactly once after the terminal emit"
  - "HOLOSCAN_KEEP_LOWRES_WEIGHTS=1 env opt-out at the injection site (kept, documented — measurement bypass for ON-vs-OFF peak comparison)"
  - "Open Q2 answered: under the rmm 26.2.0 pluggable allocator torch.cuda.empty_cache() does NOT return pool memory to the driver (silent driver-level no-op) — pool level down ~0.8 GB (derived), driver level flat (measured)"
  - "Headless unit suite scripts/test_weight_release.py (3 cases, RMM-active, pynvml-only measurement — no torch memory counters anywhere in this plan)"
  - "pynvml driver-level sampler .planning/scripts/vram_sampler.py (1–5 Hz → epoch-ns CSV, --until-file clean exit)"
  - "Gate evidence: .planning/phases/03-optimization/gates/03-GATE-mem003.json (D-25 suite fully green with the release hook active, pixel-identical to Plan 01)"
affects: [phase 3 plans 03-05 — the shipping DAG now releases aux weights; Plan 03 INFR-02 buffer reuse sees a ~0.8 GB smaller pool-occupied footprint after the lowres fragment; Plan 05 close-out report cites the Open-Q2 answer]

tech-stack:
  added: []
  patterns:
    - "Release-callback lifecycle: bound method (sw.release) captured by the downstream operator before its fragment is discarded; fires at the very end of compute() after ALL emits; guard in compute() turns a post-release invocation into a DAG-ordering-violation RuntimeError"
    - "ON-vs-OFF memory comparison via a one-line env opt-out at the injection site (HOLOSCAN_KEEP_LOWRES_WEIGHTS=1) instead of git-stash re-derivation — kept in-tree, documented"
    - "Two-level VRAM reporting under RMM: pynvml driver level (measured) + RMM pool level (derived from checkpoint arithmetic when rmm's Python API exposes no pool stats) — each labeled as measured/derived with an explicit per-level conclusion"

key-files:
  created:
    - examples/apps/cchmc-nnunet-fast/scripts/test_weight_release.py
    - .planning/scripts/vram_sampler.py
    - .planning/phases/03-optimization/evidence/mem003_vram.md
    - .planning/phases/03-optimization/evidence/mem003_vram_on.csv
    - .planning/phases/03-optimization/evidence/mem003_vram_off.csv
    - .planning/phases/03-optimization/evidence/mem003_run_on.log
    - .planning/phases/03-optimization/evidence/mem003_run_off.log
    - .planning/phases/03-optimization/evidence/sampler_on.log
    - .planning/phases/03-optimization/evidence/sampler_off.log
    - .planning/phases/03-optimization/gates/03-GATE-mem003.json
  modified:
    - examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py (release() + _released flag + compute() post-release guard)
    - examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py (release_fn kwarg captured before super().__init__ + call at compute() tail)
    - examples/apps/cchmc-nnunet-fast/my_app/app.py (the single release_fn= injection in NnUnetConfigSubgraph.compose() with the env opt-out)

key-decisions:
  - "Open Q2 resolved (measured, not assumed): torch.cuda.empty_cache() under the rmm 26.2.0 pluggable allocator is a silent driver-level no-op — the headless unit test released the real 0.8 GB 3d_lowres bundle under the 4 GiB pinned pool and pynvml moved 5,019 → 5,019 MiB (+0); the full bundle ON-vs-OFF runs give identical driver peaks (9.552 GiB both). Pool level is down ~0.8 GB but DERIVED (rmm 26.2.0 Python API exposes no pool stats — dir(rmm) probe empty), from the exact checkpoint arithmetic + the deterministic reference drop"
  - "Flat driver-level delta shipped as the valid, reportable result per research expectation — the RMM pool never shrinks its cudaMalloc-reserved blocks, so the driver keeps counting them; the peak is set by the concurrent fullres∥lowres inference phase that PRECEDES the release. MEM-003's real benefit is ~0.8 GB of pool-occupied (non-reusable) VRAM headroom during the cascade phase — meaningful for the deferred MEM-02 8 GB class, zero driver-visible or latency change on the A100-40GB (memory-lifecycle deliverable, not a speed lever — per D-23)"
  - "The env opt-out HOLOSCAN_KEEP_LOWRES_WEIGHTS=1 was added at the injection site and KEPT (plan's choice between env-var vs git-stash comparison — env var chosen for the minimal, in-tree diff); non-aux configs get release_fn=None so their behavior is byte-for-byte unchanged"

patterns-established:
  - "Hook semantics unit-tested headlessly with a bare monai.deploy Application() as fragment (holoscan 4.2 rejects None): synthetic-bundle case, real-bundle case (pynvml + compute guard), and the real compute() tail driven with a tiny synthetic logits tensor + fake op_input/op_output — no app run required"
  - "vram_sampler --until-file sentinel driven by a background log-tail helper on the app's own log (line-buffered stderr) — the sentinel fires on the post-cascade-complete timing line, so one CSV covers all three measurement moments"

requirements-completed: [MEM-003, TEST-01, TEST-002, TEST-003]

duration: ~40min
completed: 2026-08-19
---

# Phase 3 Plan 02: MEM-003 Lowres Weight Release + Pool/Driver VRAM Delta Summary

**The 3d_lowres weights (~0.8 GB: 5 × 135 MB fold state dicts + ~135 MB live network) are now freed exactly once, exactly for the aux (lowres_seg-emitting) configuration, immediately after its PostResample's terminal emit — and the memory effect is measured honestly at both levels: pool level down ~0.8 GB (derived, labeled as such), driver level flat (measured — Open Q2 answered: `torch.cuda.empty_cache()` is a silent driver-level no-op under the rmm 26.2.0 pluggable allocator), with the D-25 gate suite fully green and pixel-identical to Plan 01 with the hook active.**

## What was built

### Task 1 — Release hook + callback injection + headless unit test

- **`slidewindow_operator.py`:** `SlideWindowOperator.release()` — drops `self._bundle` to None (after capturing the fold count for the log), `del`s the bundle's `network` + `fold_state_dicts`, calls `torch.cuda.empty_cache()`, logs `weights released: <config> (folds=N) (MEM-003)`; safe no-op when already released or never loaded. New `_released` flag; `compute()` now distinguishes a post-release call (raises `compute() after release() — DAG ordering violation for <config> (MEM-003)`) from the pre-existing never-loaded guard.
- **`postresample_operator.py`:** `release_fn: Optional[Callable[[], None]] = None` kwarg, captured into `self._release_fn` **before** `super().__init__` (same discipline as the emit flags — Pitfall 7 port gating). The call sits at the very end of `compute()`, after ALL emits and the timing record: `if self._release_fn is not None: self._release_fn()`. Nothing else in the operator changed.
- **`app.py` (subgraph only):** the single injection line in `NnUnetConfigSubgraph.compose()` — `release_fn=sw.release if (self._emit_lowres_seg and os.environ.get("HOLOSCAN_KEEP_LOWRES_WEIGHTS") != "1") else None` on the PostResampleOperator construction. The bound method keeps `sw` alive via `post`; non-aux configs get `None` (byte-for-byte prior behavior). The env opt-out is the measurement bypass for Task 2's ON-vs-OFF peak comparison — kept and documented.
- **`scripts/test_weight_release.py`** (headless, exit 0, ~15 s; runs under the shipping import order — gpu_bootstrap/RMM first, device 0 pinned per Pitfall 7, pynvml-only measurement, zero torch memory counters):
  - **(a)** release() on a synthetic 2-fold bundle of 2×1×1 random CUDA tensors → `_bundle is None`, `weights released: 3d_lowres (folds=2) (MEM-003)` log line captured, second release() a no-op.
  - **(b)** release() on the REAL operator built from the airway bundle's 3d_lowres config after setup() (5-fold assert) → pynvml before/after: **5,019 → 5,019 MiB (+0 — the Open-Q2 data point under the 4 GiB pinned RMM pool)**, `_bundle is None`, and `compute()` raises the DAG-ordering-violation guard.
  - **(c)** a counter callable passed as `release_fn` to a bare PostResampleOperator (bare `Application()` fragment — holoscan 4.2 rejects `None`) fires **exactly once** when the real `compute()` tail executes (tiny synthetic (2,8,8,8) logits + fake op_input/op_output; lowres_seg emit verified) and `release_fn=None` keeps the no-callback path.

### Task 2 — Pool/driver VRAM delta + D-25 gate re-run

- **`.planning/scripts/vram_sampler.py`** (~80 lines): pynvml `nvmlDeviceGetMemoryInfo` sampler — `--device N --hz 2 --out CSV` (epoch-ns timestamps, `ts_ns,device,used_bytes,total_bytes`), clean exit on `--until-file` sentinel or SIGINT/SIGTERM, prints the CSV path.
- **Measurement runs (device 0, A100-SXM4-40GB; HOLOSCAN_MODEL_LIST unset = bundle cascade path; concurrent scheduler default ON = the Plan-01 shipping state; 2 Hz; sampler sentinel fired on the post-cascade-complete timing line so one CSV covers all 3 moments):**
  - **ON rep** (`mem003_run_on.log`, `mem003_vram_on.csv` 210 rows): exit 0; log contains `weights released: 3d_lowres (folds=5) (MEM-003)` at 11:35:15,282. Driver used VRAM at the 3 moments: **pre-lowres-inference 5.413 GiB → at release line 9.427 GiB → post-cascade-complete 9.552 GiB** (no drop after the release line). Peak **9.552 GiB** (during cascade inference).
  - **OFF rep** (`HOLOSCAN_KEEP_LOWRES_WEIGHTS=1`, `mem003_run_off.log` — verified: zero release lines, `mem003_vram_off.csv` 211 rows): exit 0; peak **9.552 GiB**.
  - **Peak ON-vs-OFF delta: 0.000 GiB.**
  - **Pool level:** DOWN ~0.8 GB after the release — **derived, labeled as such** (rmm 26.2.0's Python API exposes no pool stats — `dir(rmm)` pool/size probe returned `[]`); the derivation is the exact checkpoint arithmetic (5 × 135 MB + ~135 MB) + the deterministic reference drop + RMM pool free-list reclamation.
  - **`.planning/phases/03-optimization/evidence/mem003_vram.md`** records the device id, the 3-moment table, the ON-vs-OFF peak table, the measured/derived labeling, and the per-level conclusions — including that **flat driver level is the valid, expected result** (RMM pools do not shrink their reserved blocks; the peak is set by the concurrent inference phase that precedes the release) and the practical reading (~0.8 GB pool-occupied headroom for the cascade phase; no driver-visible or latency effect on the A100-40GB).
- **D-25 gate re-run (hook ACTIVE, `HOLOSCAN_CONCURRENT_FRAGMENTS=1` = shipping default):** `03-GATE-mem003.json` — **ALL GATES: PASS, pixel-identical to Plan 01**: fullres 99.99986%/3 (documented fp16↔fp32 boundary class), lowres 100.00000%/0, cascade 100.00000%/0, bundle 100.00000%/0; SR 0.0% delta ×4; residency static + runtime PASS; sanity (bundle ensembles: 382 differing voxels) OK.

## Must-haves check

- [x] 3d_lowres weights freed immediately after the aux fragment's PostResample emits lowres_seg (release hook at compute() tail, one line in app.py), nothing downstream touches the released bundle (unit-tested + DAG-ordering guard)
- [x] `test_weight_release.py` passes headlessly — release fires exactly once, only for the aux config (non-aux get `release_fn=None`), non-aux operators unaffected
- [x] Peak-VRAM delta measured at BOTH levels (RMM pool occupancy — derived + labeled; driver level — pynvml measured); `torch.cuda.memory_stats()`/`memory_allocated()` used NOWHERE (grep-verified across every file this plan touched)
- [x] All 4 pixel-exact config gates + SR + residency PASS with the release hook active (D-25 anchor) → `03-GATE-mem003.json`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] holoscan 4.2 rejects `None` as an Operator fragment**
- **Found during:** Task 1 (test design)
- **Issue:** the headless test suite needs bare operators, but `Operator.__init__` raises `ValueError: The first argument to an Operator's constructor must be the Fragment (Application) or Subgraph` when given `None` (live-probed).
- **Fix:** the tests construct a bare `monai.deploy.core.Application()` (no compose, no DAG) as the fragment — live-probed working; no app code affected.
- **Files modified:** `scripts/test_weight_release.py` only.
- **Commit:** 0886cc6

**2. [Rule 3 - Blocking] rmm 26.2.0 Python API exposes no pool statistics**
- **Found during:** Task 2 (pool-level measurement)
- **Issue:** the plan anticipated this ("if not, state that explicitly and derive the pool-level delta from budget math") — confirmed: `dir(rmm)` probe for pool/size attributes returns `[]`.
- **Fix:** pool level is reported as DERIVED (~0.8 GB, exact checkpoint arithmetic + deterministic reference drop) and labeled as such; driver level is directly measured at both the sampler and unit-test level — the plan's explicit fallback, taken.
- **Files modified:** evidence file only.
- **Commit:** 4244d0c

No other deviations — app.py changed by exactly the single `release_fn=` injection (with the plan-sanctioned env opt-out), the gate re-run used `phase2_gate.py` unchanged via its existing `--report` arg, and the measurement runs used the sampler's `--until-file` clean exit as designed.

## Known Stubs

None.

## Commits

- `0886cc6` perf(03-02): MEM-003 release hook — free 3d_lowres weights after aux terminal emit (HOLOSCAN_KEEP_LOWRES_WEIGHTS opt-out) + headless test
- `4244d0c` perf(03-02): MEM-003 pool/driver VRAM delta evidence (pynvml sampler, ON-vs-OFF) + D-25 gate re-run green with release hook active

## Self-Check: PASSED

All 12 referenced artifacts and both commits (0886cc6, 4244d0c) verified present on 2026-08-19.
