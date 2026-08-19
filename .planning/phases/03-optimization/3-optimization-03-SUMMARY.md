---
phase: 3-optimization
plan: "03"
subsystem: memory-lifecycle
tags: [infr-02, d-24, shape-cache, buffer-reuse, cupy, rmm, dlpack, aliasing-bug, multi-study-replay, pixel-exact-gate]

requires:
  - phase: 3-optimization (plans 01-02)
    provides: shipping state (concurrent-fragments default ON + RMM initial_pool_size pinned 4 GiB + MEM-003 release hook) that the D-25 gate and the replay replicate; the Plan-01 churn baseline (10 cudaMalloc / 1 cudaFree per bundle study)
provides:
  - "_ShapeCache (my_app/operators/buffer_cache.py): (shape, dtype) -> device buffer for torch AND cupy families — get(shape, dtype, zero=False), clear(), keys(), items(), total_bytes(), shares_storage(); no LRU (config-determined key set)"
  - "CuPy-side gap closed: PreprocessOperator vol/mask/vol_c/one_hot/vol2 come from the shape cache (D-13 scipy round-trips untouched)"
  - "Torch-side: SlideWindowOperator predicted_logits/n_predictions borrowed zero=True, per-patch workon borrowed + copy_, gaussian computed ONCE in setup(); padded explicitly NOT cached"
  - "INFR-02 proof per D-24(a)+(b): headless unit suite (7 cases incl. the multi-fold accumulation aliasing regression) + one-process 3x-same-study replay (address stability, cudaMalloc flat 0/0 on studies 2/3, byte-identical repeat outputs) -> evidence/infr02_proof.md"
  - "Gate evidence: gates/03-GATE-infr02.json (D-25 suite fully green with caches active, pixel-identical to the Phase 2 baseline)"
affects: [phase 3 plans 04-05 — the shipping DAG now shape-caches both allocator families; Plan 04's GPU-resample experiment sees the cached vol_c/vol2 as the resample input/output sites; Plan 05 close-out cites the replay proof + D-26 external-dependency record]

tech-stack:
  added: []
  patterns:
    - "Shape-keyed (shape, str(dtype)) dict cache per operator INSTANCE (no LRU, no eviction — config-determined keys); zero_()/fill(0) on borrow only where the reference allocates fresh zeros"
    - "DLPack emit-boundary rule: a cached buffer never crosses an operator boundary — emit a fresh tensor or clone() when shares_storage() detects aliasing (views caught via storage-base / cp.shares_memory)"
    - "Cross-autocast/cross-scope tensor escaping a re-borrowed buffer: clone at the accumulation site (fold-1 clone in predict_logits) — the cache's aliasing rule, unit-tested as a regression"

key-files:
  created:
    - examples/apps/cchmc-nnunet-fast/my_app/operators/buffer_cache.py
    - examples/apps/cchmc-nnunet-fast/scripts/test_buffer_cache.py
    - .planning/scripts/infr02_replay.py
    - .planning/phases/03-optimization/evidence/infr02_proof.md
    - .planning/phases/03-optimization/gates/03-GATE-infr02.json
    - .planning/profiles/phase3/infr02_replay_20260819_151533.{nsys-rep,sqlite,cuda_api_sum.txt}
  modified:
    - examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py (5 study-sized CuPy sites -> cache; per-site decision table comment)
    - examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py (4 torch sites + setup-time gaussian + fold-1 aliasing clone + emit-boundary shares_storage guard)

key-decisions:
  - "Rule 1 bug caught by the D-25 gate and fixed: the WIP's per-fold predicted_logits borrow made fold 1's returned logits a VIEW of the cached buffer that fold 2 re-borrows with zero=True — the next fold's zero_() wiped the running sum mid-accumulation (gate showed 1363 differing voxels / IoU 0.56 vs the 3-voxel baseline). Fix: predict_logits clones the fold-1 result when a cache is active (the running sum must live on a tensor the cache never touches). The emit-boundary clone alone was too late — the damage is inside the accumulation. Pinned by test_buffer_cache.py case 7c (bug reproduced without the clone)"
  - "Replay used the plan's PREFERRED direct-operator-drive path (no GXF app): real operators + SDK load chain + real DLPack handoffs, one process, 3x same study — data_ptr tables of studies 2/3 byte-identical to study 1, cudaMalloc 0/0 on studies 2/3 (bootstrap 1 + study 1 = 8 sub-1 ms pool expansions; 9/1 total process), VRAM 5,469 MiB flat (+0.00%), seg byte-identical + SR text identical"
  - "vol_c identity-crop aliasing is BY DESIGN: the airway study's crop is 256->256, so vol_c's (shape, dtype) key collides with vol's and both names use the same buffer — safe under the full-overwrite-before-read contract (documented in the proof record); non-identity crops get distinct keys (shape-key invalidation unit-tested)"
  - "padded (F.pad result) is explicitly NOT cached — the copy_ into a cache buffer costs about the RMM-pooled alloc it saves at this size; the hot per-patch workon IS cached (non-decision recorded in code)"

requirements-completed: [INFR-02, TEST-01, TEST-002, TEST-003]

duration: ~155min
completed: 2026-08-19
---

# Phase 3 Plan 03: INFR-02 Cross-Study GPU Buffer Reuse (D-24) Summary

**Both allocator families are now shape-cached at the big fixed-shape study sites — the CuPy gap (independent pool, bypasses RMM) is closed, the torch traffic is reduced, the gaussian is computed once — and INFR-02 is PROVEN per D-24(a)+(b), not asserted: headless unit suite green, and in one process the same study run 3× shows byte-identical cached-buffer addresses, cudaMalloc exactly flat (0/0) on studies 2/3, and byte-identical repeat outputs; the full D-25 gate suite passes with the caches active, pixel-identical to the Phase 2 baseline. The gate re-run also caught and fixed a real Rule 1 aliasing bug in the WIP's fold accumulation (1363 differing voxels → 3).**

## What was built

### Task 1 — `_ShapeCache` module + CuPy-side integration in PreprocessOperator + unit suite (270f3c9, pre-existing committed work of the interrupted run)

- **`my_app/operators/buffer_cache.py`**: `_ShapeCache(device, family="torch"|"cupy")` — `(tuple(shape), str(dtype))` → device buffer dict; `get(shape, dtype, zero=False)` (`zero_()`/`fill(0)` on borrow only where the reference allocates fresh zeros), `clear()`, `keys()`, `items()`, `total_bytes()`, `shares_storage()` (storage-base / `cp.shares_memory` so OFFSET VIEWS are caught, not just base pointers). No LRU/eviction — the key set is config-determined (few shapes per config); C-contiguity asserted at allocation (D-12). One cache per operator INSTANCE (a single operator's compute() is never re-entered; the concurrent scheduler runs different instances).
- **`preprocess_operator.py`** per-site decision table (also in the code comment at cache creation):

  | site | decision | zero | reason |
  |---|---|---|---|
  | `vol` transpose materialization (64 MB fp32) | CACHED | False | copy fully overwrites before any read |
  | `mask` (16 MB uint8) | CACHED | False | `cp.not_equal(out=...)` fully overwrites; ORs modify in place |
  | `vol_c` crop materialization (64 MB fp32) | CACHED | False | crop copy fully overwrites (identity crop → aliases `vol` by key, safe) |
  | `one_hot` channels (64 MB fp32, cascade) | CACHED | False | `cp.equal(out=...)` fills every channel |
  | `vol2` channel concat (128 MB fp32, cascade) | CACHED | False | both block copies fully overwrite |
  | `cp.array(raw_numpy)` H2D source copy | UNTOUCHED | — | pure input transfer, not in the D-24 inventory |
  | seg4 G2D copy + cascade seg layout | UNTOUCHED | — | input transfer |
  | per-channel normalize temporaries | UNTOUCHED | — | outside inventory; writes land in cached `vol_c`; temps ride CuPy's own pool |
  | emit boundary | FRESH by construction | — | compute() emits `torch.as_tensor(volume.get()).to(cuda)` — a cached CuPy buffer never crosses the DLPack boundary |
  | D-13 scipy resample round-trips | UNTOUCHED | — | GPUP-01 is Plan 04; OFF path stays byte-for-byte Phase 2/3 |

- **`scripts/test_buffer_cache.py`** (headless, synthetic sizes, exit 0/1): cases 1–6 (multi-call reuse, shape-key invalidation, dtype/contiguity invariants, zero semantics, clear(), accounting + shares_storage incl. offset views — both families).

### Task 2 — Torch-side cache in SlideWindowOperator + D-25 gate re-run (23d9fab, WIP reviewed, completed, and bug-fixed by this executor)

- **`slidewindow_operator.py`** per-site table (also in the code comment):

  | site | decision | zero | reason |
  |---|---|---|---|
  | `predicted_logits` (133 MB airway / 512 MB class) | CACHED | True | reference = fresh `torch.zeros`; **ALIASING RULE**: the returned view must not back the accumulation (fold-1 clone below) |
  | `n_predictions` (64 MB class) | CACHED | True | reference = fresh `torch.zeros`; consumed inside the fold (no escape) |
  | `gaussian` (8.4 MB, 128³) | computed ONCE in setup() | — | identical every fold (RESEARCH inventory); read-only in the loop (`prediction * gaussian`, `n_predictions += gaussian`) |
  | per-patch `workon` (8–16 MB × ~27 patches × 5 folds — hottest site) | CACHED | False | `copy_` fully overwrites before the forward reads it |
  | `padded` (F.pad result) | NOT CACHED (explicit non-decision, in code) | — | the extra `copy_` costs ≈ the RMM-pooled alloc it saves |
  | emit boundary | `shares_storage()` → `clone()` | — | uniform DLPack rule: a cached buffer (or a view of one) never crosses to PostResample |

- **The Rule 1 fix (the substantive work of this task):** `predict_logits` — when a cache is active, `prediction = fold_logits.clone()` on the first fold. Without it, fold 1's running sum lived on a view of the per-fold cached buffer; fold 2's `zero=True` borrow wiped it in place and `prediction += fold_logits` added the buffer to itself (each fold destroyed the partial sum). The WIP's emit-boundary clone could not save it (too late). Bisection evidence: pre-plan-03 state = 3 voxels; Task-1-only = 3 voxels; full WIP = 1363 voxels / IoU 0.56; with the fold-1 clone = 3 voxels / IoU 0.998714 (exact baseline). All 5 per-fold logits are bit-identical with/without the bug — only the cross-fold accumulation differs (fold-level dumps used to localize).
- **`test_buffer_cache.py` case 7** (torch-site semantics): 7a zero-on-borrow (3 fake folds), 7b gaussian-once (one tensor, same data_ptr, read-only loop), **7c the aliasing regression** — with-cache accumulation (fold-1 clone) == fresh-allocation reference, AND without the clone the sum IS corrupted (the caught bug, for the record).
- **D-25 gate re-run (final shipping state: caches ACTIVE + concurrent ON + RMM 4 GiB pin + release hook), device 0:** `gates/03-GATE-infr02.json` — **ALL GATES PASS, pixel-identical to the Phase 2 baseline**: fullres **99.99986 %/3** (documented fp16↔fp32 boundary class, IoU 0.998714), lowres/cascade/bundle **100.00000 %/0**, SR **0.0 %** delta ×4, residency static + runtime **PASS**, sanity (bundle ensembles: 382 differing voxels). Note: the stale mid-interruption JSON (99.95456 %/1363) was from the pre-fix code state and is replaced by this final-state run.

### Task 3 — D-24(b) multi-study replay proof + evidence record (9a4f200)

- **`.planning/scripts/infr02_replay.py`** — ONE process, 3 sequential passes over the same airway study through the REAL operators (**direct-drive path — the plan's preferred option; no GXF app**): SDK load chain (`DICOMDataLoader → Selector → SeriesToVolume`) + `Preprocess → SlideWindow → PostResample → EnsembleAverage → Postprocess` (3d_fullres) with the real DLPack `holoscan.core.Tensor` handoffs. Per study: post-preprocess+inference `data_ptr()` snapshot of every cache entry + gaussian ptr; pynvml VRAM; final seg `np.array_equal` + SR text. `--analyze` mode: per-study cudaMalloc/cudaFree from the nsys sqlite (CUPTI_ACTIVITY_KIND_RUNTIME; study windows = Nth `preprocess_3d_fullres` → Nth `postprocess` NVTX).
- **Results (full-process nsys, device 0 — `infr02_replay_20260819_151533.*`):**
  - (i) **Address stability: PASS** — studies 2/3 data_ptr tables byte-identical to study 1 (5 cached buffers + gaussian ptr: preprocess 22709673132032 / 22709740240896; slidewindow 22705631426048 / 22705764597248 / 22705831182848; gaussian 22705556451840).
  - (ii) **Churn FLAT: PASS** — bootstrap 1 cudaMalloc (RMM 4 GiB reservation, 1.51 ms) + study 1 = 8 (all < 1 ms pool expansions: 722/600/332/497/441/259/220/8 µs) / **study 2 = 0 / study 3 = 0**; total process 9 malloc / 1 free (Plan-01 baseline class: 10/1 per BUNDLE study — 3 configs; this is a single-config chain over the whole process). Never per-tile (180,538 memcpy + 43,398 memset + 627,124 kernel launches).
  - (iii) **VRAM flat: PASS** — 5,469 MiB after each study (+0.00 %/±5 % bar); **outputs byte-identical: PASS** — seg payloads `np.array_equal` across studies (2,331 post-CC voxels), SR text identical (`Airway Volume: 1 mL`).
  - Per-study wall 45.4 → 41.3 → 41.3 s (study 1 = first-touch/CuPy pool warm-up; honest scope: single-study-per-run is the clinical model — INFR-02 ships as provable reuse + reduced allocator traffic, not a first-study speed claim).
- **`evidence/infr02_proof.md`** — the (a)+(b) record: data_ptr table, per-study cudaMalloc table vs the Plan-01 baseline citation, byte-identity results, pool-occupancy row, the vol_c identity-crop aliasing note, and the **D-26 external-dependency paragraph**: the user's INFR-02 reference examples are blocked-on-external, non-blocking (arrived too late for the gate) — shipped with (a)+(b) per D-24; fold into the oracle if they land before VERIFICATION.md.

## Must-haves check

- [x] GPU buffers reused across compute() calls keyed on (shape, dtype) — Nth study reuses the 1st study's (proven: data_ptr stability + flat churn)
- [x] CuPy-side gap closed (vol/vol_c/mask/one_hot/vol2 from the shape cache; CuPy pool independent of RMM)
- [x] Torch-side big fixed-shape allocations cached (predicted_logits/n_predictions zero-on-borrow; gaussian once in setup; per-patch workon — the hottest allocation)
- [x] `test_buffer_cache.py` passes headlessly (reuse, shape-key invalidation, dtype/contiguity invariants, zero-on-borrow; both families) + torch-site semantics incl. the aliasing regression
- [x] Multi-study replay (same study 3×, one process): data_ptr stable across studies AND nsys cudaMalloc flat on studies 2/3 (Phase 2 Plan 06 churn method)
- [x] All 4 pixel-exact gates + SR + residency PASS with the caches active (D-25 anchor) → `03-GATE-infr02.json`
- [x] D-13 scipy resample path untouched (GPUP-01 → Plan 04); OFF path byte-for-byte Phase 2/3

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fold-1 aliasing corruption of the multi-fold accumulation (caught by the D-25 gate, fixed + regression-tested)**
- **Found during:** Task 2 gate re-run — full WIP state gave fullres 99.95456 %/1363 differing voxels (IoU 0.56) vs the 3-voxel baseline (the stale mid-interruption JSON was this same pre-fix state; the final gate run replaced it)
- **Issue:** `predict_logits` assigned `prediction = fold_logits`, a view of the per-fold CACHED `predicted_logits` buffer; fold 2 re-borrows that buffer with `zero=True`, wiping the running sum in place, and `prediction += fold_logits` then added the buffer to itself — deterministic cross-fold corruption (19 micro-blobs, all within 3 voxels of the airway — a logits-boundary shift, not a structural change)
- **Fix:** clone the fold-1 result when a cache is active (`prediction = fold_logits.clone() if buf_cache is not None else fold_logits`) + the per-site aliasing-rule comment + `test_buffer_cache.py` case 7c (asserts with-clone == fresh-allocation reference AND that without the clone the sum is corrupted). Post-fix gate: exact baseline (99.99986 %/3)
- **Files modified:** `slidewindow_operator.py`, `test_buffer_cache.py`
- **Commit:** 23d9fab
- **Bisection record (real-app E2E, fullres-only, device 0):** pre-plan-03 (a10b62d) = 3 voxels · Task-1-only (270f3c9) = 3 voxels · full WIP = 1363 · Variant A (workon uncached) = 1363 (workon innocent) · Variant B (logits/n_pred uncached) = 3 voxels (site localized) · fold-level dumps: all 5 folds bit-identical, final accumulation differs (aliasing confirmed)

**2. [Rule 3 - Blocking] Interrupted-execution WIP completed, not redone (and one WIP harness bug fixed)**
- **Found during:** resume — the previous executor died mid-Task-2/3 with uncommitted WIP
- **Issue:** WIP in `slidewindow_operator.py`/`test_buffer_cache.py` was reviewed and found structurally sound (kept, completed); the untracked `infr02_replay.py` had three latent bugs (never executed by the dead executor): `load_data_to_studies(str(...))` (SDK requires `Path`), `snapshot()` calling `cache.get(key)` with a `(shape, dtype_str)` key (wrong arity), `output_folder` set as `str` (operator does `Path / "temp"`); the analyze mode also had a sqlite row-unpacking bug (found on first `--analyze` run)
- **Fix:** all four fixed inline (`items()` added to `_ShapeCache` as the proof-record accessor); replay then passed first run
- **Files modified:** `.planning/scripts/infr02_replay.py`, `buffer_cache.py`
- **Commit:** 9a4f200

No other deviations. The D-26 external-dependency (user's INFR-02 reference examples) is recorded per the plan — it is an expected external item, not a code deviation.

## Known Stubs

None. (The replay is fullres-only by design — cascade-config cache sites `one_hot`/`vol2` are exercised by the D-25 gate's cascade row at 100.00000 %/0, not by the replay.)

## Commits

- `270f3c9` perf(03-03): INFR-02 _ShapeCache + CuPy-side buffer reuse in PreprocessOperator (D-24) — pre-existing committed work of the interrupted run (Task 1)
- `23d9fab` perf(03-03): INFR-02 torch-side shape cache in SlideWindowOperator (D-24) — incl. the Rule 1 fold-1 aliasing fix, case-7 tests, final green `03-GATE-infr02.json`
- `9a4f200` perf(03-03): D-24(b) multi-study replay proof — address stability + flat churn + byte-identical repeats (replay harness, `items()`, proof record, nsys artifacts)

## Self-Check: PASSED

All 9 referenced artifacts (buffer_cache.py, test_buffer_cache.py, infr02_replay.py, infr02_proof.md, 03-GATE-infr02.json, SUMMARY, nsys rep/sqlite/cuda_api_sum) verified present and all 3 commits (270f3c9, 23d9fab, 9a4f200) verified on the branch on 2026-08-19. Gate JSON re-parsed: `all_gates_pass: true`, fullres 99.99986%/3, lowres/cascade/bundle 100.0%/0.
