---
phase: 3-optimization
plan: "03"
type: execute
wave: 3
depends_on: ["3-optimization-02"]
files_modified:
  - examples/apps/cchmc-nnunet-fast/my_app/operators/buffer_cache.py
  - examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py
  - examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py
  - examples/apps/cchmc-nnunet-fast/scripts/test_buffer_cache.py
  - .planning/scripts/infr02_replay.py
autonomous: true
requirements: [INFR-02, TEST-01, TEST-002, TEST-003]

must_haves:
  truths:
    - "GPU buffers are reused across compute() calls keyed on (shape, dtype): the Nth study's preprocess/SlideWindow allocations reuse the 1st study's buffers (INFR-02/D-24)"
    - "The CuPy-side gap is closed (CuPy's pool is INDEPENDENT of RMM — its blocks bypass RMM entirely): vol/vol_c/mask/one_hot/vol2 in PreprocessOperator come from the shape cache"
    - "The torch-side big fixed-shape allocations are cached: predicted_logits + n_predictions (zero_() on borrow — the reference allocates fresh zeros), gaussian computed ONCE in setup, per-patch workon (16 MB × ~150 patches × 5 folds — the hottest allocation)"
    - "test_buffer_cache.py passes headlessly: multi-call reuse (same data_ptr), shape-key invalidation (new shape → new buffer), dtype/contiguity invariants (fp32 C-contiguous, D-12), zero-on-borrow semantics"
    - "Multi-study replay (same study 3× in ONE process): cached-buffer data_ptr() stable across studies AND nsys cudaMalloc count flat across studies 2/3 (Phase 2 Plan 06 churn method; baseline = Plan 01's rmm_openq1.md re-run)"
    - "All 4 pixel-exact config gates + SR + residency PASS with the caches active (D-25 anchor)"
  artifacts:
    - path: "examples/apps/cchmc-nnunet-fast/my_app/operators/buffer_cache.py"
      provides: "_ShapeCache class — (shape, dtype) → device buffer dict, get(shape, dtype, zero=False), torch+cupy families, no LRU (config-determined key set), explicit clear()"
    - path: "examples/apps/cchmc-nnunet-fast/scripts/test_buffer_cache.py"
      provides: "D-24(a) headless unit suite (synthetic sizes)"
    - path: ".planning/scripts/infr02_replay.py"
      provides: "D-24(b) one-process 3× same-study replay: data_ptr stability table + nsys churn window per study"
    - path: ".planning/phases/03-optimization/evidence/infr02_proof.md"
      provides: "(a)+(b) proof record: data_ptr table, cudaMalloc-per-study counts, baseline citation, and the D-26 external-dependency note for the user's reference examples"
    - path: ".planning/phases/03-optimization/gates/03-GATE-infr02.json"
      provides: "D-25 gate suite re-run with caches active"
  key_links:
    - from: "PreprocessOperator"
      to: "_ShapeCache (CuPy family)"
      via: "cp.empty allocations for vol/vol_c/mask/one_hot/vol2 replaced by cache.get(shape, dtype, zero=...)"
    - from: "SlideWindowOperator"
      to: "_ShapeCache (torch family)"
      via: "predicted_logits/n_predictions borrow with zero_(); gaussian cached in setup(); workon borrowed per patch"
    - from: "infr02_replay.py"
      to: "nsys cuda_api_sum (Plan 01 churn baseline)"
      via: "per-study NVTX-window cudaMalloc counts, flat across studies 2/3"

user_setup: []
---

<objective>
Implement INFR-02 (D-17 deferred here; D-24): shape-keyed GPU buffer reuse across compute()
calls, with the D-24(a)+(b) proof strategy — headless unit tests + one-process multi-study replay
with address-stability and cudaMalloc-flat evidence.

Purpose: RMM already makes torch-side allocations cheap (Phase 2 churn: 9 cudaMalloc in 117.5 s);
the substantive gap is the CuPy side (independent pool, bypasses RMM) and allocator-traffic
reduction + address stability. Honest scope per research: INFR-02 ships as *provable reuse +
reduced allocator traffic*, not a speed claim — single-study-per-run is the clinical model, so
first-study latency is unaffected. The user's INFR-02 reference examples are the D-26
external-dependency item (recorded, non-blocking).

Output: buffer_cache module + operator integration, unit suite, replay harness + proof record,
green gate suite.
</objective>

<execution_context>
@~/.claude/get-shit-done/workflows/execute-plan.md
@~/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/ROADMAP.md
@.planning/phases/03-optimization/03-CONTEXT.md (D-24 locked; D-26 external-dependency class)
@.planning/phases/03-optimization/03-RESEARCH.md (§D-24/INFR-02 Domain — allocation inventory table, safety rules, Pattern 3; D-24 proof strategy; Wave 0 gaps)
@.planning/phases/03-optimization/3-optimization-01-SUMMARY.md (scheduler flag state + rmm_openq1.md churn baseline the replay must compare against)
@.planning/phases/03-optimization/3-optimization-02-SUMMARY.md (release hook state — replay must run with the shipping state of all prior plans)

# Files under change (read the allocation sites before editing):
@examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py (the ~line 620+ compute path: cp.array/cp.empty sites for vol, vol_c, mask, one_hot, vol2 — exact names from the code)
@examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py (setup(); sliding_window_predict ~294: padded, per-fold prediction, per-patch workon; predict_logits ~355: predicted_logits = torch.zeros, n_predictions)

<interfaces>
<!-- Research Pattern 3 (verified shape) — the module to create: -->
```python
class _ShapeCache:
    """(shape, dtype) -> device buffer. Zero_() on borrow where the reference allocates fresh zeros."""
    def __init__(self, device: str):
        self._d: dict = {}
    def get(self, shape, dtype, zero: bool = False):
        key = (tuple(shape), str(dtype))
        buf = self._d.get(key)
        if buf is None:
            buf = torch.empty(shape, dtype=dtype, device=...)  # or cp.empty for cupy family
            self._d[key] = buf
        if zero:
            buf.zero_()
        return buf
```
Granularity/eviction (discretion, D-24): unbounded dict is fine (few config-determined shape
keys); NO LRU; `clear()` method available for peak-VRAM accounting if needed.
SAFETY (research §D-24, non-negotiable):
- key invalidation on shape change (unit-tested); NEVER cache across dtypes
- fp32 C-contiguous invariant (D-12) — assert `buf.is_contiguous()`/`buf.flags.c_contiguous`
- a cached buffer must NEVER be handed to a DLPack consumer that retains it after the study —
  where an operator EMITS a tensor to the DAG, emit a fresh copy (the Phase 1 postprocess clone
  lesson applies: `cp.from_dlpack`/`to_holoscan_gpu_tensor` ownership). Audit every emit site
  touched by this plan: if a cached buffer is on the emit path, `clone()`/`copy_()` at the boundary.
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: _ShapeCache module + CuPy-side integration in PreprocessOperator + unit suite</name>
  <files>
    examples/apps/cchmc-nnunet-fast/my_app/operators/buffer_cache.py,
    examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py,
    examples/apps/cchmc-nnunet-fast/scripts/test_buffer_cache.py
  </files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py — the full compute path (~line 620+): identify every CuPy allocation for the study-sized buffers (research inventory: vol 64 MB, vol_c 64 MB, mask 8 MB, one_hot 64 MB, vol2 128 MB) and which of them are later EMITTED as DAG tensors (those need boundary copies, not cache hand-off)
    - .planning/phases/03-optimization/03-RESEARCH.md — §D-24 (inventory table + safety) + Pattern 3
    - examples/apps/cchmc-nnunet-fast/scripts/test_gpu_bootstrap.py (headless test style template)
  </read_first>
  <action>
    1. Create `my_app/operators/buffer_cache.py`: the `_ShapeCache` from Pattern 3 with a `family` param ("torch"|"cupy") so one class serves both; `get(shape, dtype, zero=False)`, `clear()`, `keys()`, `total_bytes()` (for the proof record). dtype key = `str(dtype)`; enforce: different dtype strings never share a buffer (inherent in the key). No LRU, no eviction (document why: config-determined key set).
    2. `preprocess_operator.py`: give the operator instance a `_buf_cache = _ShapeCache("cuda", family="cupy")` created in `__init__` (BEFORE `super().__init__` — Pitfall 7 discipline, matching how the other constructor state is handled). Replace the study-sized CuPy allocations with `self._buf_cache.get(shape, dtype, zero=...)`:
       - `zero=True` only where the reference semantics allocate fresh zeroed memory (check each site: e.g. a 0-fill before insert → zero=True; a buffer fully overwritten before read → zero=False).
       - Where a buffer's bytes later cross the operator boundary into an emitted DAG tensor, EMIT A COPY (`.copy()` on CuPy) — never the cached buffer itself (DLPack retention rule).
       - Do NOT cache the one-shot `cp.array(raw_numpy)` H2D source copy if it is a pure input transfer — cache only the derived/intermediate study-sized buffers; record the per-site decision (cache / copy-at-emit / untouched + reason) in a code comment block at the cache creation.
       - The D-13 CPU round-trip (scipy resample) is UNTOUCHED in this plan (GPUP-01 is Plan 04; the OFF path must stay byte-for-byte Phase 2/3 behavior).
    3. `scripts/test_buffer_cache.py` — headless (synthetic shapes, GPU present, no app/bundle):
       - multi-call reuse: `get(s)` twice → identical `data_ptr()`;
       - shape-key invalidation: `get(s)` then `get(s')` (s'≠s) → different data_ptr, both retained (call `get(s)` again → original ptr);
       - dtype invariants: same shape, fp32 vs fp64 → different buffers; both C-contiguous (assert `flags.c_contiguous` / `is_contiguous()`);
       - zero semantics: `zero=True` → all zeros after borrow; `zero=False` → prior contents survive (documented, so a caller never relies on stale data silently — the operator comments from step 2 justify each site);
       - cupy AND torch families both exercised; `clear()` empties and reallocates.
       Exit 0/1 like the other headless tests.
    Commit after this task.
  </action>
  <verify>
    <automated>cd examples/apps/cchmc-nnunet-fast && ulimit -s unlimited && /tmp/monai-env/.venv/bin/python scripts/test_buffer_cache.py; test $? -eq 0</automated>
  </verify>
  <acceptance_criteria>
    - `buffer_cache.py` exists with `class _ShapeCache` (or public `ShapeCache`) implementing get/clear/keys/total_bytes for both families.
    - `grep -c "_buf_cache\|_ShapeCache" my_app/operators/preprocess_operator.py` ≥ 5 (creation + the study-sized sites); every emit site touched by caching has a boundary copy (grep the emit calls).
    - `test_buffer_cache.py` exits 0 and covers all six cases above (each named in a comment/`print`).
    - A fullres-only E2E run (HOLOSCAN_MODEL_LIST=3d_fullres) still exits 0 — quick smoke before the full gate in Task 2.
  </acceptance_criteria>
  <done>The CuPy-side allocation gap is closed with a provable shape-keyed cache, unit-tested headlessly, and the E2E smoke run is clean.</done>
</task>

<task type="auto">
  <name>Task 2: Torch-side cache in SlideWindowOperator + D-25 gate re-run</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py, examples/apps/cchmc-nnunet-fast/scripts/test_buffer_cache.py</files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py — setup() (~416+), sliding_window_predict (~294: padded via F.pad, per-patch workon, per-fold prediction), predict_logits (~355: `predicted_logits = torch.zeros(...)`, `n_predictions = torch.zeros(...)`, gaussian) — identify the exact fresh-allocation lines
    - .planning/phases/03-optimization/03-RESEARCH.md — §D-24 inventory (predicted_logits 512 MB, n_predictions 64 MB, gaussian 1 MB identical every fold, workon 16 MB × ~150 patches × 5 folds)
    - examples/apps/cchmc-nnunet-fast/scripts/test_buffer_cache.py (extend, don't duplicate)
  </read_first>
  <action>
    1. `slidewindow_operator.py`: instance cache `self._buf_cache = _ShapeCache("cuda", family="torch")` in `__init__` (BEFORE super().__init__, Pitfall 7 discipline). Integrate:
       - `predicted_logits` and `n_predictions`: borrow with `zero=True` (reference = fresh `torch.zeros`).
       - `gaussian`: compute ONCE per config in `setup()` (identical every fold — research inventory) and store on the operator; the per-fold loop reuses the stored tensor (read-only in the loop — verify it is never written; if any code path writes it, keep a per-use clone and note why).
       - per-patch `workon`: single cache entry borrowed per patch with the zero semantics the current code requires (check: if the code does `workon = torch.zeros(...)` then fills via padding → zero=True; if it copies fully → zero=False).
       - `padded` (F.pad result): DO NOT cache (the extra copy_ to a cache buffer costs ≈ the alloc it saves at 16 MB/patch under RMM) — record this explicit non-decision in a comment.
       - Emit-path safety: `logits` crosses the subgraph boundary via DLPack to PostResample — if a cached buffer ends up as the emitted tensor, emit a copy (same rule as Task 1). The logits tensor is consumed and not retained past the study, but the rule is uniform: boundary = copy.
    2. Extend `test_buffer_cache.py` with the torch-site semantics cases: zero-on-borrow for the predicted_logits/n_predictions pattern (simulate: borrow → write garbage → reborrow with zero=True → all zeros), gaussian-once (compute in a fake setup, 3 fake folds reuse the same ptr).
    3. D-25 gate re-run (caches active, all prior plans in shipping state — scheduler flag per Plan 01, release hook per Plan 02; pinned GPU, device recorded): `/tmp/monai-env/.venv/bin/python .planning/scripts/phase2_gate.py --json-out .planning/phases/03-optimization/gates/03-GATE-infr02.json` — all 4 pixel gates + SR + residency must pass (fullres 99.99986%/3 boundary class, lowres/cascade/bundle 100.00000%/0, SR 0.0%). Pixel-exactness is the real test that the cache did not silently change a data path.
    Commit after this task.
  </action>
  <verify>
    <automated>cd examples/apps/cchmc-nnunet-fast && ulimit -s unlimited && /tmp/monai-env/.venv/bin/python scripts/test_buffer_cache.py; test $? -eq 0 && test -f ../../.planning/phases/03-optimization/gates/03-GATE-infr02.json && /tmp/monai-env/.venv/bin/python -c "import json; d=json.load(open('../../.planning/phases/03-optimization/gates/03-GATE-infr02.json')); print('gate ok' if all((g.get('ok') or g.get('status')=='PASS') for g in d['gates'] if isinstance(g, dict)) else 'GATE FAIL')"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "gaussian" my_app/operators/slidewindow_operator.py` shows the gaussian is computed in setup() (one allocation site) and the fold loop references the stored tensor.
    - Unit suite exits 0 including the new torch-site cases.
    - `03-GATE-infr02.json` fully green.
  </acceptance_criteria>
  <done>Both allocator families are shape-cached at the big fixed-shape sites, the hottest per-patch allocation is reused, and the full pixel gate suite passes with caches active.</done>
</task>

<task type="auto">
  <name>Task 3: D-24(b) multi-study replay proof (3× same study, one process) + evidence record</name>
  <files>.planning/scripts/infr02_replay.py, .planning/phases/03-optimization/evidence/infr02_proof.md</files>
  <read_first>
    - .planning/phases/03-optimization/03-RESEARCH.md — D-24 proof strategy (b): "one process, same study 3×: cached-buffer data_ptr() stable across studies + nsys cudaMalloc count flat across studies 2/3 (Phase 2 Plan 06 churn method: CUPTI_ACTIVITY_KIND_RUNTIME, _v3020 name suffix)"
    - .planning/phases/03-optimization/3-optimization-01-SUMMARY.md + .planning/profiles/phase3/rmm_openq1.md (the churn baseline this proof compares against)
    - .planning/profiles/phase2/ phase2_bundle_*.sqlite (the Phase 2 churn query pattern to copy)
  </read_first>
  <action>
    1. `.planning/scripts/infr02_replay.py`: ONE process, 3 sequential passes over the same airway study through the REAL operators (not the full GXF app — drive `PreprocessOperator` + `SlideWindowOperator` + `PostResampleOperator` compute() directly with the real bundle and the real study volume loaded from testdata/airway_input via the app's own load path; if direct driving proves structurally impossible for an operator, fall back to the full app with a repeated-input wrapper — record which path was used). Per study: after the preprocess + inference stages complete, snapshot `data_ptr()` of every entry in each operator's `_buf_cache` (`keys()`/`total_bytes()` exist for this) + the gaussian tensor ptr; print a per-study table. Assertions: (i) data_ptr table IDENTICAL across studies 1→2→3 (study 1 = allocation, studies 2/3 = reuse); (ii) RMM pool occupancy (rmm stats if exposed, else nvidia-smi sample) does not grow across studies beyond noise (±5%); (iii) outputs of studies 2 and 3 are byte-identical to study 1's (the reuse did not corrupt a data path — np.array_equal on the final seg per study).
    2. Run the replay under full-process nsys → `.planning/profiles/phase3/infr02_replay_*.nsys-rep` + sqlite + cuda_api_sum. Per-study cudaMalloc counts from the sqlite (CUPTI_ACTIVITY_KIND_RUNTIME, `_v3020` suffix names; study windows from NVTX ranges or the script's printed timestamps): study 1 may malloc (pool expansion / first-touch CuPy pool), studies 2/3 must be FLAT (0 or pool-expansion-only, same classification rule as Phase 2 Plan 06 / Plan 01's baseline).
    3. `.planning/phases/03-optimization/evidence/infr02_proof.md`: the data_ptr table (3 studies × cached buffers), the per-study cudaMalloc counts vs the Plan-01 baseline, the byte-identity results, the pool-occupancy row, and the D-26 external-dependency record: "INFR-02 user reference examples — blocked-on-external, non-blocking (user adding during Phase 3); if they arrive before VERIFICATION.md, fold into the gate oracle; shipped with (a)+(b) per D-24." Honest-scope sentence: single-study-per-run clinical model → this is provable reuse + reduced allocator traffic, not a first-study speed claim.
    Commit after this task.
  </action>
  <verify>
    <automated>test -f .planning/scripts/infr02_replay.py && test -f .planning/phases/03-optimization/evidence/infr02_proof.md && /tmp/monai-env/.venv/bin/python - <<'EOF'
import re
t = open('.planning/phases/03-optimization/evidence/infr02_proof.md').read()
assert 'study' in t.lower() and 'cudaMalloc' in t, 'proof record missing required sections'
print('proof record ok')
EOF</automated>
  </verify>
  <acceptance_criteria>
    - `infr02_proof.md` contains: the 3-study data_ptr table with studies 2/3 identical to study 1, per-study cudaMalloc counts (2/3 flat per the Plan-01 baseline rule), byte-identity of study outputs, and the D-26 external-dependency paragraph naming the user's reference examples.
    - The nsys replay trace + sqlite are committed under .planning/profiles/phase3/ (Phase 0/1 convention).
    - No gate suite change required for this task (Task 2's gate is the D-25 anchor for the plan), but the replay itself exits 0.
  </acceptance_criteria>
  <done>INFR-02 is proven per D-24(a)+(b): unit suite green, address stability + cudaMalloc-flat measured in one process across 3 studies, external dependency recorded, evidence committed.</done>
</task>

</tasks>

<verification>
- `test_buffer_cache.py` exits 0 (all families + site-semantics cases).
- `03-GATE-infr02.json` fully green (caches active, all prior plans in shipping state).
- `infr02_proof.md` + replay trace present; cudaMalloc flat across studies 2/3 per the cited baseline.
- The D-13 scipy CPU resample path is untouched (GPUP-01 belongs to Plan 04).
</verification>

<success_criteria>
- Cross-study buffer reuse is provable (address stability + flat churn + byte-identical repeat outputs), not asserted.
- Both allocator families covered (CuPy gap closed; torch traffic reduced; gaussian once).
- Zero pixel-gate regression (D-25) with the caches active.
</success_criteria>

<output>
After completion, create `.planning/phases/03-optimization/3-optimization-03-SUMMARY.md` covering: the per-site cache/copy/non-cache decision table, the data_ptr + cudaMalloc proof numbers, gate results, commits, and deviations (including which replay path — direct operator drive vs full-app wrapper — was used).
</output>
