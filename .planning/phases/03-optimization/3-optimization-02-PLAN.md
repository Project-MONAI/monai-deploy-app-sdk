---
phase: 3-optimization
plan: "02"
type: execute
wave: 2
depends_on: ["3-optimization-01"]
files_modified:
  - examples/apps/cchmc-nnunet-fast/my_app/app.py
  - examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py
  - examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py
  - examples/apps/cchmc-nnunet-fast/scripts/test_weight_release.py
  - .planning/scripts/vram_sampler.py
autonomous: true
requirements: [MEM-003, TEST-01, TEST-002, TEST-003]

must_haves:
  truths:
    - "The 3d_lowres weights (~0.8 GB: 5 × 135 MB fold state dicts + ~135 MB live network) are freed immediately after the aux fragment's PostResample emits lowres_seg (D-23/MEM-003), and nothing downstream touches the released bundle"
    - "test_weight_release.py passes headlessly: the release hook fires exactly once, only for the aux (lowres_seg-emitting) configuration, and non-aux operators are unaffected"
    - "Peak-VRAM delta is measured at BOTH levels — RMM pool occupancy and driver-level (nvidia-smi/pynvml) — around a bundle run; torch.cuda.memory_stats()/memory_allocated() are NOT used anywhere (they raise under the RMM pluggable allocator, research Pitfall 2)"
    - "All 4 pixel-exact config gates + SR + residency PASS with the release hook active (D-25 anchor)"
  artifacts:
    - path: "examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py"
      provides: "SlideWindowOperator.release() — del network + fold_state_dicts, _bundle=None, torch.cuda.empty_cache(), log line"
    - path: "examples/apps/cchmc-nnunet-fast/scripts/test_weight_release.py"
      provides: "Headless unit suite for the hook semantics"
    - path: ".planning/scripts/vram_sampler.py"
      provides: "pynvml driver-level VRAM sampler (1–5 Hz → CSV with timestamps) for MEM-003 measurement"
    - path: ".planning/phases/03-optimization/evidence/mem003_vram.md"
      provides: "Pool-level + driver-level delta record (before/after lowres fragment completion, both levels, honest if driver-level is flat)"
    - path: ".planning/phases/03-optimization/gates/03-GATE-mem003.json"
      provides: "D-25 gate suite re-run with the release hook active"
  key_links:
    - from: "NnUnetConfigSubgraph.compose() (app.py — exactly ONE line: the `release_fn=` injection)"
      to: "PostResampleOperator"
      via: "`release_fn=sw.release if self._emit_lowres_seg else None` added to the PostResampleOperator construction when self._emit_lowres_seg is True"
    - from: "PostResampleOperator.compute() tail"
      to: "release callback"
      via: "self._release_fn() after the final op_output.emit block in compute()"
    - from: "vram_sampler.py"
      to: "bundle run log"
      via: "timestamp correlation: sampler CSV rows vs the 'weights released' log line"

user_setup: []
---

<objective>
Implement MEM-003 (D-23): free the 3d_lowres network weights and dead lowres buffers immediately
after the aux fragment's PostResample emits `lowres_seg`, and measure the peak-VRAM delta at the
pool level AND the driver level.

Purpose: `02-BENCHMARK-REPORT.md` §6 / CONTEXT D-23 — the cascade path holds ~0.8 GB of 3d_lowres
weights CUDA-resident for the whole run that is dead after the lowres fragment finishes. This is
a memory-lifecycle deliverable, NOT a speed lever (no latency impact expected — the report must
say so). MEM-01/MEM-02 are out of scope (D-20).

Output: release hook + unit test, pynvml sampler, measured pool/driver delta evidence, green gate
suite.
</objective>

<execution_context>
@~/.claude/get-shit-done/workflows/execute-plan.md
@~/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/ROADMAP.md
@.planning/phases/03-optimization/03-CONTEXT.md (D-23 locked: MEM-003 only)
@.planning/phases/03-optimization/03-RESEARCH.md (§D-23/MEM-003 Domain — release point, "what free means under RMM", Pattern 4; Pitfall 2 — torch memory API raises; Pitfall 7 — pin device)
@.planning/phases/03-optimization/3-optimization-01-SUMMARY.md (scheduler flag state shipped in Plan 01 — run the gate with that same shipping configuration so this plan's gate is apples-to-apples)

# Files under change:
@examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py (ModelBundle dataclass ~line 130–145; SlideWindowOperator __init__ ~line 416; compute(); predict_logits ~line 355)
@examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py (__init__ ~line 330–380 with the emit_lowres_seg/emit_probabilities flags processed BEFORE super().__init__ — Pitfall 7 port gating; compute() emit block ~lines 456–480)
@examples/apps/cchmc-nnunet-fast/my_app/app.py (NnUnetConfigSubgraph.compose() at lines ~235–280 builds `sw` and `post`; Task 1 injects the single `release_fn=` line INSIDE this compose — the ONLY app.py change in this plan)

<interfaces>
<!-- Extracted from the codebase — use directly. -->

From my_app/app.py (NnUnetConfigSubgraph.compose, ~line 240):
```python
sw = SlideWindowOperator(
    self, model_path=self._model_path, config_name=cfg, name=f"slidewindow_{cfg}"
)
post = PostResampleOperator(
    self,
    config_name=cfg,
    emit_probabilities=self._emit_probabilities,
    emit_lowres_seg=self._emit_lowres_seg,
    name=f"postresample_{cfg}",
)
```
→ becomes: add `release_fn=sw.release if self._emit_lowres_seg else None` to the
PostResampleOperator construction (and thread the new kwarg through the subgraph's
`__init__`/compose — the subgraph already knows `self._emit_lowres_seg`). NO other app.py change.

From my_app/operators/slidewindow_operator.py:
```python
@dataclass
class ModelBundle:
    network: torch.nn.Module  # CUDA-resident, eval()
    fold_state_dicts: List[Dict[str, torch.Tensor]]  # one per fold, CUDA-resident
```
3d_lowres checkpoints: 5 × 135 MB → ~675 MB folds + ~135 MB network ≈ 0.8 GB.

From my_app/operators/postresample_operator.py compute() tail (~lines 456–480):
```python
if self._emit_lowres_seg:
    op_output.emit(to_holoscan_gpu_tensor(seg_full), self.OUTPUT_LOWRES_SEG)
```
The release call goes AFTER the final emit of compute() (all emits done; nothing downstream
touches the lowres SlideWindowOperator again — the cascade consumes only the emitted lowres_seg
tensor).

Research Pattern 4 (release):
```python
def release(self):
    if self._bundle is not None:
        b = self._bundle; self._bundle = None
        del b.network, b.fold_state_dicts
        torch.cuda.empty_cache()   # semantics under RMM: unverified (Open Q2) — MEASURE, don't assume
```
CRITICAL (Pitfall 2): NEVER call torch.cuda.memory_stats()/memory_allocated()/reserved() in any
measurement code — they raise `RuntimeError: CUDAPluggableAllocator does not yet support
getDeviceStats` under RMM. Use pynvml driver sampling + the RMM pool stats if the installed rmm
26.2.0 exposes them (probe with `python -c "import rmm; print([a for a in dir(rmm) if 'pool' in a.lower() or 'size' in a.lower()])"`); otherwise pool-level is approximated by nvidia-smi sampling
deltas + the budget math, and the report says which level was directly measured.
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Weight release hook (release() + callback injection) + headless unit test</name>
  <files>
    examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py,
    examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py,
    examples/apps/cchmc-nnunet-fast/my_app/app.py,
    examples/apps/cchmc-nnunet-fast/scripts/test_weight_release.py
  </files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py — ModelBundle (~130–145), SlideWindowOperator __init__/compute (~416+), predict_logits (~355) to see every `self._bundle` use site
    - examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py — __init__ (the emit flags are read BEFORE super().__init__ — Pitfall 7: the same must hold for release_fn) and the compute() emit block (~456–480)
    - examples/apps/cchmc-nnunet-fast/my_app/app.py — NnUnetConfigSubgraph __init__ + compose (lines ~200–280)
    - .planning/phases/03-optimization/03-RESEARCH.md — §D-23 + Pattern 4
    - An existing headless test for the style template: examples/apps/cchmc-nnunet-fast/scripts/test_gpu_bootstrap.py (subprocess-based, venv python)
  </read_first>
  <action>
    1. `slidewindow_operator.py` — add to `SlideWindowOperator`:
       ```python
       def release(self) -> None:
           """MEM-003/D-23: free this config's weights after the aux fragment's
           terminal emit. Safe no-op if already released or never loaded."""
           if self._bundle is None:
               return
           b = self._bundle
           self._bundle = None
           del b.network, b.fold_state_dicts
           torch.cuda.empty_cache()  # RMM: may be a driver-level no-op (Open Q2) — measured in Task 2
           logging.info("weights released: %s (folds=%d) (MEM-003)", self._config_name, len(b.fold_state_dicts))
       ```
       Use the operator's actual attribute names (read __init__ first; `self._config_name` etc. are the likely names — match reality). Guard `compute()`: if `self._bundle is None`, raise `RuntimeError("compute() after release() — DAG ordering violation for <config>")` (defensive; the DAG never does this).
    2. `postresample_operator.py` — add `release_fn: Optional[Callable[[], None]] = None` to `__init__`, captured into `self._release_fn` BEFORE `super().__init__` (same discipline as the emit flags). At the very end of `compute()` — after the last emit statement — add:
       ```python
       if self._release_fn is not None:
           self._release_fn()
       ```
       No other operator change.
    3. `app.py` (subgraph only, per the interfaces block): pass `release_fn=sw.release if self._emit_lowres_seg else None` into the PostResampleOperator construction in `NnUnetConfigSubgraph.compose()`. Verify `sw.release` is bound BEFORE the subgraph object is discarded (it is — the subgraph holds `post`, which holds the bound method, which holds `sw`).
    4. `scripts/test_weight_release.py` — headless suite (no E2E app run; run with the venv python; GPU present but no bundle load needed beyond a tiny synthetic):
       - (a) a `SlideWindowOperator.release()` on a never-loaded bundle (bundle set to a tiny synthetic ModelBundle of 2×1×1 random CUDA tensors) → `self._bundle is None` after, log line present, second `release()` is a no-op (no exception).
       - (b) a real `SlideWindowOperator` built from the airway bundle's 3d_lowres config (model_path from the repo's `examples/apps/cchmc_nnunet_fifteen_ckpt_app/models`), after `setup()`: record driver memory (pynvml) before/after `release()`; assert `_bundle is None` and `compute()` now raises the guard RuntimeError.
       - (c) callback semantics: a counter callable passed as `release_fn` to a bare `PostResampleOperator` fires exactly once when the tail executes (drive the tail via a minimal fake op_output OR assert by code-inspection + (a)/(b) — pick the lighter path that still exercises the real `compute()` tail; the test must not require a full app run).
       Exit 0 on pass, nonzero with a printed failure on any assertion.
    Commit after this task.
  </action>
  <verify>
    <automated>cd examples/apps/cchmc-nnunet-fast && ulimit -s unlimited && /tmp/monai-env/.venv/bin/python scripts/test_weight_release.py; test $? -eq 0</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "def release" my_app/operators/slidewindow_operator.py` ≥ 1; `grep -c "_release_fn" my_app/operators/postresample_operator.py` ≥ 2 (capture + call); `grep -c "release_fn=sw.release" my_app/app.py` = 1.
    - `test_weight_release.py` exits 0; it contains no call to torch.cuda.memory_stats/memory_allocated/reserved (grep must return nothing).
    - `NnUnetConfigSubgraph` for non-aux configs gets `release_fn=None` (the conditional is `if self._emit_lowres_seg`).
  </acceptance_criteria>
  <done>3d_lowres weights are deterministically released exactly once, exactly for the aux configuration, with the hook semantics unit-tested headlessly and a defensive guard against post-release compute.</done>
</task>

<task type="auto">
  <name>Task 2: Pool-level + driver-level VRAM delta measurement + D-25 gate re-run</name>
  <files>.planning/scripts/vram_sampler.py, .planning/phases/03-optimization/evidence/, .planning/phases/03-optimization/gates/</files>
  <read_first>
    - .planning/phases/03-optimization/03-RESEARCH.md — §D-23 "What free means under RMM" (pool level vs driver level; RMM pools do not auto-return to the driver; empty_cache unverified = Open Q2) + Open Q2 + Pitfall 2 + Pitfall 7
    - .planning/scripts/phase2_gate.py (the D-25 invocation; use the existing --report arg with the Phase 3 JSON path)
    - .planning/phases/03-optimization/3-optimization-01-SUMMARY.md (which scheduler flag state ships — replicate it for the measurement run)
  </read_first>
  <action>
    1. `.planning/scripts/vram_sampler.py` (~30 lines): pynvml sampler — args `--device N --hz 2 --out CSV`; samples `nvmlDeviceGetMemoryInfo` used/total at 1–5 Hz; timestamps in epoch-ns; Ctrl-C or `--until-file F` (stop when F appears) clean exit; prints the CSV path on exit.
    2. Measurement run (bundle, HOLOSCAN_MODEL_LIST unset — 3-config cascade path; scheduler flag in the Plan-01 shipping state; pinned free GPU, record device id):
       - Start the sampler (`--until-file` on a sentinel written by a small log-tail helper that watches for the `weights released: 3d_lowres` line — simplest: a background `grep --line-buffered` on the app's log that touches the sentinel file).
       - Run one full bundle rep; capture the app log (it must contain `weights released: 3d_lowres ... (MEM-003)`).
       - Also record the RMM pool-level view if rmm 26.2.0 exposes pool stats (probe dir(rmm) as in the interfaces note); if not, state that explicitly and derive the pool-level delta from budget math (0.8 GB returned to pool) + the driver sampling.
    3. `.planning/phases/03-optimization/evidence/mem003_vram.md`: table with (a) driver used-VRAM at 3 moments — pre-lowres-inference, at the release log line (±1 s window), post-cascade-complete; (b) the peak driver VRAM with release ON vs a comparison rep with the hook DISABLED (run the same bundle with `release_fn` bypassed — easiest: a one-line env opt-out `HOLOSCAN_KEEP_LOWRES_WEIGHTS=1` checked at the injection site in app.py, added for this measurement only and KEPT (harmless, documented); if you prefer not to add the env var, re-derive the comparison from git stash — choose one, keep the diff minimal); (c) which level moved (pool and/or driver) and the honest conclusion (research expectation: pool level down ~0.8 GB; driver level may be flat because RMM does not return to the driver and the peak may be dominated by the pool reservation anyway — a flat driver delta is a VALID, reportable result).
    4. D-25 gate re-run (release hook ACTIVE — the shipping state): `HOLOSCAN_CONCURRENT_FRAGMENTS=<Plan-01 shipping default or explicit 1> /tmp/monai-env/.venv/bin/python .planning/scripts/phase2_gate.py --report .planning/phases/03-optimization/gates/03-GATE-mem003.json` — all 4 pixel gates + SR + residency must pass (same expected values as Plan 01: fullres 99.99986%/3 boundary class, others 100.00000%/0, SR 0.0%).
    Commit after this task.
  </action>
  <verify>
    <automated>test -f .planning/scripts/vram_sampler.py && test -f .planning/phases/03-optimization/evidence/mem003_vram.md && test -f .planning/phases/03-optimization/gates/03-GATE-mem003.json && grep -rL "memory_stats\|memory_allocated" .planning/scripts/vram_sampler.py >/dev/null && /tmp/monai-env/.venv/bin/python -c "import json; d=json.load(open('.planning/phases/03-optimization/gates/03-GATE-mem003.json')); print('gate ok' if d.get('all_gates_pass') is True else 'GATE FAIL')"</automated>
  </verify>
  <acceptance_criteria>
    - `mem003_vram.md` contains: device id, the 3-moment driver table, the peak ON-vs-OFF comparison, the pool-level delta (measured or derived — labeled as such), and an explicit conclusion per level.
    - The app log from the measurement run contains the `weights released: 3d_lowres` line.
    - `03-GATE-mem003.json` is fully green.
    - No file in this plan calls torch.cuda.memory_stats()/memory_allocated().
  </acceptance_criteria>
  <done>MEM-003 is implemented and its memory effect is measured honestly at both levels (pool and driver, with the RMM caveat stated), and the D-25 gate suite passes with the release active.</done>
</task>

</tasks>

<verification>
- `test_weight_release.py` exits 0 (headless, fast).
- `03-GATE-mem003.json` fully green in the shipping flag configuration.
- Evidence file cites the exact log line + sampler CSV path; CSV exists under the phase evidence dir.
- Only the listed files changed (app.py: the single `release_fn=` injection line only).
</verification>

<success_criteria>
- 3d_lowres weights freed exactly once, exactly for the aux config, verified by unit test and by the log line in a real bundle run.
- Both VRAM levels reported with the honest RMM caveat (driver-level flat = valid result, documented, not papered over).
- Zero regression in the 4-config pixel gate suite (D-25).
</success_criteria>

<output>
After completion, create `.planning/phases/03-optimization/3-optimization-02-SUMMARY.md` covering: the pool/driver delta numbers, the Open-Q2 answer (does empty_cache return memory to the driver under RMM?), the gate results, commits, and deviations.
</output>
