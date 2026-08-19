---
phase: "2-gpu-acceleration"
plan: "04"
type: "execute"
wave: 3
depends_on: ["02", "03"]
files_modified:
  - "examples/apps/cchmc-nnunet-fast/my_app/app.py"
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/gpu_util.py"
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py"
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py"
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py"
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py"
autonomous: true
requirements: [PIPE-03, PIPE-04, INF-009, INFR-004, INFR-005]
must_haves:
  truths:
    - "Each resolved nnUNet config runs as its OWN holoscan Fragment (name nnunet_<config>) inside one DAG; DICOM I/O, ensemble, postprocess and writers stay app-level (PIPE-03)"
    - "Fragment instantiation iterates the resolved model list — no config hard-coding in compose() (D-02); HOLOSCAN_MODEL_LIST env var selects the list, defaulting to the reference default (PIPE-03)"
    - "All four model-list configurations run end-to-end on the airway study (exit 0, SEG/SR/SC written): 3d_fullres, 3d_lowres, 3d_cascade_fullres (auto-inserts 3d_lowres), and the default bundle (PIPE-04 end-to-end: lowres_seg crosses the fragment boundary with zero disk I/O)"
    - "Every declared input port has a flow and every declared output has a receiver in ALL four configurations (conditional emit flags) — GXF port discipline (RESEARCH Pitfall 7)"
    - "The ensemble accumulates per-config probability tensors in ensemble_model_list order (fullres first by default) using the Phase 1 in-place accumulation + CuPy exact final division; the INF-009 literal 'incremental in-place averaging' is documented as met-with-deviation per D-19"
    - "NVTX range names and timing records carry the config (preprocess_3d_lowres, inference_3d_cascade_fullres, ...) and the per-study aggregate is keyed by the TOP-LEVEL application, not the sub-Fragment (INFR-005, RESEARCH Pitfall 9)"
    - "One CudaStreamPool per fragment (stream_flags=1 NonBlocking, reserved_size=1, per-fragment nvtx_identifier); stream overlap is best-effort — an honest note suffices (INFR-004, D-16)"
  artifacts:
    - path: "examples/apps/cchmc-nnunet-fast/my_app/app.py"
      provides: "fragment factory: resolve_run_model_list + Fragment per config + cross-fragment flows + per-fragment CudaStreamPool"
      contains: "HOLOSCAN_MODEL_LIST"
    - path: "examples/apps/cchmc-nnunet-fast/my_app/operators/gpu_util.py"
      provides: "root-application-keyed StudyTimingCollector + config-tagged NVTX"
      contains: "application"
    - path: "examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py"
      provides: "per-config input ports (prob_<config>) + ensemble-order accumulation"
      contains: "prob_"
  key_links:
    - from: "frag_lowres postresample (lowres_seg)"
      to: "frag_cascade preprocess (lowres_seg)"
      via: "app.add_flow(frag_lowres, frag_cascade, {('lowres_seg','lowres_seg')}) — cross-fragment zero-copy"
      pattern: "lowres_seg"
    - from: "frag_<cfg> postresample (probabilities)"
      to: "app-level ensemble (prob_<cfg>)"
      via: "app.add_flow(frag, ensemble_op, {('probabilities', f'prob_{cfg}')}) for cfg in ensemble_list; CountCondition(len(ensemble_list))"
      pattern: "prob_"
    - from: "sub-Fragment timing records"
      to: "app-level study_timing_summary"
      via: "collector key = fragment.application (top-level Application)"
      pattern: "application"
---

# Phase 2 Plan 04: Multi-Fragment DAG Assembly (one Fragment per config + cascade wiring)

## Objective
- **What:** Replace the single-config DAG in `app.py` with the reference-semantics
  multi-fragment DAG: one `Fragment` per resolved config, cross-fragment `lowres_seg`
  cascade flow, app-level ensemble with per-config ordered inputs, per-fragment
  `CudaStreamPool`, config-tagged NVTX/timing keyed on the top-level application.
- **Why:** This is the phase's architectural deliverable (PIPE-03/PIPE-04): all three
  3D configs run in one DAG with zero-disk-I/O cascade; it also fixes the timing-keying
  pitfall that would corrupt Plan 06's benchmark parsing.
- **Output:** 4 runnable model-list configurations, each E2E exit 0 on the airway study
  with correct per-config observability. (Pixel-exact gates for each configuration are
  Plan 05's job against the per-config oracles.)

## Execution Environment

- Python: `/tmp/monai-env/.venv/bin/python`. Run from the app root (my_app name
  collision):
  ```bash
  cd examples/apps/cchmc-nnunet-fast && ulimit -s unlimited
  HOLOSCAN_MODEL_LIST=<list> /tmp/monai-env/.venv/bin/python my_app \
    -i /users/srv-mde/projects/monai-deploy-app-sdk/testdata/airway_input \
    -m /users/srv-mde/projects/monai-deploy-app-sdk/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models \
    -o <scratch-out>
  ```
  Four configurations to verify (see Task 3 for the exact expected run/ensemble lists).
- CRITICAL: keep Plan 02's `gpu_bootstrap` import as the FIRST import in app.py and keep
  the `memory_allocator_backend` assertion + `compute_memory_budget`/`warm_pool` calls in
  `compose()` — extend the budget `cfgs` list to the full resolved run list.
- Commit after each major change.

## Context

@.planning/phases/02-gpu-acceleration/02-CONTEXT.md
@.planning/phases/02-gpu-acceleration/02-RESEARCH.md
@examples/apps/cchmc-nnunet-fast/my_app/app.py
@examples/apps/cchmc-nnunet-fast/my_app/operators/gpu_util.py
@examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py
@examples/apps/cchmc_nnunet_fifteen_ckpt_app/my_app/nnunet_seg_operator.py

**Verified API (holoscan-cu13 4.2.0, live-verified 2026-08-19):**
```python
from monai.deploy.core import Fragment          # Fragment(app, name="...")
from holoscan.resources import CudaStreamPool   # CudaStreamPool(fragment, dev_id=0,
                                                #   stream_flags=0, stream_priority=0,
                                                #   reserved_size=1, max_size=0,
                                                #   cuda_green_context=None,
                                                #   nvtx_identifier='nvtx_stream_pool',
                                                #   name='cuda_stream_pool')
frag = Fragment(app, name="nnunet_3d_lowres")
app.add_fragment(frag)
app.add_flow(series_to_vol_op, preprocess_op, {("image", "image")})            # app op -> fragment op
app.add_flow(frag_lowres, frag_cascade, {("lowres_seg", "lowres_seg")})        # fragment -> fragment
app.add_flow(frag_fullres, ensemble_op, {("probabilities", "prob_3d_fullres")})# fragment -> app op
```
`Fragment.application` exists (live-verified) — sub-fragment operators reach the
top-level app via `operator.fragment.application`.
holoscan 4.2 quirk (already in the codebase): `Operator.__init__` invokes `setup(spec)`
BEFORE the constructor body finishes — every per-instance flag used by `setup()` must be
initialized before `super().__init__` (pattern: `EnsembleAverageOperator`).

**Expected resolved lists for the airway bundle (from Plan 03's `resolve_run_model_list`):**

| HOLOSCAN_MODEL_LIST | run list | ensemble list |
|---|---|---|
| unset (default) | `3d_fullres, 3d_lowres, 3d_cascade_fullres` | `3d_fullres, 3d_cascade_fullres` |
| `3d_fullres` | `3d_fullres` | `3d_fullres` |
| `3d_lowres` | `3d_lowres` | `3d_lowres` |
| `3d_cascade_fullres` | `3d_lowres, 3d_cascade_fullres` | `3d_cascade_fullres` |

**Conditional port table (Pitfall 7 — every declared port must have a flow/receiver in
the configuration it is declared in):**

| config in run list | postresample `probabilities` | postresample `lowres_seg` | ensemble input ports |
|---|---|---|---|
| 3d_fullres (alone) | emit → ensemble `prob_3d_fullres` | — (not cascade) | `prob_3d_fullres` |
| 3d_lowres (alone) | emit → ensemble `prob_3d_lowres` | — (cascade not in list) | `prob_3d_lowres` |
| 3d_lowres (as auxiliary, cascade in list) | NOT declared (no consumer — D-07) | emit → cascade fragment | `prob_3d_cascade_fullres` (or + fullres) |
| 3d_cascade_fullres | emit → ensemble `prob_3d_cascade_fullres` | — (it consumes, not emits) | as above |

Locked decisions: D-02 (config-generic wiring), D-06 (bundle = fullres + cascade_fullres
probability maps, reference bundle semantics), D-07 (lowres standalone-only for the
ensemble), D-09/D-10 (argmax one-hot cascade handoff), D-16 (stream pool best-effort),
D-19 (keep Phase 1 averaging; document the INF-009 deviation).

## Tasks

<task type="auto">
  <name>Task 1: config-tagged NVTX + root-application-keyed timing (gpu_util + 3 operators)</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/gpu_util.py, examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py, examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py, examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py</files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/my_app/operators/gpu_util.py (`nvtx_range`, `set_study_id`/`get_study_id` (keyed by `id(fragment)`), `StudyTimingCollector` (records keyed by `id(fragment)`), `GpuTiming`)
    - examples/apps/cchmc-nnunet-fast/my_app/app.py (`_log_study_timing_summaries` — iterates `StudyTimingCollector.studies(self)` where self is the Application)
    - .planning/phases/02-gpu-acceleration/02-RESEARCH.md (Pitfall 9: "StudyTimingCollector ... keyed by operator.fragment, which in Phase 1 == the app. With sub-Fragments the per-study aggregate silently fragments")
  </read_first>
  <action>
1. `gpu_util.py`: add a helper
   ```python
   def _root(fragment: Any) -> Any:
       """Top-level Application for a (sub-)fragment: in Phase 1 fragment == app;
       with real sub-Fragments, Fragment.application (verified present in 4.2)
       resolves the root. Keying by the root keeps one per-study aggregate across
       fragments (Pitfall 9)."""
       return getattr(fragment, "application", None) or fragment
   ```
   and route `set_study_id` / `get_study_id` / `StudyTimingCollector.record` /
   `.studies` / `.clear` through `_root(...)` (store `id(_root(f))`). Keep the public
   signatures unchanged so every call site keeps working.
2. NVTX range names carry the config (INFR-005): in `PreprocessOperator.compute`
   change `nvtx_range("preprocess")` → `nvtx_range(f"preprocess_{self.config_name}")`;
   in `SlideWindowOperator.compute` → `nvtx_range(f"inference_{self.config_name}")`
   (it already has `config_name`); `PostResampleOperator` gains a
   `config_name: Optional[str] = None` constructor kwarg (default keeps the bare
   `"postresample"` name) and uses `nvtx_range(f"postresample_{self.config_name}" if
   self.config_name else "postresample")`. Also add a `"config": self.config_name`
   field to the structured timing records in all three operators (the app-level
   ensemble/writer records may omit it).
3. Smoke: run the single-config app (`HOLOSCAN_MODEL_LIST=3d_fullres`, Execution
   Environment) → exit 0; the log contains `study_timing_summary:` with records whose
   JSON includes `"config": "3d_fullres"` and operator names `preprocess_3d_fullres`,
   `inference_3d_fullres`, `postresample_3d_fullres`.
  </action>
  <acceptance_criteria>
    - `grep -n "_root" my_app/operators/gpu_util.py` shows the helper and that `set_study_id`, `get_study_id`, `StudyTimingCollector.record/studies/clear` use `id(_root(...))` (no bare `id(fragment)` left in the collector/study paths).
    - `grep -n "nvtx_range(f" my_app/operators/preprocess_operator.py my_app/operators/slidewindow_operator.py my_app/operators/postresample_operator.py` returns 3 lines matching `preprocess_{self.config_name}`, `inference_{self.config_name}`, `postresample_{self.config_name}`.
    - `grep -n '"config"' my_app/operators/preprocess_operator.py my_app/operators/slidewindow_operator.py my_app/operators/postresample_operator.py` >= 1 match per file (timing record field).
    - Single-config E2E exit 0; `grep -E '"config": "3d_fullres"|inference_3d_fullres'` on the run log returns matches; `study_timing_summary:` still appears exactly once per study.
  </acceptance_criteria>
  <verify>cd examples/apps/cchmc-nnunet-fast && ulimit -s unlimited && HOLOSCAN_MODEL_LIST=3d_fullres /tmp/monai-env/.venv/bin/python my_app -i /users/srv-mde/projects/monai-deploy-app-sdk/testdata/airway_input -m /users/srv-mde/projects/monai-deploy-app-sdk/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models -o /tmp/p2p4_e2e_cfg 2>&1 | grep -E "inference_3d_fullres|study_timing_summary|\"config\"" | head; ls /tmp/p2p4_e2e_cfg/SEG</verify>
  <done>Timing/NVTX survive sub-Fragments: one aggregate per study at the app level, every record/range carries its config — Plan 06's CSV parsing and Plan 05's gates can rely on it.</done>
</task>

<task type="auto">
  <name>Task 2: ensemble per-config ordered inputs (INF-009 D-19 documentation)</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py</files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py (full file: `average_probabilities`, `_divide_refparity` (CuPy in-place division — torch CUDA `/= n` is 1-ulp off, Phase 1 measurement), `_to_tensor_list`, `argmax_to_segmentation`, `__init__` (emit_averaged_probabilities flag BEFORE super()), `setup()` (conditional output), `compute()` (receives `probabilities` stream, CountCondition? — check how the current single input is conditioned), the Plan 02 `defer_strategy` flag)
    - .planning/phases/02-gpu-acceleration/02-RESEARCH.md (Pattern 6: "feed it the list of per-config probability tensors in ensemble_model_list order (order matters: first volume is the base)")
  </read_first>
  <action>
1. `EnsembleAverageOperator.__init__` gains `config_names: Optional[Sequence[str]] = None`
   (the ORDERED ensemble list) — initialize `self._config_names` BEFORE
   `super().__init__` (holoscan init quirk). Keep `emit_averaged_probabilities` and
   `defer_strategy` exactly as Plan 02 left them.
2. `setup()`: if `self._config_names` is set (multi-config mode): declare one input per
   config — `spec.input(f"prob_{cfg}")` for each cfg — and the entry condition is
   `CountCondition(self, len(self._config_names))` (import from
   `monai.deploy.conditions`); the operator must NOT run on partial arrivals. If
   `self._config_names` is None (legacy single-stream mode, keeps Phase 1 behavior for
   tests): keep the existing single `probabilities` input.
3. `compute()`: multi-config mode — receive each `f"prob_{cfg}"` port, build
   `tensors = [received[cfg] for cfg in self._config_names]` (ORDER BY THE LIST, never
   by arrival order), then call the UNCHANGED Phase 1
   `average_probabilities(tensors)` (in-place `+=` accumulation in that order +
   `_divide_refparity` CuPy final division) → `argmax_to_segmentation` → emit `seg`
   (+ optional averaged probabilities). Arrival order of GXF data is not guaranteed —
   the list-order reconstruction is what makes the mean bit-deterministic.
4. Module docstring (D-19, required verbatim intent): "INF-009 is met-with-deviation per
   D-19: the Phase 1 in-memory in-place accumulation with a single EXACT final division
   (CuPy `_divide_refparity`) is kept instead of a literal running mean
   `(acc*(k-1)+x)/k` — a running mean is not bit-identical to the reference's sum/n and
   would break the segmentation-level identity gate (D-08). VRAM intent is satisfied:
   one accumulator + one streamed input, no N-copy stack; with defer_strategy (INFR-03)
   each consumed tensor is released as it is accumulated."
5. Unit smoke: a small script inline (python -c or appended to
   `scripts/test_cascade_config.py` as `test_ensemble_order`) — call
   `average_probabilities` on a synthetic 2-tensor list and assert the result equals
   the manual reference-order accumulation
   `t0 + t1` then CuPy `/ 2` path (import the module functions directly; assert
   `torch.equal` against a numpy-reference `sum/2` computed in float64? NO — compute
   the expected with the SAME Phase 1 code path on identically-ordered tensors and
   assert invariance when the input LIST order is the ensemble order; additionally assert
   that passing the tensors in REVERSED list position but labeling them correctly yields
   the ensemble-order result (the receive-mapping behavior is covered by Task 3's
   E2E gates).
  </action>
  <acceptance_criteria>
    - `grep -n "config_names" my_app/operators/ensemble_average_operator.py` shows the flag initialized before `super().__init__`; `grep -n "CountCondition" my_app/operators/ensemble_average_operator.py` returns >= 1 line in multi-config mode.
    - `grep -n "prob_" my_app/operators/ensemble_average_operator.py` shows the per-config `spec.input(f"prob_{cfg}")` loop.
    - `grep -n "D-19" my_app/operators/ensemble_average_operator.py` returns >= 1 line in the module docstring; `_divide_refparity` and `average_probabilities` bodies are otherwise UNCHANGED (`git diff` review: no arithmetic edits).
    - `test_ensemble_order` added and `/tmp/monai-env/.venv/bin/python scripts/test_cascade_config.py` exits 0.
  </acceptance_criteria>
  <verify>cd examples/apps/cchmc-nnunet-fast && /tmp/monai-env/.venv/bin/python scripts/test_cascade_config.py</verify>
  <done>The ensemble accepts one named probability stream per ensemble config, accumulates in ensemble_model_list order via the untouched Phase 1 bit-exact code, and the INF-009/D-19 deviation is documented at the source.</done>
</task>

<task type="auto">
  <name>Task 3: app.py fragment factory + cross-fragment cascade flow + CudaStreamPool + 4-configuration E2E verification</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/app.py</files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/my_app/app.py (post-Plan-02 state: gpu_bootstrap first import, backend assertion, memory_budget call, warm_pool at end of compose(); the current single-config flow table; the timed writer subclasses — keep them)
    - examples/apps/cchmc-nnunet-fast/my_app/operators/__init__.py (what's exported — add Fragment/CudaStreamPool where appropriate or import directly in app.py)
    - .planning/phases/02-gpu-acceleration/02-RESEARCH.md (Pattern 1 "Multi-fragment DAG" — the verified API block, DAG sketch, ordering/deadlock mitigation; Open Question 2 — env var selection)
    - .planning/phases/02-gpu-acceleration/2-gpu-acceleration-03-PLAN.md (the conditional port table + expected resolved lists — implement exactly that table)
  </read_first>
  <action>
Rewrite `compose()`'s operator-chain section (keep DICOM I/O, writers, custom tags, and
all Plan 02 infra wiring intact):

1. Resolve the model list:
   ```python
   model_list_arg = os.environ.get("HOLOSCAN_MODEL_LIST")
   model_list_arg = [s.strip() for s in model_list_arg.split(",") if s.strip()] if model_list_arg else None
   plans = json.loads((find_jsonpkls_dir(model_path) / "plans.json").read_text())
   run_list, ensemble_list = resolve_run_model_list(model_list_arg, plans, model_path)
   self._logger.info("run_model_list=%s ensemble_model_list=%s", list(run_list), list(ensemble_list))
   ```
   (import `resolve_run_model_list`, `find_jsonpkls_dir` from config; `import os` at top.
   Extend the Plan 02 budget `cfgs` list to ALL configs in `run_list`.)
2. Fragment factory (iterate — NO config-name literals in this loop, D-02):
   for each `cfg` in `run_list`:
   ```python
   frag = Fragment(self, name=f"nnunet_{cfg}")
   prev_needed = cfg in run_list and any(
       _previous_stage_of(c, plans) == cfg for c in run_list)   # plans-driven, not literal
   pre = PreprocessOperator(frag, model_path=model_path, config_name=cfg, name=f"preprocess_{cfg}")
   sw  = SlideWindowOperator(frag, model_path=model_path, config_name=cfg, name=f"slidewindow_{cfg}")
   post = PostResampleOperator(frag, config_name=cfg,
                               emit_probabilities=(cfg in ensemble_list),
                               emit_lowres_seg=(cfg == _auxiliary_prev_stage(run_list, plans)),
                               name=f"postresample_{cfg}")
   frag.add_operator(pre); frag.add_operator(sw); frag.add_operator(post)
   frag.add_flow(pre, sw, {("preprocessed", "preprocessed")})
   frag.add_flow(pre, post, {("preprocessed_meta", "preprocessed_meta")})
   frag.add_flow(sw, post, {("logits", "logits")})
   self.add_fragment(frag)
   CudaStreamPool(frag, dev_id=0, stream_flags=1, reserved_size=1,
                  nvtx_identifier=f"streams_{cfg}", name=f"cuda_stream_pool_{cfg}")
   self.add_flow(series_to_vol_op, pre, {("image", "image")})
   ```
   where `_auxiliary_prev_stage(run_list, plans)` returns the unique config whose
   `previous_stage` is a member of `run_list` (i.e. `3d_lowres` when the cascade is in
   the list; `None` otherwise) and `_previous_stage_of(c, plans)` reads the raw plans
   entry — both plans-driven (D-02). `emit_probabilities=(cfg in ensemble_list)`
   implements the conditional port table (D-07: auxiliary lowres never feeds the
   ensemble; its `probabilities` output is then simply NOT declared).
3. Cascade edge (only when the auxiliary prev stage exists):
   ```python
   self.add_flow(frag_prev, frag_cascade, {("lowres_seg", "lowres_seg")})
   ```
4. Ensemble (app-level, per Task 2's new interface):
   ```python
   ensemble_op = EnsembleAverageOperator(self, config_names=ensemble_list,
                                         defer_strategy=(budget_plan.strategy == "defer_to_incremental"),
                                         emit_averaged_probabilities=False,
                                         name="ensemble_average_op")
   ```
   and for each `cfg` in `ensemble_list`:
   `self.add_flow(frag_of(cfg), ensemble_op, {("probabilities", f"prob_{cfg}")})`;
   then `self.add_flow(ensemble_op, postprocess_op, {("seg", "seg")})`.
   For a single-config ensemble list, the same code path applies (one port,
   CountCondition(1)).
5. Postprocess + all writer flows UNCHANGED (postprocess still receives `image` from
   `series_to_vol_op` for SR/SC).
6. Verification — run ALL FOUR configurations (Execution Environment command with the
   respective `HOLOSCAN_MODEL_LIST`), each must:
   - exit 0 and write SEG/SR/SC;
   - log `run_model_list=` / `ensemble_model_list=` matching the Context table exactly;
   - emit exactly one `study_timing_summary:` containing per-config records
     (`"config": "3d_lowres"` etc. present iff that config ran) — this proves each
     fragment's operators fired exactly once per study (no double-fire, no silent
     no-op from a mis-wired port);
   - the lowres-only run's SEG differs from the fullres-only SEG (sanity: different
     config, different segmentation — pixel-exactness per config is Plan 05's gate, not
     this plan's).
   If a configuration deadlocks or GXF rejects an entity, the failure mode is a
   declared-port/flow mismatch — fix the port table (Task context), not with custom
   synchronization (RESEARCH Don't-Hand-Roll: scheduler guarantees only).
7. Note (honest, D-16): after the runs, record in the plan SUMMARY whether the nsys
   trace (Plan 06) shows visible cross-fragment stream overlap; absence is acceptable.
  </action>
  <acceptance_criteria>
    - `grep -n "HOLOSCAN_MODEL_LIST" my_app/app.py` returns 1 line (env var read in compose()); `grep -n "resolve_run_model_list" my_app/app.py` returns 1 line.
    - `grep -n "Fragment(self" my_app/app.py` returns 1 line inside a `for ... in run_list` loop; `grep -n "CudaStreamPool(" my_app/app.py` returns 1 line with `nvtx_identifier=f"streams_{cfg}"`.
    - `grep -n "lowres_seg" my_app/app.py` shows the cross-fragment `add_flow(... {("lowres_seg", "lowres_seg")})` guarded by the cascade-present condition.
    - `grep -cn "emit_probabilities=(cfg in ensemble_list)\|emit_lowres_seg=" my_app/app.py` >= 2 (conditional port table implemented).
    - NO config-name literal inside the fragment factory loop other than via `run_list`/plans lookups: `sed -n '/for .*run_list/,/add_flow(series_to_vol/p' my_app/app.py | grep -n "3d_"` returns zero matches (D-02).
    - All four E2E runs exit 0 with SEG/SR/SC present; each run's log shows the expected `run_model_list=`/`ensemble_model_list=` from the Context table; each log has exactly one `study_timing_summary:` and the correct set of `"config":` values (grep counts: bundle run has 3 distinct config values incl. 3d_lowres; cascade-only run has 2; fullres/lowres-only runs have 1).
    - Plan 02 invariants survive: `head -40 my_app/app.py` still shows gpu_bootstrap before monai.deploy; `grep -n "memory_budget:\|warm_pool" my_app/app.py` both present.
  </acceptance_criteria>
  <verify>cd examples/apps/cchmc-nnunet-fast && ulimit -s unlimited && for cfg in "3d_fullres" "3d_lowres" "3d_cascade_fullres" ""; do out=/tmp/p2p4_e2e_${cfg:-bundle}; rm -rf $out; if [ -z "$cfg" ]; then /tmp/monai-env/.venv/bin/python my_app -i /users/srv-mde/projects/monai-deploy-app-sdk/testdata/airway_input -m /users/srv-mde/projects/monai-deploy-app-sdk/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models -o $out > /tmp/p2p4_log_${cfg:-bundle}.txt 2>&1; else HOLOSCAN_MODEL_LIST=$cfg /tmp/monai-env/.venv/bin/python my_app -i /users/srv-mde/projects/monai-deploy-app-sdk/testdata/airway_input -m /users/srv-mde/projects/monai-deploy-app-sdk/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models -o $out > /tmp/p2p4_log_${cfg}.txt 2>&1; fi; echo "$cfg EXIT=$?"; grep -E "run_model_list=|ensemble_model_list=" /tmp/p2p4_log_${cfg:-bundle}.txt; grep -c "study_timing_summary:" /tmp/p2p4_log_${cfg:-bundle}.txt; ls $out/SEG $out/SR $out/SC >/dev/null && echo "OUTPUTS_OK"; done</verify>
  <done>All four model-list configurations run end-to-end on the airway study: one Fragment per config, cascade lowres_seg crosses fragments with zero disk I/O, ensemble order fixed, per-fragment stream pools wired, timing/NVTX per-config and app-keyed — PIPE-03/PIPE-04/INF-009/INFR-004/INFR-005 delivered at the DAG level.</done>
</task>

## Verification
- 4/4 configurations exit 0 with SEG/SR/SC (verify block above) — expected run/ensemble
  lists logged exactly as in the Context table.
- `study_timing_summary:` count == 1 per run; per-config `"config":` values match the
  configs that ran (each fragment's operators fired exactly once).
- `head -40 app.py` still RMM-first; `memory_budget:` + `warm_pool` still present.
- `scripts/test_cascade_config.py` still exit 0 (ensemble order test included).
- git log shows atomic commits per task.

## Success Criteria
- [ ] PIPE-03: each config runs as its own Fragment in one DAG; config-generic instantiation (D-02); env-var selection with reference-default fallback
- [ ] PIPE-04: lowres argmax seg → cascade fragment input, zero `.nii.gz`/`.npz` I/O, end-to-end in the cascade-only and bundle configurations
- [ ] INF-009: ensemble in ensemble_model_list order with Phase 1 bit-exact math; D-19 deviation documented at the source
- [ ] INFR-004: per-fragment CudaStreamPool (NonBlocking, reserved_size=1, named)
- [ ] INFR-005: per-config NVTX names + app-keyed timing records
- [ ] Port discipline: no declared-without-flow port in any of the 4 configurations
