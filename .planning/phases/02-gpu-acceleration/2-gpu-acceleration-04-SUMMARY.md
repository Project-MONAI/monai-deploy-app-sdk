---
phase: 2-gpu-acceleration
plan: 04
subsystem: multi-fragment-dag
tags: [fragments, subgraph, cascade-flow, model-list, ensemble-order, cuda-stream-pool, nvtx, timing-keying, d-02, d-07, d-09, d-10, d-16, d-19, pipe-03, pipe-04, infr-004, infr-005]

requires:
  - phase: 2-gpu-acceleration (plan 01)
    provides: CuPy preprocess flow (1-channel path kept byte-for-byte), to_holoscan_gpu_tensor
  - phase: 2-gpu-acceleration (plan 02)
    provides: gpu_bootstrap first-import + RMM budget/warm_pool infra in compose() (kept and extended), defer_strategy ensemble flag
  - phase: 2-gpu-acceleration (plan 03)
    provides: resolve_run_model_list reference semantics, cascade-capable PreprocessOperator (lowres_seg in), PostResampleOperator conditional ports (lowres_seg out), the uint8-original-orientation port contract this plan wires across the fragment boundary

provides:
  - "The phase's architectural deliverable (PIPE-03/PIPE-04): one Subgraph per resolved nnUNet config in a single DAG — config-generic factory over resolve_run_model_list (zero config-name literals, D-02), HOLOSCAN_MODEL_LIST env-var selection with reference default, cross-fragment lowres_seg cascade flow with zero disk I/O, app-level ensemble with ordered prob_<cfg> ports, per-config CudaStreamPool (INFR-004), config-tagged NVTX + app-keyed timing (INFR-005, Pitfall 9 fixed)"
  - "4 runnable model-list configurations, all E2E exit 0 on the airway study with SEG/SR/SC: 3d_fullres / 3d_lowres (self-ensemble, plan-04 extension) / 3d_cascade_fullres (auto-inserts 3d_lowres) / default bundle — each with exactly one study_timing_summary and per-config records proving each fragment's operators fired exactly once"
  - "EnsembleAverageOperator multi-config mode: prob_<cfg> input ports + CountCondition(len) entry + list-order reconstruction (never arrival order) over the UNCHANGED Phase 1 bit-exact averaging; D-19 INF-009 met-with-deviation documented at the source"
  - "gpu_util._root: top-level-application keying for study ids + timing records across sub-Fragments (Pitfall 9)"
  - "Gate evidence: .planning/phases/02-gpu-acceleration/plan04-gates/ (4 E2E logs + verification.txt + fullres pixel-diff 99.99986%)"

affects: [phase 2 plan 05 — per-config pixel-exact gates run the same 4 configurations via HOLOSCAN_MODEL_LIST against ref_fullres_only/ref_lowres_only/ref_cascade_only/testdata/current_output; plan 06 — benchmark parsing relies on the per-config operator names (preprocess_<cfg>/inference_<cfg>/postresample_<cfg>) and the single app-keyed study_timing_summary this plan established]

tech-stack:
  added: []
  patterns:
    - "holoscan-cu13 4.2 multi-fragment reality (live-verified this plan): the app_driver REJECTS an app graph mixing C++ Fragment + app-level operators ('Both fragments and operators are added to the application graph'); the fragment-to-fragment add_flow overload addresses ports by bare operator name only (no op/port addressing). Subgraph + add_input_interface_port/add_output_interface_port is the supported mechanism — interface ports give exact port addressing, and subgraphs mix freely with app-level operators"
    - "Subgraph.__init__(parent, name) runs compose() during construction — constructor state before super() (same holoscan 4.2 quirk as Operator)
"
    - "Subgraph.fragment -> owning top-level Application (verified) — the _root timing key for sub-Fragment operators (Fragments use .application; apps are self-referential on both)"

key-files:
  created:
    - .planning/phases/02-gpu-acceleration/plan04-gates/e2e-3d_fullres.log
    - .planning/phases/02-gpu-acceleration/plan04-gates/e2e-3d_lowres.log
    - .planning/phases/02-gpu-acceleration/plan04-gates/e2e-3d_cascade_fullres.log
    - .planning/phases/02-gpu-acceleration/plan04-gates/e2e-bundle.log
    - .planning/phases/02-gpu-acceleration/plan04-gates/verification.txt
    - .planning/phases/02-gpu-acceleration/plan04-gates/e2e-fullres-pixel-diff.txt
  modified:
    - examples/apps/cchmc-nnunet-fast/my_app/app.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/gpu_util.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py
    - examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py
    - examples/apps/cchmc-nnunet-fast/my_app/config/__init__.py
    - examples/apps/cchmc-nnunet-fast/scripts/test_cascade_config.py

key-decisions:
  - "Subgraph, not C++ Fragment, is the per-config unit (deviation forced by the live 4.2 app_driver): the plan's verified-API block only verified SIGNATURES — the first wiring smoke (RESEARCH's flagged residual risk) hit the driver's fragment/operator mixing ban + the fragment-to-fragment port-name limitation. Subgraph interface ports deliver the same architecture (one sub-fragment per config, cross-fragment cascade flow, DICOM I/O + ensemble + postprocess + writers app-level) with exact port addressing; empirically validated with a minimal probe before touching the app"
  - "lowres-only is runnable (plan 04's table + must-haves are controlling over the reference's ValueError replicated in plan 03): resolve_run_model_list falls back to ensemble=run when the ensemble is empty but the run list is not; a truly empty list still raises the reference's exact error — unit-tested both ways"
  - "Cross-fragment edges are wired at the subgraph level with interface-port names: self.add_flow(subgraphs[aux], subgraphs[cascade], {('lowres_seg','lowres_seg')}) and self.add_flow(subgraphs[cfg], ensemble_op, {('probabilities', f'prob_{cfg}')}) — the plan's key_links realized on the 4.2-supported object types"
  - "Cascade preprocess entry condition: explicit CountCondition(subgraph, n_entry_inputs) with n_entry_inputs plans-driven (2 iff raw plans previous_stage is present) — the cascade preprocess must fire once after BOTH image and lowres_seg, never on the image alone (RESEARCH Pattern 1)"
  - "Budget cfgs extended to ALL run-list configs (the sub-Fragments coexist in one DAG so their full-volume footprints add); per-config median_image_size_in_voxels read from the PlansManager-RESOLVED configuration (inherited configs carry it via inherits_from — the plan 03 Rule 3 lesson)"
  - "D-16 honest note: CudaStreamPool wired per subgraph (NonBlocking, reserved_size=1, nvtx_identifier streams_<cfg>); whether the nsys trace shows visible cross-fragment stream overlap is Plan 06's trace work — absence would be acceptable per D-16"

patterns-established:
  - "Config-generic subgraph factory: iterate resolve_run_model_list(run_list) — NO config-name literals in the loop (D-02); conditional interface ports (probabilities only for ensemble members, lowres_seg only for the auxiliary previous stage) implement the plan's port table so no declared port is ever left without a flow/receiver (Pitfall 7)"
  - "Per-config NVTX range + timing label (preprocess_<cfg>/inference_<cfg>/postresample_<cfg>) + 'config' field in every per-config timing record; per-study aggregate keyed by the top-level Application (gpu_util._root) — one study_timing_summary per study regardless of fragment count (Pitfall 9; Plan 06's CSV parsing depends on it)"

requirements-completed: [PIPE-03, PIPE-04, INF-009, INFR-004, INFR-005]
requirements-deferred: []

deviations:
  - "API mismatch (blocked task 3, resolved): the plan's C++ Fragment + app.add_fragment + app.add_flow(fragment, operator) wiring does not exist in holoscan-cu13 4.2 — app_driver rejects mixed graphs ('Both fragments and operators are added to the application graph') and the fragment-to-fragment flow overload cannot address operator ports (live-probed: every op/port delimiter treated as a whole operator name). Implemented the architecture with Subgraph + interface ports instead (see key-decisions). Plan acceptance greps that assume the Fragment API (e.g. `Fragment(self` in the factory loop) map to `NnUnetConfigSubgraph(` inside the same run_list loop; all INTENT-level criteria (config-generic factory, env var, stream pools, cascade flow, 4 E2E runs, port discipline) verified"
  - "Rule 1 (bug, latent from plan 03, first exercised by the real cascade DAG): the cascade one-hot stacked the 4D (1,*spatial) resampled seg to 5D -> cp.concatenate 'All arrays to concatenate must have the same ndim'. Fix: one-hot seg[0] (3D), exactly the reference's convert_labelmap_to_one_hot(seg[0], ...) semantics; regression test test_preprocess_image_cascade_two_channel added (14-case suite). Plan 03's unit suite tested the seg-resample and one-hot PRIMITIVES but never the 2-channel preprocess_image path end-to-end"
  - "Plan-text contradiction (plan 03 vs plan 04 on lowres-only): plan 03 replicated the reference ValueError (empty ensemble) and unit-tested it; plan 04's Context table + must-have 'all four configurations exit 0' + acceptance (lowres-only run, ensemble_model_list=['3d_lowres'], SEG differs from fullres-only) require lowres-only to be runnable. Implemented the plan-04 behavior as a documented fast-app extension (self-ensemble fallback); the reference error is kept for truly empty run lists and both are unit-tested"
  - "gpu_util._root extended to resolve Subgraph.fragment (Fragments expose .application; Subgraphs expose .fragment -> the owning top-level app) — required for the app-keyed per-study aggregate to survive the Subgraph switch (Pitfall 9)"

metrics:
  duration: ~62min
  tasks: 3/3
  commits: 5
  unit-tests: "scripts/test_cascade_config.py — 14 cases (12 plan-03 + test_ensemble_order + test_preprocess_image_cascade_two_channel), all exit 0"
  e2e: "4/4 model-list configurations exit 0 with SEG/SR/SC; exactly one study_timing_summary per run; per-config records exactly once per fragment; lowres-only SEG != fullres-only SEG; fullres regression pixel_diff 99.99986% byte-identity vs testdata/ref_fullres_only (same 3 documented fp16<->fp32 boundary voxels, IoU 0.998714)"

completed: 2026-08-19
---

# Phase 2 Plan 04: Multi-Fragment DAG Assembly Summary

**The single-config DAG is now the reference-semantics multi-fragment DAG: one `Subgraph` per `resolve_run_model_list` entry with interface ports, the lowres argmax segmentation crosses the sub-fragment boundary to the cascade preprocess with zero disk I/O, the app-level ensemble consumes one ordered `prob_<cfg>` stream per ensemble config over the untouched Phase 1 bit-exact math, every NVTX range/timing record carries its config, and the per-study aggregate is keyed by the top-level Application — all four model-list configurations run end-to-end exit 0 on the airway study (PIPE-03/PIPE-04/INF-009/INFR-004/INFR-005 at the DAG level).**

## What was built

### Task 1 — config-tagged NVTX + root-application-keyed timing (commit f1e421b)

- `gpu_util._root(fragment)` resolves the top-level Application (`Fragment.application`, verified `frag.application is app` in 4.2); `set_study_id`/`get_study_id`/`StudyTimingCollector.record/studies/clear` all key on `id(_root(...))` — one per-study aggregate across fragments (Pitfall 9). Public signatures unchanged.
- `PreprocessOperator`/`SlideWindowOperator`: NVTX range + timing label now `preprocess_{config_name}` / `inference_{config_name}`; both operators add `"config": <config_name>` to their timing records.
- `PostResampleOperator` gained `config_name: Optional[str] = None` (initialized before `super().__init__`, the holoscan 4.2 quirk); set → `postresample_{config_name}` range/label + `"config"` record field; `None` keeps the bare Phase 1 name.
- Smoke: fullres-only E2E exit 0; log shows `preprocess_3d_fullres` / `inference_3d_fullres` / `"config": "3d_fullres"`; exactly one `study_timing_summary:`.

### Task 2 — ensemble per-config ordered inputs (commit 2bba495)

- `EnsembleAverageOperator.__init__(..., config_names: Optional[Sequence[str]] = None)` — the ORDERED ensemble list, stored BEFORE `super().__init__`; multi-config mode passes `CountCondition(fragment, len(config_names))` positionally (holoscan adds positional Conditions to the operator's conditions — verified 4.2 docs) so the operator never runs on partial arrivals.
- `setup()`: multi-config → `spec.input(f"prob_{cfg}")` per config; legacy (`config_names=None`) keeps the single `probabilities` input (Phase 1/tests).
- `compute()`: multi-config receives each `prob_<cfg>` port and reconstructs the tensor list in `config_names` ORDER (GXF arrival order is not guaranteed) → the UNCHANGED Phase 1 `average_probabilities` (in-place `+=` accumulation + CuPy `_divide_refparity` final division) → `argmax_to_segmentation`. `git diff` confirms zero arithmetic edits to the averaging bodies.
- Module docstring carries the D-19 statement (INF-009 met-with-deviation; running mean is not bit-identical to the reference's sum/n; VRAM intent satisfied; defer_strategy releases consumed tensors).
- `test_ensemble_order` (suite): result `torch.equal` to the manual reference-order `t0 + t1` then CuPy `/2`, and the list-order reconstruction from a REVERSED-arrival dict yields the identical bit-exact tensor.

### Task 3 — multi-fragment DAG + 4-configuration E2E (commits bd49c61, 2309a4d, fb46cc6)

- `compose()` now: reads `HOLOSCAN_MODEL_LIST` (comma-separated; unset = reference default) → `resolve_run_model_list` → logs `run_model_list=`/`ensemble_model_list=` → budget over ALL run-list configs (resolved-configuration medians) → per-config subgraph factory → cascade edge → app-level ensemble → unchanged postprocess/writers/`warm_pool`.
- `NnUnetConfigSubgraph` (per config, name `nnunet_<cfg>`): `preprocess_<cfg> → slidewindow_<cfg> → postresample_<cfg>`; interface ports — `image` in (always), `lowres_seg` in (cascade configs; explicit `CountCondition(subgraph, n)` with plans-driven `n_entry_inputs`), `probabilities` out (ensemble members only, D-07), `lowres_seg` out (auxiliary previous stage only). Zero config-name literals in the factory loop (D-02).
- Cross-fragment cascade edge `subgraphs[aux] → subgraphs[cascade]` on `lowres_seg` (only when the auxiliary stage is in the run list); `subgraphs[cfg] → ensemble` on `("probabilities", f"prob_{cfg}")` for every ensemble config (single-config lists take the same one-port path).
- Per-subgraph `CudaStreamPool(dev_id=0, stream_flags=1, reserved_size=1, nvtx_identifier=f"streams_{cfg}")` (INFR-004, best-effort D-16).
- `gpu_util._root` extended: `Subgraph.fragment` → top-level app (Fragments: `.application`), so the app-keyed aggregate survives the Subgraph switch.
- E2E (all on the airway study, `ulimit -s unlimited`):

| HOLOSCAN_MODEL_LIST | run / ensemble (logged) | result |
|---|---|---|
| unset | `[3d_fullres, 3d_lowres, 3d_cascade_fullres]` / `[3d_fullres, 3d_cascade_fullres]` | exit 0, SEG/SR/SC, 1 summary, 14 records (3 configs × 3 ops + ensemble + postprocess + 3 writers) |
| `3d_fullres` | `[3d_fullres]` / `[3d_fullres]` | exit 0, SEG/SR/SC, 1 summary, 8 records |
| `3d_lowres` | `[3d_lowres]` / `[3d_lowres]` | exit 0, SEG/SR/SC, 1 summary, 8 records; SEG ≠ fullres-only SEG |
| `3d_cascade_fullres` | `[3d_lowres, 3d_cascade_fullres]` (auto-insert) / `[3d_cascade_fullres]` | exit 0, SEG/SR/SC, 1 summary; 2 distinct `"config"` values |

  Each fragment's operators fired exactly once per study (no double-fire, no silent no-op); `pixel_diff` fullres run vs `testdata/ref_fullres_only`: **99.99986% byte-identity, same 3 documented fp16↔fp32 boundary voxels, IoU 0.998714** — Plan 02/03 regression intact. Bundle per-operator (single-study aggregate): inference 26.1/25.1/25.4 s (fullres/lowres/cascade), preprocess 7.6/5.0/9.3 s, postresample 1.7/3.2/1.7 s, ensemble 0.012 s, postprocess 2.4 s, writers ~1.1 s.

## Gate results

| Gate | Result |
|---|---|
| `scripts/test_cascade_config.py` (14 cases) | exit 0, all PASS |
| 4/4 model-list E2E | exit 0, SEG/SR/SC, 1 `study_timing_summary` each, correct per-config record sets |
| lowres-only SEG vs fullres-only SEG | differ (sanity: different config → different segmentation) |
| fullres regression pixel_diff vs `testdata/ref_fullres_only` | 99.99986% byte-identity, 3 boundary voxels, IoU 0.998714, PASS |
| Plan 02 invariants | `gpu_bootstrap` first import (head -40), `memory_allocator_backend` assert, `memory_budget:` (now over the full run list), `warm_pool` at end of compose — all present |

Evidence: `.planning/phases/02-gpu-acceleration/plan04-gates/`.

## Success criteria (plan)

- [x] PIPE-03: each config runs as its own sub-Fragment in one DAG; config-generic instantiation (D-02, zero literals in the factory loop); `HOLOSCAN_MODEL_LIST` selection with reference-default fallback
- [x] PIPE-04: lowres argmax seg → cascade sub-fragment input across the fragment boundary, zero `.nii.gz`/`.npz` I/O, end-to-end in the cascade-only and bundle configurations
- [x] INF-009: ensemble in ensemble_model_list order (list-order reconstruction, never arrival order) with the untouched Phase 1 bit-exact math; D-19 deviation documented at the source
- [x] INFR-004: per-fragment CudaStreamPool (NonBlocking `stream_flags=1`, `reserved_size=1`, `nvtx_identifier=streams_<cfg>`); overlap visibility is Plan 06's nsys trace (D-16 best-effort — honest note: NOT measured in this plan)
- [x] INFR-005: per-config NVTX range names + timing labels + `"config"` record fields; per-study aggregate keyed by the top-level application (single `study_timing_summary` in all 4 configurations)
- [x] Port discipline: no declared-without-flow port in any of the 4 configurations (conditional interface ports per the plan's port table; all 4 runs execute to completion — the Pitfall 7 failure modes, hang or GXF rejection, did not occur)

## Known stubs

None. All wiring is exercised by the 4 E2E runs + 14-case unit suite. The only not-yet-measured item is the nsys cross-fragment stream-overlap observation (D-16), which is explicitly Plan 06's trace work.

## Self-Check: PASSED

All key files verified present (plan04-gates evidence ×6, app.py, test_cascade_config.py) and all five task commits (f1e421b, 2bba495, bd49c61, 2309a4d, fb46cc6) found in git history.
