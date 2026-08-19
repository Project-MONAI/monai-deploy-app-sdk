# Phase 2: GPU Acceleration - Context

**Gathered:** 2026-08-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace remaining CPU-bound preprocessing (transpose, crop, normalize) with CuPy; keep resampling on the reference scipy/CPU path. Wire all available nnUNet configurations as independent Holoscan Fragments in one DAG, including the cascade path (3d_lowres seg → one-hot channel stack → 3d_cascade_fullres) with zero disk I/O. Add RMM pre-allocation, memory-budget calculator, CudaStreamPool, and ensemble infrastructure. Prove pixel-exact equivalence per config and a measurable E2E latency improvement vs Phase 1 (61.8 s) and the reference baseline (169.7 s).

</domain>

<decisions>
## Implementation Decisions

### 2D config coverage (PIPE-03, TEST-005)
- **D-01:** Gate on the THREE 3D configs only: 3d_fullres, 3d_lowres, 3d_cascade_fullres. The corpus has no 2d model (verified: `examples/apps/cchmc_nnunet_fifteen_ckpt_app/models/` contains only the three 3D config dirs, 5 folds each).
- **D-02:** Fragment wiring must stay config-generic — a 2d model must drop in later with zero code changes to the fragment instantiation logic.
- **D-03:** Do NOT synthesize or dummy a 2d model: it would require changes to the inference and ensemble configuration files, which the user has ruled out. 2D is documented as blocked-on-model in VERIFICATION; TEST-005 counts as met-with-deviation (same pattern as the Phase 0 corpus deviation).
- **D-04:** The user will locate a real 2d model to test after all phases are complete and the airway model is fully tested.

### Reference oracle & pixel-exact gate (TEST-005, task 2.6)
- **D-05:** Generate fresh per-config references for the per-config pixel-exact checks: `testdata/ref_lowres_only` and `testdata/ref_cascade_only`, produced with the same reference-harness pattern as `testdata/ref_fullres_only` (`.planning/scripts/reference_fullres_run.py` + `REFERENCE_RUN_GUIDE.md`).
- **D-06:** Final ensemble gate: fast app's bundle output (3d_fullres + 3d_cascade_fullres probability maps averaged, exactly as the reference app bundles them) vs `testdata/current_output` (the reference full-bundle run, 2447 voxels).
- **D-07:** 3d_lowres runs standalone and is correctness-checked against its per-config reference, but is NOT part of the final bundle — matching reference app semantics.
- **D-08:** Controlling equivalence level is segmentation-level identity (same as Phase 1: FP32-ours vs FP16-reference logits are bit-for-bit unreachable by design). `scripts/pixel_diff.py` is the gate tool.

### Cascade wiring (PIPE-04, task 2.5)
- **D-09:** lowres post-softmax **argmax** → one-hot float stack on GPU → concatenated as extra channels into the 3d_cascade_fullres preprocess input. This mirrors nnUNet's cascade plans exactly. Zero `.nii.gz`/`.npz` I/O between the configs; zero-copy GPU buffer handoff.
- **D-10:** (Anti-decision, locked) Do NOT feed raw lowres probabilities instead of argmax — it diverges from the nnUNet reference and would break pixel-exactness.

### CuPy preprocessing port (PREP-01, PREP-02, PREP-04; tasks 2.1–2.3)
- **D-11:** Correctness strategy is final-gate-only: NO per-op intermediate byte-identity checks. The port is validated solely by the per-config and bundle pixel-exact gates. (User explicitly chose this over per-op intermediate byte-identity checks.)
- **D-12:** Keep fp32 and C-contiguous throughout the CuPy ops — Phase 1 proved 1-ulp divergence from a single contiguity slip at 16M voxels.
- **D-13:** Resampling stays on the reference scipy/CPU path (PREP-03, locked since Phase 0; GPU resampling deferred to v2). The resulting GPU→CPU→GPU round-trip around the resample is expected and accepted (~64 MB fp32 per study).

### Infrastructure (INFR-01…INFR-004; tasks 2.7–2.10)
- **D-14:** RMM pool pre-allocation in `setup()` is required (INFR-01).
- **D-15:** Memory-budget calculator (INFR-03) required and unit-tested with synthetic large-volume sizes that force the "defer to incremental" branch. The real OOM path on a 40 GB A100 / 256-slice study is expected never to trigger — document it as unexercised, don't fake it.
- **D-16:** `CudaStreamPool` (INFR-004) is wired but best-effort: visible stream overlap in an Nsight trace is a plus, not a gate requirement; an honest note suffices if overlap isn't visible.
- **D-17:** INFR-02 (pre-allocated buffers reused across `compute()` calls) is DEFERRED to Phase 3 — the single-study dev corpus can't prove cross-study reuse. The user will add additional reference examples in Phase 3 specifically to make this provable.
- **D-18:** Latency bar: ANY positive E2E improvement vs Phase 1's 61.8 s, reported with per-operator deltas vs Phase 1 and vs the 169.7 s reference baseline. No hard percentage target.

### Ensemble (INF-009, task 2.9)
- **D-19:** Pixel-exactness takes precedence over the literal "incremental in-place averaging" wording of INF-009. Phase 1's in-memory full-copy + exact-final-division averaging is already disk-free and bit-exact. If incremental in-place averaging breaks segmentation-level identity, keep the Phase 1 approach and document the deviation. (Agent guidance, consistent with D-11/D-08.)

### Agent's Discretion
- Exact CuPy kernel choices and op fusion within the preprocess operator
- How the multi-fragment DAG config is exposed (config file shape, fragment naming)
- Exact synthetic sizes used for budget-calculator unit tests
- Structured timing log extensions for the new operators

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Roadmap & requirements
- `.planning/ROADMAP.md` §"Phase 2: GPU Acceleration" - tasks 2.1–2.11, success criteria, risk table
- `.planning/REQUIREMENTS.md` - PIPE-03, PIPE-04, PREP-01, PREP-02, PREP-03, PREP-04, INF-009, INFR-01, INFR-02, INFR-03, INFR-004, TEST-005 (lines 14–51)
- `.planning/STATE.md` - Phase 1 transition log: contiguity 1-ulp lesson, ref_fullres_only gate pattern, ensemble averaging bit-exactness notes, orientation forensics
- `.planning/PROJECT.md` - Key Decisions table (resampling stays CPU, latency-first, pixel equivalence constraint)

### Reference harness & oracles
- `.planning/scripts/REFERENCE_RUN_GUIDE.md` - how to regenerate reference outputs (pattern for `ref_lowres_only`, `ref_cascade_only`)
- `.planning/scripts/reference_fullres_run.py` - the reference runner to adapt for per-config oracles
- `testdata/current_output/` - reference full-bundle run (3d_fullres + 3d_cascade_fullres ensemble) — final gate target
- `testdata/ref_fullres_only/` - existing 3d_fullres-only reference (SC/SEG/SR)
- `testdata/airway_input/`, `testdata/airway_output/` - dev study input and historical GT

### Baselines
- `.planning/benchmarks/baseline-2026-08-18.csv` - Phase 1 numbers (61.2–62.2 s E2E; inference 27.2 s dominant) — the bar to beat
- `.planning/baseline_results.csv` - reference baseline 169,747 ± 7,274 ms

### Existing implementation (read before planning)
- `examples/apps/cchmc-nnunet-fast/my_app/app.py` - current single-config DAG (CONFIG_NAME, 15 flows), NVTX + timing wiring
- `examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py` - current CPU numpy/scipy preprocess to be ported
- `examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py` - setup-time fold loading, per-fold autocast boundary, inferenceParams-driven config
- `examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py` - current in-memory bit-exact averaging (INF-009 tension point)
- `examples/apps/cchmc-nnunet-fast/my_app/config/` - inferenceParams / detect_available_folds / load_inference_params / resolve_checkpoint_name (config-generic wiring lives here)
- `examples/apps/cchmc-nnunet-fast/scripts/pixel_diff.py` - the gate tool (extend to lowres/cascade/bundle)
- `examples/apps/cchmc-nnunet-fast/scripts/gpu_residency.py` - residency gate to re-run after the CuPy port
- `examples/apps/cchmc_nnunet_fifteen_ckpt_app/models/` - the three available config dirs (5 folds each); 2d absent

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/pixel_diff.py`: raw-byte + decoded-voxel SEG compare with exit codes and JSON — extend for lowres/cascade references and the bundle gate
- `reference_fullres_run.py` pattern: per-config reference generation with `set_num_threads` scope + fold selection — clone for lowres/cascade
- `my_app/config` inferenceParams: config- and checkpoint-driven parameter loading — the seam where multi-fragment instantiation plugs in
- `scripts/gpu_residency.py`: AST + runtime frame attribution — re-run post-port to keep the exactly-one-boundary-.cpu() invariant (note: D-13's resample round-trip adds a LEGITIMATE boundary; the test's allow-list must be updated deliberately, not silenced)

### Established Patterns
- Setup-time one-shot model load; per-fold autocast boundary outside the fold loop (torch 2.13 corrupts forwards after mid-autocast load — do not "fix")
- FP32 accumulation, CuPy exact final division for averaging (torch CUDA `/= n` is 1-ulp off)
- NVTX range per operator + structured timing {operator, study, start, end, duration_ms}
- Deterministic CuPy two-pass connected components with fixed seed (no GPU skimage in venv)

### Integration Points
- `app.py` CONFIG_NAME constant → multi-fragment DAG: one fragment per config, all in one flow; ensemble node consumes fullres + cascade_fullres probability streams (D-06)
- PreprocessOperator: transpose/crop/normalize become CuPy; the scipy resample stays, bookended by the accepted GPU↔CPU round-trip (D-13)
- New cascade channel: lowres fragment's post-softmax argmax → one-hot float (D-09) → cascade_fullres fragment's preprocess input channels

</code_context>

<specifics>
## Specific Ideas

- On the 2D deferral: "using a dummy 2d model will require changes to the inference and ensemble configuration files which is not recommended. I will find a 2d model to test after all phases are complete and the airway model is fully tested."
- On INFR-02 deferral: "I will add additional reference examples for phase 3" — Phase 3 planning should expect a multi-study corpus to exist for cross-study buffer-reuse proof.

</specifics>

<deferred>
## Deferred Ideas

- **2D config E2E validation** — user will supply a real 2d model after all phases; fragment wiring kept generic (D-02) so it is a test, not a code change, at that point.
- **INFR-02 cross-study buffer reuse proof** — moved to Phase 3; user is adding additional reference examples to make it provable.
- **≥5-study final pixel-exact gate re-run** — pre-existing deferral; corpus not yet supplied.
- **GPU resampling** — v2 (GPUP-01), unchanged.
- **Throughput / concurrent multi-study** — out of scope (PROJECT.md).

</deferred>

---

*Phase: 02-gpu-acceleration*
*Context gathered: 2026-08-18*
