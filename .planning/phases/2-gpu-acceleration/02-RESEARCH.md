# Phase 2: GPU Acceleration - Research

**Researched:** 2026-08-19
**Domain:** GPU-native nnUNet inference (CuPy preprocessing, multi-fragment Holoscan DAG, cascade wiring, RMM/stream-pool infra) inside the existing `cchmc-nnunet-fast` app
**Confidence:** HIGH (all claims verified against the live venv, vendored nnunetv2 2.8.1 source, the model bundle, and the existing Phase 1 code — no WebSearch required for core findings)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### 2D config coverage (PIPE-03, TEST-005)
- **D-01:** Gate on the THREE 3D configs only: 3d_fullres, 3d_lowres, 3d_cascade_fullres. The corpus has no 2d model (verified: `examples/apps/cchmc_nnunet_fifteen_ckpt_app/models/` contains only the three 3D config dirs, 5 folds each).
- **D-02:** Fragment wiring must stay config-generic — a 2d model must drop in later with zero code changes to the fragment instantiation logic.
- **D-03:** Do NOT synthesize or dummy a 2d model: it would require changes to the inference and ensemble configuration files, which the user has ruled out. 2D is documented as blocked-on-model in VERIFICATION; TEST-005 counts as met-with-deviation (same pattern as the Phase 0 corpus deviation).
- **D-04:** The user will locate a real 2d model to test after all phases are complete and the airway model is fully tested.

#### Reference oracle & pixel-exact gate (TEST-005, task 2.6)
- **D-05:** Generate fresh per-config references for the per-config pixel-exact checks: `testdata/ref_lowres_only` and `testdata/ref_cascade_only`, produced with the same reference-harness pattern as `testdata/ref_fullres_only` (`.planning/scripts/reference_fullres_run.py` + `REFERENCE_RUN_GUIDE.md`).
- **D-06:** Final ensemble gate: fast app's bundle output (3d_fullres + 3d_cascade_fullres probability maps averaged, exactly as the reference app bundles them) vs `testdata/current_output` (the reference full-bundle run, 2447 voxels).
- **D-07:** 3d_lowres runs standalone and is correctness-checked against its per-config reference, but is NOT part of the final bundle — matching reference app semantics.
- **D-08:** Controlling equivalence level is segmentation-level identity (same as Phase 1: FP32-ours vs FP16-reference logits are bit-for-bit unreachable by design). `scripts/pixel_diff.py` is the gate tool.

#### Cascade wiring (PIPE-04, task 2.5)
- **D-09:** lowres post-softmax **argmax** → one-hot float stack on GPU → concatenated as extra channels into the 3d_cascade_fullres preprocess input. This mirrors nnUNet's cascade plans exactly. Zero `.nii.gz`/`.npz` I/O between the configs; zero-copy GPU buffer handoff.
- **D-10:** (Anti-decision, locked) Do NOT feed raw lowres probabilities instead of argmax — it diverges from the nnUNet reference and would break pixel-exactness.

#### CuPy preprocessing port (PREP-01, PREP-02, PREP-04; tasks 2.1–2.3)
- **D-11:** Correctness strategy is final-gate-only: NO per-op intermediate byte-identity checks. The port is validated solely by the per-config and bundle pixel-exact gates. (User explicitly chose this over per-op intermediate byte-identity checks.)
- **D-12:** Keep fp32 and C-contiguous throughout the CuPy ops — Phase 1 proved 1-ulp divergence from a single contiguity slip at 16M voxels.
- **D-13:** Resampling stays on the reference scipy/CPU path (PREP-03, locked since Phase 0; GPU resampling deferred to v2). The resulting GPU→CPU→GPU round-trip around the resample is expected and accepted (~64 MB fp32 per study).

#### Infrastructure (INFR-01…INFR-004; tasks 2.7–2.10)
- **D-14:** RMM pool pre-allocation in `setup()` is required (INFR-01).
- **D-15:** Memory-budget calculator (INFR-03) required and unit-tested with synthetic large-volume sizes that force the "defer to incremental" branch. The real OOM path on a 40 GB A100 / 256-slice study is expected never to trigger — document it as unexercised, don't fake it.
- **D-16:** `CudaStreamPool` (INFR-004) is wired but best-effort: visible stream overlap in an Nsight trace is a plus, not a gate requirement; an honest note suffices if overlap isn't visible.
- **D-17:** INFR-02 (pre-allocated buffers reused across `compute()` calls) is DEFERRED to Phase 3 — the single-study dev corpus can't prove cross-study reuse. The user will add additional reference examples in Phase 3 specifically to make this provable.
- **D-18:** Latency bar: ANY positive E2E improvement vs Phase 1's 61.8 s, reported with per-operator deltas vs Phase 1 and vs the 169.7 s reference baseline. No hard percentage target.

#### Ensemble (INF-009, task 2.9)
- **D-19:** Pixel-exactness takes precedence over the literal "incremental in-place averaging" wording of INF-009. Phase 1's in-memory full-copy + exact-final-division averaging is already disk-free and bit-exact. If incremental in-place averaging breaks segmentation-level identity, keep the Phase 1 approach and document the deviation. (Agent guidance, consistent with D-11/D-08.)

### Agent's Discretion
- Exact CuPy kernel choices and op fusion within the preprocess operator
- How the multi-fragment DAG config is exposed (config file shape, fragment naming)
- Exact synthetic sizes used for budget-calculator unit tests
- Structured timing log extensions for the new operators

### Deferred Ideas (OUT OF SCOPE)
- **2D config E2E validation** — user will supply a real 2d model after all phases; fragment wiring kept generic (D-02) so it is a test, not a code change, at that point.
- **INFR-02 cross-study buffer reuse proof** — moved to Phase 3; user is adding additional reference examples to make it provable.
- **≥5-study final pixel-exact gate re-run** — pre-existing deferral; corpus not yet supplied.
- **GPU resampling** — v2 (GPUP-01), unchanged.
- **Throughput / concurrent multi-study** — out of scope (PROJECT.md).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID       | Description (from REQUIREMENTS.md)                                                        | Research Support |
| -------- | ----------------------------------------------------------------------------------------- | ---------------- |
| PREP-01  | Transpose input volume to nnUNet orientation using GPU-accelerated ops                     | §CuPy port: `cp.asarray(...).transpose(...)` + `cp.ascontiguousarray` verified bit-exact vs numpy (Code Examples) |
| PREP-02  | Per-channel mean/std normalization on GPU before resampling                                | §CuPy port: element-wise CuPy ZScore/CT ops verified bit-exact; **mean/std reductions must stay numpy** (Pitfall 4) |
| PREP-03  | Reference scipy/sk-image resampling path for pixel-exactness                               | Unchanged; `preprocess_operator._resample_to_shape` + `postresample_operator.resample_probabilities_to_shape` are the CPU path; cascade seg resample joins it (resampling_fn_seg, Pattern 4) |
| PREP-04  | Crop and pad to expected input shape using GPU operations                                  | §CuPy port: mask/bbox stay scipy-CPU (cheap, reference is CPU); the slice + contiguity materialization move to CuPy |
| PIPE-03  | Each nnUNet config as an independent Holoscan Fragment in the same DAG                     | Pattern 1: verified `Fragment`/`add_fragment`/`add_flow(frag,frag,ports)` API in holoscan-cu13 4.2; config-generic instantiation (D-02) |
| PIPE-04  | Cascade: lowres seg → one-hot channel stack → cascade_fullres input, no disk I/O           | Pattern 4: exact reference chain verified in vendored nnunetv2 2.8.1 source (resampling_fn_seg order=1/is_seg=True, one-hot after resample, 2 input channels) |
| TEST-005 | Test coverage for all four nnUNet configs (2D met-with-deviation per D-01/D-03)            | Pattern 5: `reference_fullres_run.py --config 3d_lowres/3d_cascade_fullres` already parameterized — per-config oracles are two command invocations, not new code |
| INFR-01  | RMM pool allocator, pre-allocated in `setup()`                                             | Pattern 6: verified working RMM wiring in this venv + **import-order hazard** (Pitfall 1); `rmm.mr.PoolMemoryResource` (not `PoolAllocator` — removed) |
| INFR-02  | Pre-allocated buffers reused across compute()                                              | DEFERRED to Phase 3 (D-17) — no task beyond a one-line note |
| INFR-03  | Memory budget calculator before full-volume allocation; defer-to-incremental branch        | Pattern 7: design + synthetic-size unit test plan (D-15) |
| INFR-004 | Holoscan CudaStreamPool for concurrent kernel launches                                     | Pattern 8: verified constructor signature in 4.2; best-effort (D-16) |
| INFR-005 | NVTX markers at each operator compute()                                                    | Extend existing `nvtx_range` pattern; unique per-config range names (Pitfall 9) |
| INFR-03→INFR-05 n/a | (listed IDs covered above)                                                     | — |
| TEST-01  | Bit-for-bit identical DICOM-SEG vs reference (dev-corpus deviation documented)             | Patterns 4/5: per-config gates + bundle gate vs `testdata/current_output` via existing `pixel_diff.py` |
| TEST-006 | Benchmark script, E2E + per-operator breakdown                                             | §Benchmarking: reuse `baseline_benchmark.py` subprocess pattern; parse the app's `timing: {...}` JSON logs; `phase2_results.csv` shape proposal |
| TEST-007 | Benchmark comparison vs current app: speedup ratio + absolute latency                      | Same as TEST-006; bars: 169.7 s reference bundle (primary) and 61.8 s Phase 1 (see Open Question 1 — scope tension) |
</phase_requirements>

## Project Constraints (from AGENTS.md)

- All testing/dev must use the uv venv at `/tmp/monai-env/.venv` (activate via `source activate-env.sh`). Do **not** create new venvs or use system Python.
- Do **not** install `nnunetv2` from PyPI — the vendored editable copy at `./nnUNet/` (2.8.1) must be used. All cascade/one-hot/resampling claims below were verified against that exact vendored source.
- Run commands through `/tmp/monai-env/.venv/bin/python` (or activated env).
- Commit after every major code change: short imperative messages, small atomic commits.
- Carried Phase 0/1 operational constraints (STATE.md): run apps from their own app-root (the `cchmc-nnunet-fast` editable install hijacks package name `my_app` — see Pitfall 8); `ulimit -s unlimited` (or 32768) before running Holoscan; venv monai 1.3.0 needs the two NumPy-2.0 `ptp` patches re-applied after any monai reinstall.

## Summary

Phase 2 is an extension of a proven Phase 1 pipeline, not a rewrite: the same five-operator chain becomes one **fragment per nnUNet config**, instantiated from a config-driven model list with the reference app's exact semantics (auto-insert `3d_lowres` before `3d_cascade_fullres`; ensemble = list minus lowres). The CuPy port touches only transpose/crop/normalize — resampling, the softmax, the Z-score mean/std reductions, and the cascade seg resample all stay on the verified CPU reference path (D-13, and empirically: `cupy.mean/std` are **not** bit-identical to numpy, while every element-wise CuPy op is). The cascade handoff (D-09) maps 1:1 onto the vendored nnunetv2 2.8.1 chain: lowres logits → resample → argmax (no CC — verified the reference cascade consumes the *pre-CC* seg) → transpose/crop with the image's bbox → **CPU** seg resample (`resample_data_or_seg_to_shape(is_seg=True, order=1, order_z=0)`) → one-hot float32 (bit-exact on GPU) → vstack as channel 2 of a 2-channel cascade input.

**Primary recommendation:** keep every *reduction* and every *resample* on the numpy/scipy reference path, move only element-wise and layout ops to CuPy, build the DAG as one fragment per config with reference model-list semantics, and treat the D-18 latency bar (beat 61.8 s with *more* work — 3 configs vs 1) as the phase's top scheduling risk: a serial 3-config bundle will land ~70–80 s; the plan must schedule fragment-level parallelism and an incremental benchmark early, and report bundle-vs-169.7 s (≈2.3–2.5× expected) alongside the same-scope fullres comparison (Open Question 1).

## Environment (verified 2026-08-19)

| Fact | Verified value |
|------|----------------|
| torch | 2.13.0+cu130, CUDA 13.0, `cuda.is_available()` True (autocast-per-fold caveat from Phase 1 still in force) |
| cupy | 14.1.1 (`cupy-cuda13x`), matches cu13 pin |
| holoscan | `holoscan-cu13` 4.2.0 + `holoscan-cli` 4.2.0; `monai.deploy.core` re-exports `Fragment`, `FragmentFlowGraph`, `FragmentService`; `Application.add_fragment(frag)` and the `add_flow(fragment, fragment, port_pairs)` overload **exist** (verified pybind signatures) |
| rmm | `rmm-cu13` 26.02.00. **Works only if imported before `holoscan`** (Pitfall 1). `rmm.reinitialize(pool_allocator=True)` + `rmm.allocators.torch.rmm_torch_allocator` → torch backend `"pluggable"` (re-verified today). `rmm.mr.PoolAllocator` **does not exist** in 26.x — use `rmm.mr.PoolMemoryResource` (verified present) |
| scipy / skimage | 1.15.3 / 0.25.2 (reference resample primitives unchanged) |
| monai | 1.3.0 (vendored `ptp` patches must survive) |
| nnunetv2 | 2.8.1 editable at `./nnUNet/` (all source citations below are from this tree) |
| acvl_utils | 0.2.6 (`pad_nd_image` used by SlideWindow) |
| nsys | `/usr/local/cuda/bin/nsys`, Nsight Systems **2025.6.3** (worked in Phases 0/1; harness at `.planning/scripts/nsight_profile.sh`) |
| ncu | installed (`/opt/nvidia/nsight-compute/2026.1.0/ncu`, also `/usr/local/cuda/bin/ncu`) but **blocked: `ERR_NVGPUCTRPERM`** — user lacks GPU performance-counter permission (verified with a live probe). Nsight Compute kernel profiling requires admin (`NVreg_RestrictProfilingToAdminUsers=0` or sudo). **Plan around nsys-only** for tasks 2.13/2.14. |
| GPU | A100-SXM4-40GB, driver 610.57.04 (CUDA 13.3) |
| Model bundle | `examples/apps/cchmc_nnunet_fifteen_ckpt_app/models/`: exactly `3d_fullres`, `3d_lowres`, `3d_cascade_fullres` (5 folds each) + `jsonpkls/` — **2d absent** (D-01 confirmed). `plans.json` cascade entry is `{"inherits_from": "3d_fullres", "previous_stage": "3d_lowres"}`; resolved via `PlansManager.get_configuration` → patch `[128,128,128]`, spacing = fullres spacing, `PlainConvUNet`, **2 input channels** (1 modality + 1 foreground one-hot), `num_segmentation_heads = 2` (background + airway), `foreground_labels = [1]` |

## Standard Stack

No new packages are required for this phase. Everything is already pinned in the venv.

### Core (existing, versions verified in venv)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| holoscan-cu13 / monai-deploy (SDK) | 4.2.0 | DAG runtime, `Fragment`, `CudaStreamPool`, DICOM I/O | the app already runs on it; Fragment API verified present |
| cupy-cuda13x | 14.1.1 | GPU element-wise preprocessing, one-hot, CC, final division | already used in Phase 1 (ensemble division, CC); cu13 pin |
| torch | 2.13.0+cu130 | inference, DLPack, NVTX, allocator swap | existing |
| rmm-cu13 | 26.02.00 | pool allocator (INFR-01) | already in venv from Phase 0 (test_rmm.py) |
| scipy / scikit-image | 1.15.3 / 0.25.2 | reference resample path (PREP-03) | reference parity |
| nnunetv2 (vendored editable) | 2.8.1 | plans/plans-manager, resample utils, one-hot, sliding-window utils | AGENTS.md mandate |

**Installation:** none (`pip` must not be touched; do not add `rmm-cu12x`/`cupy-cuda12x`).

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `rmm` Python pool | `torch.cuda.memory` expandable segments only (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) | roadmap risk-table mitigation; acceptable as the *torch-side* knob, but INFR-01 explicitly requires RMM and the Phase 0 test already proved the rmm→torch wiring works. Use RMM as primary, expandable_segments as a documented fallback test (Pitfall 6) |
| CuPy for normalize reductions | `torch` GPU reductions | same reduction-order problem; numpy is the reference — no alternative is bit-exact |
| ncu kernel profiling | — | blocked by permissions; nsys + `cuda_api_sum` (cudaMalloc/cudaFree counts, memcpy sizes) covers the acceptance criteria ("no per-tile cudaMalloc churn") |

## Architecture Patterns

### Pattern 1: Multi-fragment DAG (PIPE-03, task 2.4) — config-generic, reference model-list semantics

**What:** One `Fragment` per selected config; DICOM I/O stays at app level; ensemble/postprocess/writers stay at app level.

**Verified API (holoscan-cu13 4.2.0):**
```python
# monai.deploy.core (re-exported; verified dir() + pybind docs)
frag = Fragment(app, name="nnunet_3d_lowres")   # Fragment.__init__(app, name=...)
op = SomeOperator(frag, ...)                    # operators constructed with the fragment
frag.add_operator(op); frag.add_flow(a, b, {("out","in")})
app.add_fragment(frag)                          # Application.add_fragment(frag: Fragment) -> None
# cross-fragment: verified overload exists ("For the Application class there is a
# variant where the first two arguments are of type holoscan.core.Fragment")
app.add_flow(frag_lowres, frag_cascade, {("lowres_seg", "lowres_seg")})
# app-level op -> fragment entry op:
app.add_flow(series_to_vol_op, entry_op_of(frag), {("image", "image")})
```

**Reference model-list semantics to replicate** (`cchmc_nnunet_fifteen_ckpt_app/my_app/nnunet_seg_operator.py:91-96`, verified):
1. `model_list` default = configs from plans that have model folders → here `[3d_fullres, 3d_cascade_fullres]` (matches `testdata/current_output`).
2. If both `3d_lowres` and `3d_cascade_fullres` are present: remove `3d_lowres` from its position and re-insert it **immediately before** `3d_cascade_fullres`.
3. `ensemble_model_list = [m for m in run_model_list if m != "3d_lowres"]` — lowres feeds the cascade but never the ensemble.
4. At least one non-auxiliary config is required.

The fast app should get this exact list from a single source (new `my_app/config` function, e.g. `resolve_run_model_list(model_list_arg)`), with the selection exposed via an env var (e.g. `HOLOSCAN_MODEL_LIST=3d_lowres`) or module constant — agent's discretion on the shape; env var matches the reference app's `.env` pattern and needs no `app.yaml` plumbing. **Do not hard-code config names in `compose()`** (D-02): fragment instantiation must iterate the resolved list, so a future `2d` dir drops in unchanged.

**DAG sketch (default bundle run):**
```
DICOMDataLoader -> SeriesSelector -> SeriesToVolume
  ├─ (image) -> [frag 3d_fullres]  Preprocess->SlideWindow->PostResample ─ probabilities ─┐
  ├─ (image) -> [frag 3d_lowres]   Preprocess->SlideWindow->PostResample ─┬ probabilities (standalone gate only)
  │                                                                         └ lowres_seg = argmax(probs) ─┐
  └─ (image) -> [frag 3d_cascade_fullres] Preprocess(+lowres_seg) -> SlideWindow -> PostResample ─ probs ─┤
                                                                                                          v
                                                        EnsembleAverage (list order, CuPy final /n)
                                                                                  v
                                                        Postprocess (CC) -> SEG / SR / SC writers
```

**Ordering/deadlock mitigation (roadmap risk table):**
- Every declared input port MUST have a flow; every declared output MUST have a receiver (GXF rejects entities otherwise — Phase 1 hit the no-receiver case; `EnsembleAverageOperator` has an `emit_averaged_probabilities=False` flag for exactly this).
- The lowres fragment's `probabilities` port has two consumers in the full bundle (its own standalone-SEG path and the cascade fragment) — in a *lowres-only* run the standalone path must consume or be unwired. Design the fragment so `lowres_seg`/`probabilities` ports are declared conditionally (same pattern as `emit_averaged_probabilities`) based on the resolved model list.
- The cascade fragment's entry needs **two** inputs (image + lowres_seg) → entry operator with `CountCondition(2)` (or fragment `start_op()` for the image + CountCondition(2) at the preprocess op). Verify GXF fires compute exactly once per study (assert via timing records, not just absence of hang).
- Build incrementally per the roadmap risk: (a) fullres+lowres in one DAG (independent), (b) add cascade edge, (c) full bundle. One-study smoke after each.
- `StudyTimingCollector` is currently keyed by `fragment` which equals the app (Phase 1 note). With sub-Fragments, key records by the top-level app (walk `operator.application`, or store the app reference) and include `config` in each record + NVTX name (e.g. `inference_3d_lowres`) so per-study aggregates and nsys ranges stay unambiguous (Pitfall 9).

### Pattern 2: CuPort of transpose/crop/normalize (PREP-01/02/04, tasks 2.1–2.3)

**What:** In `PreprocessOperator.preprocess_image`, keep the current function structure; swap numpy element-wise/layout ops for CuPy; keep the scipy resample and the Z-score reductions on CPU. New data flow per study:

```
image.asnumpy() (DHW)
 -> torch/cp .to("cuda")                     # ONE H2D of the raw volume (new)
 -> cp astype float32 + transpose view + cp.ascontiguousarray   (PREP-01)
 -> mask = (vol != 0) on GPU; binary_fill_holes stays scipy on CPU
    (uint8 mask H2D/D2H is small; reference mask math is scipy CPU anyway)
 -> bbox on CPU (unchanged); slice on CuPy + contiguous materialize    (PREP-04)
 -> mean = vol_cpu_ref.mean(); std = ...   # computed with the SAME numpy code as today
 -> normalized = (vol - mean) / max(std, eps)  or CTNormalization element-wise  (PREP-02, CuPy)
 -> .get() -> C-contiguous fp32 numpy -> _resample_to_shape (scipy, UNCHANGED)  (PREP-03/D-13)
 -> torch.from_numpy -> .to(cuda) (new D2H of the resampled volume)
```

**Bit-exact rules (empirically verified this session — see Code Examples):**
1. `cp` element-wise `-`, `/`, `np.clip`, comparisons, `(x==1).astype(float32)`, and `transpose`+`ascontiguousarray` are **bit-identical** to numpy for fp32. ✔
2. `cp.mean` / `cp.std` / masked reductions are **NOT** bit-identical to numpy (different reduction order — verified unequal on a 1.7M-voxel volume). Mean/std must be produced by the existing numpy code and passed to the GPU as scalars. This preserves reference parity for `ZScoreNormalization` with zero new risk. (For `use_mask_for_norm`-masked normalization — inactive for this bundle (`[False]`) — the masked mean/std must likewise stay numpy.)
3. Everything that enters `_resample_to_shape` must be **C-contiguous fp32** (D-12; Phase 1's 1-ulp lesson). After the CuPy round-trip, assert `arr.flags['C_CONTIGUOUS']` before calling the resample.
4. `gpu_residency.py`'s `ALLOWED_TRANSFER_FILES` must be **deliberately** extended with `preprocess_operator.py` ("D-13 accepted GPU↔CPU resample round-trip + reference-CPU resample") and the `postprocess_operator.py` exactly-once boundary must remain the only *final*-stage `.cpu()` (CONTEXT: "updated deliberately, not silenced").

### Pattern 3: Setup-time infrastructure (INFR-01, INFR-03, INFR-004; tasks 2.7, 2.8, 2.10)

- **RMM (D-14):** wire in `my_app/app.py` `compose()` **before any operator/fragment is constructed** (i.e. before anything pulls in heavy holoscan C++ paths in a fresh process — but the hard requirement is `import rmm` before `import holoscan`, Pitfall 1). Sequence verified working today:
  ```python
  import rmm
  rmm.reinitialize(pool_allocator=True, managed_memory=False)
  from rmm.allocators.torch import rmm_torch_allocator
  import torch
  torch.cuda.memory.change_current_allocator(rmm_torch_allocator)   # backend -> "pluggable"
  ```
  For CuPy (its own allocations for the ported ops), optionally `cupy.cuda.set_allocator(mr.allocate)` with `mr = rmm.mr.PoolMemoryResource(upstream)` — `PoolMemoryResource` exists in 26.02 (`PoolAllocator` does not). Pre-allocation in `setup()`: allocate a warm tensor sized by the budget calculator (Pattern 7), `del` it, so the pool holds the memory before study 1's tiles start (acceptance: "no per-tile cudaMalloc churn" — verify with `nsys stats --report cuda_api_sum` showing cudaMalloc/cudaFree only in the warmup span).
- **CudaStreamPool (D-16, best-effort):** verified constructor in 4.2:
  `CudaStreamPool(fragment, dev_id=0, stream_flags=0, stream_priority=0, reserved_size=1, max_size=0, cuda_green_context=None, nvtx_identifier='nvtx_stream_pool', name='cuda_stream_pool')` from `holoscan.resources`. Attach one pool per fragment with `stream_flags=1` (NonBlocking), `reserved_size=1` (or 2), a per-fragment `nvtx_identifier` (e.g. `"streams_3d_lowres"`) so the nsys trace shows pool identity. Overlap is a plus, not a gate (D-16) — record an honest note either way.
- **Memory budget calculator (INFR-03, D-15):** pure-Python function `compute_memory_budget(volumes, cfgs, free_vram) -> BudgetPlan` that sums: per-config preprocessed volume (C×H×D×W×4), logits (heads×cropped×4), probabilities (heads×original×4), one model+5-fold weights per config (measure from `ModelBundle` at setup), + safety factor; returns `strategy = "full_volume" | "defer_to_incremental"`. Uses `torch.cuda.mem_get_info()`. **Unit-tested with synthetic sizes** (agent picks, e.g. a 512×512×600 4-channel volume set) that force the defer branch; the real OOM path is documented as unexercised (D-15). The planner should make the defer branch *reachable in code* (ensemble falls back to one-config-at-a-time averaging) even though the airway study never triggers it.

### Pattern 4: Cascade wiring (PIPE-04, task 2.5) — verified reference chain

Reference (vendored nnunetv2 2.8.1, all paths checked):
1. **lowres seg source:** `export_prediction_from_logits` → `convert_predicted_logits_to_segmentation_with_correct_shape(..., return_probabilities=False)`: resample **logits** to `shape_after_cropping_and_before_resampling` (`resampling_fn_probabilities`, order=1, is_seg=False) → `convert_logits_to_segmentation` (plain **argmax**; softmax is skipped in non-region training and is argmax-equivalent anyway) → insert crop → transpose-back → `write_seg(.nii.gz)`. **No `apply_postprocessing` and no KeepLargestCC on this file** — the reference app's `KeepLargestConnectedComponentd` (post_transforms) runs *after* `forward()` on the returned tensor, i.e. it does NOT reach the tmp `.nii.gz` the cascade reads (`nnunet_seg_operator.py:1145-1160` reads `tmp_dir/3d_lowres/<stem>.nii.gz`). ⇒ **the cascade input is the argmax seg WITHOUT connected-component cleanup.** (The standalone lowres SEG gate, by contrast, does include CC in both apps.)
2. **cascade preprocess** (`data_iterators.PreprocessAdapterFromNpy.generate_train_batch` + `default_preprocessor.run_case_npy`): `run_case_npy(image, seg, props)` transposes the seg with the **same** `transpose_forward`, crops seg with the **image-derived** bbox (`crop_to_nonzero(data, seg)`), then `seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)` — for this bundle's cascade config that is **`resample_data_or_seg_to_shape(is_seg=True, order=1, order_z=0, force_separate_z=None)`** (verified via `PlansManager`) — a **CPU scipy/skimage** resample, i.e. it joins the PREP-03 CPU path. Then `seg.astype(int8)`; back in the iterator: `seg_onehot = convert_labelmap_to_one_hot(seg[0], foreground_labels=[1], data.dtype=float32)` → shape `(1, *spatial)` → `data = np.vstack((data, seg_onehot))` → **2-channel** input (image channel first, one-hot last). The one-hot channels are **never normalized** (vstack happens after `_normalize`).
3. The one-hot has exactly `len(foreground_labels) = 1` extra channel (no background channel); `determine_num_input_channels` → 2 (verified).

**Implementation (zero disk I/O, D-09/D-10):**
- Lowres fragment emits an extra port `lowres_seg` = `argmax(probabilities, dim=0)` as uint8 CUDA tensor (original DICOM orientation — exactly the PostResample output orientation). This is *post-softmax argmax* = reference argmax-of-resampled-logits (softmax is monotone per voxel ⇒ identical argmax; and the lowres PostResample already resamples **logits before** softmax, same order as the reference).
- Cascade `PreprocessOperator` (config-generic: extra optional input port `lowres_seg` declared only when the config has a `previous_stage`) receives `image` + `lowres_seg`:
  1. compute the same mask/bbox from the image (identical image ⇒ identical bbox; alternatively also flow the fullres fragment's `preprocessed_meta` — either is fine, but the image-derived bbox is self-contained and keeps the fragment config-generic);
  2. transpose + slice the seg (GPU, exact 0/1 values — layout ops only, bit-exact);
  3. transfer seg to CPU (uint8, ~16 MB) and run the **existing resample primitive** with the cascade's seg kwargs (order=1, is_seg=True ⇒ anti-aliasing off, separate-z auto) — a small extension of `_resample_to_shape`/`resample_probabilities_to_shape` to accept `is_seg`/order params (they already take `order`, `order_z`, `force_separate_z`);
  4. one-hot on GPU: `(seg == 1).astype(cp.float32)` (bit-exact, verified) and concatenate with the (GPU, C-contiguous) preprocessed image volume on the channel axis ⇒ `(2, *spatial)` fp32 → the existing SlideWindow chain (the `SlideWindowOperator` is already config-driven: `num_input_channels=2` flows through `load_inference_params`).
- Zero-copy: hand the DLPack-backed `holoscan.core.Tensor` objects between fragments as-is (`torch.utils.dlpack.from_dlpack` at the consumer, as Phase 1 does everywhere). Never build *two* holoscan Tensors from the same torch buffer (Plan 03 DLPack-ownership lesson: `cp.from_dlpack` consumes; clone when a buffer is re-emitted after mutation).

### Pattern 5: Per-config reference oracles (D-05/D-06/D-07, task 2.6)

`reference_fullres_run.py` is **already parameterized** (`--config`, `--output`) — it monkey-pins `model_list=[config]` before `compose()`. Per-config oracles need **no new harness code**:
```bash
cd examples/apps/cchmc_nnunet_fifteen_ckpt_app   # my_app name-collision rule
ulimit -s unlimited
/tmp/monai-env/.venv/bin/python ../../../.planning/scripts/reference_fullres_run.py --config 3d_lowres --output testdata/ref_lowres_only
/tmp/monai-env/.venv/bin/python ../../../.planning/scripts/reference_fullres_run.py --config 3d_cascade_fullres --output testdata/ref_cascade_only
```
- `--config 3d_cascade_fullres` is cascade-correct by construction: the reference operator's own list logic (lines 92-95) auto-inserts `3d_lowres` before it and excludes lowres from the ensemble (verified).
- Fast-app side of each gate = the same app run with the matching model list (Pattern 1's env var/flag): lowres-only ⇒ SEG of `ensemble=[3d_lowres]` (includes CC, matching the reference SEG post_transforms); cascade-only ⇒ SEG of `ensemble=[3d_cascade_fullres]`; default ⇒ bundle SEG vs `testdata/current_output`.
- `pixel_diff.py` already accepts MAP output dirs and has `--exact`; no tool changes required beyond possibly a wrapper script that runs all four gates and writes a combined JSON. Phase 1 gate tolerances apply (D-08: segmentation-level identity; expect the documented ≤3-voxel fp16↔fp32 argmax boundary class, not zero).
- Reference runs are ~124 s each; budget one run per oracle + one bundle re-run if `current_output` is stale.

### Pattern 6: Ensemble (INF-009, task 2.9, D-19)

The Phase 1 `average_probabilities` is already: in-memory, in-GPU-memory, in-place `+=` accumulation in reference order, single exact final division via CuPy (`_divide_refparity` — torch CUDA `/= n` is 1-ulp off for non-power-of-2 n). The only deviation from INF-009's literal "incremental in-place averaging" is the **single final division** rather than divide-each-step; a true running mean `(acc*(k-1)+x)/k` would *not* be bit-identical to the reference's `sum/n` and is forbidden by D-19/D-08. **Recommendation (locks D-19's contingency):** keep the Phase 1 approach; document the deviation as "in-place incremental accumulation with exact final division" (satisfies the VRAM intent — 1 accumulator + 1 streamed input, no N-copy stack — and the bit-exactness mandate). In the multi-fragment DAG the ensemble operator stays app-level; feed it the list of per-config probability tensors in `ensemble_model_list` order (order matters: first volume is the base).

### Pattern 7: Benchmarking & profiling (tasks 2.12–2.14, TEST-006/007/INFR-005)

- **E2E benchmark:** extend the `baseline_benchmark.py` subprocess pattern (fresh process per rep, warmups, N reps) to the fast app. The fast app already emits per-operator `timing: {operator, study, start, end, duration_ms}` JSON logs — parse those (plus the per-study `study_timing_summary`) instead of the reference's text markers; keep the same CSV spine `study,rep,warmup,total_ms,setup_ms,inference_ms,postprocess_ms,write_ms,ok` and add per-config columns (`inference_ms_3d_fullres`, `inference_ms_3d_lowres`, `inference_ms_3d_cascade_fullres`, `preprocess_ms_<cfg>`, …) plus a speedup-ratio row/column vs 169.7 s and vs 61.8 s. Output → `.planning/benchmarks/phase2_results.csv` (roadmap acceptance). Report mean ± std over ≥3 measured reps (Phase 1 convention), warmup excluded.
- **nsys:** reuse `.planning/scripts/nsight_profile.sh` (works with nsys 2025.6.3; `--trace=cuda,nvtx,osrt,cublas,cudnn --capture-range=cudaProfilerApi`). Add unique per-operator NVTX range names including the config (`preprocess_3d_lowres`, `inference_3d_cascade_fullres`, …) so multi-fragment traces are legible. Save traces + `nsys stats` exports to `.planning/profiles/phase2/` (task 2.14).
- **cudaMalloc churn check (acceptance criterion):** `nsys stats --report cuda_api_sum <rep>` — with RMM pool active, `cudaMalloc`/`cudaFree` should appear only in the setup/warmup span, not per-tile in study compute.
- **ncu:** **blocked** (`ERR_NVGPUCTRPERM`, verified). Do not plan ncu tasks as gate items; if kernel-level detail is wanted, document the admin requirement. nsys kernel timeline + NVTX is sufficient for "identify remaining CPU-bound regions" (task 2.13).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GPU one-hot / element-wise normalize | custom CUDA kernels | CuPy element-wise ops | verified bit-exact vs numpy; zero new risk (D-11 final-gate-only still de-risked this way) |
| Previous-stage seg resampling | GPU/CuPy resampler | the existing scipy/skimage `resample_data_or_seg_to_shape` replica with `is_seg=True, order=1, order_z=0` | PREP-03/D-13 lock; the reference does it on CPU; GPU resampling is v2 (GPUP-01) |
| Z-score mean/std | CuPy/torch reductions | the existing numpy reductions | reduction order ≠ numpy ⇒ not bit-exact (verified) |
| Ensemble final division | torch `/= n` | CuPy in-place division (existing `_divide_refparity`) | torch CUDA scalar division 1-ulp off (Phase 1 measurement) |
| RMM↔torch allocator bridge | custom allocator glue | `rmm.allocators.torch.rmm_torch_allocator` + `change_current_allocator` | verified working, backend `"pluggable"` |
| Connected components | any new CC code | existing deterministic CuPy two-pass CC (Phase 1) | already voxel-identical to MONAI keep-largest + acvl |
| Fragment scheduling/deadlock fixes | custom synchronization | GXF `CountCondition` entry conditions + conditional port declaration | scheduler guarantees; custom sync would break config-genericity (D-02) |
| DLPack handoffs | `numpy()`/byte copies between fragments | `holoscan.core.Tensor` + `torch.utils.dlpack.from_dlpack` (existing `to_holoscan_gpu_tensor`) | zero-copy contract (PREP-05/PIPE-04) |

## Common Pitfalls

1. **`import rmm` after `import holoscan` → ImportError `undefined symbol: __cxa_call_terminate`** (reproduced this session; the Phase 0 smoke test passed only because it never imported holoscan). Mitigation: `import rmm` at the very top of `app.py` (or a `gpu_bootstrap` module imported first), *before* any `monai.deploy`/`holoscan` import; add an import-order self-test. This is the single most likely first-day blocker for INFR-01.
2. **`rmm.mr.PoolAllocator` does not exist in rmm 26.x** — the old RAPIDS Python API is gone; use `rmm.mr.PoolMemoryResource` / `rmm.reinitialize(pool_allocator=True)`. Trust only what's in the venv, not training-era RAPIDS docs.
3. **Contiguity slip ⇒ 1-ulp divergence at ~16M voxels** (Phase 1 gate-caught bug). Every array that crosses a device boundary or enters the resample must be C-contiguous fp32; assert it. CuPy `transpose` returns a *view* — materialize with `ascontiguousarray` before `.get()`/DLPack.
4. **GPU reductions are not bit-exact vs numpy** (reduction order). `cupy.mean/std` measurably differed from numpy this session. Keep all reductions (Z-score stats, and any future mask stats) on the numpy reference path.
5. **Cascade seg semantics**: feeding the *CC-postprocessed* lowres seg, the *probabilities*, or an *unresampled* seg into the cascade breaks pixel-exactness. The reference cascade input is argmax-of-resampled-logits, **no CC**, resampled *again* by `resampling_fn_seg` inside cascade preprocess (Pitfall of "it should already be at the right spacing" — it is at *original* spacing and gets resampled to cascade spacing inside the cascade fragment).
6. **RMM vs `expandable_segments`**: they are alternative torch allocation strategies; don't silently enable both "just in case". Test RMM primary; if any instability appears, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is the documented fallback (roadmap risk table) — record which one shipped.
7. **GXF port discipline**: a declared input with no flow ⇒ the entity never runs (silent-looking hang); a declared output with no receiver ⇒ GXF rejects the entity at build (Phase 1). In a 3-fragment DAG with conditional ports, enumerate declared-vs-flowed ports in a unit test before running the full pipeline.
8. **`my_app` package-name collision**: the fast app's editable install maps package `my_app` to the fast app; running `python -m my_app` from the wrong cwd silently executes the wrong app (Phase 0 hazard, still live). Reference runs: `cd examples/apps/cchmc_nnunet_fifteen_ckpt_app` first. Also `ulimit -s unlimited` for both.
9. **Timing/NVTX keying in multi-fragment mode**: `StudyTimingCollector`/`set_study_id` are keyed by `operator.fragment`, which in Phase 1 == the app. With real sub-Fragments the per-study aggregate silently fragments (or cross-fragments). Key by the top-level app and include `config` in records + range names before the first benchmark, or `phase2_results.csv` per-operator columns can't be parsed.
10. **torch 2.13 autocast corruption** (Phase 1): per-fold autocast boundary with `load_state_dict` outside it — do not "clean up" when refactoring `predict_logits` for reuse across fragments; the corruption reproduces deterministically (13.2 diff → 4.54e-01 after fix).
11. **DLPack ownership transfer**: `cp.from_dlpack` consumes the caller's torch buffer (Phase 1 zombie-handle incident); clone before re-emitting a buffer that will be mutated; the multi-consumer pattern (one holoscan tensor read by 2+ downstream ops) is what Phase 1 proved and should be preserved.
12. **Latency-bar scope** (see Open Question 1): plan-level expectation setting — a serial 3-config bundle is ~70–80 s by construction (inference alone ≈ 27 s fullres + ~12 s lowres + ~28 s cascade at Phase 1 per-fold rates). Positive improvement vs 61.8 s is only achievable with cross-fragment overlap of the CPU-bound spans (resample/CC/writers) and/or an apples-to-apples same-scope comparison.

## Code Examples

### Verified bit-exact CuPy element-wise ops (measured 2026-08-19, fp32, 1×120×140×100)

```python
import numpy as np, cupy as cp
eps = 1e-8
mean, std = x.mean().astype(np.float32), x.std().astype(np.float32)  # CPU numpy — reference
out_gpu = (cp.asarray(x) - cp.asarray(mean)) / (float(max(std, eps)))   # == numpy: bitwise True
ct = cp.clip((cp.asarray(x) - 123.5) / 110.25, 0, 1)                     # == np.clip path: True
oh = (cp.asarray(seg) == 1).astype(np.float32)                           # == reference one-hot: True
t  = cp.ascontiguousarray(cp.asarray(x).transpose(0, 3, 2, 1))           # == np.ascontiguousarray(x.T...): True
# counter-example:
cp.mean(x_gpu) == mean   # False — reduction order differs; keep reductions on CPU
```

### RMM wiring that works in this venv (re-verified 2026-08-19)

```python
# MUST be before any holoscan/monai.deploy import
import rmm
rmm.reinitialize(pool_allocator=True, managed_memory=False)
from rmm.allocators.torch import rmm_torch_allocator
import torch
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
torch.zeros(1024, device="cuda")
torch.cuda.memory.get_allocator_backend()   # -> "pluggable"
# CuPy pool (optional): rmm.mr.PoolMemoryResource exists in 26.02 (PoolAllocator does NOT)
```

### Holoscan 4.2 fragment API (verified pybind signatures/docs)

```python
from monai.deploy.core import Fragment
frag = Fragment(app, name="nnunet_3d_lowres")
op   = PreprocessOperator(frag, model_path=mp, config_name="3d_lowres", name=...)
app.add_fragment(frag)
app.add_flow(series_to_vol_op, entry, {("image", "image")})           # app op -> fragment op
app.add_flow(frag_lowres, frag_cascade, {("lowres_seg", "lowres_seg")})  # fragment -> fragment
# resources:
from holoscan.resources import CudaStreamPool
CudaStreamPool(frag, dev_id=0, stream_flags=1, reserved_size=1,
               nvtx_identifier="streams_3d_lowres", name="cuda_stream_pool")
```

### Cascade reference chain (vendored nnunetv2 2.8.1, verified)

```
# lowres side (fast app):  logits -> resample(order 1, is_seg False, CPU) -> softmax -> argmax -> uint8 seg (original orientation, NO CC)
# cascade side (fast app), inside cascade PreprocessOperator:
seg_t   = seg.transpose([0, *tf])                 # same transpose_forward
seg_c   = seg_t[(slice(None),) + bbox_slicer]     # image-derived bbox
seg_r   = resample_ref(seg_c, new_shape, orig_spacing, cascade_spacing,
                       order=1, order_z=0, force_separate_z=None, is_seg=True)  # CPU scipy
oh      = (cp.asarray(seg_r) == 1).astype(cp.float32)      # (1, *spatial) — bit-exact
cascade_input = cp.concatenate([preprocessed_image, oh], axis=0)   # (2, *spatial), channel order: image, one-hot
# matches: data_iterators.generate_train_batch + default_preprocessor.run_case_npy
#   (resampling_fn_seg = resample_data_or_seg_to_shape(is_seg=True, order=1, order_z=0, force_separate_z=None);
#    convert_labelmap_to_one_hot(seg, [1], float32); np.vstack((data, seg_onehot)))
```

## Runtime State Inventory

Not applicable — this phase adds capabilities (no rename/refactor/migration of identifiers, services, or stored data). The only "runtime state" changes are: (1) `gpu_residency.py` `ALLOWED_TRANSFER_FILES` gains `preprocess_operator.py` with a documented D-13 reason (a deliberate allow-list edit, not a silencing), and (2) `testdata/` gains two new oracle dirs (`ref_lowres_only`, `ref_cascade_only`). No stored data, OS registrations, secrets, or build artifacts embed renamed strings.

## State of the Art / Deprecated in This Stack

| Old | Current (in this venv) | Impact |
|-----|------------------------|--------|
| `rmm.mr.PoolAllocator` (RAPIDS ≤24.x Python API) | `rmm.reinitialize(pool_allocator=True)` + `rmm.mr.PoolMemoryResource` / `rmm.allocators.torch` | any training-era RAPIDS snippet will not import — use the verified Pattern 6/Code Examples sequence |
| `MemoryData(DeviceType::GPU)` (SDK docs) | `holoscan.core.Tensor` via DLPack (Phase 1 decision, already in code) | keep `to_holoscan_gpu_tensor`; do not "fix" toward MemoryData — it does not exist in 4.2 Python |
| nsys `--trace=cub` | removed in nsys ≥2024 (harness already fixed 2026-08-17) | keep current flag set |
| MONAI `sliding_window_inference` as-is | nnUNet pure utilities + MONAI-style loop (Phase 1, measured divergence) | do not revisit for Phase 2 |
| torch CUDA `/= n` for ensemble | CuPy in-place division (Phase 1 measurement) | keep `_divide_refparity` |

## Open Questions

1. **D-18 scope tension — 61.8 s bar vs 3-config bundle (planner must decide and document).** Phase 1's 61.8 s was *single-config fullres*. The Phase 2 bundle does ~2× the inference work; a serial bundle is estimated 70–80 s E2E, so "any positive improvement vs 61.8 s" may be unattainable for the bundle even with perfect overlap (inference alone ≈ 65–70 s at Phase 1 per-fold rates). *What we know:* per-operator deltas vs Phase 1 and vs 169.7 s are explicitly required by D-18. *What's unclear:* whether the user's intent is bundle-vs-61.8 or same-scope (fullres-only) vs 61.8. *Recommendation:* benchmark three runs — (a) fullres-only fast app vs Phase 1's 61.8 s (apples-to-apples, should show the CuPy/preprocess win), (b) bundle fast app vs 169.7 s reference bundle (the headline speedup, ≈2.3–2.5× expected), (c) bundle fast app reported against 61.8 s with the scope difference documented; spend parallelism effort (CudaStreamPool, fragment concurrency) only on (b)/(c) if cheap, and report honestly if (c) is negative.
2. **Config-selection mechanism for per-config gate runs.** Env var vs `app.yaml` vs module constant is agent's discretion. Recommendation: env var (`HOLOSCAN_MODEL_LIST`) read in `compose()`, defaulting to the reference default list — simplest, matches the reference `.env` pattern, zero new CLI parsing.
3. **Where the cascade fragment gets the seg resample kwargs.** `PreprocessParams` currently carries only the *data* resample kwargs. The cascade needs the *seg* resample kwargs (order=1, is_seg=True, order_z=0, force_separate_z=None) — extend `load_preprocess_params` to also expose `resampling_fn_seg` parameters from the resolved `ConfigurationManager` (verified available) so nothing is hard-coded (INF-006/D-02).
4. **Lowres fragment's `probabilities` port consumers.** In the default bundle, lowres probabilities are not ensembled (D-07) but the argmax seg must reach the cascade. The fragment must conditionally emit `lowres_seg` (needed by cascade) vs `probabilities` (needed only by the standalone lowres SEG path). A small `emit_*` flag pair (existing operator pattern) resolves it — confirm no declared-port-left-unwired case in any of the 3 model-list configurations (fullres-only, lowres-only, cascade-only, bundle).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| nsys | tasks 2.13/2.14, acceptance (cudaMalloc churn, stream overlap) | ✓ (worked Phases 0/1) | 2025.6.3 | — |
| ncu | task 2.13 (kernel-level) | ✗ **blocked by permissions** (`ERR_NVGPUCTRPERM`, verified) | installed 2026.1.0 | nsys kernel timeline + NVTX only; document admin requirement |
| rmm-cu13 | INFR-01 (D-14) | ✓ **with import-order constraint** (must precede holoscan) | 26.02.00 | torch `expandable_segments` fallback (document which shipped) |
| cupy-cuda13x | CuPy port, one-hot, CC, division | ✓ | 14.1.1 | — |
| holoscan-cu13 Fragment API + CudaStreamPool | PIPE-03/04, INFR-004 | ✓ (signatures verified) | 4.2.0 | — |
| torch (CUDA 13) | inference | ✓ | 2.13.0+cu130 | — |
| nvidia-smi / A100 40 GB | memory budget | ✓ | driver 610.57.04 | — |

**Missing dependencies with no fallback:** none blocking.
**Missing dependencies with fallback:** ncu (nsys-only profiling).

## Sources

### Primary (HIGH confidence — direct verification this session)
- Live venv probes: torch/cupy/rmm/holoscan versions; rmm import-order failure reproduction + workaround; `rmm.mr` API surface; `CudaStreamPool` constructor doc; `Application.add_fragment` / `add_flow(fragment, fragment, ...)` pybind docs; ncu `ERR_NVGPUCTRPERM` probe; CuPy-vs-numpy bit-exactness measurements (element-wise ops True, reductions False).
- Vendored nnunetv2 2.8.1 (`./nnUNet/`): `inference/data_iterators.py` (`PreprocessAdapterFromNpy.generate_train_batch` one-hot+vstack order), `preprocessing/preprocessors/default_preprocessor.py` (`run_case_npy` seg transpose/crop/resample), `inference/export_prediction.py` (resample→argmax→no CC on exported seg), `utilities/label_handling/label_handling.py` (`determine_num_input_channels`, `convert_labelmap_to_one_hot`), `utilities/plans_handling/plans_handler.py` + bundle `plans.json` (resolved cascade config: patch/spacing/2 channels/seg-resample kwargs).
- Reference app `cchmc_nnunet_fifteen_ckpt_app/my_app/`: `nnunet_seg_operator.py:91-96,1145-1160` (model-list logic, cascade tmp-file read), `nnunet_bundle.py` (forward/save path, `PostProcessNNUnet`).
- Existing Phase 1 code: `app.py`, `preprocess_operator.py`, `postresample_operator.py`, `slidewindow_operator.py`, `ensemble_average_operator.py`, `config/__init__.py`, `scripts/pixel_diff.py`, `scripts/gpu_residency.py`, `scripts/nsight_profile.sh`, `.planning/scripts/{reference_fullres_run.py,REFERENCE_RUN_GUIDE.md,baseline_benchmark.py,test_rmm.py}`.
- `.planning/{ROADMAP.md §Phase 2, REQUIREMENTS.md, STATE.md, PROJECT.md}`; baseline CSVs.

### Secondary / Tertiary
- None — no external (web) sources were needed; all claims verified in-repo or in-venv.

## Metadata

**Confidence breakdown:**
- Standard Stack / environment: **HIGH** — every version and API signature verified live in `/tmp/monai-env/.venv`.
- Architecture (multi-fragment DAG, cascade, CuPy port): **HIGH** — API existence and reference math verified from source; GXF multi-fragment *runtime* behavior (entry conditions, concurrent fragments) not yet exercised in this codebase → first wiring smoke is where residual risk lives.
- Pitfalls: **HIGH** for the reproduced ones (rmm import order, ncu permissions, contiguity, reductions); **MEDIUM** for the estimate-dependent ones (bundle E2E ~70–80 s serial — derived from Phase 1 per-fold timings, not measured).
- Performance outcomes: **MEDIUM** — depends on the as-yet-unbuilt parallelism; no Phase 2 numbers exist yet.

**Research date:** 2026-08-19
**Valid until:** ~30 days (stable environment; re-verify rmm/holoscan if the venv is rebuilt — /tmp scratch venv)
