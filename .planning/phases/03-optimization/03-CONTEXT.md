# Phase 3: Optimization - Context

**Gathered:** 2026-08-19
**Status:** Ready for planning
**Mode:** auto (yolo auto-advance chain; all choices are recommended options, logged in 03-DISCUSSION-LOG.md for user review)

<domain>
## Phase Boundary

Data-driven performance optimization of the Phase 2 multi-fragment pipeline.
Only optimizations **validated by Phase 2 profiling data** (the ranked,
trace-cited bottleneck list in `02-BENCHMARK-REPORT.md` §6) are in scope, and
**pixel-exact equivalence is maintained at all times** — every change is gated
by the Phase 2 pixel-exact suite (4 config gates + bundle vs
`testdata/current_output`) before it ships. This phase delivers measured
improvement over Phase 2 (bundle 129.54 s / fullres-only 57.14 s) on the same
hardware (A100-SXM4-40GB) and corpus.

In scope:
- INFR-02 cross-study buffer reuse (deferred here by D-17)
- Serial fragment scheduling / stream overlap (bottleneck #2, ~25 s)
- GPU resampling as a **gated experiment** (bottleneck #1, ~28.8 s; ROADMAP 3.4)
- Memory lifecycle: MEM-003 (free lowres weights after inference)
- Final benchmarks + full pixel-exact validation suite (ROADMAP 3.9/3.10)

Out of scope (deferred, with reasons — see <deferred>):
- TensorRT (ACCEL-01/02), torch.compile (ACCEL-03), MEM-01 weight sharing,
  MEM-02 8 GB VRAM target, inference-kernel tuning (ncu admin-blocked)
</domain>

<decisions>
## Implementation Decisions

### Scope sizing
- **D-20:** Phase 3 is a **trimmed, data-driven sweep** — not skipped, not full. Phase 2 met both D-18 bars (1.082× same-scope, 1.310× bundle), so per ROADMAP the phase trims to: the two low-correctness-risk measurable levers (fragment scheduling overlap, memory lifecycle) + INFR-02 + a gated GPU-resampling experiment + final benchmark/validation. High-risk items (TensorRT, torch.compile, weight sharing) are deferred — their expected wins (inference-kernel level) cannot be validated without ncu access anyway.

### Bottleneck strategy
- **D-21:** Serial fragment scheduling (bottleneck #2, up to ~25 s): implement **concurrent independent-fragment execution** (e.g. fullres ∥ lowres) as a DAG/executor-level change using the existing per-config `CudaStreamPool`s — NOT a math change. Hard gates: all 4 pixel-exact gates + residency must pass with overlap enabled; if holoscan-cu13 4.2's app_driver cannot express the needed concurrency, document the measured ceiling and ship the fallback (serial) with a trace citation.
- **D-22:** GPU resampling (bottleneck #1, ~28.8 s; D-13 locked it to CPU in Phase 1–2 with GPU resampling explicitly the v2 item — Phase 3 is that v2 slot, ROADMAP 3.4): implement a **gated experiment**, not a commitment. A CuPy (or monai.data) resampler for the image path and the seg path behind a config flag (default OFF = scipy). Gate = per-tensor `np.array_equal` vs the scipy reference on the dev corpus across all 4 configs. If byte-identity is not achieved, the CPU path stays default, the experiment + measured divergence is documented, and the phase still ships. The pixel-exact gates must pass with the flag both off and on (on only if byte-identical).

### Memory lifecycle
- **D-23:** Implement **MEM-003 only**: free lowres network weights (and intermediate lowres buffers) immediately after 3d_lowres inference in cascade configurations; measure peak-VRAM delta (nvidia-smi / torch memory snapshot around the bundle run). MEM-01 (shared weights across the 15-ckpt ensemble) is **evaluation-note only** — architectural, no profiling evidence it is the bottleneck (models load once, Phase 2 Plan 02 already eliminated cold-start). MEM-02 (8 GB VRAM target) is **hardware-unverifiable** on the A100-40GB and deferred (see <deferred>).

### INFR-02 (D-17 deferred here)
- **D-24:** Cross-study buffer reuse: operators pre-allocate/reuse GPU buffers across `compute()` calls keyed on shape (dict-cached allocations), so the Nth study reuses the 1st study's buffers. Proof strategy: (a) unit tests with synthetic sizes (multi-call reuse, shape-key invalidation, dtype/contiguity invariants); (b) multi-study replay of the dev corpus (same study 3× sequentially — buffer addresses stable across studies, cudaMalloc count flat across studies, per the RMM-churn nsys check from Phase 2 Plan 06); (c) the **user is adding additional reference examples in Phase 3 to make it provable** — if they arrive before the gate runs, fold them into the verification oracle; if not, ship with (a)+(b) and record the examples as an external-dependency item (same class as the 2d model, D-01/D-03).
- **D-25:** The Phase 2 gate/verify infrastructure is the correctness anchor: `phase2_gate.py` (4 config pixel gates + SR + residency), `pixel_diff.py`, `gpu_residency.py`, `phase2_benchmark.py`. Every Phase 3 plan ships behind a re-run of the gate suite; the final plan re-runs it for the close-out report.

### Gates & external dependencies
- **D-26:** External-dependency items are **blocked-on-external, non-blocking** (mirrors the D-01/D-03 precedent): (1) ≥5-CT corpus re-run (TEST-01 final gate) — blocked on CT data; (2) ncu kernel profiling — blocked on `ERR_NVGPUCTRPERM` admin access; (3) user's INFR-02 reference examples. Phase 3 completes on the dev corpus with these three recorded as external-dependency items in VERIFICATION.md; they re-open as gap plans if the dependencies land.

### the agent's Discretion
- Concrete concurrency mechanism for D-21 (threads vs holoscan scheduling vs explicit stream sync) — choose by what holoscan-cu13 4.2 actually supports (live-probe first, as Phase 2 Plan 04 did for Subgraph).
- Exact CuPy resample algorithm choice for D-22 (map_coordinates on-GPU vs monai.data.Resample) — pick whichever is closest to the scipy reference path; the byte-identity gate is the arbiter.
- Buffer-cache granularity and LRU/eviction policy for D-24.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 2 evidence (the profiling basis for this phase)
- `.planning/phases/02-gpu-acceleration/02-BENCHMARK-REPORT.md` — §6 ranked, trace-cited bottleneck list (resample 28.8 s → serial scheduling ~25 s → inference kernels (ncu-blocked) → postprocess 9.9 s → bootstrap ~20.8 s); D-16 stream-overlap note; D-18 two-bar definitions; carried items
- `.planning/phases/02-gpu-acceleration/02-CONTEXT.md` — all locked Phase 2 decisions (D-01…D-19) that Phase 3 must not regress
- `.planning/phases/02-gpu-acceleration/02-RESEARCH.md` — patterns and pitfalls (CuPy non-bit-identical reductions, D-13 boundary semantics, holoscan 4.2 API realities)
- `.planning/profiles/phase2/` — `phase2_bundle_*.nsys-rep` + `.sqlite` + `*_nvtx_sum.txt` / `*_nvtx_kern_sum.txt` / `*_cuda_gpu_kern_sum.txt` / `ncu_status.txt` — the trace citations
- `.planning/benchmarks/phase2_results.csv` — Phase 2 per-rep/per-config numbers (the "before" for Phase 3)
- `.planning/benchmarks/baseline-2026-08-18.csv` + `.planning/baseline_results.csv` — Phase 1 fast-app and reference-app baselines

### Gate infrastructure (reused, not rebuilt)
- `.planning/scripts/phase2_gate.py` — 4-config pixel gate + SR + residency → `02-GATE-RESULTS.json`
- `examples/apps/cchmc-nnunet-fast/scripts/pixel_diff.py` — byte-identity / IoU comparator
- `examples/apps/cchmc-nnunet-fast/scripts/gpu_residency.py` — static + runtime residency gate with reason-string allow-list
- `.planning/scripts/phase2_benchmark.py` + `.planning/scripts/reference_fullres_run.py` — benchmark + reference-oracle harnesses
- `.planning/scripts/REFERENCE_RUN_GUIDE.md` — how to regenerate reference outputs

### Code under change
- `examples/apps/cchmc-nnunet-fast/my_app/app.py` — multi-fragment DAG assembly, per-config Subgraph factory, CudaStreamPool wiring
- `examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py` — `_resample_to_shape` (scipy image path) + cascade seg-resample replica
- `examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py` — scipy resample + lowres_seg emit
- `examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py` — weight loading (MEM-003 hook)
- `examples/apps/cchmc-nnunet-fast/my_app/gpu_bootstrap.py` + `mem_budget.py` — RMM pool, budget calculator
- `nnUNet/nnunetv2/` (vendored) — the reference semantics source of truth for any resample/schedule behavior

### Requirements
- `.planning/REQUIREMENTS.md` — INFR-02 (deferred here), MEM-003, GPUP-01/02, TEST-002/003 rows and traceability table
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `phase2_gate.py` / `pixel_diff.py` / `gpu_residency.py` / `phase2_benchmark.py` — the full verify+measure loop; Phase 3 re-runs them unchanged except flag/env additions
- `gpu_util.py` — app-keyed timing aggregation (Subgraph-safe via `_root`), config-tagged NVTX — reuse for overlap measurement
- `mem_budget.py` — pure-arithmetic budget; extendable for peak-VRAM accounting
- Per-config `CudaStreamPool`s (NonBlocking, reserved_size=1, `streams_<cfg>`) already wired in app.py — the streams exist; only the scheduling is serial (D-16)
- RMM pluggable allocator + pool warm-up (`gpu_bootstrap.py`) — makes cudaMalloc-churn checks meaningful across studies

### Established Patterns
- Pixel-exact final gates as the sole correctness arbiter (D-11) — no per-operator correctness gates; every Phase 3 change re-runs the 4-config suite
- Reason-string allow-list for deliberate transfers (D-13) — any new deliberate CPU step must be added there, not by weakening entries
- Reference-semantic replication against vendored nnunetv2 2.8.1 with `np.array_equal` unit tests (Phase 2 Plan 03 pattern) — the template for the GPU-resample experiment
- Live-probe holoscan 4.2 API support before planning around it (Phase 2 Plan 04 learned this the hard way — the plan's C++ Fragment API did not exist)

### Integration Points
- app.py Subgraph factory + `HOLOSCAN_MODEL_LIST` — where fragment concurrency attaches
- `_resample_to_shape` in preprocess_operator.py + the seg-resample replica in the same file + postresample_operator.py — the three scipy call sites a GPU resampler would replace behind a flag
- slidewindow_operator.py one-shot model load — where MEM-003 weight-freeing hooks
- `compose()` RMM warm-up — where per-study buffer reuse (D-24) registration happens
</code_context>

<specifics>
## Specific Ideas

No specific user requirements beyond the locked decisions above — the phase is data-driven by construction (D-20). The D-18 two-bar reporting convention carries over: Phase 3 reports per-config deltas vs Phase 2's `phase2_results.csv` with per-operator breakdown, same fresh-process + warmup-excluded methodology.
</specifics>

<deferred>
## Deferred Ideas

- **ACCEL-01/ACCEL-02 (TensorRT export):** deferred — inference kernels are 91–96% GPU-busy; the per-kernel profile needed to justify a TRT reimplementation is ncu-blocked (admin access). TensorRT also risks losing dynamic tiling logic (ROADMAP risk: High). Re-open when ncu unblocks.
- **ACCEL-03 (torch.compile):** deferred — same ncu gap; also `cudnn.benchmark` already had to be disabled under the RMM pluggable allocator (Phase 2 Plan 02) — dynamic-shape compile has a hostile environment here.
- **MEM-01 (shared weights across 15-ckpt ensemble):** deferred — architectural; profiling shows models load once (bootstrap ~20.8 s, amortized) and inference is saturated, so weight sharing is not a measured bottleneck.
- **MEM-002 / 8 GB VRAM target:** deferred — **hardware-unverifiable** on the A100-SXM4-40GB; testing it requires an 8 GB machine. Incremental strategies (defer branch) already exist and are unit-tested (Phase 2 Plan 02).
- **Inference-kernel optimization:** blocked-on-external — `ERR_NVGPUCTRPERM` (ncu admin). `phase2_bundle_cuda_gpu_kern_sum.txt` is the current ceiling.
- **≥5-CT corpus re-run (TEST-01 final):** blocked-on-external — CT data not yet supplied (carried from Phase 0/1/2).
- **INFR-02 user reference examples:** blocked-on-external — user is adding them during Phase 3 (D-24); fold in when they arrive.
- **2d model validation:** blocked-on-model (D-01/D-03) — config-generic wiring exists; a test, not a code change.

### Bootstrap caching
The ~20.8 s bootstrap (DICOM load + 3× model load + RMM warm) is a repeat-study concern only — the clinical use case is single-study latency with warm models (Phase 2 Plan 02 already eliminated per-study cold start). Note for a future phase if the usage model shifts.
</deferred>

---

*Phase: 03-optimization*
*Context gathered: 2026-08-19 (auto)*
