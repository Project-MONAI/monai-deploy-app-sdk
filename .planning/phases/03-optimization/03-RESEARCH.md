# Phase 3: Optimization — Research

**Researched:** 2026-08-19
**Domain:** Holoscan-cu13 4.2 DAG scheduling / CuPy-scipy bit-exact GPU resampling / RMM+CuPy memory lifecycle on A100-SXM4-40GB
**Confidence:** HIGH (mechanisms live-probed on the installed stack + v4.2.0 source; realized speedups MEDIUM until measured in-pipeline)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-20:** Phase 3 is a **trimmed, data-driven sweep** — not skipped, not full. Phase 2 met both D-18 bars (1.082× same-scope, 1.310× bundle), so per ROADMAP the phase trims to: the two low-correctness-risk measurable levers (fragment scheduling overlap, memory lifecycle) + INFR-02 + a gated GPU-resampling experiment + final benchmark/validation. High-risk items (TensorRT, torch.compile, weight sharing) are deferred — their expected wins (inference-kernel level) cannot be validated without ncu access anyway.
- **D-21:** Serial fragment scheduling (bottleneck #2, up to ~25 s): implement **concurrent independent-fragment execution** (e.g. fullres ∥ lowres) as a DAG/executor-level change using the existing per-config `CudaStreamPool`s — NOT a math change. Hard gates: all 4 pixel-exact gates + residency must pass with overlap enabled; if holoscan-cu13 4.2's app_driver cannot express the needed concurrency, document the measured ceiling and ship the fallback (serial) with a trace citation.
- **D-22:** GPU resampling (bottleneck #1, ~28.8 s; D-13 locked it to CPU in Phase 1–2 with GPU resampling explicitly the v2 item — Phase 3 is that v2 slot, ROADMAP 3.4): implement a **gated experiment**, not a commitment. A CuPy (or monai.data) resampler for the image path and the seg path behind a config flag (default OFF = scipy). Gate = per-tensor `np.array_equal` vs the scipy reference on the dev corpus across all 4 configs. If byte-identity is not achieved, the CPU path stays default, the experiment + measured divergence is documented, and the phase still ships. The pixel-exact gates must pass with the flag both off and on (on only if byte-identical).
- **D-23:** Implement **MEM-003 only**: free lowres network weights (and intermediate lowres buffers) immediately after 3d_lowres inference in cascade configurations; measure peak-VRAM delta (nvidia-smi / torch memory snapshot around the bundle run). MEM-01 (shared weights across the 15-ckpt ensemble) is **evaluation-note only** — architectural, no profiling evidence it is the bottleneck (models load once, Phase 2 Plan 02 already eliminated cold-start). MEM-02 (8 GB VRAM target) is **hardware-unverifiable** on the A100-40GB and deferred (see <deferred>).
- **D-24:** Cross-study buffer reuse: operators pre-allocate/reuse GPU buffers across `compute()` calls keyed on shape (dict-cached allocations), so the Nth study reuses the 1st study's buffers. Proof strategy: (a) unit tests with synthetic sizes (multi-call reuse, shape-key invalidation, dtype/contiguity invariants); (b) multi-study replay of the dev corpus (same study 3× sequentially — buffer addresses stable across studies, cudaMalloc count flat across studies, per the RMM-churn nsys check from Phase 2 Plan 06); (c) the **user is adding additional reference examples in Phase 3 to make it provable** — if they arrive before the gate runs, fold them into the verification oracle; if not, ship with (a)+(b) and record the examples as an external-dependency item (same class as the 2d model, D-01/D-03).
- **D-25:** The Phase 2 gate/verify infrastructure is the correctness anchor: `phase2_gate.py` (4 config pixel gates + SR + residency), `pixel_diff.py`, `gpu_residency.py`, `phase2_benchmark.py`. Every Phase 3 plan ships behind a re-run of the gate suite; the final plan re-runs it for the close-out report.
- **D-26:** External-dependency items are **blocked-on-external, non-blocking** (mirrors the D-01/D-03 precedent): (1) ≥5-CT corpus re-run (TEST-01 final gate) — blocked on CT data; (2) ncu kernel profiling — blocked on `ERR_NVGPUCTRPERM` admin access; (3) user's INFR-02 reference examples. Phase 3 completes on the dev corpus with these three recorded as external-dependency items in VERIFICATION.md; they re-open as gap plans if the dependencies land.

### Claude's Discretion
- Concrete concurrency mechanism for D-21 (threads vs holoscan scheduling vs explicit stream sync) — choose by what holoscan-cu13 4.2 actually supports (live-probe first, as Phase 2 Plan 04 did for Subgraph).
- Exact CuPy resample algorithm choice for D-22 (map_coordinates on-GPU vs monai.data.Resample) — pick whichever is closest to the scipy reference path; the byte-identity gate is the arbiter.
- Buffer-cache granularity and LRU/eviction policy for D-24.

### Deferred Ideas (OUT OF SCOPE)
- **ACCEL-01/ACCEL-02 (TensorRT export):** deferred — inference kernels are 91–96% GPU-busy; the per-kernel profile needed to justify a TRT reimplementation is ncu-blocked (admin access). TensorRT also risks losing dynamic tiling logic (ROADMAP risk: High). Re-open when ncu unblocks.
- **ACCEL-03 (torch.compile):** deferred — same ncu gap; also `cudnn.benchmark` already had to be disabled under the RMM pluggable allocator (Phase 2 Plan 02) — dynamic-shape compile has a hostile environment here.
- **MEM-01 (shared weights across 15-ckpt ensemble):** deferred — architectural; profiling shows models load once (bootstrap ~20.8 s, amortized) and inference is saturated, so weight sharing is not a measured bottleneck.
- **MEM-002 / 8 GB VRAM target:** deferred — **hardware-unverifiable** on the A100-SXM4-40GB; testing it requires an 8 GB machine. Incremental strategies (defer branch) already exist and are unit-tested (Phase 2 Plan 02).
- **Inference-kernel optimization:** blocked-on-external — `ERR_NVGPUCTRPERM` (ncu admin). `phase2_bundle_cuda_gpu_kern_sum.txt` is the current ceiling.
- **≥5-CT corpus re-run (TEST-01 final):** blocked-on-external — CT data not yet supplied (carried from Phase 0/1/2).
- **INFR-02 user reference examples:** blocked-on-external — user is adding them during Phase 3 (D-24); fold in when they arrive.
- **2d model validation:** blocked-on-model (D-01/D-03) — config-generic wiring exists; a test, not a code change.
- Bootstrap caching: ~20.8 s bootstrap is a repeat-study concern only — clinical use case is single-study latency with warm models. Note for a future phase if the usage model shifts.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **INFR-02** | Pre-allocate GPU buffers in `setup()`, reuse across `compute()` calls | Allocation-site inventory (§D-24 domain); RMM vs CuPy pool semantics (torch allocs go through RMM; **CuPy has its own independent pool**); shape-keyed cache pattern + proof strategy (a)+(b) per D-24 |
| **MEM-003** | Free lowres weights after 3d_lowres inference | 3d_lowres = 5 folds × 135 MB = **~675 MB fp32 weights + ~135 MB live network ≈ 0.8 GB**; release hook site identified; RMM `empty_cache`/driver-reclaim semantics flagged as open question (measurement must be pool-level AND driver-level) |
| **GPUP-01** | GPU resampler, pixel-exact | All 3 call sites reduce to ONE primitive: `scipy.ndimage.zoom(grid_mode=True)` (skimage.resize with `anti_aliasing=False`); separate-z `map_coordinates` is **inactive in this bundle**. Byte-identity verdicts: order 0 = byte-identical on CuPy (measured); order ≥ 1 = **NOT** byte-identical with stock `cupyx.scipy.ndimage.map_coordinates` (fp32 or fp64, measured). Byte-identity requires a scipy-faithful CUDA kernel (full spec in §D-22); monai.data.Resample is a non-candidate (different grid math) |
| **GPUP-02** | All preprocessing on GPU, zero CPU-GPU transfers before inference | Achieved only when the GPUP-01 flag is ON and byte-identical (removes the 2× ~64 MB D2H/H2D resample round-trips per preprocess). Residual: numpy mean/std reductions (CuPy non-bit-identical) + ~8 MB mask round-trip remain on CPU by Phase 1/2 decisions |
| **TEST-01** | Pixel-exact vs ≥5 CT corpus | Dev-corpus re-run via `phase2_gate.py` (D-25); ≥5-CT half is blocked-on-external (D-26) — record in VERIFICATION.md |
| **TEST-002** | SR within 0.1% | `phase2_gate.py` SR check (Phase 2: 0.0% delta on all 4 rows); re-run per plan |
| **TEST-003** | Automated pixel-diff tool fails on divergence | `pixel_diff.py` already exists (Phase 1); Phase 3 re-runs it as part of every gate; no new tool needed |
| **TEST-006** | Benchmark script, E2E + per-operator | `phase2_benchmark.py` re-run; Phase 3 adds the 2×2 matrix (resample flag OFF/ON × serial/concurrent) + comparison vs `phase2_results.csv` (D-18 convention) |
| **TEST-007** | Comparison vs reference app, speedup ratio | Final two-bar report vs 169.7 s reference + 61.8 s/57.14 s fast-app baselines |

**Deferred-with-reason (D-20):** ACCEL-01/02/03 (ncu-blocked + hostile compile env — see Deferred), MEM-01 (no profiling evidence; models load once), MEM-02 (needs an 8 GB machine).
</phase_requirements>

## Project Constraints (from AGENTS.md)

- All dev/test in the uv venv at `/tmp/monai-env/.venv`; activate via `source activate-env.sh`; run with `/tmp/monai-env/.venv/bin/python`.
- **Do not** create new venvs or use system Python. **Do not** install `nnunetv2` from PyPI — vendored editable copy at `./nnUNet/`.
- Commit after every major code change (short imperative messages, atomic).
- The venv is scratch (`/tmp`) — it can drift between sessions (see Open Question 1: RMM behavior). Re-verify environment facts at plan start.

## Summary

Phase 3 has three real levers, all now mechanism-verified on the installed stack:

1. **Concurrent fragments (D-21) — the mechanism exists and works.** holoscan-cu13 4.2's default `GreedyScheduler` runs operators serially on one thread (live-probed: 2×2 s branches = 5.3 s). Swapping in `EventBasedScheduler(app, worker_thread_number=N)` (or `MultiThreadScheduler`) makes independent DAG branches run on N worker threads concurrently (live-probed 2-way and 3-way: different thread IDs, overlapping spans, `CountCondition` joins fire exactly once). `PyOperator::compute` is invoked with the GIL released (v4.2.0 source: `py::call_guard<py::gil_scoped_release>`), so two Python operators genuinely progress in parallel. **However, the gain is NOT overlapping two GPU-saturated inferences** — a GPU probe shows two 2 s saturated-matmul branches on separate threads gain only ~10 % wall time (4.86 s vs 4.0 s serial; per-branch slowdown cancels the overlap). The real ~25 s win is **hiding the 22.2 s CPU-bound preprocess + 6.6 s CPU-bound postresample spans behind other fragments' GPU inference**, which concurrency delivers for free once the scheduler is multi-threaded. This is a scheduler swap (~5 lines in `app.py`), not a math change — D-21 compliant.
2. **GPU resampling (D-22) — the byte-identity gate is winnable, but only with a custom kernel.** In this bundle every resample goes through one primitive — `scipy.ndimage.zoom(grid_mode=True)` via `skimage.transform.resize(anti_aliasing=False)` (the separate-z `map_coordinates` branch never fires; spacing is near-isotropic). Measured byte-identity: `cupyx.scipy.ndimage.map_coordinates` is **byte-identical at order 0** (fp32 and fp64) but **not at order 1 or 3** (55–100 % of voxels differ; scipy computes internally in double with a fixed per-voxel accumulation order; CuPy computes in float32, and even CuPy-fp64 differs at 1-ulp-fp64 from scipy-fp64 — accumulation order). A CuPy `RawKernel` replicating scipy 1.15.3's `NI_GeometricTransform` per-voxel arithmetic in double with `--fmad=false` (verified: FMA contraction flips bit-exactness; `--fmad=false` restores it) is the only byte-identical GPU path. Full porting spec in §D-22. If the kernel fails the gate, D-22's fallback (scipy stays default, document divergence) ships — the phase is not blocked.
3. **Memory lifecycle (MEM-003, D-24) — scoped and cheap, with one measurement caveat.** 3d_lowres weights are ~0.8 GB; freeing them after the lowres fragment's `PostResample` emits `lowres_seg` is a small hook. INFR-02 shape-keyed buffer caching has an identified allocation inventory (~2–3 GB of per-study torch-side big blocks covered by the RMM pool; ~0.5 GB CuPy-side blocks in CuPy's *separate* pool). **Caveat:** `torch.cuda.memory_stats()`/`memory_allocated()` are **unsupported under the RMM pluggable allocator** (measured: `RuntimeError: CUDAPluggableAllocator does not yet support getDeviceStats`) — VRAM measurement must use nvidia-smi/pynvml sampling + RMM pool stats, not the torch counters.

**Primary recommendation:** Ship in dependency order — (1) scheduler-swap concurrency (biggest, lowest-risk, measurable immediately), (2) MEM-003 release hook, (3) INFR-02 buffer cache, (4) GPU-resample gated experiment (the only item with real byte-exactness risk — budget one full plan for the kernel + identity tests), (5) final 2×2 benchmark + gate close-out. Expect the bundle to land near 100–110 s E2E from concurrency alone (in-study 108.7 s → ~75–85 s if CPU spans hide well), with the GPU-resample flag ON shaving another 20+ s **only if** byte-identity passes.

## Standard Stack

No new packages. Phase 3 uses only what is installed (verified 2026-08-19):

### Core (already installed — do not upgrade mid-phase)
| Library | Version | Purpose | Notes |
|---|---|---|---|
| holoscan-cu13 | 4.2.0 | DAG runtime, schedulers, resources | `EventBasedScheduler`/`MultiThreadScheduler` in `holoscan.schedulers`; `make_thread_pool` on Application |
| torch | 2.13.0+cu130 | inference, DLPack | `torch.cuda.ExternalStream` exists (verified) |
| cupy-cuda13x | 14.1.1 | GPU numpy, `RawKernel` | `cupyx.scipy.ndimage.map_coordinates` exists but is **not** byte-identical at order ≥ 1; `RawKernel(options=...)` takes a **tuple** |
| scipy | 1.15.3 | reference resample semantics | C source `scipy/ndimage/src/ni_interpolation.c` (`NI_GeometricTransform`) — the porting reference |
| scikit-image | 0.25.2 | `resize` wrapper | `resize(anti_aliasing=False)` == `scipy.ndimage.zoom(grid_mode=True)` exactly (no gaussian) |
| rmm-cu13 / librmm-cu13 | 26.2.0 | pluggable torch allocator | `rmm.reinitialize(pool_allocator=True)` — **default initial pool = ½ total GPU memory (20 GiB on A100-40GB)** (see Open Q1) |
| monai | 1.3.0 | deploy core (thin wrapper over holoscan) | monai.deploy.core classes are holoscan classes |
| numpy | 2.2.6 | reference reductions | CuPy reductions not bit-identical (Phase 2 Pitfall 4) |

**Installation:** none required.

### Supporting
| Tool | Purpose |
|---|---|
| nsys (Nsight Systems 2026.1.0) | cudaMalloc-churn checks, stream/overlap traces (Phase 2 harness reusable) |
| nvidia-smi / pynvml | driver-level VRAM sampling for MEM-003 (torch memory_stats unavailable under RMM) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|---|---|---|
| `EventBasedScheduler` | `MultiThreadScheduler` | polling-style scheduler (recession-period checks) adds CPU contention against the CPU-bound scipy spans; EventBased is what NVIDIA's own 4.2 examples use — preferred |
| CuPy `RawKernel` scipy-clone (fp64) | monai.data.Resample / MONAI `Spacing` transform | MONAI resamples via affine grids / torch `grid_sample` — different coordinate math, **zero** byte-identity chance; rejected for the gate |
| stock `cupyx.scipy.ndimage.map_coordinates` | — | byte-identical at order 0 only (measured); fine for the unused separate-z order_z=0 path, wrong for orders 1/3 |

## D-21 Domain: Concurrent Fragments on holoscan-cu13 4.2

### What the 4.2 app_driver actually supports (verified: v4.2.0 source + live probes)

- **Default scheduler = `GreedyScheduler`** (single-threaded, serial). `Application.scheduler()` docstring: "If unspecified, the default is a `holoscan.gxf.GreedyScheduler`". Live probe: two independent 2 s branches = 5.30 s wall, same thread ID for both, back-to-back spans.
- **`EventBasedScheduler(fragment, clock=None, worker_thread_number=N, ...)`** — "Event-based multi-thread scheduler". `worker_thread_number` "creates a default thread pool. Operators not explicitly assigned to a user-defined thread pool (via `make_thread_pool`) will use this default pool." Live probe: 2 branches, `worker_thread_number=2` → 3.40 s wall, **different thread IDs, identical 2.00 s spans** (true overlap; ~1.4 s fixed GXF start/teardown). 3 branches, `worker_thread_number=3` → 2.13 s vs 4.5 s serial, 3 distinct thread IDs, `CountCondition(3)` join fired **exactly once**.
- **`MultiThreadScheduler(fragment, worker_thread_number=N, ...)`** — same mechanism, polling style. Probed: also concurrent (3.40 s). Not preferred (CPU polling overhead vs CPU-bound operators).
- **`fragment.make_thread_pool(name, initialize_size=1)` → `ThreadPool`** with `.add(op, pin: bool)` for pinning individual operators to a pool (official example `examples/resources/thread_pool/python/ping_simple_thread_pool.py` in the v4.2.0 tree). Not needed for Phase 3 — the app-level default pool suffices. Constraint from that example: all operators in a pool using a GPU allocator (CudaStreamPool/RMMAllocator/etc.) must share one CUDA device id — trivially true here (dev 0).
- **The app's scheduler propagates to fragments** that don't have their own (`app_driver.cpp` v4.2.0: `Application::set_scheduler_for_fragments(target_fragments, app_->scheduler_)`). Our app is a single top-level Application (Subgraphs, not separate fragments) → one scheduler covers all operators.
- **GIL:** `python/holoscan/core/operator.cpp` (v4.2.0) binds `compute`/`start`/`stop` with `py::call_guard<py::gil_scoped_release>()` — Python `compute()` runs without holding the GIL at the C++ boundary; torch/CuPy re-release it during GPU work. Two computes genuinely run in parallel (probed at thread and GPU level).
- **`CudaGreenContextPool` / `CudaGreenContext`** exist in 4.2 (CUDA green-context SM partitioning, driver ≥ 12.4 — this box is 610.57.04/CUDA 13.3, supported). **Recommendation: do NOT use them in Phase 3.** The measured overlap problem is CPU-span-hiding, not SM partitioning; green contexts add partition-size tuning risk (the official example needs an `is_partitioning_supported` probe loop) for zero expected gain on saturated conv kernels.

### The GPU-overlap reality (probe evidence — plan around this)

Two branches each doing 2 s of saturating 4096×4096 matmuls, `EventBasedScheduler` 2 workers:
- both on the torch default stream: wall 4.86 s (per-branch 3.54 s) — ~10 % gain over 4.0 s serial;
- each on its own `context.allocate_cuda_stream(...)`: wall 6.33 s (per-branch 5.00 s) — no gain (same context, SM time-sharing; contention overhead).

Conclusion: **concurrent inference∥inference ≈ no win** (both are 91–96 % GPU-busy; total GPU work is conserved). The concurrency win is: fragment A's **CPU-bound** spans (scipy resample in preprocess/postresample — 99.9 % CPU per the Phase 2 trace) run while fragment B's kernels saturate the GPU. Expected hidden work in the bundle: lowres preprocess (5.0 s) + cascade preprocess (9.4 s, after lowres_seg) + all three postresample spans (6.6 s) + postprocess (2.1 s) can partially slip under inference windows. Conservative expectation: 15–35 s of the 108.7 s in-study bundle.

### Wiring (verified pattern — official v4.2.0 example + probes)

```python
# in CCHMCNNUnetFastApp.compose() (after all operators/subgraphs are added), before run():
from holoscan.schedulers import EventBasedScheduler
self.scheduler(EventBasedScheduler(self, worker_thread_number=5, name="concurrent"))
```

`worker_thread_number=5` = 3 fragment chains + app-level postprocess + writers (each chain has at most one ready operator at a time; headroom is free). The scheduler object is set after `compose()`, before `run()` — verified working. (A `worker_thread_number` C++ option exists, but the supported Python path is the scheduler object; `CLIOptions` exposes only distributed-mode fields.)

### Thread-safety audit of the existing operators

| Shared state | Under concurrency | Verdict |
|---|---|---|
| `StudyTimingCollector._records` (class dict, `setdefault().append()`) | concurrent appends | `list.append` is GIL-atomic; record **order becomes nondeterministic** (start_ns preserves causal order) — cosmetic; tests must not assert record order |
| `_study_by_fragment[id(app)]` | 3 PreprocessOperators write the **same** study string concurrently | harmless (identical value) |
| `torch.set_num_threads(...)` scopes in `predict_logits` + `postresample_reference` | interleaved set/restore | both scopes set the **same** value (`default_num_processes`) — reference-parity CPU-softmax thread count preserved; no numeric change expected, but keep an eye on the fullres 3-voxel gate |
| torch CUDA (allocations, kernels) from 2 threads | standard multi-threaded torch | safe; RMM pool allocator is thread-safe by design |
| CuPy ops from 2 threads | both on CuPy's legacy default stream | serialized at CUDA level (that's where the CPU-span overlap still happens); CuPy is thread-safe |
| logging / DLPack | stdlib logging; per-tensor DLPack | thread-safe / no shared state |

**Stream note (keep it simple):** keep all torch/CuPy work on the legacy default streams for Phase 3. Moving fragments onto per-fragment non-blocking streams (`context.allocate_cuda_stream` + `torch.cuda.ExternalStream` — both verified available) adds D2H-copy sync subtleties (legacy default stream = implicit sync) for the measured-≈-zero GPU-overlap gain. If the 2×2 benchmark shows the serial-GPU cost is material, revisit as a documented follow-up — do not bake it into the first cut.

## D-22 Domain: GPU Resampling — Byte-Identity Analysis

### The actual call sites and what they compile to (this bundle, measured shapes)

All three "scipy call sites" reduce to **one primitive**: `scipy.ndimage.zoom(grid_mode=True)` with `mode='nearest'` (skimage maps its `'edge'` to scipy `'nearest'`), because:
- `skimage.transform.resize(..., anti_aliasing=False)` (skimage 0.25.2 `_warps.py`) = **exactly** `scipy.ndimage.zoom(filtered=image, zoom_factors, order, mode=ndi_mode, grid_mode=True)` + clip — no gaussian, no map_coordinates;
- the separate-z branch (`scipy.ndimage.map_coordinates` with the `(c+0.5)*s−0.5` grid) **never executes in this bundle**: the airway bundle spacings are near-isotropic so `_determine_do_sep_z_and_axis` returns `(False, None)` for every config.

Per-study execution (identity crop 256³, from Phase 2 timing logs):

| Site | Actual work | Span |
|---|---|---|
| `_resample_to_shape` (preprocess image) | fullres 256³→(255,256,255) o3; lowres 256³→(201,202,201) o3; cascade 256³→(255,256,255) o3 | part of 7.6 / 5.0 / 9.4 s |
| `_resample_seg_to_shape` (cascade seg) | 256³→(255,256,255) per-label multihot, o1, `>=0.5` threshold | part of 9.4 s |
| `resample_probabilities_to_shape` (postresample) | (255,256,255)→256³ / (201,202,201)→256³ / (255,256,255)→256³, 2 ch, o1 | 1.7 / 3.2 / 1.7 s |

Total ~28.8 s of the 129.5 s E2E (Phase 2 report §6, item 1). One GPU implementation of `zoom(grid_mode=True, mode='nearest')` at order ∈ {0, 1, 3} covers **every** active call site; the separate-z `map_coordinates` replica stays scipy (inactive here; order_z=0 would be byte-identical on CuPy anyway — measured).

### Byte-identity verdicts (measured on A100, venv stack)

| Comparison | order | Result |
|---|---|---|
| scipy `map_coordinates` fp32-in vs explicit fp64 compute cast back to fp32 | 1, 3 | **byte-equal** (scipy computes internally in double for fp32 input) |
| scipy vs `cupyx.scipy.ndimage.map_coordinates` (fp32 and fp64) | 0 | **byte-equal** (0/517,440 differ) |
| scipy vs cupyx (fp32) | 1 | **NOT equal** — 283,623/517,440 voxels differ, max 1.8e-4 |
| scipy vs cupyx (fp64) | 1 | **NOT equal** — 517,414/517,440 differ at 1-ulp-fp64 (max 3.1e-5) — accumulation order differs even in double |
| scipy vs cupyx (fp32 / fp64) | 3 | **NOT equal** (455,434 / 517,440 differ) |

Conclusion: **stock CuPy is dead at order ≥ 1; monai.data is dead by construction** (affine-grid math ≠ scipy zoom math). Byte-identity requires replicating scipy's own per-voxel arithmetic on GPU.

### The scipy-faithful kernel spec (scipy 1.15.3, `scipy/ndimage/src/ni_interpolation.c`, `NI_GeometricTransform`, lines 256–640)

Verified for `zoom(zoom_factors, order, mode='nearest', grid_mode=True)`:

1. **Per output index `kk` per axis** (the zoom path precomputes per-axis, per-index): `cc = kk; cc += 0.5; cc *= zoom; cc -= 0.5;` (in skimage's call the factor is output/input — i.e. `cc = (kk+0.5) * (out_dim/in_dim) − 0.5`, identical to the app's existing `map_rows` formula). Boundary: `nearest` clamps `cc` to `[0, len−1]` (`map_coordinate`: `in<0 → 0`, `in>len−1 → len−1`).
2. **Spline weights** `splvals[axis][kk][0..order]` computed **in double** by `get_spline_interpolation_weights(cc, order, splvals)` (`ni_splines.c`, closed-form B-spline); tap start: `order` odd → `(int)floor(cc) − order/2`; even → `(int)floor(cc + 0.5) − order/2`.
3. **Per-voxel accumulation** (the exact order to replicate): `t = 0.0;` then iterate the `(order+1)^ndim` tap grid `ff` in **C order (last axis fastest)**: `coeff = (double)input[idx]` (fp32 upcast to double); `for ll in 0..ndim−1: coeff *= splvals[ll][ff[ll]];`; then `t += coeff;`. Edge taps map their index through `map_coordinate(idx, len, spline_mode)` per axis/tap (nearest → clamp).
4. **Output**: `out = (float32) t` — double→fp32 round-to-nearest-even (`__double2float_rn` in CUDA).
5. **No FMA contraction.** Measured: a CuPy `RawKernel` computing `a*b + c` in double fuses to FMA by default and is NOT bit-equal to separate mul+add; the same kernel with `options=("--fmad=false",)` (tuple, not list — verified API quirk) IS bit-equal. Compile the kernel with `--fmad=false` (or explicit `__dmul_rn`/`__dadd_rn`) for every double op.
6. fp64 throughput on A100 (1/64 rate) is fine: 256³ × 27 taps × ~4 flops ≈ 2 GFLOP « 1 s.

**Implementation shape:** one `RawKernel` module (`gpu_zoom.py` in the app) exposing `gpu_zoom_grid_mode(cupy_fp32, zoom_factors, order) -> cupy_fp32`, splvals precomputed per axis (tiny: 3 × 256 × 4 doubles). Unit-test `np.array_equal` vs scipy at the three real shapes (256³→255²·256 o3; 256³→201·202·201 o3; 256³→255²·256 o1 × 2ch; seg multihot 0/1 masks o1) plus randomized shapes/orders 0–3 — the Phase 2 Plan 03 "reference-replica `np.array_equal`" template. Gate per D-22: all 4 config pixel gates with the flag ON; if any `np.array_equal` fails, flag stays OFF, divergence documented, phase ships.

**Flag placement (D-22):** default OFF (e.g. `HOLOSCAN_GPU_RESAMPLE=1`), consulted at the three call sites. Flag ON keeps the preprocess path fully on GPU (no D2H for resample → GPUP-02 for that span); numpy mean/std reductions stay CPU (Phase 1 bit-exactness). The OFF path must remain byte-for-byte Phase 2 behavior.

## D-23 / MEM-003 Domain: Freeing lowres weights

- **What's held:** `SlideWindowOperator._bundle` (per config) = `network` (torch module, fp32, CUDA) + `fold_state_dicts` (5 folds, `torch.load(map_location=cuda)`). 3d_lowres checkpoints: 5 × 135 MB → **~675 MB fold weights + ~135 MB live network ≈ 0.8 GB** CUDA-resident for the whole run, dead after the lowres fragment finishes.
- **Release point:** the lowres fragment's terminal operator is its `PostResampleOperator` (`emit_lowres_seg=True`, i.e. `cfg == aux_prev`). After it emits `lowres_seg`, nothing touches the lowres `SlideWindowOperator` again. Cleanest wiring: the `NnUnetConfigSubgraph` factory (already knows `emit_lowres_seg`) injects a release callback into that PostResample; PostResample invokes it **after** its final `op_output.emit` in `compute()`. `SlideWindowOperator.release()` = `del bundle.network, bundle.fold_state_dicts; self._bundle = None` + `torch.cuda.empty_cache()`.
- **What "free" means under RMM (measure both levels, per D-23):**
  - *Pool level:* the 0.8 GB returns to the RMM pool → headroom for the cascade fragments' volumes. The real functional benefit (lower pool high-water).
  - *Driver level (nvidia-smi):* RMM pools do **not** auto-return memory to the driver; `torch.cuda.empty_cache()` under the pluggable allocator is **unverified** (Open Q2) and may be a driver-level no-op. The gate samples nvidia-smi/pynvml AND RMM pool occupancy before/after and reports both — if driver-level is flat, that is the honest result (peak driver usage on this 40 GB box is dominated by the RMM pool reservation anyway — Open Q1).
- **No latency impact expected** (freeing = `del` + optional pool release) — a memory-lifecycle deliverable, not a speed lever.

## D-24 / INFR-02 Domain: Shape-keyed buffer reuse

**Allocation inventory per study (airway shapes), by allocator:**

| Allocator | Big per-study / per-call allocations |
|---|---|
| **torch (→ RMM pool)** | SlideWindow: `padded` (~64 MB), `predicted_logits` (2×256³ = 512 MB), `n_predictions` (64 MB), per-fold `prediction` (64 MB), per-patch `workon` (16 MB × ~150 patches × 5 folds), `gaussian` (1 MB — **identical every fold: computable once in setup**); PostResample: `full` fill (512 MB), `lowres_seg` (16 MB); Ensemble: per-config probability tensors (512 MB each); Preprocess final tensor (64–128 MB) |
| **CuPy (own pool, NOT RMM)** | `vol` (64 MB), `vol_c` (64 MB), `mask` (8 MB), one-hot (64 MB), `vol2` (128 MB) — CuPy allocations never touch the RMM pool (independent `cudaMalloc` + CuPy LRU pool) |
| **numpy (host)** | `raw` (16 MB), `vol_np`/`vol_out` (64 MB each) — host staging, not VRAM |

**Key facts for the plan:**
- The RMM pool already makes torch-side allocations cheap (Phase 2 churn check: 9 cudaMalloc / 117.5 s, pool expansions only). INFR-02's torch-side benefit is **allocation-traffic reduction and address stability**, not cudaMalloc elimination — the deliverable is D-24's proof (a)+(b), not a speed claim.
- **CuPy-side caching is the substantive gap** (its blocks bypass RMM entirely): a `(shape, dtype) → cp.ndarray` cache in `PreprocessOperator` covers vol/vol_c/mask/one_hot/vol2.
- Natural torch-side cache candidates (big, fixed-shape, re-allocated 5× per study per config): `predicted_logits`, `n_predictions`, `gaussian` (→ setup-time once), `padded`. **Zero the buffer where the reference allocates fresh** (`predicted_logits`/`n_predictions` are `torch.zeros` → `zero_()` on borrow; PostResample's `full` is explicit zero+fill → `zero_()`/re-fill).
- **Single-study-per-run is the clinical model** — cross-study reuse only matters for repeat-study processes; first-study latency is unaffected. Honest scope: INFR-02 ships as *provable reuse + reduced allocator traffic*.
- **Safety:** key invalidation on shape change (D-24(a) test); never cache across dtypes; C-contiguous fp32 invariant (D-12); a cached buffer must never be handed to a DLPack consumer that retains it after the study (Phase 1 ownership lesson — the postprocess clone exists for a reason).
- Granularity/eviction (discretion): unbounded dict keyed `(shape, dtype)` is fine here (few config-determined shape keys); no LRU; explicit `clear()` at study end only if peak-VRAM accounting requires it.

## Measurement Architecture (Phase 2 methodology, extended)

- **Wall-clock / per-operator:** re-run `phase2_benchmark.py` unchanged (fresh process/rep, 32 MB stack, warmup excluded, mean±std) vs `phase2_results.csv` (D-18 convention). **Phase 3 extension:** the 2×2 matrix (resample flag OFF/ON × serial/concurrent), 4 cells × (1 warmup + 3 measured) per scope; per-cell per-operator columns from the existing `timing:` JSON logs.
- **Concurrency proof (D-21):** nsys trace of one bundle run — overlapping NVTX ranges `preprocess_<cfg>`/`inference_<cfg>`/`postresample_<cfg>` across threads + overlapping kernel windows (vs Phase 2's single-stream back-to-back); record the measured overlap as the trace citation (also the evidence for the fallback clause if the ceiling is low).
- **INFR-02 proof (D-24):** (a) unit tests (multi-call reuse, shape-key invalidation, dtype/contiguity); (b) one process, same study 3×: cached-buffer `data_ptr()` stable across studies + nsys cudaMalloc count flat across studies 2/3 (Phase 2 Plan 06 churn method: CUPTI_ACTIVITY_KIND_RUNTIME, `_v3020` name suffix).
- **MEM-003 proof (D-23):** nvidia-smi/pynvml sampling (1–5 Hz) around the bundle run + RMM pool occupancy before/after lowres-fragment completion; record both driver-level and pool-level deltas. **Never `torch.cuda.memory_stats()`/`memory_allocated()`** — they raise under RMM (measured).
- **GPU-resample gate (D-22):** per-tensor `np.array_equal` unit suite (kernel vs scipy, real + random shapes) + full 4-config pixel gate with the flag ON (and OFF regression), per D-25.

## Architecture Patterns

### Recommended change surface

```
examples/apps/cchmc-nnunet-fast/my_app/
├── app.py                       # D-21: EventBasedScheduler wiring in compose(); MEM-003: release-callback injection
├── gpu_bootstrap.py             # (possible) initial_pool_size pin — Open Q1
├── operators/
│   ├── preprocess_operator.py   # D-22: flag at _resample_to_shape (+seg replica); D-24: CuPy shape-cache
│   ├── postresample_operator.py # D-22: flag at resample_probabilities_to_shape; MEM-003: release hook call
│   ├── slidewindow_operator.py  # MEM-003: release(); D-24: cache predicted_logits/n_predictions/gaussian
│   └── gpu_zoom.py              # NEW (D-22): CuPy RawKernel scipy-faithful zoom
examples/apps/cchmc-nnunet-fast/scripts/
├── test_gpu_zoom.py             # NEW: np.array_equal vs scipy (real + random shapes)
├── test_buffer_cache.py         # NEW: D-24(a) unit tests (headless, synthetic sizes)
└── test_weight_release.py       # NEW: MEM-003 hook semantics
.planning/scripts/
├── phase3_benchmark.py          # NEW or extension of phase2_benchmark.py: 2×2 matrix
└── probes-phase3/               # copy the /tmp probe scripts (scratch-safe provenance)
```

### Pattern 1: Scheduler swap (D-21) — minimal diff
Two lines in `compose()` + one import (§D-21 wiring). No operator changes. Config flag (e.g. `HOLOSCAN_CONCURRENT_FRAGMENTS`, default ON once gates pass; OFF = today's GreedyScheduler) so the serial fallback ships testable — D-21's fallback clause.

### Pattern 2: Gated GPU resample (D-22)
```python
# _resample_to_shape's non-separate-z branch (per channel):
if _gpu_resample_enabled():
    from my_app.operators.gpu_zoom import gpu_zoom_grid_mode
    zooms = [float(n) / float(o) for o, n in zip(data[c].shape, new_shape)]
    out = gpu_zoom_grid_mode(cp.asarray(data[c]), zooms, order)   # cp fp32 in/out
else:
    out = resize(data[c], new_shape, order, mode="edge", anti_aliasing=False)
```
Same flag at the seg multihot site and in `resample_probabilities_to_shape`. Default OFF = byte-for-byte Phase 2 behavior.

### Pattern 3: Shape-keyed cache (D-24)
```python
class _ShapeCache:
    """(shape, dtype) -> device buffer; zero_() on borrow where the reference zeros."""
    def __init__(self, device: str): self._d: dict = {}
    def get(self, shape, dtype, zero=False):
        key = (tuple(shape), str(dtype))
        buf = self._d.get(key)
        if buf is None:
            buf = <torch.empty / cp.empty per dtype family>(shape, dtype=dtype, device=...)
            self._d[key] = buf
        if zero: buf.zero_()
        return buf
```
Operator instance attribute (lives as long as the operator; single process). Never returns a buffer still referenced by an emitted DLPack tensor (boundary clone/copy, Phase 1 lesson).

### Pattern 4: Weight release (MEM-003)
```python
# slidewindow_operator.py
def release(self):
    if self._bundle is not None:
        b = self._bundle; self._bundle = None
        del b.network, b.fold_state_dicts
        torch.cuda.empty_cache()          # semantics under RMM: verify (Open Q2)
# postresample (aux config only, after final emit in compute()):
if self._release_fn is not None: self._release_fn()
```

### Anti-patterns to avoid
- **Per-fragment non-blocking streams in the first cut** — measured ≈0 GPU-overlap gain + D2H sync subtleties.
- **Green contexts / SM partitioning** — no expected gain on saturated convs; partition-tuning risk.
- **Replacing scipy with cupyx/monai for order ≥ 1** — not byte-identical (measured); fails the D-22 gate and burns the experiment.
- **`torch.cuda.memory_stats()` in any Phase 3 measurement** — raises under RMM (measured).
- **Caching buffers across dtypes or reusing a buffer still owned by a downstream DLPack consumer.**

## Don't Hand-Roll

| Problem | Don't build | Use instead | Why |
|---|---|---|---|
| Concurrent operator execution | hand-rolled threads around `compute()` / `run_async` tricks | `EventBasedScheduler(worker_thread_number=N)` | the 4.2-supported, GIL-released, condition-safe path (probed) |
| Cross-stream GPU sync | manual cudaEvent plumbing | torch default-stream serialization (Phase 3 cut) | implicit default-stream ordering is already correct |
| GPU resample math | cupyx map_coordinates / monai.data Resample | CuPy `RawKernel` replicating `NI_GeometricTransform` | measured non-identity of both stock alternatives — the only byte-exact path |
| VRAM accounting | torch memory counters | nvidia-smi/pynvml + RMM pool stats | torch counters raise under the pluggable allocator |
| Benchmark/gate harness | new tools | `phase2_benchmark.py` / `phase2_gate.py` / `pixel_diff.py` / `gpu_residency.py` (D-25) | reuse, not rebuild |
| Memory pooling | per-operator cudaMalloc pools | RMM (torch) + CuPy pool + shape-cache for the gaps | INFR-01 in place; INFR-02 fills the CuPy gap |

## Common Pitfalls (all live-reproduced this session unless noted)

### Pitfall 1: `Application(name=...)` does not exist in 4.2
`holoscan.core.Application.__init__` forwards kwargs to the C++ ctor which accepts only `argv` — `Application(name="x")` raises `TypeError`. Use `Application()` (or the monai.deploy subclass ctor as the app already does).

### Pitfall 2: torch memory API is degraded under RMM
`torch.cuda.memory_stats()` / `memory_allocated()` raise `RuntimeError: CUDAPluggableAllocator does not yet support getDeviceStats` once the RMM pluggable allocator is installed. Any MEM-003/INFR-02 measurement code that assumes these APIs will crash at runtime, not at import. Use nvidia-smi/pynvml + RMM pool stats. (Also: the pluggable-allocator swap must happen **before** any `torch.cuda.*` context-init call — `change_current_allocator` raises "Can't swap an already initialized allocator" otherwise; `gpu_bootstrap` already gets this right, and probes must too.)

### Pitfall 3: FMA contraction breaks bit-exactness
A CUDA kernel computing `a*b + c` in double fuses to an FMA by default (nvcc) and is NOT bit-equal to scipy's separate mul+add; `--fmad=false` (or `__dmul_rn`/`__dadd_rn`) is mandatory for the D-22 kernel. CuPy `RawKernel` accepts `options` as a **tuple** (a list raises `TypeError: expected tuple`).

### Pitfall 4: scipy computes resamples in double, CuPy in float32
`scipy.ndimage.map_coordinates`/`zoom` upcast fp32 input to double internally (measured: fp32-input output == explicit-fp64 compute). Any "just compute in fp32 on GPU" shortcut diverges by ~55 % of voxels at order 1 and ~88 % at order 3. Even fp64-on-GPU diverges from scipy at 1-ulp-fp64 unless the accumulation order matches exactly.

### Pitfall 5: `torch.set_num_threads` is global
Concurrent fragments interleave their thread-count scopes. Harmless here (both set the same `default_num_processes` value) but any future per-operator thread tuning would be a correctness landmine for the CPU-softmax reference parity (Phase 1: CPU softmax is thread-count sensitive at 2-ulp).

### Pitfall 6: Timing-record order becomes nondeterministic under concurrency
`StudyTimingCollector` record order follows thread scheduling, not DAG order. Any Phase 3 test or benchmark parse that assumes a fixed record order in `study_timing_summary` must sort by `start_ns` (the existing `phase2_benchmark.py` parses by operator name — safe; new asserts must be).

### Pitfall 7: GPU tenancy on this box is not stable
During this session `nvidia-smi` showed 2 GPUs, then 8 (4 of them fully occupied by other tenants), and `mem_get_info` readings were anomalous at times. **Pin `CUDA_VISIBLE_DEVICES` to a known-free device for every gate/benchmark run and record the device in the results CSV** (provenance, same class as Phase 2's venv pinning).

### Pitfall 8: The venv is scratch and has drifted before
`/tmp/monai-env/.venv` has been rebuilt/patched during the project (monai ptp patches re-applied after reinstall; holoscan wheel force-reinstalled in Phase 0). Phase 2-era assumptions (e.g. the nsys trace's ~1.98 GB of cudaMalloc implying a small initial RMM pool) may not hold under today's rmm 26.2.0 (default initial pool = ½ total GPU memory = 20 GiB, measured). Re-verify at plan start (Open Q1).

## State of the Art

| Old approach | Current approach | When | Impact |
|---|---|---|---|
| `rmm.mr.PoolAllocator` class API (legacy) | `rmm.reinitialize(pool_allocator=True, ...)` + `rmm.allocators.torch.rmm_torch_allocator` | rmm 26.x (installed 26.2.0) | gpu_bootstrap already uses the new API; `initial_pool_size` default = ½ total GPU memory |
| Serial GXF scheduling (default GreedyScheduler) | `EventBasedScheduler`/`MultiThreadScheduler` with `worker_thread_number` thread pools | holoscan 4.x (in 4.2) | the D-21 lever; app scheduler propagates to fragments |
| CUDA stream pools only | + `CudaGreenContextPool` (green-context SM partitioning, driver ≥ 12.4) | holoscan 4.2 | available but NOT recommended for Phase 3 (no expected gain, tuning risk) |
| `cupyx.scipy.ndimage` stock kernels | `cupy.RawKernel` with `--fmad=false` for bit-exact ports | CuPy 14.x | the D-22 path |

**Deprecated/outdated:**
- `MemoryData(DeviceType::GPU)` — absent from the 4.2 Python API; `holoscan.core.Tensor` (DLPack) is the equivalent (established Phase 1).
- C++ `Fragment`-mixed graphs in Python — rejected by the 4.2 app_driver (established Phase 2 Plan 04); `Subgraph` + interface ports is the mechanism.

## Open Questions

1. **RMM initial pool = 20 GiB (measured today) vs Phase 2's 1.98 GB total cudaMalloc (traced 2026-08-19).** What we know: rmm 26.2.0 `reinitialize(pool_allocator=True)` reserves ½ total GPU memory at init (measured: 20.0 GiB used immediately post-reinit, before any torch allocation). The Phase 2 bundle trace shows ~1.98 GB total cudaMalloc — inconsistent with a 20 GiB reservation at bootstrap. What's unclear: whether the venv's rmm/librmm changed between the Phase 2 runs and now (scratch /tmp venv), or the pool allocates lazily in some configurations. **Recommendation:** first task of the first plan — run one bundle rep with nsys and check the bootstrap cudaMalloc size; if 20 GiB is real, (a) record the shifted peak-VRAM baseline for MEM-003, (b) consider pinning `initial_pool_size=plan.total_bytes` in `gpu_bootstrap` (also the natural INFR-02 integration point), and (c) re-run the Phase 2 churn check so the INFR-02 "cudaMalloc flat" proof has a correct baseline.
2. **`torch.cuda.empty_cache()` under the RMM pluggable allocator — unverified.** `memory_stats`/`memory_allocated` raise (measured); `empty_cache` was never observed to error, but also never confirmed to release anything (mem_get_info was unreliable on a contended GPU during the session). **Recommendation:** a 10-line probe at plan start (allocate 2 GiB under RMM, del, empty_cache, compare nvidia-smi before/after); the MEM-003 plan then measures at the right level (pool vs driver) with correct expectations.
3. **Worker count sizing.** 2-way and 3-way probes passed; the app needs ≥5 (3 chains + postprocess + writers). EventBasedScheduler has no published cap; 5 is low-risk. Confirm via the nsys overlap trace (if a worker ever blocks the whole run, raise the count).
4. **GPU-resample kernel effort estimate.** The spec (§D-22) is precise, but the tap-order replication is subtle (C-order over the `(order+1)^3` grid, edge-tap remapping). Budget one full plan for kernel + identity tests; if identity fails after a good-faith effort, D-22's fallback ships the experiment write-up (measured divergence table) and the phase is unblocked.
5. **Concurrent CPU spans and CPU-core contention.** Two single-threaded scipy resamples in parallel need 2 cores (fine on this box); the 2×2 benchmark will show if CPU contention erodes the D-21 win (watch per-operator deltas, not just E2E).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|---|---|---|---|---|
| A100-SXM4-40GB (free device) | all GPU gates/benchmarks | ✓ (tenancy fluctuating — pin device, see Pitfall 7) | driver 610.57.04 / CUDA 13.3 | re-run when a device frees |
| nsys (Nsight Systems) | churn checks, overlap trace | ✓ | 2026.1.0 | nvidia-smi + NVTX timing logs (weaker evidence) |
| ncu | inference-kernel counters | ✗ (blocked `ERR_NVGPUCTRPERM`, carried) | 2026.1.0 installed | out of scope (D-20/D-26) |
| venv `/tmp/monai-env/.venv` | everything | ✓ (scratch — re-verify, Pitfall 8) | torch 2.13.0, holoscan-cu13 4.2.0, cupy 14.1.1, rmm 26.2.0, scipy 1.15.3, skimage 0.25.2, monai 1.3.0, numpy 2.2.6 | — |
| `ulimit -s 32768` (32 MB stack) | every app/probe run | ✓ (shell setting) | — | app warns and may segfault below 32 MB |

**Missing dependencies with no fallback:** none blocking (ncu is out of scope by decision).
**Missing dependencies with fallback:** none.

## Validation Architecture

### Test Framework
| Property | Value |
|---|---|
| Framework | **No pytest** — standalone scripts run directly with the venv python (project convention since Phase 0/1) |
| Config file | none (`examples/apps/cchmc-nnunet-fast/scripts/test_*.py` + `.planning/scripts/`) |
| Quick run command | `/tmp/monai-env/.venv/bin/python examples/apps/cchmc-nnunet-fast/scripts/<test>.py` |
| Full suite command | `/tmp/monai-env/.venv/bin/python .planning/scripts/phase2_gate.py` (4 config pixel gates + SR + residency → `03-GATE-RESULTS.json`) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|---|---|---|---|---|
| GPUP-01 | GPU resample byte-identical to scipy | unit (`np.array_equal`) | `…/scripts/test_gpu_zoom.py` | ❌ Wave 0 |
| GPUP-01/02 | 4-config pixel gate with flag ON + OFF | integration (E2E) | `phase2_gate.py` with `HOLOSCAN_GPU_RESAMPLE=1` / unset | ✅ (gate script; flag plumbing ❌ Wave 0) |
| D-21 (TEST-01/002/003/005 rows) | 4-config pixel gate + SR + residency with concurrency ON | integration (E2E) | `phase2_gate.py` with `HOLOSCAN_CONCURRENT_FRAGMENTS=1` | ✅ (flag plumbing ❌ Wave 0) |
| INFR-02 | buffer reuse semantics (multi-call, invalidation, invariants) | unit, headless | `…/scripts/test_buffer_cache.py` | ❌ Wave 0 |
| INFR-02 | address stability + cudaMalloc flat over 3× same study | integration (replay + nsys) | 3× replay harness + nsys churn export (Phase 2 Plan 06 method) | ❌ Wave 0 |
| MEM-003 | lowres weights released after aux fragment; pool/driver delta measured | unit + integration | `…/scripts/test_weight_release.py` + pynvml sampler around bundle run | ❌ Wave 0 |
| TEST-006 | 2×2 benchmark matrix, per-operator, vs phase2_results.csv | benchmark | `.planning/scripts/phase3_benchmark.py` | ❌ Wave 0 (extend phase2_benchmark.py) |
| TEST-007 | two-bar speedup report vs 169.7 s / 57.14 s | report | final plan writes `03-BENCHMARK-REPORT.md` from CSV | ❌ (report, not test) |
| TEST-003 | pixel_diff fails on divergence | unit | `…/scripts/pixel_diff.py` (existing; re-run in gate) | ✅ |
| TEST-002 | SR ≤ 0.1% delta | integration (in gate) | `phase2_gate.py` SR section | ✅ |
| TEST-01 | pixel-exact on dev corpus (≥5-CT blocked-on-external, D-26) | integration (in gate) | `phase2_gate.py` bundle row vs `testdata/current_output` | ✅ |

### Sampling Rate
- **Per task commit:** relevant unit script (headless where possible: `test_gpu_zoom.py`, `test_buffer_cache.py`, `test_weight_release.py`).
- **Per wave merge:** full `phase2_gate.py` suite (4 configs + SR + residency) — the D-25 anchor.
- **Phase gate:** full gate suite green (flags both ON and OFF where applicable) + `phase3_benchmark.py` matrix + nsys churn/overlap artifacts → `03-BENCHMARK-REPORT.md` + `03-GATE-RESULTS.json`.

### Wave 0 Gaps
- [ ] `examples/apps/cchmc-nnunet-fast/scripts/test_gpu_zoom.py` — covers GPUP-01 identity (requires `gpu_zoom.py` kernel)
- [ ] `examples/apps/cchmc-nnunet-fast/scripts/test_buffer_cache.py` — covers INFR-02 unit semantics (headless, synthetic sizes)
- [ ] `examples/apps/cchmc-nnunet-fast/scripts/test_weight_release.py` — covers MEM-003 hook semantics
- [ ] `HOLOSCAN_CONCURRENT_FRAGMENTS` + `HOLOSCAN_GPU_RESAMPLE` flag plumbing in app.py/operators (each plan that touches a lever adds its flag with default preserving Phase 2 behavior)
- [ ] `.planning/scripts/probes-phase3/` — commit the live probe scripts from `/tmp` (`probe_concurrency.py`, `probe_gpu_overlap.py`, `probe3.py`, `probe_resample_identity.py`) as reproducible provenance before /tmp is cleaned
- [ ] nvidia-smi/pynvml sampler helper for MEM-003 (10–30 lines, `.planning/scripts/`)

## Code Examples (verified this session)

### Concurrency probe result (EventBasedScheduler, 3 branches, worker_thread_number=3)
```
=== 3-branch wall=2.13s (serial 4.5s, concurrent ~1.5s) joins=1
  w1 tid=139844135155264 dur=1.50
  w3 tid=139844269364800 dur=1.50
  w2 tid=139844000937536 dur=1.50
```
vs default GreedyScheduler (2 branches): `wall=5.30s`, both spans on the same tid, back-to-back.

### FMA-contraction probe (CuPy RawKernel, double)
```
fma-allowed == separate mul+add: False     # default nvcc fuses a*b+c to FMA
fmad=false  == separate mul+add: True      # options=("--fmad=false",)  [tuple!]
```

### Byte-identity probe highlights (scipy 1.15.3 vs cupy 14.1.1, A100)
```
Q1 zoom o1: scipy(f32) vs scipy(f64).astype(f32): byte_equal=True    # scipy computes in double
Q2 mapc o0: byte_equal=True (fp32 and fp64)                          # nearest is portable
Q2 mapc o1 fp32: n_diff=283623/517440  max_abs=1.831e-04             # NOT portable
Q2 mapc o1 fp64: n_diff=517414/517440  max_abs=3.052e-05             # order mismatch even in double
```

## Sources

### Primary (HIGH confidence)
- **Live probes on the installed stack (this session):** scheduler concurrency (2-way/3-way, thread IDs, join counts), GPU-overlap wall times, scipy-vs-CuPy byte-identity at orders 0/1/3 (fp32/fp64), `RawKernel --fmad=false` bit-exactness, RMM reinit 20 GiB reservation, torch memory-API failures under RMM. Probe scripts in `/tmp/probe_*.py` (commit to `.planning/scripts/probes-phase3/`).
- **holoscan-sdk v4.2.0 source** (git clone, tag v4.2.0, nvidia-holoscan/holoscan-sdk): `python/holoscan/core/operator.cpp` (GIL release on compute), `include/holoscan/core/schedulers/gxf/{event_based,multithread,greedy}_scheduler.hpp`, `src/core/app_driver.cpp` (scheduler propagation), `examples/resources/thread_pool/python/ping_simple_thread_pool.py`, `examples/resources/cuda_green_context/python/cuda_green_context.py`.
- **scipy 1.15.3 source** (PyPI sdist): `scipy/ndimage/src/ni_interpolation.c` `NI_GeometricTransform` (per-voxel double arithmetic, tap order, nearest clamping), `scipy/ndimage/src/ni_splines.c` (spline weights).
- **scikit-image 0.25.2** `skimage/transform/_warps.py` `resize` body (the `ndi.zoom(grid_mode=True)` reduction).
- **Project artifacts:** `.planning/phases/02-gpu-acceleration/02-BENCHMARK-REPORT.md` (§5/§6), `02-RESEARCH.md`, `.planning/benchmarks/phase2_results.csv`, Phase 2 timing logs (real per-study shapes), `02-CONTEXT.md` (D-01…D-19).
- **Installed package introspection:** `holoscan.schedulers` / `holoscan.resources` docstrings (4.2.0 wheel), `rmm.reinitialize` docstring (26.2.0: initial_pool_size default ½ total GPU memory), app source (`app.py`, `preprocess_operator.py`, `postresample_operator.py`, `slidewindow_operator.py`, `gpu_bootstrap.py`, `gpu_util.py`).

### Secondary (MEDIUM confidence)
- Realized D-21 speedup (15–35 s) — inferred from probe wall times + Phase 2 per-operator spans, not yet measured in-pipeline.
- RMM 20 GiB reservation vs Phase 2 trace discrepancy (Open Q1) — unresolved until re-traced.

### Tertiary (LOW confidence)
- Nothing material — web docs were unreachable (GitHub raw/API blocked from this network); all claims rest on installed-package source, official v4.2.0 examples, or live probes.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every version verified in the venv today; no new packages needed.
- Architecture (D-21/D-24): HIGH for mechanism (probes + source), MEDIUM for realized in-pipeline speedup (probe used matmul/sleep proxies, not the pipeline).
- D-22 byte-identity: HIGH for the verdicts (empirical) and the kernel spec (C source read); MEDIUM for kernel feasibility (unimplemented; tap-order edge cases).
- MEM-003: MEDIUM — `empty_cache` semantics under RMM unverified (Open Q2); the 0.8 GB figure is exact from checkpoint sizes.
- Pitfalls: HIGH — all eight live-reproduced or carried from instrumented Phase 1/2 incidents.

**Research date:** 2026-08-19
**Valid until:** ~7 days (fast-moving: scratch venv can drift; GPU tenancy fluctuates — re-verify Open Q1/Q2 at plan start)
