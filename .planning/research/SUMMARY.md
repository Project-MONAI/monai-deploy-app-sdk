# SUMMARY — Holoscan-Native nnUNet Inference Pipeline

> Synthesis of STACK.md, FEATURES.md, ARCHITECTURE.md, and PITFALLS.md.
> Covers the `cchmc-nnunet-fast` replacement of the nnUNet Python wrapper with Holoscan-native GPU operators.

---

## 1. Stack Recommendations

### Use (Phase 1)

| Component | Choice | Why |
|-----------|--------|-----|
| Inference backend | **PyTorch eager-mode** (`torch.no_grad()`) | Native nnUNet compatibility, pixel-exact guaranteed, Holoscan `InferenceOp` supports `"torch"` natively |
| GPU array ops | **CuPy** (`cupy-cuda13x >=13.4.0`) | NumPy-compatible API on GPU; covers resampling, smoothing, connected components, morphology |
| Memory management | **RMM** (`rmm-cu13 >=25.10.0`) + Holoscan `CudaStreamPool` | Pool allocator reduces fragmentation for variable-sized 3D volumes |
| Tiling | **MONAI `sliding_window_inference`** on GPU | Handles tiling, overlap blending, Gaussian weighting; already GPU-aware |
| Profiling | **Nsight Systems** (`nsys`) + **Nsight Compute** (`ncu`) + **NVTX** | Pipeline-level (nsys), kernel-level (ncu), in-code markers (nvtx) |
| DICOM I/O | `pydicom` + `highdicom` + `nvidia-nvimgcodec-cu13` | GPU-accelerated DICOM decompression already in use |

### Avoid (Phase 1–2)

| Component | Reason |
|-----------|--------|
| TensorRT / ONNX | Loses dynamic tiling logic, adds ONNX export + engine build pipeline; defer to Phase 3+ after profiling |
| `cupy-cuda12x` | Conflicts with cu13x; uninstall from venv |
| `scikit-image` (CPU) | Replace with CuPy equivalents for GPU path |
| `pylibraft` | Evaluate only if CuPy connected-components is a measured bottleneck (>5% of latency) |
| `torch.compile()` | nnUNet's dynamic control flow in tiling/mirroring may not compile cleanly |
| `cProfile` / `nvprof` / `py-spy` | CPU-only or deprecated; don't see GPU kernels |

### Key Constraint

Every library must have a **CUDA 13 (cu13)** variant — Holoscan 4.2 pins `holoscan-cu13 >=4.0.0, <4.3.0`.

---

## 2. Features: Table Stakes vs. Differentiators

### Table Stakes (P0–P1, must-have)

| Feature | Priority | Impact |
|---------|----------|--------|
| **Zero-copy GPU data movement** between operators | P0 | Eliminates PCIe roundtrips — the single biggest win |
| **In-memory ensemble averaging** (no `.npz` I/O) | P0 | Eliminates ~1.5 GB disk I/O per study (15-checkpoint ensemble) |
| **Tile-based inference on GPU** | P0 | nnUNet already tiles; integrate with Holoscan async execution |
| **GPU preprocessing** (resampling, normalization) | P1 | Resampling is hard — may stay CPU-bound initially; normalization is easy |
| **Profiling infrastructure** | P1 | Needed to validate any improvement claim |

### Differentiators (P1–P2, competitive edge)

| Feature | Priority | Impact |
|---------|----------|--------|
| **In-place probability averaging** | P1 | Saves ~14 GB VRAM vs. keeping N copies |
| **Pre-allocated GPU memory pool** | P1 | Eliminates per-tile `cudaMalloc` churn |
| **Deferred CPU-GPU transfers** (boundary-only) | P1 | Only 2 PCIe crossings: decode→GPU and GPU→DICOM-SEG |
| **Double-buffered pipeline overlap** | P2 | Hides latency: preprocess M+1 while inferring M |
| **Multi-model weight sharing** | P2 | 15 checkpoints share architecture — load once, switch weights via views |
| **Structured operator-level timing** | P1 | Enables targeted optimization |

### Anti-features (deliberately avoid)

- SDK-level operator changes (scoped to example app only)
- Python multiprocessing for inference (serialization triggers CPU copies)
- Batch / concurrent study processing (Phase 2+)
- Triton / remote inference (out of scope)
- Dynamic operator graph reconstruction per study
- Lossy compression of intermediates (pixel-exact equivalence required)

---

## 3. Architecture

### Operator Boundaries (5 core operators + 2 downstream)

```
DICOMSeriesToVolume (existing, unchanged)
    │
    ▼
① PreprocessOperator    — transpose, crop, normalize, resample  [CPU→GPU]
    │
    ▼
② SlideWindowOperator   — tiled inference + TTA + fold averaging [GPU]
    │
    ▼
③ PostResampleOperator  — resample probs, softmax, argmax, revert crop/transpose [GPU]
    │
    ├──────── (parallel config branches) ────────┐
    │                                            ▼
④ EnsembleAverageOperator   — in-memory element-wise mean + argmax  [GPU]
    │
    ▼
⑤ PostprocessOperator       — connected component cleanup           [GPU/CPU hybrid]
    │
    ▼
MeasurementOperator / OverlayOperator   — downstream, no correctness impact
    │
    ▼
DICOMSegWriter (existing, unchanged)
```

Each config (e.g., `3d_fullres`, `3d_lowres`, `2d`) gets its own instance of operators ①–③, grouped as a Holoscan **Fragment**. Operators ④–⑤ are shared.

### Data Flow Principle

**Data stays on GPU from DICOM decode to DICOM-SEG write.** The only PCIe crossings are:
1. CPU numpy → GPU at the `PreprocessOperator` entry
2. GPU tensor → CPU numpy at the `DICOMSegWriter` entry

Inter-operator handoffs use `holoscan.core.MemoryData` with `DeviceType::GPU` for zero-copy buffer references.

### Cascade Handling

`3d_lowres` → `PostResampleOperator` produces a GPU segmentation → fed directly to `3d_cascade_fullres` `PreprocessOperator` as a one-hot channel stack. No `.nii.gz` disk I/O.

### Build Order

| Phase | Steps | Milestone |
|-------|-------|-----------|
| **Phase 1** | ① Preprocess → ② SlideWindow → ③ PostResample (single config) | Single-config end-to-end, pixel-exact vs. reference |
| **Phase 2** | ④ EnsembleAverage + parallel config branches + cascade wiring | Multi-config ensemble, pixel-exact vs. reference |
| **Phase 3** | ⑤ Postprocess + Measurement + Overlay | Full pipeline, polish |
| **Phase 4** | DAG assembly, replace `NNUnetSegOperator` in `app.py` | Production integration |

---

## 4. Critical Pitfalls & Prevention

### Data Transfer (silent performance loss)

| Pitfall | Severity | Prevention |
|---------|----------|------------|
| **Silent CPU fallback** — nnUNet catches OOM and reruns on CPU without signaling | Critical | Assert `tensor.device.type == 'cuda'` at every operator boundary; never swallow `RuntimeError` |
| **Hidden `.numpy()` / `.cpu()` round-trips** — synchronous PCIe transfer masquerading as "just a cast" | High | Audit every cross-device call; accumulate on GPU, transfer only once at pipeline end |
| **`MemoryData` device confusion** — `map()` returns CPU view even for GPU buffers | High | Assert `device_type() == DeviceType::GPU` at operator entry; use `gpu_memory()` APIs explicitly |

### Numerical Equivalence (correctness)

| Pitfall | Severity | Prevention |
|---------|----------|------------|
| **Resampler differences** — GPU resampling ≠ `scipy.ndimage.map_coordinates` | Critical | **Don't reimplement resampling.** Use the reference path. Resampling isn't the bottleneck. |
| **Autocast scope changes** — splitting FP16/FP32 boundaries across operators | Medium | Keep autocast at the outermost inference scope; don't split across operators |
| **Gaussian rounding** — recomputing Gaussian on GPU produces different FP16 weights | Medium | Precompute from reference `scipy.ndimage.gaussian_filter` path; treat as model metadata |
| **TTA accumulation order** — FP16 addition is non-associative | High | Preserve sequential `+=` order; use FP32 accumulators |
| **Normalize→resample reordering** — normalization MUST precede resampling | Critical | Enforce as strict DAG dependency |
| **Probability vs. logit averaging** — nnUNet averages probabilities (post-softmax), not logits | Critical | Replicate `average_probabilities` path: softmax → average → argmax |

### Memory Management

| Pitfall | Severity | Prevention |
|---------|----------|------------|
| **Tile pre-allocation OOM** — full-volume logits buffer can be 7.8 GB alone | High | Memory budget calculator before allocation; incremental ensemble averaging |
| **CUDA fragmentation** — per-study alloc/free fragments the heap over time | Medium | Pre-allocate in `setup()`, reuse across `compute()`; use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| **Cascade doubles peak memory** — lowres output + fullres input must coexist | Medium | Free lowres network weights immediately after inference |

### Operator Lifecycle

| Pitfall | Severity | Prevention |
|---------|----------|------------|
| **Model loading in `compute()`** — 10–30 s cold-start per study | Critical | Load models in `setup()` or `on_insert()`; never in `compute()` |
| **Mutable state leaks** — nnUNet predictor mutates thread count, device placement | High | Save/restore state at `compute()` boundaries; test double-call scenarios |
| **DataStore serializes CPU numpy** — CUDA tensors aren't picklable | High | Use `MemoryData` with GPU device type; don't emit raw tensors to DataStore |

---

## 5. Cross-Cutting Insights

1. **nnUNet's internals are the spec.** Every preprocessing step, tiling parameter, and postprocessing rule is part of the model's "signature." Don't optimize correctness-critical paths — wrap or reuse the reference code. The optimization surface is data movement and scheduling, not numerical algorithms.

2. **The bottleneck is I/O, not compute.** The current pipeline's worst offenders are `.npz` disk I/O for ensemble averaging (~1.5 GB/study) and CPU↔GPU bounces at every stage. The forward pass itself is well-optimized. Zero-copy GPU flow + in-memory ensemble eliminates the dominant costs.

3. **GPU memory is the tight constraint, not GPU compute.** On an A100-40GB there's headroom. On an 8 GB target GPU, full-volume logits buffers + ensemble probability maps + cascade intermediaries push close to OOM. Design for incremental averaging and memory pooling from day one.

4. **Holoscan's built-in GPU operators are for autonomous vehicles, not medical imaging.** `InferenceOp` is the only one worth considering (and even it expects ONNX/TRT). The real GPU work happens in CuPy/PyTorch — Holoscan manages scheduling, async execution, and the data flow DAG.

5. **Pixel-exact equivalence is non-negotiable and hard to maintain.** Every change to resampling, autocast scope, accumulation order, or padding strategy can silently diverge. Establish a reference corpus early, automate pixel-level diffs, and treat equivalence as a first-class test.

6. **TensorRT is a Phase 3+ option, not a Phase 1 shortcut.** The ONNX export + engine build overhead outweighs benefits until PyTorch is proven and profiling confirms the forward pass (not data movement) is the bottleneck. nnUNet's tiling and Gaussian weighting are PyTorch-native.

7. **Connect the full picture: stack enables architecture, architecture exposes pitfalls.** The choice of PyTorch + CuPy (stack) enables the 5-operator decomposition (architecture), which in turn creates device-boundary assertions and memory lifecycle concerns (pitfalls). These documents are interdependent.

---

*Synthesized: 2025-08-13*
*Sources: STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md*
*Status: Draft — for review before implementation begins*
