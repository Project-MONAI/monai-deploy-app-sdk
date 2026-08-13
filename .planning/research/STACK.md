# STACK: Holoscan-Native nnUNet Inference — Optimal 2025 Stack

> Decision log for the technology stack powering `cchmc-nnunet-fast`. Prescriptive — specific libraries, versions, what to use and what NOT to use.

---

## 1. Core Runtime Constraints (Immutable)

| Constraint | Value | Rationale |
|---|---|---|
| Holoscan SDK | `holoscan-cu13 >=4.0.0, <4.3.0` | SDK pins this range; our installed version is **4.2.0** |
| CUDA Toolkit | 13.x (13.2 on this machine) | Holoscan cu13 variant requires CUDA 13 |
| Python | 3.10–3.13 | Per pyproject.toml |
| PyTorch | >=2.6.0 (installed: 2.13.0+cu130) | nnUNet v2 requires PyTorch; CUDA 13 build needed |
| GPU | NVIDIA A100-SXM4-40GB ×2 | Available; 80 GB total VRAM. Target minimum: 8 GB |

**Implication:** Every library in this stack must have a CUDA 13 (cu13) variant. No CUDA 12-only packages.

---

## 2. Inference Backend: Direct PyTorch, NOT TensorRT

### Decision: Use PyTorch eager-mode inference with `torch.no_grad()`

### NOT TensorRT for now

**Rationale:**

| Factor | PyTorch (direct) | TensorRT |
|---|---|---|
| nnUNet compatibility | Native — nnUNet v2 already outputs `torch.nn.Module` | Requires ONNX export → conversion; loss of dynamic tiling logic |
| Pixel-exact equivalence | Guaranteed — same code path as current app | Risk of floating-point divergence across backends |
| Holoscan integration | Holoscan's `InferenceOp` supports `"torch"` backend natively | Holoscan's `InferenceOp` supports `"trt"` backend, but requires pre-built engines |
| Dynamic shapes | nnUNet tiling is dynamic per-study (varies by CT size) | TensorRT requires optimization profiles; adds complexity |
| nnUNet-specific ops | Tiling, Gaussian weighting, mirror augmentation — already in PyTorch | Must be reimplemented as custom TRT plugins or pushed outside TRT |
| Maintenance burden | Low — one framework | High — ONNX export pipeline + TRT engine build + profile management |
| First-pass latency target | Achievable — PyTorch 2.x has torch.compile | Overkill for phase 1 |

### When to consider TensorRT (Phase 3+)

Only after pixel-exact equivalence is proven with PyTorch and profiling shows the forward pass itself (not data movement) is the bottleneck. At that point:

1. Export nnUNet network to ONNX via `torch.onnx.export()`
2. Build TensorRT engines with `trtexec` at target batch size (1) and representative ROI sizes
3. Use Holoscan's `InferenceOp` with `backend="trt"` and `is_engine_path=True`
4. Handle tiling and Gaussian weighting outside TRT (in CuPy or PyTorch)

### Version recommendation

```
torch>=2.6.0,<2.14.0       # CUDA 13 build
torch-tensorrt>=2.13.0     # Phase 3+ only, NOT installed now
```

**NOT to install:** `onnx`, `onnxruntime`, `tensorrt`, `torch-tensorrt` (Phase 1–2)

---

## 3. GPU Operator Primitives for Image Processing

### The Core Strategy: CuPy for Pre/Post-processing on GPU

Holoscan 4.2 provides a limited set of GPU operators — they are optimized for **video streaming** (camera pipelines), not medical imaging. Here is what's available and whether it's useful:

#### Holoscan Built-in GPU Operators

| Operator | Relevance | Verdict |
|---|---|---|
| `InferenceOp` | Runs ONNX/TensorRT/LibTorch models on GPU. Supports `"torch"`, `"trt"`, `"onnxrt"` backends. | **USE** — for the forward pass if we want to use Holoscan's native operator (see §4) |
| `InferenceProcessorOp` | Pre/post-processing for inference tensors (format conversion, normalization). Designed for camera images (2D, RGB/YUV). | **NOT USE** — not designed for 3D medical volumes, DICOM spacing, or medical image normalization |
| `SegmentationPostprocessorOp` | Connected-component cleanup on 2D tensors. Outputs `(H, W, 1)` uint8. | **NOT USE** — 2D only; medical imaging is 3D |
| `FormatConverterOp` | YUV/RGB/Bayer format conversion. | **NOT USE** — wrong domain |
| `BayerDemosaicOp` | Camera sensor demosaicing. | **NOT USE** — irrelevant |
| `RawImageProcessorOp` | Raw camera image processing. | **NOT USE** — irrelevant |
| `GXFCodeletOp` | Generic C++ GXF codelet wrapper. | **CONSIDER** — if we need a custom C++ GPU kernel and want it in the Holoscan pipeline |

**Conclusion on Holoscan GPU operators:** They are purpose-built for autonomous vehicle camera pipelines. None are directly useful for 3D medical image segmentation. The `InferenceOp` is the only one worth considering, and even that is limited — it expects ONNX/TensorRT models, not raw PyTorch modules with nnUNet's complex tiling logic.

#### CuPy: The Right Tool for GPU Array Operations

**USE CuPy** (`cupy-cuda13x >=13.4.0`) as the GPU array backbone for all non-inference operations.

CuPy provides a NumPy-compatible API that runs on CUDA. It's already installed in the venv (`13.6.0`) and is used by the existing `decoder_nvimgcodec.py` operator.

**Key capabilities for medical imaging:**

| Operation | CuPy / cupyx | Replaces |
|---|---|---|
| Array indexing, slicing, broadcasting | `cupy` (NumPy-compatible) | NumPy on CPU |
| Resampling / interpolation | `cupyx.scipy.ndimage.map_coordinates` | `scipy.ndimage.map_coordinates` (CPU) |
| Gaussian smoothing | `cupyx.scipy.ndimage.gaussian_filter` | `scipy.ndimage.gaussian_filter` |
| Connected components (labeling) | `cupyx.scipy.ndimage.label` | `scipy.ndimage.label` |
| Distance transform | `cupyx.scipy.ndimage.distance_transform_edt` | `scipy.ndimage.distance_transform_edt` |
| Morphological ops (erosion, dilation) | `cupyx.scipy.ndimage.binary_erosion`, `binary_dilation` | scikit-image (CPU) |
| FFT | `cupyx.scipy.fft` | `scipy.fft` |
| Array I/O (numpy interop) | `cupy.asnumpy()`, `cupy.asarray()` | Manual `torch.Tensor.cpu().numpy()` |

**Version:** `cupy-cuda13x >=13.4.0` (NOT `cupy-cuda12x` — uninstall that from the venv)

### GPU Memory Allocation

#### Use: Holoscan's `RMMAllocator` for GPU buffer management

Holoscan 4.2 ships with RMM (RAPIDS Memory Manager) integration. Available allocators:

| Allocator | Use Case | Verdict |
|---|---|---|
| `RMMAllocator` | CUDA pool allocator via RAPIDS RMM. Reduces fragmentation for large allocations. | **USE** — for GPU buffers managed by Holoscan operators |
| `CudaAllocator` | Raw CUDA malloc/mfree. | NOT USE — inferior to RMM for large/variable allocations |
| `BlockMemoryPool` | Fixed-size block pool. Good for predictable tensor sizes. | CONSIDER — if tile sizes are fixed across studies |
| `StreamOrderedAllocator` | Allocates in stream order, frees in reverse. | NOT USE — overkill for single-study pipeline |

Install: `rmm-cu13 >=25.10.0`

#### CUDA Stream Management

Holoscan's `CudaStreamPool` is available as a resource. **USE** it for async operations within operators — particularly if preprocessing and inference overlap.

### PyTorch CUDA Memory Management

For PyTorch tensors specifically:

- **USE** `torch.cuda.empty_cache()` sparingly — only between studies if memory fragmentation is observed
- **USE** `torch.cuda.memory_allocated()` / `torch.cuda.memory_reserved()` for monitoring
- **DO** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (PyTorch 2.0+) to reduce fragmentation for variable-sized 3D volumes
- **DO** use `torch.inference_mode()` instead of `torch.no_grad()` where possible (slightly faster, prevents graph building)
- **DO NOT** use `torch.compile()` for nnUNet yet — nnUNet's dynamic control flow in tiling/mirroring may not compile cleanly

---

## 4. Profiling & Performance Tools

### Nsight Systems (`nsys`) — Pipeline-level profiling

- **Installed:** `2025.6.3` (CLI at `/usr/local/cuda/bin/nsys`)
- **Use for:** End-to-end pipeline timing, operator scheduling, CPU-GPU overlap, data transfer bottlenecks
- **Integration:** Wrap the Holoscan `app.run()` call:
  ```bash
  nsys profile --stats=true \
    -o holoscan_pipeline \
    --trace=cuda,nvtx,osrt/python \
    --capture-range-end=stop \
    python -m my_app.entry
  ```
- **Read output with:** `nsys-ui` (requires display) or `nsys stats holoscan_pipeline.nsys-rep` (CLI)

### Nsight Compute (`ncu`) — Kernel-level profiling

- **Installed:** `2026.1.0` (CLI at `/usr/local/cuda/bin/ncu`)
- **Use for:** GPU kernel analysis, occupancy, memory throughput, identifying slow kernels
- **Integration:** Profile specific kernels or the full application:
  ```bash
  ncu --set full --output nnunet_inference \
    python -m my_app.entry
  ```
- **Focus areas:** Convolution kernels (nnUNet forward pass), memory copies (preprocessing→inference)

### NVTX — Marking regions in profiling output

- **Available:** `torch.cuda.nvtx` (built into PyTorch, no separate install needed)
- **USE** extensively to annotate regions in Nsight Systems/Compute:
  ```python
  import torch.cuda.nvtx as nvtx

  with nvtx.range("DICOM decode"):
      # decode DICOM pixels
      pass

  with nvtx.range("GPU preprocessing"):
      # resample, normalize on GPU
      pass

  with nvtx.range("nnUNet forward - tile"):
      # forward pass for one tile
      pass

  with nvtx.range("ensemble averaging"):
      # average probability maps
      pass

  with nvtx.range("postprocessing"):
      # connected component cleanup
      pass
  ```

### What NOT to use

| Tool | Reason |
|---|---|
| `cProfile` / `line_profiler` | CPU-only profilers; miss GPU activity entirely |
| `nvprof` | Deprecated, replaced by Nsight Compute |
| `py-spy` | Doesn't see GPU kernels; limited value for GPU-bound workloads |
| `torch.profiler` | Useful for PyTorch-only profiling but doesn't see Holoscan operators; use Nsight instead |

---

## 5. GPU Memory Patterns for Large 3D Medical Images

### Problem Space

A single CT study: ~512×512×300 voxels × 2 bytes (int16) ≈ **150 MB raw**
Preprocessed float32: ~512×512×300 × 4 bytes ≈ **300 MB**
nnUNet tile (e.g., 256×256×256, 4 channels in + C channels out): ~256 MB/tile
Probability maps (ensemble): N_configs × N_classes × volume ≈ **hundreds of MB**

**Total peak: 1–2 GB for a single study** — easily fits in 40 GB A100, but memory management matters for correctness and avoiding OOM on smaller GPUs.

### Pattern 1: Tile-Based Inference (Required)

nnUNet uses tiling to handle volumes larger than GPU memory. **Keep this.**

- **USE** MONAI's `sliding_window_inference()` — it handles tiling, overlap blending, and Gaussian weighting. Already GPU-aware (`sw_device`, `device` params).
- **DO** run `sliding_window_inference` with `sw_device="cuda"` and `device="cuda"` to keep tiles on GPU during inference.
- **DO** set `sw_batch_size` to fit 2–4 tiles in GPU memory simultaneously.
- **DO NOT** try to load the entire volume into the model at once — even on A100, this wastes compute (better utilization with tiling).

### Pattern 2: Zero-Copy Data Flow

**Goal:** Data stays on GPU from DICOM decode → preprocessing → inference → postprocessing → DICOM-SEG write.

```
CPU: DICOM decode (pydicom + nvimgcodec)
  ↓ cudaMemcpy (unavoidable for pixel data)
GPU: Preprocessing (CuPy: resample, normalize)
  ↓ (zero-copy: CuPy → PyTorch via torch.as_tensor on cupy array)
GPU: Inference (PyTorch sliding_window_inference)
  ↓ (zero-copy: PyTorch output stays on GPU)
GPU: Postprocessing (CuPy: connected components, cleanup)
  ↓ cudaMemcpy (to write DICOM-SEG)
CPU: DICOM-SEG writer
```

**Key interop pattern:**
```python
import cupy as cp
import torch

# CuPy → PyTorch (zero-copy, same device)
cupy_array = cp.random.rand(512, 512, 300).astype(cp.float32)
torch_tensor = torch.as_tensor(cupy_array, device="cuda")

# PyTorch → CuPy (zero-copy)
gpu_tensor = torch.randn(256, 256, 256, device="cuda")
cupy_back = cp.asarray(gpu_tensor)
```

### Pattern 3: Probable Map Memory Management for Ensemble

Current app writes `.npz` to disk for each ensemble member. **Replace with in-memory GPU storage.**

```python
# Pre-allocate GPU memory for ensemble probabilities
num_configs = len(model_list)  # e.g., 2-3
num_classes = 2  # background + airway
volume_shape = (D, H, W)

# Store on GPU as float32
prob_accumulator = torch.zeros(
    (num_classes, *volume_shape),
    device="cuda",
    dtype=torch.float32
)

for model in ensemble_models:
    prob = sliding_window_inference(...)  # returns GPU tensor
    prob_accumulator += prob  # in-place GPU addition

# Average on GPU
segmentation = prob_accumulator.argmax(dim=0)  # GPU argmax
```

### Pattern 4: GPU Memory Pooling Across Studies

If processing multiple studies sequentially (future throughput phase):

- **USE** `torch.cuda.caching_allocator` — PyTorch's built-in allocator caches freed memory
- **DO** pre-allocate a reusable buffer for the largest expected volume
- **DO NOT** call `torch.cuda.empty_cache()` after every study — it defeats the caching allocator

---

## 6. Connected Components on GPU (Postprocessing)

nnUNet's `postprocessing.pkl` specifies connected-component cleanup rules. The current app uses `scipy.ndimage.label` on CPU.

### USE: CuPy's GPU-connected components

```python
from cupyx.scipy.ndimage import label

# Input: GPU binary mask (0/1), uint8 or bool
# structure: connectivity (e.g., 26-connected for 3D)
structure = cp.ones((3, 3, 3), dtype=bool)  # 26-connected
labeled_array, num_features = label(binary_mask_gpu, structure=structure)
```

### Alternative (if CuPy label is too slow for large volumes):

Consider `pylibraft` (RAPIDS) which has a highly optimized GPU connected components:
```python
from pylibraft.graphs import connected_components
```

**Version:** `pylibraft-cu13 >=25.10.0` (evaluate performance vs CuPy; start with CuPy, swap if needed)

**Recommendation:** Start with CuPy. Profile. Only swap to pylibraft if profiling shows CC is >5% of total latency.

---

## 7. Resampling & Interpolation on GPU

nnUNet preprocessing requires resampling to target spacing. Current approach: `scipy.ndimage.zoom` on CPU.

### USE: CuPy `map_coordinates` for GPU resampling

```python
from cupyx.scipy.ndimage import map_coordinates

def resample_to_spacing(cupy_volume, from_spacing, to_spacing, order=3):
    """GPU resampling using map_coordinates (trilinear for order=1, cubic for order=3)."""
    from_shape = cupy_volume.shape
    to_shape = tuple(int(s * f / t) for s, f, t in zip(from_shape, from_spacing, to_spacing))

    # Generate target coordinates
    grids = [cp.linspace(0, from_s - 1, to_s)
             for from_s, to_s in zip(from_shape, to_shape)]
    coordinates = cp.meshgrid(*grids, indexing='ij')
    coordinates = cp.stack(coordinates, axis=-1)

    return map_coordinates(cupy_volume, coordinates, order=order, mode='reflect')
```

### NOT to use:

| Library | Reason |
|---|---|
| `torch.nn.functional.interpolate` | Limited to specific interpolation modes; not equivalent to `scipy.ndimage.map_coordinates` |
| `scipy.ndimage.zoom` on CPU | Requires CPU-GPU transfer for every resample |
| `SimpleITK.Resample` | CPU-based; not GPU-accelerated |

---

## 8. Summary: Prescriptive Stack

### Install (Phase 1)

```toml
# Core (already in pyproject.toml)
holoscan-cu13 = ">=4.0.0,<4.3.0"
torch = ">=2.6.0"         # CUDA 13 build
monai = ">=1.3.0"
numpy = ">=1.21.6"

# GPU array processing
cupy-cuda13x = ">=13.4.0"          # GPU NumPy-compatible arrays
rmm-cu13 = ">=25.10.0"             # RAPIDS memory manager

# DICOM I/O (already in requirements-examples.txt)
pydicom = ">=3.0.0"
highdicom = ">=0.18.2"
nvidia-nvimgcodec-cu13 = ">=0.6.1"  # GPU-accelerated DICOM decompression
python-gdcm = ">=3.0.10"

# nnUNet (vendored, editable install — already present)
# nnunetv2 = { path = "./nnUNet", editable = true }
```

### Install (Phase 3+ only, if profiling warrants)

```toml
torch-tensorrt = ">=2.13.0"   # PyTorch → TensorRT compilation
onnx = ">=1.17.0"             # ONNX export
pylibraft-cu13 = ">=25.10.0"  # GPU connected components (if CuPy label is too slow)
```

### Do NOT install (Phase 1–2)

- `tensorrt` — adds build complexity, no net benefit over PyTorch for nnUNet
- `onnx` / `onnxruntime` — not needed unless going to TensorRT
- `cupy-cuda12x` — **uninstall** from venv (conflicts with cupy-cuda13x, causes runtime warnings)
- `scikit-image` — CPU-based; replace with CuPy equivalents
- `pylibraft` — evaluate only if CuPy label is a bottleneck

### Profiling tools (already installed)

| Tool | CLI | Version | Purpose |
|---|---|---|---|
| Nsight Systems | `nsys` | 2025.6.3 | Pipeline-level profiling |
| Nsight Compute | `ncu` | 2026.1.0 | Kernel-level profiling |
| NVTX | `torch.cuda.nvtx` | Built into PyTorch 2.13 | Mark profiling regions in code |

### Environment variables

```bash
# Reduce PyTorch CUDA memory fragmentation for variable-sized volumes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 9. Architecture Implications

Given the stack above, the Holoscan-native app should be structured as:

```
DICOMSeriesToVolumeOperator (SDK, CPU→GPU pixel decode via nvimgcodec)
    ↓ [GPU buffer via DataStore]
GpuPreprocessOperator (NEW, CuPy-based Holoscan operator)
    - Resampling: cupyx.scipy.ndimage.map_coordinates
    - Normalization: CuPy array ops (z-score, intensity clipping)
    - GPU→GPU transfer to PyTorch tensor via torch.as_tensor(cupy_array)
    ↓ [GPU tensor via Holoscan Message]
GpuInferenceOperator (NEW, PyTorch-based Holoscan operator)
    - MONAI sliding_window_inference on GPU
    - Returns probability maps as GPU tensors
    ↓ [GPU tensor via Holoscan Message]
GpuEnsembleOperator (NEW, in-memory)
    - Sum probabilities on GPU (torch.add)
    - Average on GPU (torch.div)
    - argmax on GPU → segmentation tensor
    ↓ [GPU tensor via Holoscan Message]
GpuPostprocessOperator (NEW, CuPy-based Holoscan operator)
    - Connected components: cupyx.scipy.ndimage.label
    - Keep largest component per label
    - Convert to uint8
    ↓ [GPU buffer via DataStore]
DICOMSegmentationWriterOperator (SDK, GPU→CPU for DICOM write)
```

**Key principle:** Each operator is a thin Holoscan wrapper around CuPy/PyTorch GPU ops. The heavy computation stays on GPU; Holoscan manages the data flow, scheduling, and async execution.

---

*Created: 2025-08-13*
*Author: Research agent*
*Status: Draft — requires architecture review before implementation*
