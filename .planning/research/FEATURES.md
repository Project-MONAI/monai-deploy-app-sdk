# FEATURES — Holoscan-Native nnUNet Inference Pipeline

_Research date: 2025-08-12_
_Context: cchmc-nnunet-fast, replacing the nnUNet Python wrapper (cchmc_nnunet_fifteen_ckpt_app) with Holoscan-native GPU operators._

---

## 1. Table Stakes

Features that **must** be present for the pipeline to be viable. These address known bottlenecks in the current app and are non-negotiable given the PROJECT.md constraints.

### 1.1 Zero-copy GPU data movement between operators

**Current state:** `DICOMSeriesToVolumeOperator` produces a CPU numpy array. `NNUnetSegOperator.compute()` passes it to `nnUNetPredictor.predict_from_list_of_npy_arrays()` which copies it to GPU, infers, then copies logits back to CPU. Every handoff is a PCIe roundtrip.

**Requirement:** Data must stay on GPU from decode → preprocess → infer → postprocess → encode. Holoscan provides the primitives:
- `CudaStreamPool` / `CudaAllocator` for GPU memory management across operators
- `DoubleBufferTransmitter` / `DoubleBufferReceiver` for lock-free handoff between pipeline stages
- `RMMAllocator` (RAPIDS Memory Manager) for pooled GPU allocations to avoid per-tile `cudaMalloc`/`cudaFree` churn

**Evidence from codebase:** The existing `MonaiSegInferenceOperator` already supports `cupy` (`cp, has_cp = optional_import("cupy")`) — indicating the SDK is aware of GPU-resident arrays, but the nnUNet wrapper never uses this path. The `MemoryDatastore` is a Python dict with no GPU awareness.

**Holoscan integration point:** Operators exchange `Tensor` objects backed by `CudaAllocator` rather than numpy arrays. The DataStore must be extended or bypassed for GPU tensor passing.

### 1.2 Tile-based (sliding window) inference on GPU

**Current state:** nnUNet's `predict_sliding_window_return_logits` already does sliding window, but it runs as a monolithic Python call inside the operator. Tiles are managed by nnUNet's internal `_internal_predict_sliding_window_return_logits` with a producer-consumer thread and `Queue(maxsize=2)`. Results accumulate on GPU (`perform_everything_on_device=True`) but the **entire volume** must reside in GPU memory for the accumulation arrays (`predicted_logits`, `n_predictions`).

**Requirement:** The Holoscan operator must:
- Support all tile sizes from `configuration_manager.patch_size` (derived from plans.json)
- Implement the Gaussian-weighted tile averaging (sigma_scale=1/8) on GPU
- Handle 2D slicing (when `len(patch_size) < len(image_size)`) and full 3D tiling
- Support mirror augmentation (TTA) per `allowed_mirroring_axes` — nnUNet applies up to 2^N mirror flips per tile

**Evidence from codebase:** `sliding_window_prediction.py` shows `compute_gaussian` is `@lru_cache(maxsize=2)` and cached as fp16 on GPU — this is a small tensor that can be pre-allocated. `_internal_get_sliding_window_slicers` handles both 2D (per-slice) and 3D tiling.

**Critical:** nnUNet's current producer thread (`producer` function) clones each tile with `torch.clone(..., memory_format=torch.contiguous_format).to(self.device)` — each clone triggers a memory copy. In the Holoscan pipeline, tiles should be views/slices of the GPU-resident volume, not clones.

### 1.3 In-memory ensemble averaging (no .npz I/O)

**Current state:** This is the single largest bottleneck. `ModelnnUNetWrapper.forward()` sets `save_probabilities=True, save_files=True` which causes nnUNet to write `.npz` files to disk via `export_prediction_from_logits`. The `EnsembleProbabilitiesToSegmentation` transform then reads these files back with `average_probabilities(all_files)` from nnUNet's ensembling module.

**Evidence from codebase:** `nnunet_bundle.py` line: `predictor.predict_from_list_of_npy_arrays(..., save_probabilities=True, truncated_ofname=outfile)` — this writes the file. Then `EnsembleProbabilitiesToSegmentation.__call__` loads: `avg_probs = average_probabilities(all_files)` where each file is loaded from disk.

For a 15-checkpoint ensemble on a 512³ volume with ~20 classes, each `.npz` probability file is ~100 MB. Writing and reading 15 × 100 MB = **1.5 GB of disk I/O per study**.

**Requirement:**
- Each model in the ensemble outputs logits/probabilities as a GPU tensor
- An ensemble operator accumulates sums in a single GPU buffer: `accumulated += probs_i`
- After all models vote, divide by N: `accumulated /= N`
- No intermediate file writes

### 1.4 GPU-accelerated preprocessing (resampling, normalization)

**Current state:** Preprocessing is done by nnUNet's CPU-side preprocessor (`preprocessing_iterator_fromnpy`) which handles:
- Resampling to target spacing via `configuration_manager.resampling_fn_probabilities`
- Cropping to bounding box
- Channel normalization (per-channel mean/std from dataset.json)
- One-hot encoding of previous-stage segmentation (cascade)

**Evidence from codebase:** `predict_from_raw_data.py` calls `preprocessing_iterator_fromnpy(...)` which runs in multiprocessing worker processes — data is serialized between processes. The `resampling_fn_probabilities` is typically a Lanczos-based resampler from `scipy.ndimage` (CPU).

**Requirement:** At minimum, normalization and cropping must run on GPU. Resampling (the heavy part) is the hardest — nnUNet's Lanczos kernel is scipy-based. Options:
- Use MONAI's `ResampleToImgd` which supports GPU via PyTorch
- Use cupy/cucim for GPU resampling
- Accept that resampling may remain CPU-bound in Phase 1 but ensure all subsequent steps are GPU-native

### 1.5 GPU-accelerated connected component postprocessing

**Current state:** `PostProcessNNUnet` wraps `apply_postprocessing` from `nnunetv2.postprocessing.remove_connected_components` which runs `scipy.ndimage.label` and size filtering on CPU. Additionally, the MONAI `KeepLargestConnectedComponentd` transform runs on CPU numpy arrays.

**Evidence from codebase:** `nnunet_seg_operator.py` uses `KeepLargestConnectedComponentd(keys=pred_key, applied_labels=[1])` — MONAI's implementation uses `scipy.ndimage.label`.

**Requirement:** Use a GPU-connected-components algorithm:
- Holoscan's `SegmentationPostprocessorOp` — NVIDIA's C++ GPU operator for connected component analysis, label smoothing, etc. This is the **primary candidate** — it's a native Holoscan operator, already installed
- Alternative: `torch` + custom CUDA kernel, or cupy's `cupy.sparse.csr.coo_matrix` based approach

### 1.6 Profiling and benchmarking infrastructure

**Current state:** No systematic profiling exists. The app prints shapes and timestamps but has no structured benchmarking.

**Requirement:**
- NVIDIA Nsight Systems integration (`nsys`) for end-to-end timeline profiling including CPU-GPU transfers, operator boundaries, and kernel launches
- Per-operator timing (Holoscan `RealtimeClock` or Python `time.perf_counter` at operator entry/exit)
- GPU memory utilization tracking (`torch.cuda.max_memory_allocated()`, `nvidia-smi` sampling)
- Structured output: CSV/JSON with study ID, operator timings, total latency, GPU peak memory
- Baseline comparison tool: run current app and new app on same inputs, diff outputs for pixel equivalence

---

## 2. Differentiators

Features that separate a well-optimized pipeline from a naive "just move things to GPU" approach.

### 2.1 Pipeline-level overlapping (double buffering across stages)

**Concept:** While tiles for model M are being inferred, preprocessing for model M+1 should be running, and postprocessing for model M-1 should be executing. The current architecture is strictly sequential: preprocess → infer → ensemble (file I/O) → postprocess.

**Implementation:**
- `DoubleBufferTransmitter` / `DoubleBufferReceiver` between each operator pair
- Holoscan's scheduler naturally overlaps when operators have different resources (e.g., preprocessing on stream A, inference on stream B)
- The Gaussian-weighted tile accumulation buffer should be double-buffered so accumulation of tile N doesn't stall the producer for tile N+1

### 2.2 Pre-allocated GPU memory pool (no per-tile malloc)

**Concept:** nnUNet's `_internal_predict_sliding_window_return_logits` pre-allocates `predicted_logits` and `n_predictions` for the full volume, which is good. But each tile is still `torch.clone()`'d which allocates. For the full pipeline, each operator should pre-allocate its working buffers during `setup()` and reuse them across studies.

**Implementation:**
- `BlockMemoryPool` resource with known max allocation sizes (derived from plans.json patch_size + volume max dims)
- `RMMAllocator` for cuda-aware pooling with arena pre-allocation
- Pool sizes tuned to: `2 × patch_size_tensor + accumulation_buffer + probability_buffer_per_ensemble_model`

### 2.3 Deferred CPU-GPU transfer (transfer at pipeline boundaries only)

**Concept:** The only points where data must cross the PCIe bus are:
1. DICOM decode → GPU (input)
2. GPU → DICOM-SEG encoding (output)

Everything between these boundaries must be zero-copy. This means the `DICOMSeriesToVolumeOperator` output and `DICOMSegmentationWriterOperator` input are the **only** transfer points.

**Implementation:**
- GPU-resident tensor passes through `preprocess_op → infer_op → ensemble_op → postprocess_op`
- Use `cudaMemcpyAsync` with stream-ordered allocator for the input transfer (decode results → GPU)
- Defer output transfer until the final operator; batch it with the DICOM encoding if possible

### 2.4 Multi-model checkpoint sharing on GPU

**Concept:** All 15 checkpoints share the same network architecture (same layer shapes, same parameter counts). Only the weights differ. Loading each checkpoint sequentially means 15 separate `load_state_dict()` calls with CPU→GPU transfers of weights.

**Implementation:**
- Load all checkpoint weights into a single pooled GPU buffer: `weights_pool[N_models, *flat_params]`
- Switch between checkpoints by pointer arithmetic or `torch.narrow` views
- For TTA mirror flips within a single model, the network forward pass reuses the same weights — no extra loading needed
- This also means the network architecture is instantiated once, not 15 times

### 2.5 In-place probability averaging

**Concept:** Instead of keeping N separate probability volumes in memory (one per ensemble model), accumulate in a single buffer.

**Memory savings:** For a 512³ volume with 2 classes and fp16: `512³ × 2 × 2 bytes ≈ 1 GB`. Keeping 15 copies = 15 GB (exceeds 8 GB VRAM). In-place accumulation = 1 GB.

**Implementation:**
- Pre-allocate `accumulation_buffer` as fp32 (to avoid precision loss from repeated fp16 adds)
- After each model: `accumulation_buffer += probs / N_models` (scale by 1/N to avoid overflow)
- Final: convert fp32 accumulation to segmentation via argmax (GPU)

### 2.6 Structured timing with operator-level attribution

**Concept:** Beyond "total latency," the benchmark should attribute time to: preprocessing, per-model inference, ensemble averaging, postprocessing, and I/O. This enables targeted optimization.

**Implementation:**
- Holoscan `RealtimeClock` at operator boundaries
- NVIDIA Nsight Systems markers (`nsight::profiler::setMarker`) for kernel-level detail
- Output structured JSON: `{study_id, total_ms, preprocess_ms, inference_per_model_ms[], ensemble_ms, postprocess_ms, io_ms, gpu_peak_mb}`

### 2.7 Fallback path for OOM conditions

**Concept:** Large volumes (e.g., whole-body CT with 1000+ slices) may exceed GPU memory even with tiling. The pipeline must degrade gracefully.

**Implementation:**
- Detect volume size vs. available VRAM at pipeline entry
- If OOM risk: reduce tile overlap, use fp16 accumulation, or fall back to CPU for accumulation arrays (matching nnUNet's existing fallback pattern: `except RuntimeError: ... perform_everything_on_device=False`)
- Never crash — log warning and use fallback

---

## 3. Anti-Features (Deliberately Avoid)

### 3.1 SDK-level operator changes

**Avoid:** Modifying `monai/deploy/operators/` to add GPU tensor support.

**Rationale:** PROJECT.md explicitly scopes this as "example app only." SDK changes require upstream review, break compatibility, and create merge conflicts. The GPU operator logic lives in the app directory.

### 3.2 Multiprocessing for inference

**Avoid:** nnUNet's current `multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export)` for exporting results.

**Rationale:** Serialization of torch tensors across processes triggers CPU copies. The Holoscan scheduler handles parallelism at the operator level — no need for Python multiprocessing. `predict_from_data_iterator` uses this pattern and it's a major source of the "convert to numpy to prevent uncatchable memory alignment errors" workaround.

### 3.3 File-based probability caching

**Avoid:** Writing intermediate `.npz` or `.pkl` files for probability maps.

**Rationale:** This is the current bottleneck. The 15-checkpoint ensemble writes ~1.5 GB per study. Even with NVMe, this is 500–1000 ms of pure I/O that can be eliminated with in-memory averaging.

### 3.4 Over-customized preprocessing

**Avoid:** Re-implementing nnUNet's preprocessing from scratch in CUDA C++.

**Rationale:** nnUNet's preprocessing is correctness-critical (spacing transposition, cropping, normalization must match training exactly). Use MONAI's GPU transforms (`ResampleToImgd`, `NormalizeIntensityd`) which are well-tested, or wrap nnUNet's preprocessor and only move the data path to GPU. Pixel-equivalence is a hard requirement.

### 3.5 Batch inference

**Avoid:** Processing multiple studies concurrently in the same pipeline.

**Rationale:** PROJECT.md explicitly marks throughput as "Phase 2+". Single-study latency is the target. Adding batch support complicates memory management and operator design.

### 3.6 Triton / remote inference integration

**Avoid:** Adding Triton gRPC calls or model server integration.

**Rationale:** Out of scope per PROJECT.md. In-process inference is simpler and avoids network latency. Triton adds deployment complexity (container orchestration, model repository management) that doesn't help the latency target.

### 3.7 Dynamic operator graph reconstruction per study

**Avoid:** Building the pipeline graph dynamically based on each study's characteristics.

**Rationale:** The graph topology (preprocess → infer × N → ensemble → postprocess) is fixed. Only the parameters (tile size, number of models, output labels) change. Dynamic reconstruction adds Holoscan scheduler overhead and complicates profiling.

### 3.8 Lossy compression of intermediate tensors

**Avoid:** Quantizing probability maps to int8 or using JPEG compression for intermediate results.

**Rationale:** Pixel-exact equivalence is required. Any compression introduces numerical artifacts that break the bit-for-bit comparison with the reference app.

### 3.9 Async I/O for DICOM writing during inference

**Avoid:** Starting DICOM-SEG encoding before the full segmentation is ready.

**Rationale:** DICOM-SEG requires complete volumetric segmentation. Streaming DICOM-SEG output requires frame-level completeness which conflicts with the ensemble averaging step that needs all model outputs. The DICOM write must be the final sequential step.

---

## 4. Summary Matrix

| Feature | Category | Priority | Effort | Impact |
|---------|----------|----------|--------|--------|
| Zero-copy GPU data movement | Table Stake | P0 | High | High — eliminates PCIe roundtrips |
| Tile-based inference on GPU | Table Stake | P0 | Medium | High — nnUNet already tiles, need to integrate |
| In-memory ensemble averaging | Table Stake | P0 | Medium | High — eliminates 1.5 GB disk I/O |
| GPU preprocessing | Table Stake | P1 | High | Medium — resampling is hard, normalization is easy |
| GPU connected components | Table Stake | P1 | Medium | Medium — Holoscan `SegmentationPostprocessorOp` available |
| Profiling infrastructure | Table Stake | P1 | Low | High — needed to validate improvements |
| Double-buffered pipeline overlap | Differentiator | P2 | High | High — hides latency across stages |
| Pre-allocated GPU memory pool | Differentiator | P1 | Medium | Medium — eliminates allocation churn |
| Deferred CPU-GPU transfers | Differentiator | P1 | Low | Medium — boundary-only transfers |
| Multi-model weight sharing | Differentiator | P2 | High | Medium — reduces 15× weight loads |
| In-place probability averaging | Differentiator | P1 | Low | High — saves 14 GB VRAM |
| Operator-level timing | Differentiator | P1 | Low | High — enables optimization targeting |
| OOM fallback | Differentiator | P2 | Medium | Low — robustness, not performance |

---

*Derived from codebase analysis: `cchmc_nnunet_fifteen_ckpt_app`, nnUNet vendored inference modules, Holoscan SDK 4.2 capabilities, and MONAI Deploy App SDK operator architecture.*
