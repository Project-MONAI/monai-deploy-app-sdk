# PITFALLS: Holoscan-Native nnUNet Inference

Research dimension for `cchmc-nnunet-fast` — documented pitfalls when moving nnUNet inference from the Python wrapper path to Holoscan-native GPU operators.

---

## 1. CPU↔GPU Data Transfer Pitfalls

### 1.1 Silent CPU fallback erases performance gains

**What goes wrong:** nnUNet's `nnUNetPredictor` already has a `perform_everything_on_device` flag and a fallback path — when GPU OOM occurs, `_internal_predict_sliding_window_return_logits` catches `RuntimeError` and silently re-runs on CPU. In a Holoscan operator that assumes GPU-resident buffers, this fallback is invisible. The operator completes "successfully" but runs on CPU, producing no latency improvement.

**Warning signs:**
- End-to-end timing doesn't improve despite GPU operators
- `torch.cuda.memory_allocated()` is lower than expected
- Nsight Systems shows the kernel launch region is much smaller than the total compute time
- `results_device` silently switches from `cuda` to `cpu` in the exception handler (line ~495 of `predict_from_raw_data.py`)

**Prevention:**
- Explicitly assert that `results_device.type == 'cuda'` in every operator's compute path
- Never swallow `RuntimeError` from GPU kernels — surface it as an operator-level failure
- Log the device of every tensor at operator boundaries: `assert tensor.device.type == 'cuda'`
- Phase: **Design** — write device-invariant test assertions before implementation

### 1.2 The hidden cost of `.numpy()` / `torch.from_numpy()` round-trips

**What goes wrong:** The current pipeline calls `prediction.cpu().detach().numpy()` after every ensemble member's forward pass (line ~374 of `predict_from_raw_data.py`). This is a synchronous PCIe transfer. If your Holoscan operators replicate this pattern instead of keeping accumulators on GPU, you lose the zero-copy benefit entirely. Even worse: `torch.from_numpy()` on a non-pinned CPU buffer creates a *copy*, not a view.

**Warning signs:**
- Profiler shows large `cudaMemcpyHtoD` / `cudaMemcpyDtoH` segments
- GPU utilization graph shows long idle gaps between kernel bursts
- Memory bandwidth is dominated by PCIe transfers, not DRAM accesses

**Prevention:**
- Accumulate ensemble logits on GPU across all folds, transfer to CPU only once at the end
- Use `pin_memory=True` on any CPU tensor destined for GPU (Holoscan's `MemoryData` should handle this natively)
- Audit every `.numpy()`, `.cpu()`, `torch.from_numpy()` call — each is a potential data transfer
- Phase: **Implementation** — add a lint rule or pre-commit hook flagging cross-device calls

### 1.3 Assuming Holoscan `MemoryData` is GPU-contiguous

**What goes wrong:** Holoscan's `MemoryData` can wrap either CPU or GPU memory. When you receive a `MemoryData` from a preceding operator, its `map()` returns a CPU view even if the underlying buffer was allocated on GPU. The data must be explicitly mapped to the GPU domain via `MemoryData.to_device()` or by using `gpu_memory()` APIs. If you skip this, you get a CPU buffer and the operator silently processes on CPU.

**Warning signs:**
- `MemoryData.device_type()` returns `DeviceType::CPU` when GPU was expected
- Operator runs without error but Nsight shows no custom kernels
- Shape/values are correct but timing matches the CPU baseline

**Prevention:**
- At the top of every `compute()` method, assert `input_memory.device_type() == DeviceType::GPU`
- If the upstream operator may produce CPU memory, add an explicit `copy_to_device()` step with timing
- Phase: **Implementation** — write a small harness that feeds CPU and GPU MemoryData through every operator and asserts the device type

---

## 2. Numerical Equivalence Pitfalls

### 2.1 Resampler implementation differences are not cosmetic

**What goes wrong:** nnUNet's preprocessing uses `skimage.transform.resize` for data (order=3, anti_aliasing=False) and `batchgenerators.augmentations.utils.resize_segmentation` for labels (nearest-neighbor). When you move resampling to the GPU, common choices are CUDA-based grid-sample or cuSignal. Even with the same "trilinear" label, these produce different interpolation weights. The difference accumulates through normalization → resampling → inference → inverse-resampling, and the final segmentation diverges.

**Critical detail from the codebase:** `default_resampling.py` has a `do_separate_z` path that does 2D reslicing per-slice then 1D interpolation along Z — this is fundamentally non-separable and hard to replicate exactly on GPU. The `map_coordinates` call with `order=order_z` on a hand-constructed coordinate grid (lines 98-128 of `default_resampling.py`) is particularly fragile to reproduce.

**Warning signs:**
- Dice scores match (0.98) but pixel-level comparison fails
- `np.allclose()` passes with `rtol=1e-5` but not with `rtol=0`
- Differences cluster at tissue boundaries and in low-intensity regions

**Prevention:**
- **Do not reimplement resampling.** Use the exact same `skimage.resize` / `map_coordinates` calls. Resampling is not the compute bottleneck — inference is.
- If GPU resampling is mandatory for latency: validate against the CPU path on a representative corpus with `np.array_equal()` on the final segmentation mask (not just logits)
- Store the reference output (pre-optimization) and diff every test run
- Phase: **Design** — identify which preprocessing steps are "numerically critical" vs. "tolerable to approximate"

### 2.2 Float16 autocast changes the computation graph

**What goes wrong:** nnUNet wraps inference in `torch.autocast(device_type='cuda', enabled=True)` (line ~517 of `predict_from_raw_data.py`). Inside autocast, convolutions run in FP16 and reductions in FP32. When you extract individual operators or change the kernel composition (e.g., fusing normalization into the first conv), the autocast policy may apply differently, changing which operations use FP16. This produces different rounding.

**Warning signs:**
- Outputs differ by ~1e-4 even with identical inputs and model weights
- The difference is stable (deterministic) — not random
- Disabling autocast makes outputs match but kills performance

**Prevention:**
- Keep the autocast context manager at the outermost inference scope — don't split it across operators
- If you must split: explicitly set dtype on every intermediate tensor (`tensor.to(torch.float16)` or `.to(torch.float32)`)
- Validate with autocast ON for both reference and optimized paths
- Phase: **Implementation** — don't change autocast boundaries; accept the FP16 rounding as the reference

### 2.3 Gaussian importance map rounding in sliding window

**What goes wrong:** `compute_gaussian` in `sliding_window_prediction.py` builds the Gaussian on CPU with `scipy.ndimage.gaussian_filter`, converts to `torch.float16`, then applies `value_scaling_factor=10`. The FP16 truncation of the Gaussian is cached via `@lru_cache(maxsize=2)`. If your GPU operator recomputes the Gaussian (e.g., using `torch.exp` of a quadratic), rounding at the half-precision boundary produces different weights. Since the Gaussian divides accumulated logits (`torch.div(predicted_logits, n_predictions, out=predicted_logits)`), even tiny weight differences affect boundary voxels.

**Warning signs:**
- Divergence concentrated at tile boundaries (visible as grid artifacts in the probability map)
- Increasing `tile_step_size` toward 1.0 (fewer tiles) reduces divergence
- Switching Gaussian dtype to FP32 makes outputs match the reference

**Prevention:**
- Precompute the Gaussian once in FP32, cast to the target dtype, and pass it as operator input (don't recompute per-volume)
- Use the exact same `scipy.ndimage.gaussian_filter` path to generate the weight map, then upload to GPU
- Phase: **Implementation** — treat the Gaussian as model metadata, not a computed value

### 2.4 Mirror-augmentation (TTA) accumulation order matters

**What goes wrong:** nnUNet's TTA (`_internal_maybe_mirror_and_predict`) averages `2^N + 1` forward passes (original + all mirror axis combinations). Each pass adds to a running sum. FP16 addition is non-associative: `(a + b) + c ≠ a + (b + c)`. If your GPU operator changes the order of accumulation (e.g., using a parallel reduction tree instead of sequential `+=`), results diverge.

**Warning signs:**
- Divergence is stable per-input but unpredictable in magnitude
- Disabling mirroring (`use_mirroring=False`) eliminates divergence
- Divergence scales with the number of mirror axes (3 axes → 8× more terms)

**Prevention:**
- Preserve the exact accumulation order: sequential `+=` on the full-tensor, not element-wise parallel reduction
- If parallel accumulation is necessary: use FP32 for the accumulator, FP16 only for the forward pass
- Phase: **Implementation** — keep TTA as a single monolithic loop; don't parallelize the averaging

### 2.5 Normalization must precede resampling (order is enforced)

**What goes wrong:** The nnUNet preprocessor has an explicit comment: "normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no longer fitting the images perfectly!" (line 53 of `default_preprocessor.py`). If a Holoscan operator pipeline reorders steps for pipelining (e.g., resample all channels in parallel, then normalize), the crop mask and normalization interact differently.

**Warning signs:**
- Segmentation mask has artifacts near the crop boundary
- Intensity values in the resampled volume don't match expected range
- The issue appears only for volumes with significant cropping (small anatomy in large FOV)

**Prevention:**
- Enforce the ordering: transpose → crop → normalize → resample as a strict sequence
- In the DAG, add data dependencies that prevent out-of-order execution
- Phase: **Design** — document this ordering as an immutable constraint in the operator spec

### 2.6 Ensemble averaging of probabilities vs. logits

**What goes wrong:** The current pipeline saves `.npz` probability files per ensemble member, then loads and averages them with `average_probabilities()` from `nnunetv2.ensembling.ensemble`. If the optimized pipeline averages *logits* (pre-softmax) instead of *probabilities* (post-softmax), results differ. nnUNet's `average_probabilities` applies the label manager's `apply_inference_nonlin` (softmax or sigmoid depending on region-based training) *before* averaging.

**Warning signs:**
- Dice changes systematically across labels (some go up, some down)
- The difference is most pronounced for rare labels with low probabilities
- The issue disappears when there's only one ensemble member

**Prevention:**
- Replicate the exact `average_probabilities` path: softmax → average → convert to segmentation
- If averaging logits on GPU for performance: validate pixel-exact equivalence against the probability-averaging path
- Phase: **Implementation** — unit test the ensemble operator with a 2-member synthetic ensemble

---

## 3. GPU Memory Management Pitfalls

### 3.1 Sliding window tile pre-allocation causes silent OOM

**What goes wrong:** `predict_sliding_window_return_logits` pre-allocates `predicted_logits` (dtype=half) and `n_predictions` arrays at the full resampled volume size (line ~485). For a 3D volume of shape (1, 512, 512, 384) with 20 segmentation heads, this is `20 × 512 × 512 × 384 × 2 bytes ≈ 7.8 GB`. Combined with the network weights, the input tile, and the Gaussian, this easily exceeds 8 GB. The fallback to CPU hides this, producing correct but slow results.

**Warning signs:**
- First volume succeeds, second fails (fragmentation)
- `torch.cuda.memory_allocated()` is close to total VRAM before inference starts
- OOM only appears for large studies (e.g., full-body CT at 0.5mm spacing)

**Prevention:**
- Compute the memory budget before allocation: `logits_mem + n_pred_mem + tile_mem + model_mem < 0.85 * total_vram`
- If over budget: reduce `tile_step_size` (larger strides = fewer overlapping tiles), or process slices independently
- Use `torch.cuda.empty_cache()` between ensemble members (nnUNet already does this)
- Phase: **Design** — add a memory budget calculator that takes volume shape + model config as input

### 3.2 Multiple ensemble members multiply memory needs

**What goes wrong:** The current pipeline runs each ensemble member sequentially, writing results to disk between them. The optimized version that keeps everything in memory must hold *all* probability maps simultaneously if ensemble averaging is deferred. For 3 ensemble members × 20 channels × full volume, this triples the pre-allocation from pitfall 3.1.

**Warning signs:**
- OOM appears only when ensemble size ≥ 3
- OOM appears at the averaging step, not during inference
- Reducing ensemble size to 1 eliminates OOM

**Prevention:**
- Average incrementally: keep only the running sum and count on GPU, don't store per-member maps
- Process ensemble members one at a time: softmax → accumulate into running sum → discard
- Phase: **Design** — design the ensemble operator for incremental averaging from day one

### 3.3 Cascaded models (3d_lowres → 3d_cascade_fullres) double the peak memory

**What goes wrong:** Cascade configurations require the lowres segmentation as additional input channels to the fullres network. The `3d_lowres` output must be one-hot encoded and concatenated with the input data. This means both the lowres probability map AND the fullres input tile must be resident simultaneously.

**Warning signs:**
- OOM only on cascade configurations (works fine for single-stage models)
- Error appears during the one-hot encoding step, not the forward pass
- The lowres output shape is smaller but the one-hot expansion can be large for many labels

**Prevention:**
- One-hot encode on GPU using sparse representations (don't materialize the full dense tensor if label count is high)
- Free the lowres network weights immediately after its inference completes (`del predictor.network`)
- Phase: **Implementation** — test cascade configs early; they are the memory stress test

### 3.4 CUDA memory fragmentation kills long-running pipelines

**What goes wrong:** Holoscan operators that allocate and free GPU memory on every `compute()` call (per-study) fragment the CUDA heap. After 50-100 studies, allocation of a single large contiguous block (e.g., the pre-allocated logits array) fails with OOM even though total free memory is sufficient. This is specific to the CUDA allocator's behavior — it doesn't compact.

**Warning signs:**
- `nvidia-smi` shows 4 GB free but allocation of 2 GB fails
- First 50 studies succeed, then intermittent failures
- Restarting the process fixes it

**Prevention:**
- Pre-allocate operator buffers in `setup()` and reuse across `compute()` calls
- Use `torch.cuda.caching_allocator()` with a fixed capacity
- Call `torch.cuda.empty_cache()` only between studies, not between operators
- Phase: **Implementation** — use a memory pool pattern; never allocate in `compute()`

---

## 4. Holoscan Operator Lifecycle Pitfalls

### 4.1 Model loading in `compute()` is a latency landmine

**What goes wrong:** The existing `NNUnetSegOperator.compute()` calls `_load_nnunet_models()` on every invocation (line ~208 of `nnunet_seg_operator.py`). Model loading (checkpoint deserialization + network construction) takes 10-30 seconds. In a Holoscan pipeline where `compute()` should be the fast hot path, this turns every study into a cold-start event.

**Warning signs:**
- First call to `compute()` takes 20+ seconds, subsequent calls are fast (if cached)
- In production, every study pays the cold-start cost
- CPU spikes during model loading block other operators

**Prevention:**
- Load models in `setup()` or `on_insert()` — never in `compute()`
- For ensemble models: pre-initialize all predictors during setup
- Cache the `PlansManager`, `ConfigurationManager`, and `LabelManager` as operator instance variables
- Phase: **Design** — enforce the "setup = slow, compute = fast" discipline by code review

### 4.2 Holoscan `setup()` runs once, `compute()` runs many times — mutable state leaks

**What goes wrong:** nnUNet's predictor has mutable state: `self.network` device placement, `self.list_of_parameters`, thread count (`torch.set_num_threads()`). If `compute()` modifies these (e.g., changes the number of threads, moves the network to a different device), subsequent calls to `compute()` inherit the mutated state. This produces correct results for the first study and incorrect results for the second.

**Warning signs:**
- First volume produces correct output, second volume has artifacts
- Thread count drifts (starts at default, ends at 1)
- Device placement changes between calls

**Prevention:**
- Save and restore mutable state at the boundaries of `compute()` (nnUNet's `predict_logits_from_preprocessed_data` already saves/restores thread count — replicate this)
- Mark predictor methods that mutate state and wrap them in context managers
- Phase: **Implementation** — write a test that calls `compute()` twice in a row and diffs outputs

### 4.3 Operator shutdown doesn't free CUDA context

**What goes wrong:** When a Holoscan pipeline tears down, the CUDA context and its allocations may not be released if the operator holds references to CUDA tensors. This matters in a long-running service where pipelines are created/destroyed per request. Each cycle leaks ~500 MB of VRAM.

**Warning signs:**
- VRAM usage grows monotonically across pipeline restarts
- `nvidia-smi` shows the process holding memory after Python exit
- `del predictor` doesn't free memory (circular references in the state dict)

**Prevention:**
- Implement an explicit `on_remove()` or destructor that calls `empty_cache(self.device)` and deletes large tensors
- Break circular references: `self.network = None`, `self.list_of_parameters = []`
- Phase: **Implementation** — add explicit cleanup; test with a loop that creates/destroys pipelines

### 4.4 DataStore messages carry CPU numpy arrays, not GPU tensors

**What goes wrong:** MONAI Deploy's `DataStore` (the mechanism operators use to pass data) serializes data as Python objects. CUDA tensors are not picklable. If your GPU operator emits a tensor directly to the DataStore, it will either fail at serialization or silently fall back to `.cpu().numpy()`, adding an unintended data transfer.

**Warning signs:**
- `pickle.PicklingError` on CUDA tensors
- Serialization time is unexpectedly long (implicit `.cpu()` + copy)
- Downstream operator receives CPU array when GPU was expected

**Prevention:**
- Convert to `MemoryData` explicitly before emitting (this makes the CPU↔GPU transfer intentional and measurable)
- Use `holoscan.core.MemoryData` with `DeviceType::GPU` for GPU-to-GPU operator chains
- Phase: **Design** — document the DataStore serialization contract and its GPU implications

---

## 5. nnUNet-Specific Gotchas

### 5.1 Tiling step size is not free to change

**What goes wrong:** `tile_step_size=0.5` means 50% overlap between adjacent tiles. Halving it to 0.25 for "speed" doubles the number of forward passes and the Gaussian-weighted overlap zone, producing different boundary artifacts. nnUNet's validation postprocessing was tuned for the default 0.5.

**Warning signs:**
- Grid-pattern artifacts at tile boundaries in the output
- Dice drops for small structures near tile edges
- Changing step size changes the output (it shouldn't, theoretically, but numerical rounding makes it so)

**Prevention:**
- Do not change `tile_step_size` from 0.5 — it is part of the model's "signature"
- If tiling strategy must change for GPU efficiency: retrain or re-validate on the target corpus
- Phase: **Design** — treat `tile_step_size` as a model hyperparameter, not a performance knob

### 5.2 Padding-to-tile-size changes the effective field of view

**What goes wrong:** `pad_nd_image` pads the input to at least `patch_size` in each dimension (line ~528). For volumes smaller than the patch size (e.g., thin 2D slices fed to a 3D model, or small ROIs), padding adds zeros that the network processes. The `slicer_revert_padding` at the end crops them back. If the GPU operator changes the padding strategy (e.g., reflects instead of constant-zero), or if the revert slice is off by one, output values change.

**Warning signs:**
- Outputs differ only for small volumes (< patch_size in any dimension)
- Boundary voxels have incorrect values
- 2D configuration works fine, 3D configuration diverges

**Prevention:**
- Use the exact same `pad_nd_image` from `acvl_utils` — don't reimplement padding
- Validate the `slicer_revert_padding` round-trip: `padded[slicer_revert_padding].shape == original.shape`
- Phase: **Implementation** — unit test padding with edge-case shapes

### 5.3 Cascade stage dependency is a data serialization trap

**What goes wrong:** The `3d_cascade_fullres` configuration requires the `3d_lowres` segmentation as input. In the current pipeline, this flows through a file on disk (`rw.read_seg(seg_file)`). In the Holoscan pipeline, the lowres output must be passed in-memory to the fullres operator. The one-hot encoding step (`convert_labelmap_to_one_hot`) changes the channel count, and the tensor must be resampled to the fullres spacing before concatenation.

**Warning signs:**
- Shape mismatch at the concatenation step (lowres one-hot shape ≠ fullres input shape)
- The cascade output is identical to fullres-only output (lowres input was silently dropped)
- Memory usage spikes at the concatenation point

**Prevention:**
- Design the cascade as two operators with an explicit DataStore connection between them
- Validate that the one-hot channels are appended (not prepended) — nnUNet expects `data = np.vstack((data, seg_onehot))`
- Phase: **Design** — document the cascade data contract between lowres and fullres operators

### 5.4 Postprocessing (`postprocessing.pkl`) is configuration-specific and non-trivial

**What goes wrong:** nnUNet's learned postprocessing (`postprocessing.pkl`) contains rules like "remove connected components smaller than X voxels for label Y". The `apply_postprocessing` function applies these rules sequentially. If the GPU operator approximates connected-component analysis (e.g., uses a different algorithm or different connectivity — 6-neighbor vs 26-neighbor), results diverge.

**Warning signs:**
- Small structures disappear or appear inconsistently
- Dice is high overall but drops for rare/small labels
- The postprocessing step removes more (or less) than the reference

**Prevention:**
- Use the exact same `apply_postprocessing` from `nnunetv2.postprocessing` — don't reimplement
- If GPU connected-components is used: validate against the CPU implementation on a test corpus
- Connectivity must match: nnUNet uses 26-neighbor (3D) / 8-neighbor (2D)
- Phase: **Implementation** — treat postprocessing as a "run the reference code" step, not an optimization target

### 5.5 Region-based training changes the output interpretation

**What goes wrong:** Some nnUNet models use "region-based" label handling instead of per-class segmentation. The `label_manager` in `PlansManager` determines whether `convert_logits_to_segmentation` does argmax (per-class) or threshold-and-combine (regions). If the GPU operator assumes argmax for all models, region-based models produce wrong outputs.

**Warning signs:**
- Output has unexpected label values (e.g., label 255 for background in a region model)
- The issue appears only for specific datasets, not all
- `label_manager.has_regions` is True but the operator doesn't check it

**Prevention:**
- Query `label_manager.has_regions` and dispatch to the correct conversion path
- Use `label_manager.convert_logits_to_segmentation()` directly — it handles both cases
- Phase: **Implementation** — never assume the label scheme; read it from `dataset.json`

---

## 6. Profiling and Measurement Pitfalls

### 6.1 Measuring only kernel time ignores data movement

**What goes wrong:** Nsight Compute profiles individual kernel execution time. The "optimized" GPU operator may show a 5× faster kernel, but if 60% of the wall-clock time is data transfer (DICOM decode → CPU numpy → GPU tensor) and postprocessing export (GPU tensor → CPU numpy → DICOM-SEG write), the end-to-end improvement is 1.2×. Optimizing the kernel is the wrong lever.

**Warning signs:**
- Kernel time improved dramatically but total latency barely changed
- Nsight Systems timeline shows long gaps (PCIe transfers) between kernel regions
- GPU utilization is < 50% because it's waiting on data

**Prevention:**
- Profile the *entire* operator: from `compute()` entry to `op_output.emit()` exit
- Use Nsight Systems (not Compute) for end-to-end profiling — it shows PCIe, CPU, and GPU together
- Identify the critical path: is it data transfer, kernel execution, or serialization?
- Phase: **Validation** — profile before optimizing; optimize the bottleneck, not the kernel

### 6.2 First-call cold start contaminates benchmarks

**What goes wrong:** The first call to any GPU operator triggers CUDA context creation (~200ms), cuDNN benchmark tuning (~50ms per layer), and memory allocation. Subsequent calls are faster. If you average timings across N calls including the first, the benchmark is pessimistic. If you exclude the first call, the benchmark is optimistic compared to production (where each study may be a cold start).

**Warning signs:**
- Timing varies by 10× between runs
- Benchmark result changes when you add a "warm-up" call
- Production latency is worse than benchmark latency

**Prevention:**
- Report both cold-start and warm-start timings
- Match the benchmark protocol to the production deployment (one-shot vs. streaming)
- Phase: **Validation** — define the measurement protocol before collecting data

### 6.3 Holoscan pipeline overhead is invisible to PyTorch profilers

**What goes wrong:** `torch.profiler` only sees PyTorch operations. Holoscan's operator scheduling, message passing, and DataStore serialization happen outside PyTorch's visibility. A "fast" PyTorch operator may be bottlenecked by Holoscan's DAG scheduler waiting for upstream operators.

**Warning signs:**
- PyTorch profiler shows the operator should be fast but wall-clock says otherwise
- Holoscan's internal traces show the operator is idle waiting for input
- Adding `time.sleep(0)` (yielding) before the operator changes timing

**Prevention:**
- Use Holoscan's built-in tracing (if available) or manual timing at operator boundaries
- Profile the DAG, not just individual operators
- Phase: **Validation** — instrument operator entry/exit with timestamps

### 6.4 Comparing "disk I/O path" to "GPU path" without controlling for cache

**What goes wrong:** The current pipeline writes `.npz` files to disk and reads them back. On SSDs, the OS page cache may serve the read from RAM, making the "disk" path artificially fast. The GPU path (which bypasses disk) looks slower in comparison because it does real computation. Conversely, if the GPU path's output is compared against a cold-disk read, the comparison is inverted.

**Warning signs:**
- Benchmark result flips when you clear the page cache (`echo 3 > /proc/sys/vm/drop_caches`)
- Second run is faster than first (cache effect)
- Disk I/O metrics show near-zero writes during "slow" benchmark

**Prevention:**
- Clear OS caches before each benchmark run
- Use `sysbench` or similar to confirm I/O is not cached
- Compare equivalent paths: both cold, both warm, or both with cache cleared
- Phase: **Validation** — define the cache state as part of the benchmark protocol

---

## Quick Reference: Pitfall Matrix

| # | Pitfall | Severity | Phase to Address | Key Prevention |
|---|---------|----------|-----------------|----------------|
| 1.1 | Silent CPU fallback | Critical | Design | Assert GPU device everywhere |
| 1.2 | Hidden `.numpy()` transfers | High | Implementation | Audit cross-device calls |
| 1.3 | `MemoryData` device confusion | High | Implementation | Assert device type at operator entry |
| 2.1 | Resampler differences | Critical | Design | Don't reimplement; use reference |
| 2.2 | Autocast changes graph | Medium | Implementation | Keep autocast scope unchanged |
| 2.3 | Gaussian rounding | Medium | Implementation | Precompute from reference path |
| 2.4 | TTA accumulation order | High | Implementation | Sequential `+=`, not parallel reduce |
| 2.5 | Normalize→resample order | Critical | Design | Enforce as DAG dependency |
| 2.6 | Probability vs. logit averaging | Critical | Implementation | Replicate reference path exactly |
| 3.1 | Tile pre-allocation OOM | High | Design | Memory budget calculator |
| 3.2 | Ensemble × memory multiply | High | Design | Incremental averaging |
| 3.3 | Cascade doubles peak memory | Medium | Implementation | Free lowres weights early |
| 3.4 | CUDA fragmentation | Medium | Implementation | Memory pool, allocate in setup() |
| 4.1 | Model loading in compute() | Critical | Design | Load in setup(), never compute() |
| 4.2 | Mutable state leaks | High | Implementation | Save/restore state, test double-call |
| 4.3 | CUDA context leaks | Medium | Implementation | Explicit on_remove() cleanup |
| 4.4 | DataStore CPU serialization | High | Design | Use MemoryData with GPU device type |
| 5.1 | Changing tile_step_size | Critical | Design | Treat as immutable hyperparameter |
| 5.2 | Padding strategy changes | Medium | Implementation | Use reference pad_nd_image |
| 5.3 | Cascade data serialization | High | Design | Explicit DataStore connection |
| 5.4 | Postprocessing algorithm diff | High | Implementation | Run reference postprocessing code |
| 5.5 | Region-based label handling | Critical | Implementation | Query has_regions, use label_manager |
| 6.1 | Kernel time ≠ wall-clock time | High | Validation | Profile end-to-end with Nsight Systems |
| 6.2 | Cold start in benchmarks | Medium | Validation | Report cold + warm timings |
| 6.3 | Holoscan overhead invisible to PT | Medium | Validation | Instrument operator boundaries |
| 6.4 | Disk cache skews comparison | Medium | Validation | Clear caches, define protocol |

---

*Research compiled: 2025-07-20 | Source analysis: nnUNet v2 vendored codebase, MONAI Deploy App SDK operators, Holoscan SDK patterns*
