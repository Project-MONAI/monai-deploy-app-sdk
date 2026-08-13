# Architecture Research: Holoscan-Native nnUNet Inference Pipeline

## 1. Problem Statement

The current `cchmc_nnunet_fifteen_ckpt_app` wraps nnUNet as a monolithic Python call inside `NNUnetSegOperator`. This causes:
- **CPU bounce**: `DICOMSeriesToVolumeOperator` outputs CPU numpy → nnUNet works on CPU for preprocessing → GPU for forward pass → CPU for export
- **Disk I/O in the critical path**: Probability maps written to `.npz` files on disk, then read back for ensemble averaging
- **No pipeline parallelism**: The entire ensemble (all configs, all folds) runs synchronously in a single operator's `compute()`
- **Cascade dependency on disk**: `3d_lowres` writes `.nii.gz` → `3d_cascade_fullres` reads it back from disk
- **Holoscan unused**: GPU memory pools, async execution, and streaming are all idle

## 2. nnUNet Inference Deconstruction

Understanding nnUNet's internals is prerequisite to decomposing it into operators.

### 2.1 What nnUNet does end-to-end (for a single config)

```
Raw numpy array (C, Z, Y, X) in SimpleITK axis order
  │
  ├── PREPROCESSING (CPU)
  │     1. Transpose axes (transpose_forward from plans.json)
  │     2. Crop to nonzero bbox (stores bbox + shape_before_cropping in properties)
  │     3. Normalize per-channel (e.g., CT → mean=0/std=1 using training stats)
  │     4. Resample to config spacing (e.g., 0.8mm³ → config-specific)
  │     5. For cascade: stack one-hot seg from prev stage onto channels
  │
  │   Output: torch.Tensor (C', Z', Y', X') at config resolution
  │
  ├── SLIDING-WINDOW INFERENCE (GPU)
  │     1. Pad image to tile_size if smaller
  │     2. Compute sliding window steps (tile_step_size=0.5 by default)
  │     3. For each tile:
  │        a. Copy tile to GPU
  │        b. Forward pass through network
  │        c. TTA mirroring (if enabled): flip along allowed axes → forward → flip back → average
  │        d. Gaussian-weighted accumulation into output buffer
  │        e. Divide by Gaussian overlap counts
  │     4. Revert padding
  │
  │   Output: logits tensor (num_classes, Z', Y', X') at config resolution
  │
  ├── POST-RESAMPLING (CPU/GPU)
  │     1. Resample logits back to original image spacing
  │     2. Apply softmax/sigmoid (label_manager.apply_inference_nonlin)
  │     3. Convert probabilities → segmentation (argmax per voxel)
  │     4. Insert segmentation back into crop bbox
  │     5. Transpose axes back (transpose_backward from plans.json)
  │
  │   Output: segmentation (Z_orig, Y_orig, X_orig) uint8
  │
  └── EXPORT (CPU, to disk)
        1. Write segmentation as .nii.gz
        2. Write probabilities as .npz (save_probabilities=True)
        3. Write properties as .pkl
```

### 2.2 Multi-fold averaging

nnUNet's `predict_logits_from_preprocessed_data` iterates over `list_of_parameters` (one per fold), loading weights each time and accumulating logits on CPU:

```python
for params in self.list_of_parameters:
    network.load_state_dict(params)
    if prediction is None:
        prediction = predict_sliding_window_return_logits(data).to('cpu')
    else:
        prediction += predict_sliding_window_return_logits(data).to('cpu')
prediction /= len(self.list_of_parameters)
```

### 2.3 Ensemble averaging

`average_probabilities()` from `nnunetv2/ensembling/ensemble.py`:
1. Loads `.npz` files from disk for each config
2. Element-wise average: `avg += np.load(f)['probabilities']` then `avg /= len(files)`
3. `label_manager.convert_logits_to_segmentation(avg_probs)` → argmax

### 2.4 Postprocessing

`apply_postprocessing()` from `nnunetv2/postprocessing/remove_connected_components.py`:
1. Loads `postprocessing.pkl` → `(pp_fns, pp_fn_kwargs)` tuples
2. For each (fn, kwargs): `segmentation = fn(segmentation, **kwargs)`
3. Typical fn: `remove_all_but_largest_component_from_segmentation`

## 3. Natural Operator Boundaries

Based on the nnUNet deconstruction, these are the natural decomposition points:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         HOLOSCAN OPERATOR PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  [DICOM I/O — existing, unchanged]                                              │
│  DICOMDataLoader → DICOMSeriesSelector → DICOMSeriesToVolume                   │
│                                             │                                    │
│                                             ▼                                    │
│                                  ┌──────────────────────┐                       │
│                                  │ ① PreprocessOperator │                       │
│                                  │ (CPU→GPU, per-config) │                      │
│                                  │ - transpose          │                       │
│                                  │ - crop               │                       │
│                                  │ - normalize          │                       │
│                                  │ - resample           │                       │
│                                  │ Output: GPU tensor   │                       │
│                                  │   (C', Z', Y', X')   │                       │
│                                  └──────────┬───────────┘                       │
│                                             │                                    │
│                                             ▼                                    │
│                                  ┌──────────────────────┐                       │
│                                  │ ② SlideWindowOperator│                       │
│                                  │ (GPU inference)      │                       │
│                                  │ - tile computation   │                       │
│                                  │ - forward + TTA      │                       │
│                                  │ - gaussian weighting │                       │
│                                  │ - fold averaging     │                       │
│                                  │ Output: GPU logits   │                       │
│                                  │   (K, Z', Y', X')    │                       │
│                                  └──────────┬───────────┘                       │
│                                             │                                    │
│                                             ▼                                    │
│                                  ┌──────────────────────┐                       │
│                                  │ ③ PostResampleOperator│                      │
│                                  │ (GPU→original space) │                       │
│                                  │ - resample probs     │                       │
│                                  │ - softmax/sigmoid    │                       │
│                                  │ - logits→seg         │                       │
│                                  │ - revert crop        │                       │
│                                  │ - revert transpose   │                       │
│                                  │ Output: GPU tensor   │                       │
│                                  │   (Z, Y, X) uint8    │                       │
│                                  └──────────┬───────────┘                       │
│                                             │                                    │
│                           ┌─────────────────┼─────────────────┐                 │
│                           │                 │                 │                 │
│                           ▼                 ▼                 ▼                 │
│                    ┌────────────┐  ┌────────────┐  ┌────────────┐              │
│                    │ Config A   │  │ Config B   │  │ Config C   │              │
│                    │ pre→slide  │  │ pre→slide  │  │ (cascade   │              │
│                    │ →post      │  │ →post      │  │  needs B   │              │
│                    │ (parallel) │  │ (parallel) │  │  output)   │              │
│                    └─────┬──────┘  └─────┬──────┘  └─────┬──────┘              │
│                          │               │               │                      │
│                          └───────┬───────┘───────┬───────┘                      │
│                                  ▼               ▼                              │
│                          ┌─────────────────────────────────┐                    │
│                          │ ④ EnsembleAverageOperator       │                    │
│                          │ (GPU, in-memory)                │                    │
│                          │ - receive prob maps from all    │                    │
│                          │   non-aux configs               │                    │
│                          │ - element-wise mean on GPU      │                    │
│                          │ - softmax + argmax → seg        │                    │
│                          │ Output: GPU seg (Z, Y, X)       │                    │
│                          └────────────────┬────────────────┘                    │
│                                           │                                     │
│                                           ▼                                     │
│                                  ┌──────────────────────┐                       │
│                                  │ ⑤ PostprocessOperator│                       │
│                                  │ (GPU)                │                       │
│                                  │ - connected component │                       │
│                                  │ - label filtering     │                       │
│                                  │ Output: GPU seg       │                       │
│                                  └──────────┬───────────┘                       │
│                                             │                                    │
│                                             ▼                                    │
│  [DICOM I/O — existing, unchanged]                                              │
│  DICOMSegWriter ← GPU seg tensor                                                 │
│  DICOMSRWriter  ← volume measurements                                            │
│  DICOMSCWriter  ← overlay images                                                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Operator Specifications

#### ① PreprocessOperator

**Purpose**: Convert raw volume to nnUNet-ready GPU tensor for a specific config.

**Input**:
- `volume`: GPU tensor from `DICOMSeriesToVolumeOperator` (CPU numpy → needs GPU transfer)
- `config_name`: String identifying the nnUNet configuration (e.g., "3d_fullres")
- `plans_config`: PlansManager + ConfigurationManager (shared reference, loaded once in setup)
- `dataset_json`: Dataset metadata (shared reference)
- `prev_stage_seg` (optional): GPU segmentation from previous cascade stage

**Output**:
- `preprocessed`: GPU tensor (C', Z', Y', X') float32, contiguous format
- `properties`: Dict with `spacing`, `shape_before_cropping`, `bbox_used_for_cropping`, `shape_after_cropping_and_before_resampling`, `transpose_forward`/`backward`

**Key design decisions**:
- One instance per config. Each config has different spacing, normalization, patch size.
- Preprocessing steps that nnUNet does on CPU:
  - **Transpose**: `torch.permute()` — trivially GPU
  - **Crop**: Boolean mask + bbox extraction — GPU via `torch.nonzero()` on a thresholded volume
  - **Normalize**: Per-channel scheme lookup from plans — GPU via element-wise ops
  - **Resample**: nnUNet uses `resample_torch` (already torch-based) or SimpleITK-based. For GPU-native, use `torch.nn.functional.interpolate` or MONAI's `Resample` with GPU backend.
- Cascade input: For `3d_cascade_fullres`, the previous stage's segmentation is one-hot encoded and stacked as extra channels. This happens **inside** this operator so the downstream operator sees a uniform tensor shape.

**GPU data flow**: Input CPU numpy → `torch.as_tensor().to('cuda')` → all preprocessing on GPU → output GPU tensor. No CPU bounce.

#### ② SlideWindowOperator

**Purpose**: Execute tiled inference with TTA mirroring and multi-fold averaging.

**Input**:
- `preprocessed`: GPU tensor (C', Z', Y', X') float32
- `properties`: Dict from PreprocessOperator
- `network`: PyTorch module (loaded in setup, shared across calls)
- `fold_weights`: List of state dicts (one per fold)

**Output**:
- `logits`: GPU tensor (K, Z', Y', X') float16/float32 — at config resolution

**Key design decisions**:
- **Tile computation**: `compute_steps_for_sliding_window` is pure arithmetic — compute once per unique image size, cache per config.
- **Inference loop**: Port nnUNet's `_internal_predict_sliding_window_return_logits` to use a CUDA stream. The producer/consumer queue pattern (tiles → GPU → accumulate) maps naturally to Holoscan's async execution model.
- **TTA mirroring**: `_internal_maybe_mirror_and_predict` generates axis combinations, flips, forwards, flips back, averages. All `torch.flip` — native GPU.
- **Fold averaging**: Instead of the current pattern (forward each fold → CPU accumulate), keep all fold predictions on GPU and average in-place: `accumulated_logits += tile_prediction` then `accumulated_logits /= num_folds`.
- **Gaussian weighting**: Precompute the Gaussian kernel once per config (LRU-cached in nnUNet, make it a setup-time constant). Apply via broadcasting.

**Memory management**: The accumulator tensors (`predicted_logits`, `n_predictions`) are allocated once per inference session. For large volumes on 8GB GPUs, the `perform_everything_on_device` fallback to CPU for accumulators should be preserved as a configurable option.

#### ③ PostResampleOperator

**Purpose**: Convert logits from config space back to original image space.

**Input**:
- `logits`: GPU tensor (K, Z', Y', X') from SlideWindowOperator
- `properties`: Dict from PreprocessOperator (carried through the pipeline)

**Output**:
- `segmentation`: GPU tensor (Z, Y, X) uint8 — at original image resolution
- `probabilities` (optional): GPU tensor (K, Z, Y, X) float32 — for ensemble averaging

**Key design decisions**:
- **Resample probabilities**: `configuration_manager.resampling_fn_probabilities` — nnUNet uses `resample_torch` which is already torch-compatible. For full GPU: `torch.nn.functional.interpolate`.
- **Non-linearity**: `label_manager.apply_inference_nonlin()` — softmax for standard, custom for region-based. GPU native.
- **Logits → segmentation**: `label_manager.convert_logits_to_segmentation()` — argmax along class dim. GPU native.
- **Revert cropping**: `insert_crop_into_image` — place segmentation into zero-padded tensor at original bbox. GPU via indexing.
- **Revert transpose**: `torch.permute()` with `transpose_backward`. GPU native.

#### ④ EnsembleAverageOperator

**Purpose**: Average probability maps from multiple configs in memory.

**Input**:
- `probabilities_*`: GPU tensors (K, Z, Y, X) float32 — one per non-auxiliary config
- `label_manager`: Shared reference for label conversion

**Output**:
- `segmentation`: GPU tensor (Z, Y, X) uint8

**Key design decisions**:
- Replaces the `.npz` file I/O entirely. Probability maps flow through Holoscan's DataStore (GPU buffer references).
- When all inputs arrive, compute: `avg = (prob_A + prob_B + ...) / N`
- Then: `segmentation = label_manager.convert_logits_to_segmentation(avg)`
- **Count condition**: Holoscan's `CountCondition` or `ConditionalWait` ensures all config branches complete before ensemble.

#### ⑤ PostprocessOperator

**Purpose**: Apply nnUNet's learned postprocessing rules.

**Input**:
- `segmentation`: GPU tensor (Z, Y, X) uint8
- `pp_rules`: Postprocessing rules from `postprocessing.pkl` (loaded in setup)

**Output**:
- `segmentation`: GPU tensor (Z, Y, X) uint8 (cleaned)

**Key design decisions**:
- nnUNet's postprocessing is: `remove_all_but_largest_component` per label.
- Current implementation: `acvl_utils.morphology.morphology_helper.remove_all_but_largest_component` — CPU-based.
- GPU alternative: Connected components via `scipy.ndimage.label` has no direct GPU equivalent in standard PyTorch. Options:
  - **Option A**: Use MONAI's `KeepLargestConnectedComponentd` — it wraps `skimage.measure.label` (CPU). Acceptable for small post-processing volume relative to inference.
  - **Option B**: Use a GPU connected components library (e.g., `gpu-cc` or custom CUDA kernel). More complex, marginal gain.
  - **Recommendation**: Option A for Phase 1. The post-processing step operates on the final segmentation (small, uint8, single-channel) — the CPU cost is negligible compared to inference. Keep the data on CPU only for this step, then transfer back if needed.

### 3.2 Data Flow Between Operators

```
Current (broken) flow:
  DICOMSeriesToVolume ──CPU numpy──→ NNUnetSegOperator
                                        │
                                        ├── nnUNet preprocessor (CPU numpy)
                                        ├── nnUNet predictor (GPU tensor)
                                        ├── .npz write (disk)
                                        ├── .npz read (disk)
                                        ├── ensemble average (CPU numpy)
                                        ├── postprocessing (CPU numpy)
                                        │
                                        └── CPU numpy ──→ DICOMSegWriter

Target (GPU-native) flow:
  DICOMSeriesToVolume ──CPU numpy──→ [GPU transfer boundary]
                                        │
                                        ├── PreprocessOperator ──GPU tensor──→ SlideWindowOperator
                                        │         (torch.permute, torch.interpolate, element-wise) │
                                        │                                                    │
                                        │                                           GPU logits ──┐
                                        │                                                       ▼
                                        │                                              PostResampleOperator
                                        │                                                       │
                                        │                                              GPU seg + probs
                                        │                                                       │
                                        │                    ┌──────────────────────────────────┤
                                        │                    ▼                                  ▼
                                        │              (Config A branch)                  (Config B branch)
                                        │              Same operators,                   Same operators,
                                        │              different config.                 different config.
                                        │                    │                                  │
                                        │                    └──────────┬──────────────────────┘
                                        │                               ▼
                                        │                      EnsembleAverageOperator
                                        │                      (GPU element-wise mean + argmax)
                                        │                               │
                                        │                               ▼
                                        │                      PostprocessOperator
                                        │                      (GPU/CPU hybrid)
                                        │                               │
                                        │                    [GPU→CPU boundary]
                                        │                               │
                                        ▼                               ▼
  DICOMSegWriter  ◄── GPU seg tensor (uint8, Z, Y, X)
  DICOMSRWriter   ◄── volume measurements (computed from seg + affine)
  DICOMSCWriter   ◄── overlay images (computed from seg + original)
```

### 3.3 Data Types on the Wire

| Boundary | Current | Target |
|----------|---------|--------|
| DICOMSeriesToVolume → Preprocess | CPU numpy (float32, C, Z, Y, X) | Same (GPU transfer at operator boundary) |
| Preprocess → SlideWindow | N/A (internal to monolithic op) | GPU tensor (float32, C', Z', Y', X') + properties dict |
| SlideWindow → PostResample | N/A (internal) | GPU tensor (float16/32, K, Z', Y', X') |
| PostResample → Ensemble | N/A (internal, via .npz) | GPU tensor (float32, K, Z, Y, X) — probabilities |
| Ensemble → Postprocess | CPU numpy (via .npz roundtrip) | GPU tensor (uint8, Z, Y, X) |
| Postprocess → DICOMSegWriter | CPU numpy (uint8, Z, Y, X) | GPU tensor → `.cpu().numpy()` at emit |

## 4. Handling nnUNet's Tiling Strategy Within Holoscan

### 4.1 Current nnUNet tiling

nnUNet uses **sliding window inference** with:
- **Patch size**: Config-dependent (e.g., 384×256×256 for 3d_fullres on a 1mm-spaced volume)
- **Step size**: 0.5× patch size (50% overlap)
- **Gaussian weighting**: Edge tiles weighted less than center tiles
- **TTA mirroring**: Flip along allowed axes (from training), forward, flip-back, average

For a 512×512×100 volume with patch 384×256×256 and step 0.5:
- ~2×3×3 = 18 tile positions (rough estimate)
- With TTA mirroring on 2 axes: 18 × 4 = 72 forward passes

### 4.2 Holoscan integration options

**Option A: Single operator, async tile dispatch (recommended for Phase 1)**

The `SlideWindowOperator` handles all tiling internally but uses Holoscan's async execution model:
- Operator receives the full preprocessed volume
- Computes tile schedule in `setup()` or on first call
- Processes tiles sequentially within the operator but yields control between tiles via Holoscan's async mechanism
- GPU stream ensures overlap between tile copy-to-device and forward pass

**Option B: Tile-level parallelism across Holoscan operators (Phase 2)**

Each tile becomes a Holoscan message:
- A `TileDispatcherOperator` emits tile tensors
- Multiple `SlideWindowOperator` instances (or a pool) process tiles in parallel
- A `TileAccumulatorOperator` collects and Gaussian-weights results

This is complex and provides marginal benefit for single-study latency (the target metric). The network forward pass is the bottleneck, not tile scheduling.

**Recommendation**: Option A for Phase 1. The bottleneck is GPU compute, not CPU scheduling. Parallelize at the config level (multiple configs running simultaneously via Holoscan DAG parallelism), not the tile level.

### 4.3 TTA mirroring

Keep TTA inside `SlideWindowOperator`. It's a per-tile operation that's tightly coupled with the forward pass. Extracting it into a separate operator adds unnecessary data movement.

## 5. Handling Ensemble Averaging Without Disk I/O

### 5.1 Current disk-based approach

```
Config A inference → .npz on disk → np.load() → avg += prob_A
Config B inference → .npz on disk → np.load() → avg += prob_B
avg /= 2 → argmax → segmentation
```

### 5.2 GPU-native approach

```
Config A PostResample → GPU probability tensor ─┐
                                                ├──→ EnsembleAverageOperator
Config B PostResample → GPU probability tensor ─┘     (avg = (A + B) / 2, on GPU)
                                                              │
                                                       argmax on GPU
                                                              │
                                                       GPU segmentation
```

### 5.3 Implementation details

- Each config branch (Preprocess → SlideWindow → PostResample) runs as a parallel branch in the Holoscan DAG
- `EnsembleAverageOperator` uses a `CountCondition` to wait for all N config branches
- Probability maps are passed via Holoscan's DataStore — zero-copy GPU buffer references (no serialization)
- If probability maps are too large for available VRAM, fall back to CPU accumulation with pinned memory transfers
- The `label_manager.convert_logits_to_segmentation()` or `convert_probabilities_to_segmentation()` runs on the averaged probabilities

### 5.4 Memory consideration

For a 512×512×300 volume with 2 classes (background + airway):
- Probability map: 2 × 512 × 512 × 300 × 4 bytes ≈ 1.2 GB per config
- For a 2-config ensemble: 2.4 GB of probability maps in flight
- For a 3-config ensemble: 3.6 GB
- **Mitigation**: If VRAM is constrained, accumulate incrementally — `EnsembleAverageOperator` can emit a running average and accept configs one at a time (sequential within the operator, not parallel in the DAG).

## 6. Handling Cascade Models

### 6.1 nnUNet cascade architecture

```
3d_lowres (coarse)                    3d_cascade_fullres (refined)
┌─────────────────┐                  ┌──────────────────────────┐
│ Low-resolution   │                  │ Full-resolution, small   │
│ patch size       │                  │ receptive field          │
│ e.g., 1.2mm³     │                  │ e.g., 0.8mm³ + seg input │
│ Config spacing   │────seg───────────│ Prev stage seg → one-hot │
│                  │                  │ stacked as extra channels│
└─────────────────┘                  └──────────────────────────┘
```

- `3d_lowres` runs first, produces a segmentation
- `3d_cascade_fullres` takes that segmentation, converts to one-hot, stacks onto the preprocessed image as additional channels
- The cascade config's `previous_stage_name` in `ConfigurationManager` = "3d_lowres"

### 6.2 Current disk-based cascade

In `nnunet_bundle.py` `ModelnnUNetWrapper.forward()`:
```python
if self.configuration_name == "3d_cascade_fullres":
    seg_file = os.path.join(lowres_predictions_folder, outfile_name + ".nii.gz")
    rw = self.predictor.plans_manager.image_reader_writer_class()
    previous_segmentation, _ = rw.read_seg(seg_file)
```
The 3d_lowres writes a `.nii.gz` file → 3d_cascade_fullres reads it back.

### 6.3 GPU-native cascade

The `3d_lowres` branch's PostResampleOperator already produces the segmentation in original image space. Feed this GPU tensor directly to the `3d_cascade_fullres` PreprocessOperator:

```
3d_lowres branch:
  Preprocess(3d_lowres) → SlideWindow(lowres network) → PostResample → GPU seg (Z, Y, X)
                                                                        │
                                                                        ▼ (GPU tensor, zero-copy)
3d_cascade_fullres branch:                                              │
  Preprocess(3d_cascade) ◄─────────────────────────────────────────────┘
    (takes seg, one-hot encodes, stacks with image)
  → SlideWindow(cascade network) → PostResample → GPU seg (Z, Y, X)
```

- The cascade PreprocessOperator receives the GPU segmentation from 3d_lowres
- It performs: `convert_labelmap_to_one_hot(seg, foreground_labels, dtype)` → `np.vstack((image, seg_onehot))`
- This is all torch-operations: `torch.nn.functional.one_hot()` + `torch.cat()`
- No disk I/O

### 6.4 Cascade in the DAG

```
                              ┌─→ Preprocess(2d) ──→ SlideWindow(2d) ──→ PostResample ─┐
                              │                                                        │
  Preprocess(3d_lowres) ──→ SlideWindow(lowres) ──→ PostResample(lowres) ─────────────┼──→ EnsembleAverage ──→ Postprocess
                              │                        │                               │
                              │                        ▼ (GPU seg)                      │
                              │              Preprocess(3d_cascade) ──→ SlideWindow(cascade) ──→ PostResample │
                              │                                                        │
                              └────────────────────────────────────────────────────────┘
```

- `3d_lowres` output feeds both the ensemble (if it's listed as a non-aux config) AND the cascade input
- When `3d_lowres` is purely auxiliary (only feeds cascade), it doesn't participate in ensemble averaging — the current code already handles this by excluding it from `ensemble_model_list`

## 7. Component Boundaries and Data Flow Directions

### 7.1 Component Summary

| Component | Type | Input | Output | Device |
|-----------|------|-------|--------|--------|
| PreprocessOperator | Holoscan Op | CPU numpy volume + config | GPU tensor + properties dict | CPU→GPU |
| SlideWindowOperator | Holoscan Op | GPU tensor + network | GPU logits | GPU |
| PostResampleOperator | Holoscan Op | GPU logits + properties | GPU seg + GPU probs | GPU |
| EnsembleAverageOperator | Holoscan Op | N× GPU probs | GPU seg | GPU |
| PostprocessOperator | Holoscan Op | GPU seg | GPU seg (cleaned) | GPU/CPU |
| MeasurementOperator | Holoscan Op | GPU seg + affine | Volume measurements | GPU/CPU |
| OverlayOperator | Holoscan Op | GPU seg + original image | Overlay images | GPU |

### 7.2 Config-agnostic vs. config-specific

| Component | Config-specific? | Notes |
|-----------|-----------------|-------|
| PreprocessOperator | Yes — one instance per config | Different spacing, normalization, patch size |
| SlideWindowOperator | Yes — one instance per config | Different network architecture, weights, patch size |
| PostResampleOperator | Yes — one instance per config | Different resampling functions |
| EnsembleAverageOperator | No — single instance | Aggregates across configs |
| PostprocessOperator | No — single instance | Postprocessing rules are ensemble-level |

## 8. Suggested Build Order

### Phase 1: Foundation (can be developed sequentially)

**Step 1: PreprocessOperator**
- Why first: Isolated from other new operators. Can be validated by comparing preprocessed tensor against nnUNet's CPU preprocessor output (pixel-exact).
- Dependencies: None (reads plans.json, dataset.json, accepts CPU numpy)
- Validation: Compare output tensor shape and values against `DefaultPreprocessor.run_case_npy()`

**Step 2: SlideWindowOperator**
- Why second: Needs preprocessed input from Step 1. Core inference logic.
- Dependencies: PreprocessOperator output, nnUNet model weights
- Validation: Compare logits against nnUNet's `predict_sliding_window_return_logits()` output

**Step 3: PostResampleOperator**
- Why third: Completes a single-config inference pipeline. End-to-end pixel validation possible.
- Dependencies: SlideWindowOperator output, properties from PreprocessOperator
- Validation: Compare final segmentation against nnUNet's `convert_predicted_logits_to_segmentation_with_correct_shape()` — must be pixel-exact for the config

### Phase 2: Multi-config (requires Phase 1)

**Step 4: EnsembleAverageOperator**
- Why fourth: Needs multiple PostResampleOperator outputs.
- Dependencies: Multiple PostResampleOperator instances (one per config)
- Validation: Compare ensemble segmentation against `average_probabilities()` + `convert_logits_to_segmentation()`

**Step 5: Cascade support**
- Why fifth: Extends Step 4 with cross-config data flow.
- Dependencies: 3d_lowres branch (Steps 1-3), 3d_cascade_fullres branch (Steps 1-3), plus the cascade input wire
- Validation: Compare cascade output against nnUNet's disk-based cascade output

### Phase 3: Polish

**Step 6: PostprocessOperator**
- Why last: Operates on the final segmentation. Smallest computational impact. Can be deferred and use the existing MONAI transform as a shim.
- Dependencies: EnsembleAverageOperator or PostResampleOperator (single config)
- Validation: Compare against nnUNet's `apply_postprocessing()` output

**Step 7: MeasurementOperator + OverlayOperator**
- Why last: Purely downstream. No impact on segmentation correctness.
- Dependencies: Final segmentation + original image
- Validation: Compare measurements against existing `post_transforms.py` output

### Phase 4: Integration

**Step 8: Full DAG assembly**
- Wire all operators into Holoscan fragments
- Replace `NNUnetSegOperator` in `app.py` with the new operator graph
- Pixel-exact validation of full pipeline end-to-end

### Dependency Graph

```
Step 1: PreprocessOperator
    │
    ▼
Step 2: SlideWindowOperator
    │
    ▼
Step 3: PostResampleOperator ────► (single config works end-to-end)
    │
    ├──► Step 4: EnsembleAverageOperator ──► Step 6: PostprocessOperator
    │       (needs N× Steps 1-3)              Step 7: Measurement + Overlay
    │
    └──► Step 5: Cascade ───────────────────► (feeds into Step 4 ensemble)
            (3d_lowres Steps 1-3 →
             3d_cascade Steps 1-3)
```

## 9. Risk Areas and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU memory exhaustion with large volumes | OOM crash | Precompute memory requirements from plans.json; fall back to CPU accumulation; configurable `perform_everything_on_device` |
| Resampling differences (GPU vs CPU) | Pixel mismatch | Use nnUNet's `resample_torch` functions initially; validate against SimpleITK-based reference; only switch to pure torch after validation |
| Connected components on GPU | Complexity | Accept CPU for post-processing (Step 6); volume is small relative to inference |
| Multi-config VRAM pressure | OOM with 3+ configs | Incremental ensemble accumulation; process configs sequentially within EnsembleAverageOperator if needed |
| Cascade one-hot encoding dtype | Shape mismatch | Validate channel count at operator boundary; fail fast with clear error |
| Fold averaging numerical precision | Pixel difference | Use float32 accumulation for fold averaging; match nnUNet's accumulation pattern |

## 10. Holoscan Integration Notes

### Fragment vs. Operator topology

Each config's Preprocess → SlideWindow → PostResample chain should be grouped into a single **Fragment** for clean DAG composition. The Fragment exposes:
- Input: `volume` (CPU numpy), `config_name`
- Output: `probabilities` (GPU tensor), `segmentation` (GPU tensor)

The top-level app composes:
- One Fragment per config (N fragments)
- EnsembleAverageOperator consuming all N fragment outputs
- PostprocessOperator consuming ensemble output
- MeasurementOperator + OverlayOperator consuming final output

### DataStore usage

Holoscan's DataStore (already used by MONAI Deploy) passes data between operators. For GPU tensors, the DataStore should hold `torch.Tensor` objects directly — the tensor's `.device` property determines GPU residency. No copy occurs on emit/receive as long as the tensor stays on GPU.

### CountCondition for ensemble

```python
ensemble_op = EnsembleAverageOperator(
    self,
    CountCondition(self, num_configs),  # wait for all N config branches
    config_names=ensemble_config_names,
)
```

### Async execution

Holoscan 4.x supports async operator execution. The SlideWindowOperator's tile loop can use `holoscan.core.send()` between tiles to allow other operators (e.g., the next config's preprocessing) to run concurrently on the GPU.
