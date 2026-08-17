# Requirements: cchmc-nnunet-fast

**Defined:** 2025-08-13
**Core Value:** Single-study inference latency without sacrificing correctness — a CT study goes in, a pixel-identical DICOM-SEG comes out, faster than before, with every step between staying on GPU.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Pipeline — DAG assembly and operator composition

- [ ] **PIPE-01**: The app assembles a Holoscan DAG that connects existing DICOM I/O operators to new GPU-native operators without modifying SDK core code
- [ ] **PIPE-02**: The app replaces `NNUnetSegOperator` with the new operator chain (Preprocess → SlideWindow → PostResample → EnsembleAverage → Postprocess) in `app.py`
- [ ] **PIPE-03**: Each nnUNet configuration (2D, 3D_fullres, 3D_lowres, 3D_cascade_fullres) runs as an independent Holoscan Fragment within the same DAG
- [ ] **PIPE-04**: For 3D_cascade_fullres, the 3D_lowres segmentation output feeds directly into the full-res preprocessing operator as a one-hot channel stack without disk I/O
- [ ] **PIPE-05**: The app runs end-to-end from DICOMDIR input to DICOM-SEG output on a single study without operator errors

### Preprocessing — GPU resampling and normalization

- [ ] **PREP-01**: The system transposes input volume to match nnUNet training orientation using GPU-accelerated operations
- [ ] **PREP-02**: The system applies nnUNet normalization (per-channel mean/std from training plans) on GPU before any resampling step
- [ ] **PREP-03**: The system uses the reference nnUNet resampling path (scipy/scikit-image) to guarantee pixel-exact equivalence with the current app
- [ ] **PREP-04**: The system crops and pads the volume to match nnUNet's expected input shape using GPU operations
- [ ] **PREP-05**: The preprocessing operator emits a `MemoryData` buffer with `DeviceType::GPU` for zero-copy handoff to the inference operator

### Inference — tile-based model execution and ensemble

- [ ] **INF-001**: The system runs nnUNet model inference in PyTorch eager mode (`torch.no_grad()`) on GPU without falling back to CPU
- [ ] **INF-002**: The system uses MONAI `sliding_window_inference` for tile-based inference with the same patch size, overlap, and Gaussian weighting as the reference nnUNet predictor
- [ ] **INF-003**: The system applies Test-Time Augmentation (TTA) mirror flips in the same order as the reference nnUNet predictor
- [ ] **INF-004**: The system accumulates TTA results using FP32 accumulators and preserves sequential `+=` addition order to maintain numerical equivalence
- [ ] **INF-005**: The system asserts `tensor.device.type == 'cuda'` at every operator boundary and never swallows `RuntimeError` exceptions that would indicate silent CPU fallback
- [ ] **INF-006**: The system runs all nnUNet model configurations (2D, 3D_fullres, 3D_lowres, 3D_cascade_fullres) through the same inference operator with configuration-driven parameters
- [ ] **INF-007**: The system supports custom nnUNet trainer variants by loading model weights from checkpoint paths without hard-coded trainer assumptions
- [ ] **INF-008**: The system loads all model weights and architecture during `setup()` or `on_insert()`, not during `compute()`, to avoid cold-start latency per study
- [ ] **INF-009**: Ensemble averaging computes element-wise mean of probability maps (post-softmax) in GPU memory using incremental in-place averaging, not disk-based `.npz` I/O
- [ ] **INF-010**: Ensemble averaging applies `argmax` after probability averaging (not before), matching nnUNet's `average_probabilities` reference path
- [ ] **INF-011**: The system keeps autocast scope at the outermost inference boundary and does not split FP16/FP32 regions across operator boundaries

### Postprocessing — connected component cleanup

- [ ] **POST-01**: The system applies connected component analysis to remove false positives based on volume size thresholds from nnUNet's postprocessing configuration
- [ ] **POST-02**: The system reverts preprocessing transformations (crop, transpose) to map the segmentation back to the original DICOM volume orientation and shape
- [ ] **POST-03**: The postprocessing operator produces output as a GPU tensor that is transferred to CPU numpy exactly once at the pipeline boundary before DICOM-SEG write

### Infrastructure — GPU memory management and profiling

- [ ] **INFR-01**: The system uses RMM (RAPIDS Memory Manager) as the CUDA memory allocator with pool allocation to reduce heap fragmentation across sequential studies
- [ ] **INFR-02**: The system pre-allocates GPU buffers during operator `setup()` and reuses them across `compute()` calls instead of allocating per-tile or per-study
- [ ] **INFR-03**: The system computes a memory budget before allocating full-volume logits or probability buffers and defers to incremental strategies when the budget would exceed available VRAM
- [ ] **INFR-004**: The system uses Holoscan `CudaStreamPool` for concurrent kernel launches across operators
- [ ] **INFR-005**: The system emits NVTX markers (via `torch.cuda.nvtx`) at the start and end of each operator's `compute()` method for correlation with Nsight Systems traces
- [ ] **INFR-006**: The system provides structured operator-level timing logs (start, end, duration in ms) for each operator in the pipeline for per-study latency reporting

### Testing — equivalence validation and benchmarks

- [ ] **TEST-01**: The system produces DICOM-SEG output with pixel values that are bit-for-bit identical to the current `cchmc_nnunet_fifteen_ckpt_app` on a validated reference corpus of at least 5 representative CT studies
  - **Deviation (2026-08-17):** the working corpus is the single UTE-MR series in `testdata/input` (62 slices, patient 01153813) with complete SC/SEG(2)/SR ground truth in `testdata/output` (SEG source series UID verified against input). The ≥5-CT-study bar is deferred to the final Phase 1 acceptance gate (or Phase 2) when CT data is supplied. TEST-01 remains **unsatisfied** until re-validated on ≥5 studies.
- [ ] **TEST-002**: The system produces DICOM-SR measurements that match the current app to within 0.1% tolerance on the same reference corpus
- [ ] **TEST-003**: The test suite includes an automated pixel-level diff tool that compares new app output against reference output and fails on any divergence
- [ ] **TEST-004**: The test suite verifies that all intermediate tensors remain on GPU (`device == 'cuda'`) throughout the pipeline and flags any `.cpu()` or `.numpy()` calls before the final output stage
- [ ] **TEST-005**: The test suite covers all four nnUNet configurations (2D, 3D_fullres, 3D_lowres, 3D_cascade_fullres) with at least one test case each
- [ ] **TEST-006**: The system provides a benchmark script that measures end-to-end latency (DICOM input to DICOM-SEG write) and per-operator breakdown for a single study
- [ ] **TEST-007**: The system provides a benchmark comparison against the current `cchmc_nnunet_fifteen_ckpt_app` on identical hardware and input data, reporting speedup ratio and absolute latency

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Throughput

- **THRU-01**: The system processes multiple studies concurrently using Holoscan pipeline-level parallelism
- **THRU-02**: The system implements double-buffered pipeline overlap (preprocess study M+1 while inferring study M)
- **THRU-03**: The system provides throughput metrics (studies/minute) at sustained load

### Acceleration

- **ACCEL-01**: The system supports TensorRT engine execution as an alternative inference backend
- **ACCEL-02**: The system generates ONNX export and TensorRT engine files from nnUNet checkpoint during model setup
- **ACCEL-03**: The system supports `torch.compile()` for forward pass optimization where nnUNet dynamic control flow permits

### Memory Optimization

- **MEM-01**: The system shares model architecture weights across the 15-checkpoint ensemble, loading weights once and switching via tensor views
- **MEM-02**: The system runs the full pipeline on a target GPU with 8 GB VRAM without OOM for standard chest CT volumes (512×512×300)
- **MEM-003**: The system frees low-res network weights immediately after 3D_lowres inference in cascade configurations to reduce peak memory

### GPU Preprocessing

- **GPUP-01**: The system replaces the reference CPU resampling path with a GPU-accelerated resampler (CuPy or monai.data) while maintaining pixel-exact equivalence
- **GPUP-02**: The system performs all preprocessing steps (transpose, crop, normalize, resample) on GPU with zero CPU-GPU transfers before inference

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| SDK-level operator changes | Example app only — new operators live in the app directory, not `monai/deploy/operators/` |
| Modifications to existing nnUNet example apps | This is a new app, not a refactor of `cchmc_nnunet_fifteen_ckpt_app` |
| Training pipeline | Inference only — no training, validation, or hyperparameter tuning |
| Remote/Triton inference | In-process PyTorch only — no Triton server integration |
| Batch / concurrent study processing | Phase 2+ — v1 targets single-study latency |
| GPU memory efficiency optimization | Phase 3+ — v1 uses conservative allocation, not aggressive pooling |
| TensorRT / ONNX inference | Phase 3+ — deferred until PyTorch baseline is proven and profiled |
| Dynamic operator graph reconstruction per study | DAG is static — configuration varies parameters, not topology |
| Lossy compression of intermediate data | Pixel-exact equivalence is non-negotiable |
| Python multiprocessing for inference | Serialization triggers CPU copies — antipattern for GPU-native flow |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PIPE-01 | 0, 1 | Pending |
| PIPE-02 | 1 | Pending |
| PIPE-03 | 2 | Pending |
| PIPE-04 | 2 | Pending |
| PIPE-05 | 1 | Pending |
| PREP-01 | 1, 2 | Pending |
| PREP-02 | 1, 2 | Pending |
| PREP-03 | 1 | Pending |
| PREP-04 | 1, 2 | Pending |
| PREP-05 | 1 | Pending |
| INF-001 | 1 | Pending |
| INF-002 | 1 | Pending |
| INF-003 | 1 | Pending |
| INF-004 | 1 | Pending |
| INF-005 | 1 | Pending |
| INF-006 | 1 | Pending |
| INF-007 | 1 | Pending |
| INF-008 | 1 | Pending |
| INF-009 | 1, 2 | Pending |
| INF-010 | 1 | Pending |
| INF-011 | 1 | Pending |
| POST-01 | 1 | Pending |
| POST-02 | 1 | Pending |
| POST-03 | 1 | Pending |
| INFR-01 | 0, 2 | Pending |
| INFR-02 | 2 | Pending |
| INFR-03 | 2 | Pending |
| INFR-004 | 2 | Pending |
| INFR-005 | 0, 1 | Pending |
| INFR-006 | 1 | Pending |
| TEST-01 | 1, 2 | Pending (deviation: single-MR-study dev corpus, 2026-08-17) |
| TEST-002 | 1, 2 | Pending |
| TEST-003 | 1 | Pending |
| TEST-004 | 1 | Pending |
| TEST-005 | 2 | Pending |
| TEST-006 | 0, 1, 2, 3 | Pending |
| TEST-007 | 0, 2, 3 | Pending |

**Coverage:**
- v1 requirements: 36 total
- Mapped to phases: 36
- Unmapped: 0

---
*Requirements defined: 2025-08-13*
*Last updated: 2026-08-17 — traceability matrix synced from ROADMAP.md; TEST-01 corpus deviation documented*
