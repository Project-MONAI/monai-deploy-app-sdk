# Requirements: cchmc-nnunet-fast

**Defined:** 2025-08-13
**Core Value:** Single-study inference latency without sacrificing correctness — a CT study goes in, a pixel-identical DICOM-SEG comes out, faster than before, with every step between staying on GPU.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Pipeline — DAG assembly and operator composition

- [ ] **PIPE-01**: The app assembles a Holoscan DAG that connects existing DICOM I/O operators to new GPU-native operators without modifying SDK core code
- [ ] **PIPE-02**: The app replaces `NNUnetSegOperator` with the new operator chain (Preprocess → SlideWindow → PostResample → EnsembleAverage → Postprocess) in `app.py`
- [x] **PIPE-03**: Each nnUNet configuration (2D, 3D_fullres, 3D_lowres, 3D_cascade_fullres) runs as an independent Holoscan Fragment within the same DAG
- [x] **PIPE-04**: For 3D_cascade_fullres, the 3D_lowres segmentation output feeds directly into the full-res preprocessing operator as a one-hot channel stack without disk I/O
- [ ] **PIPE-05**: The app runs end-to-end from DICOMDIR input to DICOM-SEG output on a single study without operator errors

### Preprocessing — GPU resampling and normalization

- [x] **PREP-01**: The system transposes input volume to match nnUNet training orientation using GPU-accelerated operations
- [x] **PREP-02**: The system applies nnUNet normalization (per-channel mean/std from training plans) on GPU before any resampling step
- [x] **PREP-03**: The system uses the reference nnUNet resampling path (scipy/scikit-image) to guarantee pixel-exact equivalence with the current app
- [x] **PREP-04**: The system crops and pads the volume to match nnUNet's expected input shape using GPU operations
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
- [x] **INF-009**: Ensemble averaging computes element-wise mean of probability maps (post-softmax) in GPU memory using in-place accumulation in ensemble_model_list order, not disk-based `.npz` I/O (met-with-deviation per D-19: Phase 1 in-place accumulation + CuPy exact final division instead of a literal running mean — bit-exactness over literal wording)
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
- [x] **INFR-004**: The system uses Holoscan `CudaStreamPool` for concurrent kernel launches across operators (one per model-config subgraph — NonBlocking, reserved_size=1, per-config nvtx_identifier; overlap visibility is Plan 06's nsys trace, best-effort per D-16)
- [x] **INFR-005**: The system emits NVTX markers (via `torch.cuda.nvtx`) at the start and end of each operator's `compute()` method for correlation with Nsight Systems traces (per-config range names `preprocess_<cfg>`/`inference_<cfg>`/`postresample_<cfg>` + `"config"` timing record fields; per-study aggregate keyed by the top-level application)
- [ ] **INFR-006**: The system provides structured operator-level timing logs (start, end, duration in ms) for each operator in the pipeline for per-study latency reporting

### Testing — equivalence validation and benchmarks

- [ ] **TEST-01**: The system produces DICOM-SEG output with pixel values that are bit-for-bit identical to the current `cchmc_nnunet_fifteen_ckpt_app` on a validated reference corpus of at least 5 representative CT studies
  - **Deviation (2026-08-17):** the working corpus is the single airway MR series in `testdata/airway_input` (256 slices, 256×256) with complete SC/SEG/SR ground truth in `testdata/airway_output` (airway model `MRI_NICU-Airway_TRAINv2`; models at `examples/apps/cchmc_nnunet_fifteen_ckpt_app/models`). The ≥5-CT-study bar is deferred to the final Phase 1 acceptance gate. **RESOLVED 2026-08-17 (20:40 UTC):** the earlier concern that the fresh reference app differs from historical GT (~45 mm world-COM, 0 overlap) was a **decoding artifact** — a fresh run (`testdata/current_output`) is **99.902% byte-identical** to historical GT (segment voxel counts 2430 vs 2447, Δ0.7%), so a freshly regenerated reference is a valid pixel-exact gate target. TEST-01 remains **unsatisfied** until `cchmc-nnunet-fast` itself is validated on the corpus.
- [ ] **TEST-002**: The system produces DICOM-SR measurements that match the current app to within 0.1% tolerance on the same reference corpus
- [ ] **TEST-003**: The test suite includes an automated pixel-level diff tool that compares new app output against reference output and fails on any divergence
- [ ] **TEST-004**: The test suite verifies that all intermediate tensors remain on GPU (`device == 'cuda'`) throughout the pipeline and flags any `.cpu()` or `.numpy()` calls before the final output stage
- [ ] **TEST-005**: The test suite covers all four nnUNet configurations (2D, 3D_fullres, 3D_lowres, 3D_cascade_fullres) with at least one test case each
- [x] **TEST-006**: The system provides a benchmark script that measures end-to-end latency (DICOM input to DICOM-SEG write) and per-operator breakdown for a single study
  - **Satisfied (Phase 0, 2026-08-17):** `.planning/scripts/baseline_benchmark.py` measures E2E + per-stage latency (setup/inference/postprocess/write) for a single study → `.planning/baseline_results.csv` (169,747 ± 7,274 ms, n=3).
- [x] **TEST-007**: The system provides a benchmark comparison against the current `cchmc_nnunet_fifteen_ckpt_app` on identical hardware and input data, reporting speedup ratio and absolute latency
  - **Baseline satisfied (Phase 0, 2026-08-17):** absolute latency of the current/reference app established (169,747 ± 7,274 ms/study) → `.planning/baseline_results.csv`. The **speedup-ratio** half (new app vs baseline) is completed in Phase 2/3 when the faster pipeline exists — this is a shared requirement (Phases 0/2/3).

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
| PIPE-03 | 2 | **Done (Phase 2, Plans 03–04)** — resolve_run_model_list reference semantics (Plan 03) + one Subgraph per config in one DAG, config-generic factory, HOLOSCAN_MODEL_LIST selection, 4/4 configurations E2E (Plan 04) |
| PIPE-04 | 2 | **Done (Phase 2, Plans 03–04)** — lowres_seg port contract + 2-channel input (Plan 03) + cross-fragment flow with zero disk I/O, end-to-end in cascade-only and bundle runs (Plan 04) |
| PIPE-05 | 1 | Pending |
| PREP-01 | 1, 2 | **Done (Phase 2, Plan 01)** — CuPy transpose, fp32 C-contiguous |
| PREP-02 | 1, 2 | **Done (Phase 2, Plan 01)** — element-wise on GPU, numpy reductions |
| PREP-03 | 1 | **Done (Phase 2, Plan 01)** — unchanged scipy/skimage path (D-13) |
| PREP-04 | 1, 2 | **Done (Phase 2, Plan 01)** — CuPy crop-slice, materialized |
| PREP-05 | 1 | Pending |
| INF-001 | 1 | Pending |
| INF-002 | 1 | Pending |
| INF-003 | 1 | Pending |
| INF-004 | 1 | Pending |
| INF-005 | 1 | Pending |
| INF-006 | 1 | Pending |
| INF-007 | 1 | Pending |
| INF-008 | 1 | Pending |
| INF-009 | 1, 2 | **Done (Phase 2, Plan 04)** met-with-deviation per D-19 — ordered per-config `prob_<cfg>` inputs + list-order reconstruction over Phase 1 in-place accumulation + CuPy exact final division (bit-exact; deviation documented at source) |
| INF-010 | 1 | Pending |
| INF-011 | 1 | Pending |
| POST-01 | 1 | Pending |
| POST-02 | 1 | Pending |
| POST-03 | 1 | Pending |
| INFR-01 | 0, 2 | Pending |
| INFR-02 | 2 | Pending |
| INFR-03 | 2 | Pending |
| INFR-004 | 2 | **Done (Phase 2, Plan 04)** — CudaStreamPool per model-config subgraph (NonBlocking, reserved_size=1, streams_<cfg>) |
| INFR-005 | 0, 1 | **Done (Phase 2, Plan 04)** — per-config NVTX range names + `"config"` timing fields + app-keyed per-study aggregate (sub-Fragment-safe via gpu_util._root) |
| INFR-006 | 1 | Pending |
| TEST-01 | 1, 2 | Pending (deviation: single-MR-study dev corpus, 2026-08-17) |
| TEST-002 | 1, 2 | Pending |
| TEST-003 | 1 | Pending |
| TEST-004 | 1 | Pending |
| TEST-005 | 2 | Pending |
| TEST-006 | 0, 1, 2, 3 | **Done (Phase 0)** — baseline_benchmark.py + baseline_results.csv |
| TEST-007 | 0, 2, 3 | **Baseline done (Phase 0)** — speedup-ratio pending Ph2/3 |

**Coverage:**
- v1 requirements: 36 total
- Mapped to phases: 36
- Unmapped: 0

---
*Requirements defined: 2025-08-13*
*Last updated: 2026-08-17 — traceability matrix synced from ROADMAP.md; TEST-01 corpus deviation documented*
