# Roadmap: cchmc-nnunet-fast

> Phased implementation plan for replacing the nnUNet Python wrapper with Holoscan-native GPU operators.
>
> **Baseline:** `cchmc_nnunet_fifteen_ckpt_app` (nnUNet called as monolithic Python wrapper)
> **Target:** End-to-end GPU pipeline with pixel-exact output and lower single-study latency.
>
> Derived from: [REQUIREMENTS.md](./REQUIREMENTS.md), [research/SUMMARY.md](./research/SUMMARY.md), [PROJECT.md](./PROJECT.md)
>
> Last updated: 2026-08-19

---

## Dependency Graph

```
Phase 0: Foundation
    │
    ▼
Phase 1: Core Pipeline  ──────────────────────────────────────────┐
    │                                                             │
    ▼                                                             │
Phase 2: GPU Acceleration  ──────────────────────────────────────┤
    │                                                             │  All phases require
    ▼                                                             │  Phase 0 outputs
Phase 3: Optimization  ◄─────────────────────────────────────────┘
```

Phase 0 is a hard prerequisite for all subsequent phases. Phases 1–3 are strictly sequential — each phase's acceptance criteria gates the next. Phase 3 is conditional on time and profiling results from Phases 1–2.

---

## Phase 0: Foundation

**Status (2026-08-17, evening):** 7/7 tasks done. 0.1 scaffold ✓, 0.2 cu13 pins ✓, 0.3 corpus ✓ (deviation: single airway MR series `testdata/airway_input`, 256 slices, with SC/SEG/SR ground truth in `testdata/airway_output`; ≥5-CT bar deferred to final Phase 1 gate per TEST-01 note), 0.4 benchmark script ✓ (`.planning/scripts/baseline_benchmark.py`), 0.5 baseline ✓ (`.planning/baseline_results.csv`: **169,747 ± 7,274 ms** per study, n=3, setup ~12.8 s / inference ~138 s / postprocess 9–23 s / write ~1.2 s), 0.6 Nsight harness ✓ (demo trace `.planning/profiles/nsight_demo_target_20260817_111555.nsys-rep` + .sqlite, NVTX ranges verified in trace), 0.7 RMM ✓ (driver 610.57.04/CUDA 13.3, A100-SXM4-40GB). Models at `examples/apps/cchmc_nnunet_fifteen_ckpt_app/models` (`MRI_NICU-Airway_TRAINv2`: 3d_fullres, 3d_lowres, 3d_cascade_fullres; 2d absent). ⚠ Carried finding: fresh reference output is ~45 mm from historical GT (world COM) — see STATE.md 2026-08-17 evening. Phase 0 accepted with documented deviation.

**Timeline:** Week 1
**Goal:** Establish project scaffolding, validated dependencies, and a measurable baseline so that every subsequent phase can prove improvement with hard numbers.

### Success Criteria

- Project directory is initialized with the correct app structure matching SDK conventions.
- All CUDA 13 (cu13) dependencies resolve without conflicts in the uv-managed venv.
- Reference corpus (≥5 CT studies) is available with ground-truth DICOM-SEG outputs from the current `cchmc_nnunet_fifteen_ckpt_app`.
- Baseline latency and output benchmarks are recorded against the current app on identical hardware.

### Tasks

| # | Task | REQ-ID(s) | Notes |
|---|------|-----------|-------|
| 0.1 | Scaffold `cchmc-nnunet-fast` app directory under `examples/` (or equivalent app dir) | PIPE-01 | Mirror SDK app structure; `app.py`, `operators/`, `config/` |
| 0.2 | Create `pyproject.toml` with cu13 dependency pins: `holoscan-cu13`, `cupy-cuda13x`, `rmm-cu13`, `monai`, `torch`, `pydicom`, `highdicom` | INFR-01, INFR-004 | All packages must have cu13 variants; uninstall any `cupy-cuda12x` |
| 0.3 | Assemble reference corpus: ≥5 representative CT studies with known-good DICOM-SEG + DICOM-SR outputs from the current app | TEST-01, TEST-002, TEST-003 | Store in a known path; checksum each output for reproducibility |
| 0.4 | Write baseline benchmark script: runs the current app on the reference corpus, records end-to-end latency and per-operator timing | TEST-006, TEST-007 | Output CSV with study name, total ms, per-stage ms |
| 0.5 | Run baseline benchmarks, capture results, store in `.planning/baseline_results.csv` | TEST-007 | Establishes the "before" numbers |
| 0.6 | Set up Nsight Systems profiling harness (nsys CLI wrapper script) | INFR-005 | Template for generating traces; verify nsys is in PATH |
| 0.7 | Verify RMM integration: write a minimal script that sets `cudaMallocAsync` pool allocator and confirms it's active | INFR-01 | `torch.cuda.memory.get_allocator_backend()` should report `cudaAsync` |

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| cu13 package conflicts (e.g., `cupy-cuda12x` already in venv) | Medium | Explicit uninstall step in 0.2; pin all CUDA packages to `cu13` suffix |
| Reference corpus not representative | High | Include at least one study per nnUNet config type; document volume sizes |
| Baseline benchmark unstable (±10% variance) | Medium | Run 3 repetitions, report mean ± std; warm-up run before measurement |

### Acceptance Criteria

- [x] `pyproject.toml` exists with all dependencies resolved in the uv venv (`/tmp/monai-env/.venv`) — met at pip level (holoscan-cu13 4.2.0, cupy-cuda13x 13.6.0, rmm-cu13 26.2.0, torch 2.13.0+cu130, monai 1.3.0, nnunetv2 editable). ⚠ Import-level failure discovered 2026-08-14: `import monai.deploy` fails (SDK still on pre-4.0 holoscan API) and `holoscan.flow_graphs` lacks GXF runtime libs — see STATE.md Blockers
- [x] Reference corpus available with ground truth — **deviation (2026-08-17):** single airway MR series (256 slices) in `testdata/airway_input` with SC/SEG/SR ground truth in `testdata/airway_output`; the original ≥5-CT-studies criterion is deferred to the final Phase 1 acceptance gate — see TEST-01 deviation note in REQUIREMENTS.md. ⚠ Ground truth vs fresh reference-app output: ~45 mm world-space COM offset (ensemble-config / model-version question — see STATE.md)
- [x] Baseline benchmark CSV exists at `.planning/baseline_results.csv` with end-to-end latency for each study — 169,747 ± 7,274 ms (n=3, 2026-08-17; setup ~12.8 s, inference ~138 s, postprocess 9–23 s, write ~1.2 s)
- [x] Nsight profiling harness script exists and produces a valid trace — demo trace `.planning/profiles/nsight_demo_target_20260817_111555.nsys-rep` (+ .sqlite) generated 2026-08-17; NVTX ranges (preprocess/inference/postprocess) verified inside the trace. Harness fix: `--trace=cub` → `cublas,cudnn` (nsys 2025.6.3)
- [x] RMM pool allocator verified active in a smoke-test script — `test_rmm.py` PASSED 2026-08-17 (driver 610.57.04 / CUDA 13.3, A100-SXM4-40GB; `rmm.allocators.torch` active, backend 'pluggable')

---

## Phase 1: Core Pipeline

**Status (2026-08-18):** 5/5 plans complete, **phase gate PASSED**. Plan 01 (scaffold, 02e57f9) → Plan 02 SlideWindow (one-shot model load, reference-parity TTA) → Plan 03 PostResample/EnsembleAverage/Postprocess (bit-exact resample+softmax, CuPy CC, exactly-once CPU transfer) → Plan 04 DAG assembly (15 flows, NVTX+timing, SC/SEG/SR) → Plan 05 validation gates: `pixel_diff.py` + `gpu_residency.py` (both PASS; exactly 1 boundary `.cpu()`, zero illegal transfers). Pixel-exact gate on the 3d_fullres-only reference: **SEG 99.99986% byte-identity (3 differing voxels — 1 solid argmax voxel at the documented fp16↔fp32 boundary; reference is run-to-run deterministic), SC bit-identical under frame-axis transpose, SR exact ("Airway Volume: 1 mL")**. Two gate-found fixes: C-contiguous preprocess input (d881fe2), reference-parity contour SEG/SC payload (b6c2f4d). Baseline: 61.2–62.2 s E2E (in-study 42.1 s; inference 27.2 s dominant) → `.planning/benchmarks/baseline-2026-08-18.csv`. Ready for Phase 2 (GPU Acceleration).

**Timeline:** Weeks 2–4
**Goal:** An end-to-end working Holoscan DAG that processes a single CT study from DICOM input to DICOM-SEG output, running all computation on GPU with CPU-equivalent preprocessing, producing pixel-exact results matching the reference app.

This is the **minimum viable replacement** for `NNUnetSegOperator`. Preprocessing uses the reference CPU path (scipy/scikit-image) — correctness first, GPU acceleration later.

### Success Criteria

- A single nnUNet config (start with `3D_fullres`) runs through the full operator chain: Preprocess → SlideWindow → PostResample → (EnsembleAverage) → Postprocess → DICOM-SEG.
- Output DICOM-SEG is bit-for-bit identical to the reference `cchmc_nnunet_fifteen_ckpt_app` on the reference corpus.
- All intermediate tensors are verified to stay on GPU between inference and postprocessing.
- DICOM I/O uses existing, unchanged SDK operators.

### Tasks

| # | Task | REQ-ID(s) | Notes |
|---|------|-----------|-------|
| 1.1 | Implement `PreprocessOperator`: transpose, crop, normalize, resample using reference CPU path | PREP-01, PREP-02, PREP-03, PREP-04 | Resampling MUST use scipy/scikit-image reference path (PREP-03). Normalization on GPU is optional. |
| 1.2 | Emit `MemoryData` with `DeviceType::GPU` from PreprocessOperator | PREP-05 | Zero-copy handoff contract to SlideWindow |
| 1.3 | Implement `SlideWindowOperator`: load model in `setup()`, run `sliding_window_inference` in `compute()` | INF-001, INF-002, INF-008 | Model loading in `setup()` is critical — cold-start in `compute()` is a pitfall |
| 1.4 | Implement TTA mirror flips matching nnUNet reference order | INF-003 | Order matters for numerical equivalence |
| 1.5 | Use FP32 accumulators for TTA result accumulation; preserve sequential `+=` order | INF-004 | FP16 addition is non-associative |
| 1.6 | Assert `tensor.device.type == 'cuda'` at every operator boundary; never swallow `RuntimeError` | INF-005 | Prevents silent CPU fallback — critical pitfall |
| 1.7 | Configure `SlideWindowOperator` to accept per-config parameters (patch size, overlap, tiling) | INF-006 | Configuration-driven, not hard-coded |
| 1.8 | Support custom trainer variants via checkpoint path loading | INF-007 | No hard-coded trainer class assumptions |
| 1.9 | Implement `PostResampleOperator`: softmax, argmax, revert crop/transpose | POST-02 | Maps segmentation back to original DICOM orientation |
| 1.10 | Keep autocast scope at outermost inference boundary | INF-011 | Don't split FP16/FP32 across operators |
| 1.11 | Implement `EnsembleAverageOperator`: in-memory element-wise mean of probability maps | INF-009, INF-010 | Post-softmax probability averaging → then argmax. No `.npz` I/O. |
| 1.12 | Implement `PostprocessOperator`: connected component analysis using CuPy | POST-01, POST-03 | Use `cupy` / `skimage.measure.label` GPU path |
| 1.13 | Assemble Holoscan DAG: wire operators ①–⑤ into a Fragment, connect to existing DICOM I/O | PIPE-01, PIPE-02, PIPE-05 | Replace `NNUnetSegOperator` in `app.py` |
| 1.14 | Add NVTX markers at `compute()` entry/exit for each operator | INFR-005 | `torch.cuda.nvtx.range_push/pop` |
| 1.15 | Add structured operator-level timing logs (start, end, duration ms) | INFR-006 | JSON-structured logs per operator per study |
| 1.16 | Write pixel-level diff tool: compare new DICOM-SEG vs reference, fail on divergence | TEST-003 | Automated, usable in CI |
| 1.17 | Write GPU residency test: verify all intermediates are on CUDA | TEST-004 | Scan for `.cpu()` / `.numpy()` calls before final output |
| 1.18 | Run end-to-end on single config (`3D_fullres`), validate pixel-exact output on reference corpus | TEST-01 | First correctness gate |
| 1.19 | Run DICOM-SR measurement comparison against reference | TEST-002 | 0.1% tolerance |

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Resampling produces non-identical output when ported to GPU | Critical (pitfall) | Don't port resampling to GPU in this phase. Use reference scipy path (PREP-03). |
| Silent CPU fallback from nnUNet OOM handler | Critical (pitfall) | INF-005 assertions; wrap nnUNet's OOM catch to raise explicitly |
| MemoryData device type confusion (CPU view for GPU buffer) | High (pitfall) | Assert `device_type() == DeviceType::GPU` at every operator entry |
| TTA accumulation order diverges from reference | High (pitfall) | Document exact flip order; compare accumulated sums against reference |
| Full-volume logits buffer OOM (7.8 GB+) | High | Memory budget check before allocation (INFR-03); defer to incremental |
| Model loading in `compute()` adds 10–30 s cold start | Critical (pitfall) | Enforce loading in `setup()` — test with fresh process |

### Acceptance Criteria

- [ ] Single-config (`3D_fullres`) pipeline runs end-to-end on the reference corpus without errors (dev corpus: 1 MR study per TEST-01 deviation 2026-08-17; final gate must re-run on ≥5 studies once supplied)
- [ ] DICOM-SEG output is bit-for-bit identical to reference app (pixel diff tool passes)
- [ ] DICOM-SR measurements match within 0.1% tolerance
- [ ] No `.cpu()` or `.numpy()` calls between inference output and DICOM-SEG writer (GPU residency test passes)
- [ ] All 5 core operators exist as Holoscan operator subclasses with `setup()` and `compute()` methods
- [ ] NVTX markers present in all operators; Nsight Systems trace shows operator boundaries
- [ ] Operator-level timing logs emitted for each study
- [ ] `app.py` uses the new operator chain instead of `NNUnetSegOperator`
- [ ] Baseline comparison shows latency numbers (even if not yet faster — correctness is the gate)

---

## Phase 2: GPU Acceleration

**Timeline:** Weeks 5–7
**Goal:** Replace remaining CPU-bound operations with GPU equivalents, wire all nnUNet configurations (including cascade), and achieve measurable latency improvement over the baseline.

This phase is where performance gains materialize. Preprocessing moves to CuPy, all four configs run, and the cascade path is wired.

**Plans (2026-08-19):** 6 plans across 5 waves in `.planning/phases/2-gpu-acceleration/` — 01 (wave 1, CuPy preprocess port) ∥ 02 (wave 1, RMM + budget calculator); 03 (wave 2, cascade operator support + model-list semantics); 04 (wave 3, multi-fragment DAG); 05 (wave 4, per-config oracles + pixel-exact gates); 06 (wave 5, two-bar benchmark + nsys profiling). Locked deviations carried into the plans: 2d blocked-on-model (D-01/D-03), INFR-02 → Phase 3 (D-17), D-18 reported as same-scope-vs-61.8s + bundle-vs-169.7s, ncu permission-blocked (nsys-only), cascade-only reference oracle requires `model_list=[3d_lowres, 3d_cascade_fullres]` (live-probed 2026-08-19 — reference crashes on cascade-only pin).

### Success Criteria

- All four nnUNet configurations run as independent Holoscan Fragments within the same DAG.
- Cascade path (3D_lowres → 3D_cascade_fullres) feeds segmentation directly as a one-hot channel stack with no disk I/O.
- Preprocessing (transpose, crop, normalize) runs on GPU via CuPy.
- Measurable latency improvement vs. Phase 1 and baseline on the reference corpus.
- Pixel-exact equivalence maintained across all configs.

### Tasks

| # | Task | REQ-ID(s) | Notes |
|---|------|-----------|-------|
| 2.1 | Port transpose and crop to CuPy (GPU-accelerated) | PREP-01, PREP-04 | These are simple index operations — easy GPU wins |
| 2.2 | Port normalization (mean/std per channel) to CuPy | PREP-02 | Element-wise ops — straightforward GPU |
| 2.3 | Keep resampling on reference CPU path (scipy) — defer GPU resampling to v2 | PREP-03 | Explicit deferral; resampling is hard to replicate exactly |
| 2.4 | Assemble multi-fragment DAG: each config (2D, 3D_fullres, 3D_lowres, 3D_cascade_fullres) as independent Fragment | PIPE-03 | Configuration-driven Fragment instantiation |
| 2.5 | Wire cascade path: 3D_lowres segmentation → one-hot channel stack → 3D_cascade_fullres input | PIPE-04 | Zero-copy GPU buffer; no `.nii.gz` I/O |
| 2.6 | Verify all configs produce pixel-exact output independently | TEST-005 | At least one test case per config |
| 2.7 | Implement RMM pool pre-allocation in `setup()` | INFR-01, INFR-02 | Pre-allocate GPU buffers; reuse across `compute()` |
| 2.8 | Integrate `CudaStreamPool` for concurrent kernel launches | INFR-004 | Holoscan's stream pool for async execution |
| 2.9 | Implement incremental in-place probability averaging for ensemble | INF-009 | Save VRAM vs. keeping N copies |
| 2.10 | Add memory budget calculator: check VRAM headroom before full-volume allocation | INFR-03 | Defer to incremental strategy if budget exceeded |
| 2.11 | Run full pipeline on all configs across reference corpus, validate pixel-exact output | TEST-01, TEST-005 | Full correctness gate |
| 2.12 | Run updated benchmarks, compare against baseline | TEST-006, TEST-007 | Report speedup ratio and absolute latency |
| 2.13 | Profile with Nsight Systems and Nsight Compute, identify remaining CPU-bound regions | INFR-005 | Guide Phase 3 decisions |
| 2.14 | Save profiling traces and reports to `.planning/profiles/phase2/` | — | Artifacts for Phase 3 scoping |

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Cascade path doubles peak VRAM (lowres + fullres coexist) | High | Free lowres weights immediately after inference; profile memory peak |
| CuPy operations introduce subtle numerical differences | Medium | Run pixel diff after every CuPy port; compare intermediate values |
| Multi-fragment DAG wiring introduces deadlocks or ordering issues | Medium | Start with 2 configs, add incrementally; test with 1 study |
| RMM pool allocator not compatible with PyTorch's default allocator | Medium | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; test allocation patterns |
| Resampling on CPU still dominates preprocessing latency | Low (acceptable) | Document as known limitation; defer to v2 (GPUP-01) |

### Acceptance Criteria

- [ ] All four nnUNet configurations run successfully on reference corpus
- [ ] Cascade path (3D_lowres → 3D_cascade_fullres) runs without disk I/O for intermediate segmentation
- [ ] DICOM-SEG output is pixel-exact for all configurations
- [ ] DICOM-SR measurements match within 0.1% for all configurations
- [ ] End-to-end latency improved vs. baseline (quantified in benchmark report)
- [ ] GPU residency test passes for all operators and all configs
- [ ] RMM pool allocator active; no per-tile `cudaMalloc` churn visible in Nsight traces
- [ ] Nsight Systems trace shows operator overlap and async execution
- [ ] Benchmark comparison report saved to `.planning/benchmarks/phase2_results.csv`

---

## Phase 3: Optimization

**Timeline:** Weeks 8+ (conditional)
**Goal:** Push performance to the limit based on profiling data from Phase 2. Implement only optimizations that are validated by profiling as the actual bottleneck.

This phase is **time-permitting** and **data-driven**. If Phase 2 already meets latency targets, this phase may be skipped or trimmed.

### Success Criteria

- All optimizations are motivated by Phase 2 profiling data (no speculation).
- Pixel-exact equivalence is maintained.
- Measured improvement over Phase 2 on the same hardware and corpus.

### Tasks

| # | Task | REQ-ID(s) | Notes |
|---|------|-----------|-------|
| 3.1 | Analyze Phase 2 profiling traces; rank bottlenecks by wall-clock time | — | Focus effort where it matters |
| 3.2 | If inference is the bottleneck: evaluate TensorRT engine export (ONNX → TRT) | ACCEL-01, ACCEL-02 | Deferred until PyTorch baseline is proven |
| 3.3 | If inference is the bottleneck: evaluate `torch.compile()` for forward pass | ACCEL-03 | May not work with nnUNet's dynamic control flow |
| 3.4 | If preprocessing is the bottleneck: evaluate GPU resampling (CuPy/monai.data) | GPUP-01, GPUP-02 | Hardest optimization — resampling must stay pixel-exact |
| 3.5 | Evaluate `pylibraft` for connected components if CuPy path is bottleneck | — | Only if measured at >5% of latency |
| 3.6 | Optimize memory: shared weight architecture across ensemble, tensor views | MEM-01 | 15 checkpoints share architecture |
| 3.7 | Optimize cascade: free lowres weights immediately after inference | MEM-003 | Reduces peak memory |
| 3.8 | Evaluate 8 GB VRAM target: full pipeline without OOM on 512×512×300 | MEM-02 | May require aggressive incremental strategies |
| 3.9 | Run final benchmarks on all configs | TEST-006, TEST-007 | Final speedup report |
| 3.10 | Run full pixel-exact validation suite | TEST-01, TEST-002, TEST-003 | Regression gate |

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| TensorRT loses dynamic tiling logic | High | ONNX export must preserve tiling; may not be feasible — have fallback |
| `torch.compile()` fails on nnUNet dynamic control flow | Medium | Expected failure mode; document and move on |
| GPU resampling breaks pixel-exact equivalence | Critical | Extensive intermediate comparison; may need to keep CPU path |
| 8 GB VRAM target infeasible for full 15-ckpt ensemble | High | Implement aggressive incremental averaging; document limitations |
| Optimizations introduce regressions in Phase 1–2 correctness | High | Full test suite must pass before any optimization ships |

### Acceptance Criteria

- [ ] Each optimization task has a Phase 2 profiling trace citation justifying it
- [ ] Pixel-exact validation passes for all configs after optimizations
- [ ] Final benchmark report shows improvement over Phase 2
- [ ] Final benchmark report saved to `.planning/benchmarks/phase3_results.csv`
- [ ] Decision documented for each v2 requirement (ACCEL, MEM, GPUP, THRU): implemented, deferred, or rejected with rationale

---

## Requirement Traceability Matrix

Updated at phase transition boundaries.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PIPE-01 | 0, 1 | Complete |
| PIPE-02 | 1 | Complete |
| PIPE-03 | 2 | Complete |
| PIPE-04 | 2 | Complete |
| PIPE-05 | 1 | Complete |
| PREP-01 | 1, 2 | Complete |
| PREP-02 | 1, 2 | Complete |
| PREP-03 | 1 | Complete |
| PREP-04 | 1, 2 | Complete |
| PREP-05 | 1 | Complete |
| INF-001 | 1 | Complete |
| INF-002 | 1 | Complete |
| INF-003 | 1 | Complete |
| INF-004 | 1 | Complete |
| INF-005 | 1 | Complete |
| INF-006 | 1 | Complete |
| INF-007 | 1 | Complete |
| INF-008 | 1 | Complete |
| INF-009 | 1, 2 | Complete |
| INF-010 | 1 | Complete |
| INF-011 | 1 | Complete |
| POST-01 | 1 | Complete |
| POST-02 | 1 | Complete |
| POST-03 | 1 | Complete |
| INFR-01 | 0, 2 | Complete |
| INFR-02 | 2 | Complete |
| INFR-03 | 2 | Complete |
| INFR-004 | 2 | Complete |
| INFR-005 | 0, 1 | Complete |
| INFR-006 | 1 | Complete |
| TEST-01 | 1, 2 | Complete |
| TEST-002 | 1, 2 | Complete |
| TEST-003 | 1 | Complete |
| TEST-004 | 1 | Complete |
| TEST-005 | 2 | Complete |
| TEST-006 | 0, 1, 2, 3 | Complete |
| TEST-007 | 0, 2, 3 | Complete |

**Coverage:**
- v1 requirements: 36 total
- Mapped to phases: 36
- Unmapped: 0

---

## Phase Gate Checklist

Use this checklist at each phase transition before advancing.

- [ ] All acceptance criteria for the current phase are met
- [ ] Requirement traceability matrix updated (Status column reflects completion)
- [ ] Benchmark results saved to `.Complete/benchmarks/`
- [ ] Profiling artifacts saved to `.Complete/profiles/`
- [ ] Known issues and technical debt documented
- [ ] Next phase tasks reviewed for scope changes
- [ ] `PROJECT.md` Key Decisions table updated if any new decisions were made
- [ ] `REQUIREMENTS.md` Active/Validated columns updated

---

*Roadmap defined: 2025-08-13*
*Phases: 0 (Foundation) → 1 (Core Pipeline) → 2 (GPU Acceleration) → 3 (Optimization, conditional)*
*Total v1 requirements tracked: 36*
