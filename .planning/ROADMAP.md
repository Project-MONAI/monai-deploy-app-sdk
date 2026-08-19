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

**Plans (2026-08-19):** 6 plans across 5 waves in `.planning/phases/02-gpu-acceleration/` — 01 (wave 1, CuPy preprocess port) ∥ 02 (wave 1, RMM + budget calculator); 03 (wave 2, cascade operator support + model-list semantics); 04 (wave 3, multi-fragment DAG); 05 (wave 4, per-config oracles + pixel-exact gates); 06 (wave 5, two-bar benchmark + nsys profiling). Locked deviations carried into the plans: 2d blocked-on-model (D-01/D-03), INFR-02 → Phase 3 (D-17), D-18 reported as same-scope-vs-61.8s + bundle-vs-169.7s, ncu permission-blocked (nsys-only), cascade-only reference oracle requires `model_list=[3d_lowres, 3d_cascade_fullres]` (live-probed 2026-08-19 — reference crashes on cascade-only pin).

**Status (2026-08-19):** 6/6 plans complete, **PHASE 2 COMPLETE** (correctness gate plan 05 + performance gate plan 06). **Plan 06 (two-bar benchmark + nsys profiling) DONE** (fc7e977, fb6cbea, f405d83): `.planning/scripts/phase2_benchmark.py` (fresh-process reps, 1 warmup + 3 measured, per-config columns parsed from the config-tagged `timing:` JSON logs) → `.planning/benchmarks/phase2_results.csv` (10 rows, both speedups, mean±std summaries): same-scope fullres **57,140.3 ± 250.4 ms = 1.082× vs Phase 1 61.8 s — the D-18 positive-improvement bar is MET**; headline bundle **129,542.9 ± 896.4 ms = 1.310× vs 169.7 s reference** (inference stage 76.7 s vs 138.2 s = −44.5%; the bundle-vs-61.8 s scope asymmetry documented as the D-18 deviation, report §4). nsys bundle trace in `.planning/profiles/phase2/` (rep + sqlite + cuda_api_sum/nvtx_sum/nvtx_kern_sum/cuda_gpu_kern_sum): RMM churn check PASS (9 cudaMalloc / 1 cudaFree vs 629,929 kernels — pool expansions only, never per-tile), 14/14 per-config NVTX ranges legible, stream overlap NOT visible (single stream, serial fragment scheduling — D-16 honest note), top CPU-bound regions ranked (scipy resample ~28.8 s, postprocess 9.9 s, bootstrap 20.8 s); ncu documented unavailable (ERR_NVGPUCTRPERM). `02-BENCHMARK-REPORT.md` = the two-bar report + ranked Phase 3 handoff. See 2-gpu-acceleration-06-SUMMARY.md. **Plan 01 (CuPy preprocess port) DONE** (215b66a, 303a248): transpose/crop/element-wise-normalize on CuPy (fp32, C-contiguous, D-12), Z-score/CT reductions stay numpy (CuPy reductions not bit-exact), scipy resample unchanged (D-13, C_CONTIGUOUS assert at the transfer), `preprocess_reference` CPU fallback kept. Gates (D-11 final-gate-only): fullres E2E exit 0, SEG **99.99990% byte-identity vs ref_fullres_only (2 documented fp16↔fp32 boundary voxels)**, SR exact ("Airway Volume: 1 mL"); gpu_residency static + runtime + self-test all PASS with `preprocess_operator.py` deliberately allow-listed (D-13 reason string). Evidence: `.planning/phases/02-gpu-acceleration/plan01-gates/`. See 2-gpu-acceleration-01-SUMMARY.md. **Plan 02 (RMM pool + memory budget) DONE** (1a27e45, 88773fa): RMM is torch's active allocator (backend `pluggable`), imported before holoscan via `gpu_bootstrap` (first import) with the undefined-symbol hazard pinned by a subprocess self-test; `memory_budget: {JSON}` logged in compose() (INFR-03, headless unit tests force the defer branch); `warm_pool(budget.total_bytes)` at end of compose (D-14); ensemble `defer_strategy` branch code-reachable with D-19-identical math; INFR-02 deferred to Phase 3 (D-17), real OOM documented unexercised (D-15). Deviation: cudnn benchmark disabled under the pluggable allocator (torch 2.13 cacheInfo incompatibility — RMM wins over benchmark-mode parity). Gates: fullres E2E exit 0 + `strategy: full_volume`; precautionary pixel diff 99.99986% vs ref_fullres_only (3 boundary voxels). Evidence: `.planning/phases/02-gpu-acceleration/plan02-gates/`. See 2-gpu-acceleration-02-SUMMARY.md. **Plan 03 (cascade operator plumbing + model-list semantics) DONE** (0b265e1, c930280, 1d5d340): `resolve_run_model_list` replicates `nnunet_seg_operator.py:91-99` exactly (2d filtered, lowres-before-cascade reorder, ensemble = run minus 3d_lowres, reference ValueError when empty) + data-driven previous-stage auto-insertion (reference CRASHES on cascade-only — documented divergence; D-02: zero config-name literals in the insertion step); `PreprocessParams` cascade fields from the PlansManager-resolved configuration; `PreprocessOperator` optional `lowres_seg` input (cascade configs only) → image-bbox crop on GPU → CPU seg-resample replica (bit-exact vs vendored `resample_data_or_seg` seg path, np.array_equal on 3 regimes) → GPU one-hot (bit-exact vs `convert_labelmap_to_one_hot`) → 2-channel (image, one-hot) fp32 C-contiguous, zero disk I/O (D-09/D-10, integer-dtype guard); `PostResampleOperator` conditional `lowres_seg` output (emit_lowres_seg/emit_probabilities flags) via `revert_crop_gpu` (argmax uint8, original DICOM orientation, NO CC). Gates: 12-case unit suite exit 0 (real bundle + vendored nnunetv2 2.8.1); fullres-only E2E ×2 exit 0 + pixel_diff 99.99986% (3 boundary voxels). Deviations: reference-semantics ValueError controls over the plan's inconsistent 'standalone lowres ensemblable' bullet; resolved-config loading (raw cascade entry inherits fields). Evidence: `.planning/phases/02-gpu-acceleration/plan03-gates/`. See 2-gpu-acceleration-03-SUMMARY.md. **Plan 04 (multi-fragment DAG assembly) DONE** (f1e421b, 2bba495, bd49c61, 2309a4d, fb46cc6): one Subgraph per `resolve_run_model_list` entry (config-generic factory, zero config-name literals, D-02; `HOLOSCAN_MODEL_LIST` env var with reference default), cross-fragment `lowres_seg` cascade flow (zero disk I/O), app-level ensemble with ordered `prob_<cfg>` ports + `CountCondition(len)` + list-order reconstruction over the untouched Phase 1 bit-exact averaging (D-19 documented at source), per-subgraph `CudaStreamPool` (NonBlocking, named), config-tagged NVTX + app-keyed `StudyTimingCollector` (`_root` resolves sub-Fragments to the top-level app — Pitfall 9). Gates: 4/4 model-list configurations E2E exit 0 on the airway study (SEG/SR/SC, exactly one `study_timing_summary` each, per-config records exactly once); lowres-only SEG ≠ fullres-only SEG; fullres regression 99.99986% vs ref_fullres_only (3 boundary voxels); 14-case unit suite. Deviations: C++ Fragment + mixed app operators rejected by the 4.2 app_driver → Subgraph interface ports (the 4.2-supported mechanism, live-probed); Rule 1 fix: cascade one-hot 5D stack bug (seg[0], plan-03 latent); lowres-only runnable via documented self-ensemble fallback (plan 04 table controls over the reference ValueError). Evidence: `.planning/phases/02-gpu-acceleration/plan04-gates/`. See 2-gpu-acceleration-04-SUMMARY.md. **Plan 05 (per-config oracles + pixel-exact gates) DONE** (8adeef8, b423800, 4a956e4, bf0c39f, 1f2aa69, e53e026, c5cc593, 07fde68, 257fdb4): `reference_fullres_run.py --config` now takes a comma-separated model list; both missing per-config oracles generated with the reference app UNMODIFIED (`testdata/ref_lowres_only` 2404 post-CC voxels — reference raises on lowres-only, so the harness subclass reproduces the fast app's documented self-ensemble fallback; `testdata/ref_cascade_only` 2519 — `3d_lowres,3d_cascade_fullres` pin, lowres auxiliary); `phase2_gate.py` runs all 4 fast-app configurations + pixel_diff + SR compare + model-list assertions + residency static/runtime → `gates/02-GATE-RESULTS.json`. **ALL 4 GATES PASS: fullres 99.99986%/3 (documented fp16↔fp32 boundary, IoU 0.998714), lowres 100.00000%/0 (2404=2404), cascade 100.00000%/0 (2519=2519), BUNDLE 100.00000%/0 vs testdata/current_output (2447=2447, D-06)**; SR airway volume 0.0% delta on all rows (bar 0.1%); residency static + runtime PASS (multi-fragment bundle, D-13 boundary deliberate); sanity: bundle SEG ≠ fullres-only SEG (382 voxels). Deviations recorded for VERIFICATION.md: TEST-005 2d blocked-on-model (met-with-deviation, D-01/D-03), TEST-01 single dev-corpus (≥5-CT re-run deferred). Evidence: `.planning/phases/02-gpu-acceleration/gates/`. See 2-gpu-acceleration-05-SUMMARY.md.

### Success Criteria

- All four nnUNet configurations run as independent Holoscan Fragments within the same DAG.
- Cascade path (3D_lowres → 3D_cascade_fullres) feeds segmentation directly as a one-hot channel stack with no disk I/O.
- Preprocessing (transpose, crop, normalize) runs on GPU via CuPy.
- Measurable latency improvement vs. Phase 1 and baseline on the reference corpus.
- Pixel-exact equivalence maintained across all configs.

### Tasks

| # | Task | REQ-ID(s) | Notes |
|---|------|-----------|-------|
| 2.1 ✓ | Port transpose and crop to CuPy (GPU-accelerated) | PREP-01, PREP-04 | Done in plan 01 (215b66a) |
| 2.2 ✓ | Port normalization (mean/std per channel) to CuPy | PREP-02 | Done in plan 01 — element-wise on GPU, reductions on numpy (bit-exact) |
| 2.3 ✓ | Keep resampling on reference CPU path (scipy) — defer GPU resampling to v2 | PREP-03 | Done in plan 01 — unchanged, C_CONTIGUOUS assert, D-13 round-trip documented |
| 2.4 ✓ | Assemble multi-fragment DAG: each config (2D, 3D_fullres, 3D_lowres, 3D_cascade_fullres) as independent Fragment | PIPE-03 | DONE in plan 04 (bd49c61) — one Subgraph per config (4.2 app_driver forbids C++ Fragment + app-operator mixing), config-generic factory, `HOLOSCAN_MODEL_LIST` selection; 4/4 configurations E2E exit 0 |
| 2.5 ✓ | Wire cascade path: 3D_lowres segmentation → one-hot channel stack → 3D_cascade_fullres input | PIPE-04 | DONE: operator-level plumbing plan 03 (c930280, 1d5d340) + cross-fragment flow plan 04 (bd49c61) — lowres_seg crosses the sub-fragment boundary, zero disk I/O; Rule 1 fix 2309a4d (one-hot 5D stack) |
| 2.6 ✓ | Verify all configs produce pixel-exact output independently | TEST-005 | DONE in plan 05 — per-config fresh oracles (ref_lowres_only 2404 / ref_cascade_only 2519 post-CC voxels) + `phase2_gate.py`: lowres 100.00000%/0, cascade 100.00000%/0, fullres 99.99986%/3 (boundary class); 2d blocked-on-model = met-with-deviation (D-01/D-03, recorded in 02-GATE-RESULTS.json) |
| 2.7 ~ | Implement RMM pool pre-allocation in `setup()` | INFR-01, INFR-02 | INFR-01 done in plan 02 (1a27e45, 88773fa) — RMM pluggable allocator + warm pool sized by the budget; INFR-02 (cross-`compute()` reuse) deferred to Phase 3 (D-17) |
| 2.8 ✓ | Integrate `CudaStreamPool` for concurrent kernel launches | INFR-004 | DONE in plan 04 (bd49c61) — one pool per subgraph (NonBlocking, reserved_size=1, `streams_<cfg>`); overlap visibility = plan 06 nsys trace (D-16 best-effort) |
| 2.9 ✓ | Implement incremental in-place probability averaging for ensemble | INF-009 | DONE in plan 04 (2bba495) met-with-deviation per D-19 — per-config ordered `prob_<cfg>` inputs over the Phase 1 in-place accumulation + CuPy exact final division (bit-exact; running mean forbidden); deviation documented at the source |
| 2.10 ✓ | Add memory budget calculator: check VRAM headroom before full-volume allocation | INFR-03 | Done in plan 02 (88773fa) — BudgetPlan + headless unit tests forcing the defer branch; defer branch wired to the ensemble (D-15/D-19) |
| 2.11 ✓ | Run full pipeline on all configs across reference corpus, validate pixel-exact output | TEST-01, TEST-005 | DONE in plan 05 — all 4 configurations E2E + per-config/bundle pixel gates PASS (bundle 100.00000%/0 vs testdata/current_output, D-06); dev-corpus deviation + ≥5-CT re-run deferred, recorded in 02-GATE-RESULTS.json |
| 2.12 ✓ | Run updated benchmarks, compare against baseline | TEST-006, TEST-007 | DONE in plan 06 (fc7e977) — `.planning/scripts/phase2_benchmark.py` (fresh-process reps, 1 warmup + 3 measured, per-config columns from the app's `timing:` JSON logs): same-scope fullres **57,140.3 ± 250.4 ms = 1.082× vs Phase 1 61.8 s (D-18 positive-improvement bar MET)**; headline bundle **129,542.9 ± 896.4 ms = 1.310× vs 169.7 s reference** (inference stage 76.7 s vs 138.2 s, −44.5%); results in `.planning/benchmarks/phase2_results.csv` + two-bar report (bundle-vs-61.8 s scope asymmetry documented as the D-18 deviation) |
| 2.13 ✓ | Profile with Nsight Systems and Nsight Compute, identify remaining CPU-bound regions | INFR-005 | DONE in plan 06 (fb6cbea) — nsys bundle trace in `.planning/profiles/phase2/` (cuda_api_sum: 9 cudaMalloc / 1 cudaFree vs 629,929 kernels = no per-tile churn, RMM pool expansions only; 14/14 per-config NVTX ranges legible; top CPU-bound: scipy resample spans ~28.8 s, postprocess 9.9 s, bootstrap 20.8 s). **ncu: BLOCKED — `ERR_NVGPUCTRPERM`** (admin `NVreg_RestrictProfilingToAdminUsers=0` required; documented in `ncu_status.txt`, no fake kernel metrics). Stream overlap NOT visible (single stream, serial fragment scheduling — D-16 honest note, report §5) |
| 2.14 ✓ | Save profiling traces and reports to `.planning/profiles/phase2/` | — | DONE in plan 06 — `phase2_bundle_20260819_071235.nsys-rep` (+ .sqlite) + 4 stats exports + `ncu_status.txt` + `README.md` committed (Phase 0/1 convention: rep + sqlite tracked); Phase 3 scoping list in `02-BENCHMARK-REPORT.md` §6 |

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Cascade path doubles peak VRAM (lowres + fullres coexist) | High | Free lowres weights immediately after inference; profile memory peak |
| CuPy operations introduce subtle numerical differences | Medium | Run pixel diff after every CuPy port; compare intermediate values |
| Multi-fragment DAG wiring introduces deadlocks or ordering issues | Medium | Start with 2 configs, add incrementally; test with 1 study |
| RMM pool allocator not compatible with PyTorch's default allocator | Medium | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; test allocation patterns |
| Resampling on CPU still dominates preprocessing latency | Low (acceptable) | Document as known limitation; defer to v2 (GPUP-01) |

### Acceptance Criteria

- [x] All four nnUNet configurations run successfully on reference corpus — dev-corpus deviation: single airway study (≥5-CT re-run deferred, TEST-01 note; 2d blocked-on-model, D-01/D-03)
- [x] Cascade path (3D_lowres → 3D_cascade_fullres) runs without disk I/O for intermediate segmentation
- [x] DICOM-SEG output is pixel-exact for all configurations — plan 05: fullres 99.99986%/3 (documented fp16↔fp32 boundary class), lowres/cascade/bundle 100.00000%/0
- [x] DICOM-SR measurements match within 0.1% for all configurations — 0.0% delta on all 4 gate rows
- [x] End-to-end latency improved vs. baseline (quantified in benchmark report) — same-scope 57.14 s vs 61.8 s (1.082×, D-18 bar MET); headline bundle 129.54 s vs 169.7 s (1.310×) — report §2–§4
- [x] GPU residency test passes for all operators and all configs — plan 05 (static + runtime, multi-fragment bundle)
- [x] RMM pool allocator active; no per-tile `cudaMalloc` churn visible in Nsight traces — plan 06 cuda_api_sum: 9 mallocs total (pool expansions), 629,929 kernels, zero per-tile
- [~] Nsight Systems trace shows operator overlap and async execution — PARTIAL (D-16 honest note): overlap visible at light boundaries (postresample_3d_fullres ∥ lowres inference) but the heavy inference spans run serially on a single stream (back-to-back, 0.2 ms gap) — serial fragment scheduling is the blocker; Phase 3 candidate (report §5/§6)
- [x] Benchmark comparison report saved to `.planning/benchmarks/phase2_results.csv` — plan 06 (10 rows: 2 scopes × (1 warmup + 3 measured) + 2 mean±std summaries, both speedups)

---

## Phase 3: Optimization

**Timeline:** Weeks 8+ (conditional)
**Goal:** Push performance to the limit based on profiling data from Phase 2. Implement only optimizations that are validated by profiling as the actual bottleneck.

This phase is **time-permitting** and **data-driven**. If Phase 2 already meets latency targets, this phase may be skipped or trimmed.

**Plans (2026-08-19):** 5 plans across 5 waves in `.planning/phases/03-optimization/` — 01 (wave 1, D-21 EventBasedScheduler concurrency + RMM Open-Q1 re-verify/pin); 02 (wave 2, MEM-003 lowres weight release + pool/driver VRAM delta); 03 (wave 3, INFR-02 shape-keyed buffer reuse + D-24 (a)+(b) proof); 04 (wave 4, D-22 gated GPU-resample experiment — CuPy RawKernel scipy-faithful zoom behind HOLOSCAN_GPU_RESAMPLE, default OFF); 05 (wave 5, close-out: 2×2 benchmark + final gate suite + 03-BENCHMARK-REPORT.md). Locked scope (03-CONTEXT.md D-20..D-26): trimmed data-driven sweep — every plan ships behind a re-run of `phase2_gate.py` (D-25); TensorRT/torch.compile/MEM-01/MEM-02/ncu OUT of scope; ≥5-CT corpus, ncu admin, INFR-02 user examples = blocked-on-external non-blocking (D-26, recorded in the close-out report for VERIFICATION.md).

Plans:
- [ ] 3-optimization-01-PLAN.md — concurrent independent fragments (D-21) + RMM initial-pool re-verification (Open Q1)
- [ ] 3-optimization-02-PLAN.md — MEM-003: free 3d_lowres weights after the aux fragment + pool/driver VRAM measurement
- [ ] 3-optimization-03-PLAN.md — INFR-02: shape-keyed GPU buffer reuse (D-24 a+b proof)
- [ ] 3-optimization-04-PLAN.md — D-22: gated GPU-resample experiment (byte-identity vs scipy; default OFF)
- [ ] 3-optimization-05-PLAN.md — close-out: 2×2 benchmark (D-18 two bars) + final gate + 03-BENCHMARK-REPORT.md

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

- [x] Each optimization task has a Phase 2 profiling trace citation justifying it (completed 2026-08-19)
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
