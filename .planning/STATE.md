---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 2-gpu-acceleration-03-PLAN.md (cascade operator plumbing + reference model-list semantics; PIPE-03/PIPE-04 operator level, 12-case unit suite, fullres E2E clean)
last_updated: "2026-08-19T08:15:38.000Z"
last_activity: 2026-08-19
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 12
  completed_plans: 9
  percent: 75
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-14)

**Core value:** Single-study inference latency without sacrificing correctness — CT in, pixel-identical DICOM-SEG out, faster, every intermediate step stays on GPU.
**Current focus:** Phase 1 (Core Pipeline) — **COMPLETE** (plans 01–05 all done; Phase 2 next)

## Current Position

Phase: 02 of 3 (gpu acceleration)
Plan: 4 of 6
Status: Executing Phase 02
Last activity: 2026-08-19

Progress: [███████░░░░] 75% (Phase 1 complete; Phase 2 plans 01–03 done, plans 04–06 pending)

## Transition Log

- **2026-08-19 Phase 2 Plan 03 complete (cascade operator plumbing + model-list semantics):** `resolve_run_model_list` (config/__init__.py) replicates `nnunet_seg_operator.py:91-99` exactly — plans.json order filtered to model dirs (2d filtered out), reference lowres-before-cascade reorder, ensemble = run minus 3d_lowres, reference's exact ValueError when empty — plus the DATA-DRIVEN previous-stage auto-insertion (`['3d_cascade_fullres']` → `['3d_lowres','3d_cascade_fullres']`; the reference CRASHES on cascade-only — the documented fast-app divergence; zero config-name literals in the insertion step, D-02). `PreprocessParams` gained previous_stage / resample_seg_order(_z) / resample_seg_force_separate_z / foreground_labels, loaded from the PlansManager-RESOLVED configuration (Rule 3: the raw cascade entry inherits spacing + seg kwargs — raw `cfg['spacing']` crashed). `PreprocessOperator`: optional `lowres_seg` input declared ONLY for cascade configs (params.previous_stage, Pitfall 7), seg takes the image's exact layout chain + image-derived bbox crop (GPU), ~16 MB uint8 D2H (D-13), `_resample_seg_to_shape` = bit-exact replica of the vendored `resample_data_or_seg` seg path (resize_segmentation per-label multihot, no anti-aliasing, integer dtype preserved, short-circuit) — `np.array_equal` vs vendored on 3 random uint8 regimes (short-circuit / plain / separate-z+map_coordinates); GPU one-hot `(seg==lbl).astype(fp32)` per foreground_labels — `np.array_equal` vs vendored `convert_labelmap_to_one_hot` (incl. the CuPy op itself); `cp.concatenate([image, one_hot], axis=0)` → (2,*new_shape) fp32 C-contiguous, one-hot never normalized, zero disk I/O; D-10 integer-dtype guard at the boundary (probabilities forbidden). `PostResampleOperator`: `emit_lowres_seg`/`emit_probabilities` flags (before super().__init__, Pitfall 7 port gating), `revert_crop_gpu` (0-fill + bbox insert + transpose_backward, Rule 1 fix: 3D perm has no channel +1), seg = `torch.argmax(softmax-probs, dim=0).to(uint8)` BEFORE revert (== reference argmax-of-resampled-logits, softmax monotone; NO CC — reference cascade input is pre-CC, D-09), 3D uint8 CUDA in original DICOM orientation = same array order as image.asnumpy() (the system-wide orientation contract for lowres_seg). Gates: 12-case unit suite exit 0 vs the REAL bundle + vendored nnunetv2 2.8.1; fullres-only E2E ×2 (post-Task 2, post-Task 3) exit 0 + pixel_diff **99.99986% byte-identity vs ref_fullres_only (3 documented fp16↔fp32 boundary voxels, same coordinates as Phase 1/Plan 02)**, SR exact. Plan-text deviations (see SUMMARY): the task-1 bullet 'standalone lowres is ensemblable' contradicts the replicated reference + the plan's own must-have → implemented/tested the reference ValueError; the revert_crop_gpu test's 'output shape == shape_before_cropping' is only true for identity tf → test asserts np.array_equal vs the numpy reference. Evidence: `.planning/phases/02-gpu-acceleration/plan03-gates/`. Commits 0b265e1 + c930280 + 1d5d340. See 2-gpu-acceleration-03-SUMMARY.md.
- **2026-08-19 Phase 2 Plan 02 complete (RMM + memory budget):** `gpu_bootstrap.py` is the FIRST import in app.py (rmm before holoscan — the `undefined symbol: __cxa_call_terminate` hazard is live-pinned by `scripts/test_gpu_bootstrap.py` subprocesses: rmm-first → `pluggable`, holoscan-first → `undefined symbol`); compose() logs `memory_allocator_backend: pluggable` + asserts, then logs `memory_budget: {JSON}` from the new pure-Python `compute_memory_budget` (INFR-03: per-config preprocessed+logits+probabilities fp32 estimate × 1.15 vs free VRAM → `full_volume` | `defer_to_incremental`), and warms the RMM pool to `plan.total_bytes` at the end of compose (D-14). Ensemble gained `defer_strategy` (default False = byte-for-byte Phase 1 path; defer branch frees each consumed per-config tensor with IDENTICAL accumulation order + CuPy exact final division — D-19 bit-exactness; real OOM documented UNEXERCISED, D-15). INFR-02 explicitly DEFERRED to Phase 3 (D-17) — documented, not implemented. Headless unit tests (`test_mem_budget.py`) force the defer branch with synthetic (4,600,512,512) volumes. **Blocker found + fixed:** torch 2.13's cudnn benchmark search calls the pluggable allocator's unsupported `cacheInfo` (RuntimeError on the first conv, size-independent, reproduced outside holoscan) → `cudnn.benchmark` disabled when backend is `pluggable` (RMM wins over benchmark-mode reference parity); expandable_segments NOT shipped (Pitfall 6). Gates: fullres-only E2E exit 0 with `strategy: full_volume`; precautionary pixel diff after the benchmark change: SEG **99.99986% byte-identity vs ref_fullres_only (3 documented fp16↔fp32 boundary voxels)**, IoU 0.998714. Evidence: `.planning/phases/02-gpu-acceleration/plan02-gates/`. Commits 1a27e45 + 88773fa. See 2-gpu-acceleration-02-SUMMARY.md.
- **2026-08-19 Phase 2 Plan 01 complete (CuPy preprocess port):** `preprocess_image` now runs transpose (PREP-01) + crop (PREP-04) + element-wise normalize (PREP-02) on CuPy in fp32 C-contiguous (D-12); one H2D of the raw volume via `cp.array` (not `cp.from_dlpack` — ownership), on-device int→fp32 cast; Z-score/CT mean-std reductions stay numpy (CuPy reductions not bit-identical — Pitfall 4); masked-assignment semantics preserved on GPU via boolean indexing (inactive for this bundle but config-generic); scipy `_resample_to_shape` byte-identical (PREP-03/D-13) with `C_CONTIGUOUS` fp32 assert + D-13 comments at both transfer sites; `preprocess_reference` kept as CPU fallback. Gates (D-11 final-gate-only, no per-op checks): fullres E2E exit 0 (SEG/SR/SC); pixel_diff vs `testdata/ref_fullres_only`: **99.99990% byte-identity, 2 differing voxels** (documented fp16↔fp32 argmax-boundary class; Phase 1 measured 99.99986%/3), IoU 0.999142; SR exact ("Airway Volume: 1 mL"); gpu_residency static + runtime + self-test all PASS — `ALLOWED_TRANSFER_FILES` deliberately gained `preprocess_operator.py` with a D-13 reason string (no existing entry weakened). Evidence: `.planning/phases/02-gpu-acceleration/plan01-gates/`. Commits 215b66a + 303a248. No deviations. See 2-gpu-acceleration-01-SUMMARY.md.

- **2026-08-19 Phase 2 context gathered:** `02-CONTEXT.md` locked: gate on 3×3D configs (2D blocked-on-model, generic wiring, no dummy model); per-config fresh references (ref_lowres_only, ref_cascade_only) + final bundle gate vs testdata/current_output; cascade = lowres argmax → one-hot float → cascade_fullres channels (zero disk I/O); CuPy port validated by final pixel-exact gates only (fp32/C-contiguous, scipy resample stays CPU with accepted GPU↔CPU round-trip); RMM + budget calculator (synthetic-size unit tests) required, CudaStreamPool best-effort, latency bar = any positive E2E vs 61.8 s with per-operator deltas; INFR-02 cross-study buffer reuse deferred to Phase 3 (user adding reference examples then).

- **2026-08-17 Phase 0 → Phase 1:** gate passed (VERIFICATION passed, 5/5 acceptance, TEST-01 corpus deviation documented). PROJECT.md updated: Phase 0 requirements moved to Validated (TEST-006/TEST-007), cu13/resampling/Docker-deferral/baseline decisions logged. Phase 1 is now active and ready for planning (inputs staged in "Next — Phase 1").
- **2026-08-18 Plan 02 complete:** SlideWindowOperator — setup-time one-shot model load (architecture from plans.json, 5 fold weights from resolved checkpoint path), TTA in exact nnUNet order with FP32 sequential accumulation, reference-parity steps/gaussian/autocast boundaries, config- and checkpoint-driven (InferenceParams). Airway study: logits max abs diff 4.539e-01 vs reference (fp16-ref vs fp32-ours; plan's ~1e-6 assumes an fp32 reference — unreachable, seg gate is controlling) and **100.00000% voxel-identical segmentation** (16,646,400/16,646,400). Inference 27.1 s/study, zero cold start on study 2+. Deviations: nnUNet pure SW utilities instead of MONAI swi (monai 1.3.0 kernel/step divergence, measured); per-fold autocast boundary (torch 2.13 corrupts forwards following a mid-autocast load_state_dict — one-loop autocast gave 13.2 diff). See 1-core-pipeline-02-SUMMARY.md.
- **2026-08-18 Plan 03 complete:** PostResampleOperator + EnsembleAverageOperator + PostprocessOperator. Resample+softmax bit-exact vs the reference export path (required replicating the reference's `set_num_threads(default_num_processes)` scope around torch CPU softmax); in-memory GPU ensemble average in reference accumulation order with CuPy bit-exact final division (torch CUDA `/= n` is 1-ulp off numpy for non-power-of-2 n) and argmax-after-average; postprocess = custom deterministic CuPy two-pass min-seed CC (26-conn; no GPU skimage in venv) with MONAI keep-largest + acvl rule parity, zero-copy DLPack, exactly-once CPU transfer. Gates: identical label mask vs reference postprocess (exact), pre-CC segs 100% identical, **full E2E final seg 100.00000% voxel-identical vs a fresh in-harness reference run**, SR text exact. 6 auto-fixes incl. CC component-count off-by-one (hid 2-component inputs) and DLPack ownership (cp.from_dlpack consumes the caller's torch buffer — clone added). See 1-core-pipeline-03-SUMMARY.md.
- **2026-08-18 Plan 04 complete:** DAG assembly in app.py — NNUnetSegOperator replaced by Preprocess → SlideWindow → PostResample → EnsembleAverage → Postprocess (15 flows), reference DICOMSeriesSelector/SC-writer copied, SC overlay side-output added to PostprocessOperator (LabelToContour + jet + alpha, SaveImaged .dcm), ensemble emits uint8 seg. NVTX verified in an Nsight trace (preprocess/inference/postresample/ensemble_average/postprocess); structured timing records {operator, study, start, end, duration_ms} + per-study aggregate incl. writers. E2E airway run exit 0, SC/SEG/SR, 3655 voxels, SR exact. **Gate-calibration finding:** testdata/current_output is the reference app's FULL-BUNDLE run (3d_fullres + 3d_cascade_fullres ensemble → 2447 voxels); Phase 1 scope is single-config 3d_fullres (3655) — Plan 05's pixel-exact gate must use a 3d_fullres-only reference. See 1-core-pipeline-04-SUMMARY.md.
- **2026-08-18 Plan 05 complete (Phase 1 gate):** pixel_diff.py (raw-byte + decoded-voxel SEG comparison, exit codes, JSON) + gpu_residency.py (static AST + runtime frame-attribution + self-test; PASS: exactly 1 boundary .cpu(), 0 illegal transfers). Gate oracle = reference_fullres_run.py (3d_fullres-only, testdata/ref_fullres_only). **Two gate-found correctness fixes:** (1) F-contiguous preprocess input → 1-ulp divergence at ~16M voxels — np.ascontiguousarray after transpose → preprocessed tensor now bit-identical to reference (d881fe2); (2) SEG payload must be the reference's per-slice 2D-Laplacian contour in reference-internal orientation seg.transpose(2,1,0) (reference LabelToConturd before the SEG write; SC uses the same contour — fast used a 3D Laplacian); orientation chain forensically recovered (SDK writer x→x.transpose(2,0,1)[::-1]) (b6c2f4d). **Gate result:** SEG 99.99986% byte-identity vs fresh fullres-only reference (3 differing voxels = 1 solid argmax voxel at the documented fp16↔fp32 boundary; reference itself is run-to-run deterministic — two fresh runs bit-identical); SC bit-identical under frame-axis transpose; SR exact ("Airway Volume: 1 mL"). Baseline: 5 measured runs 61.2–62.2 s E2E (in-study 42.1 s: inference 27.2 s dominant) → .planning/benchmarks/baseline-2026-08-18.csv. **PHASE 1 COMPLETE.**

## Phase 0 Acceptance Status

| # | Criterion | Status |
|---|-----------|--------|
| 1 | pyproject.toml + all cu13 deps resolved in venv | ✓ met — import chain verified 2026-08-14; monai 1.3.0 + itk 5.4.7 restored/added 2026-08-17 |
| 2 | Reference corpus with DICOM-SEG/SR ground truth | ✓ met (deviation) — single airway MR series (256 slices) in `testdata/airway_input` + SC/SEG/SR in `testdata/airway_output`; original ≥5-CT bar deferred to final Phase 1 gate (TEST-01 note) |
| 3 | Baseline benchmark CSV at .planning/baseline_results.csv | ✓ met — 2026-08-17: 169,747 ± 7,274 ms/study (n=3, warmup excluded); per-stage columns populated |
| 4 | Nsight harness produces valid trace | ✓ met — demo trace .planning/profiles/nsight_demo_target_20260817_111555.nsys-rep (+.sqlite); NVTX preprocess/inference/postprocess ranges verified in trace |
| 5 | RMM pool allocator verified active | ✓ met — 2026-08-17: driver 610.57.04 (CUDA 13.3), A100-SXM4-40GB, test_rmm.py PASSED (backend 'pluggable') |

Done: app scaffold (examples/apps/cchmc-nnunet-fast, standard MAP layout, app.py skeleton), cu13 pins (commit 7d5e687), Nsight/NVTX harness (d99deb4), RMM + venv validation scripts (8910ae8, 205fcd3), reference-app E2E env repairs + baseline + demo trace (2026-08-17, commits 668ef0f..e2c38be + close-out).

## Performance Metrics

**Velocity:** 8 planned GSD plans executed (Phase 1, Plans 01–05; Phase 2 Plans 01–03): 57 min + ~105 min + ~228 min + ~40 min + ~95 min + ~40 min + ~40 min + ~31 min wall (subagent, 3–5 tasks / 2–4 commits each). Phase 1 gate passed. Phase 2 Plan 03 (cascade plumbing): ~31 min, 3 tasks / 3 commits — interface-first plan, so gates were the 12-case unit suite (vs real bundle + vendored reference) + 2× fullres E2E regressions; no cascade E2E yet (Plan 04 assembles the DAG).

**By Phase:**

| Phase | Plans | Total     | Avg/Plan |
| ----- | ----- | --------- | -------- |
| 0     | 0     | 0         | -        |
| 1     | 5     | 525 min   | 105 min  |
| 2     | 3/6   | 111 min   | 37 min   |

**Recent Trend:** plan 05 (validation gate) is where the real correctness work happened — the gate tools did their job, catching two bugs that 4 plans of component-level verification had missed (F-contiguous preprocess view; solid-vs-contour SEG payload + orientation). Lesson: component bit-exactness ≠ end-to-end bit-exactness; the last-mile transforms (writer axis mapping, transform order) need the full oracle in the loop.

## Accumulated Context

### Decisions

Logged in PROJECT.md Key Decisions table. Recent:

- [Phase 2, Plan 03]: System-wide lowres_seg orientation contract: uint8 CUDA, 3D, same array order as image.asnumpy() — PostResample's revert_crop_gpu emits it (transpose_backward, 3D form with NO channel +1), Preprocess consumes it via the image's exact layout chain (so the image-derived bbox applies verbatim); the cascade 2-channel input is (image, one-hot) fp32 C-contiguous with the one-hot channel never normalized
- [Phase 2, Plan 03]: Reference semantics control over plan text when they conflict: resolve_run_model_list(['3d_lowres']) raises the reference's exact ValueError (empty ensemble) — the plan's 'standalone lowres is ensemblable' bullet contradicted both the replicated reference and the plan's own must-haves; the auto-insertion step is data-driven off plans 'previous_stage' (D-02: zero config-name literals)
- [Phase 2, Plan 03]: load_preprocess_params reads ALL per-config fields from the PlansManager-resolved configuration (the raw cascade entry carries only inherits_from + previous_stage — spacing/normalization/resampling kwargs are inherited); post-softmax argmax == reference argmax-of-resampled-logits (softmax monotone per voxel, D-09), no CC on the cascade input (reference KeepLargestCC runs only on the final output)
- [Phase 2, Plan 02]: cudnn benchmark mode is INCOMPATIBLE with the RMM pluggable allocator on torch 2.13 (benchmark search calls unsupported `cacheInfo` → RuntimeError on the first conv, size-independent) — `cudnn.benchmark` is disabled in slidewindow setup when the backend is `pluggable`; RMM (INFR-01/D-14) wins over the reference's benchmark=True parity; pixel-exactness held (99.99986% SEG, 3 boundary voxels)
- [Phase 2, Plan 02]: the setup-time budget estimate uses plans.json `median_image_size_in_voxels` as the preprocessed shape (study volume size is unknown at compose time; inherited configs fall back to the original median shape × spacing ratio); crop bounded by the resampled volume (crop ⊆ image)
- [Phase 2, Plan 02]: the defer branch releases per-config tensors as consumed but keeps the exact accumulation order + CuPy final division (bit-identical to the full_volume path — D-19); the defer unit test uses 5 configs of the synthetic (4,600,512,512) shape because the plan's 3-config sketch totals only ~24.4 GB at 40 GB free under its own formula
- [Phase 2, Plan 01]: CuPy port keeps every *reduction* on numpy (CuPy reduction order ≠ numpy — bit-exactness verified empirically in 02-RESEARCH) and materializes every transpose/slice with `cp.ascontiguousarray` (D-12); H2D uses plain `cp.array` (CUDA array interface copy), deliberately not `cp.from_dlpack` (ownership-transfer pitfall from Phase 1)
- [Phase 2, Plan 01]: Masked Z-score normalization (inactive for this bundle, `use_mask_for_norm=[False]`) implemented with the reference's masked-assignment semantics via GPU boolean indexing — config-generic for future bundles
- [Phase 2, Plan 01]: `gpu_residency.py` allow-list extended deliberately (not silenced) — `preprocess_operator.py` entry carries the D-13 reason string; the exactly-once postprocess boundary `.cpu()` remains the only final-stage transfer

- [Init]: New example app, not SDK core modification (lower risk, prove concept first)
- [Init]: Latency first, throughput later (single-study clinical workflow)
- [Phase 0]: Pin every CUDA dependency to cu13; cupy-cuda12x must not coexist
- [Phase 0]: Venv repair 2026-08-14: holoscan-cu13 4.2.0 force-reinstalled (install was corrupted — 118 files from the wheel's RECORD missing on disk, incl. all GXF runtime .so and the holoscan.core/conditions/executors modules); holoscan-cli 4.2.0 added; pydicom → 3.0.2 (SDK uses 3.x API), highdicom → 0.28.1 (0.22.x breaks on pydicom 3.x). GXF runtime libs are bundled in the holoscan-cu13 wheel — no separate package. Documented in app README.
- [Phase 0]: Resampling stays on reference CPU (scipy) path in Phases 1–2 — pixel-exactness over speed
- [Phase 0]: Driver cleared 2026-08-17: host updated to 610.57.04 (CUDA 13.3); A100-SXM4-40GB; test_rmm.py passes (pluggable allocator, not literal 'cudaAsync' string — the criterion's intent, RMM active, is met)
- [Phase 0]: Reference-app env repairs 2026-08-17 (needed to run the baseline): (1) `nnunet_bundle.py` `build_network_architecture` call adapted to vendored nnunetv2 2.8.1 signature (glue only, no math); (2) venv monai reverted 1.6.0 → 1.3.0 per the app's own pin (`monai[einops]==1.3.0`) — 1.6.0 had drifted in and removed/changed writer plumbing; (3) `itk 5.4.7` installed so MONAI `SaveImaged(output_ext='.dcm')` (SC overlay write) resolves an ITKWriter — MONAI has no native DICOM writer, the original env must have had ITK; (4) patched 2 NumPy-2.0-removed `ndarray.ptp()` call sites in venv monai 1.3.0 (`data/utils.py:906`, `transforms/spatial/functional.py:376` → `np.ptp(arr, axis=1)`) — venv is scratch (/tmp), re-apply after any monai reinstall
- [Phase 0]: Baseline provenance caveat: testdata/output ground truth was generated by an earlier env (pre-2.8.1 nnunetv2, unknown MONAI). After the repairs, Phase 1 validation should diff the new app against FRESHLY regenerated reference output (same env, same nnunetv2 2.8.1) as the primary gate, with testdata/output as the historical reference
- [Phase 0, 2026-08-17 evening]: testdata replaced by user with airway-consistent data: `testdata/airway_input` (256 MR slices, 256×256, patient 12345678) + `testdata/airway_output/{SC,SEG,SR}` (1 airway SEG, 256 frames, 2430 voxels); models moved to `examples/apps/cchmc_nnunet_fifteen_ckpt_app/models` = `MRI_NICU-Airway_TRAINv2` (airway=1), configs present: 3d_fullres, 3d_lowres, 3d_cascade_fullres (2d listed in plans.json but absent). Reference app runs end-to-end on it (exit 0, SC/SEG/SR produced, ~124 s).
- [Phase 0, FINDING — RESOLVED 2026-08-17 20:40 UTC]: the earlier "reference app does NOT reproduce GT (~45 mm world-COM offset, zero overlap)" note was a **decoding artifact**, not a real mismatch. A fresh reference run (`testdata/current_output`) was made and compared to the historical GT (`testdata/airway_output`): DICOM-SEG segment pixel data is **99.902% byte-identical** (2,095,090/2,097,152), segment voxel counts 2430 (GT) vs 2447 (fresh, Δ0.7%), all differences confined to the airway band (frames 106–225). The airway is a thin 1-voxel structure so IoU is hypersensitive to sub-voxel registration — the raw pixel data confirms a match. The ensemble-config question is moot for the correctness gate. See `.planning/phases/00-foundation/0-foundation-VERIFICATION.md` + `.planning/scripts/REFERENCE_RUN_GUIDE.md`.
- [Phase 0]: **Docker build + container test deferred** until after Phase 3 optimizations are in place (decided 2026-08-17). Pythonic reference runs (`.planning/scripts/REFERENCE_RUN_GUIDE.md`) are the validation path until then.
- [Phase 0]: Phase 0 **closed in the GSD cycle** 2026-08-17 — backfilled `.planning/phases/00-foundation/` (01-PLAN, 01-SUMMARY, VERIFICATION status: passed). Credits TEST-006 (benchmark script) + TEST-007 (baseline) as satisfied; PIPE-01/INFR-01/INFR-005 tracked as partial (scaffold/feasibility only, full satisfaction in Phases 1–2).
- [Phase 1, Plan 01]: GPU handoff contract uses `holoscan.core.Tensor` (DLPack, `device_type == kDLDeviceCUDA`) because `MemoryData` does not exist in the holoscan-cu13 4.2 Python API — zero-copy equivalent. Plan 02+ consume the `preprocessed` tensor (back to torch via DLPack) + `preprocessed_meta` dict (bbox, pre-crop shape, spacing, permute) for post-revert.
- [Phase 1, Plan 01]: Resampling output dtype captured from plans `dtype_out` (float32) before nnUNet's internal float64 upcast — Rule 1 bug fix; verified bit-exact (max abs diff 0) vs `DefaultPreprocessor.run_case_npy` on the airway study (3D_fullres).
- [Phase 1, Plan 02]: MONAI `sliding_window_inference` not used as-is — monai 1.3.0's step generator and Gaussian kernel measurably diverge from the vendored nnunetv2 2.8.1 (normalized-kernel max abs diff 0.034 on the 128³ patch; different step sets on non-dev shapes); the operator runs the same MONAI-style loop with nnUNet's pure utilities (`compute_steps_for_sliding_window`, `compute_gaussian`, `pad_nd_image`).
- [Phase 1, Plan 02]: Autocast boundary = per-fold (reference parity); each fold's `load_state_dict` runs OUTSIDE any active autocast. A single autocast around the whole fold loop corrupts the forward following mid-autocast weight loads on torch 2.13 (reproduced on a minimal net; fold-loop diff 13.2 → 4.54e-01 after fix).
- [Phase 1, Plan 02]: The vendored nnUNet 2.8.1 reference accumulates in FP16 (sliding window, visit counts, TTA sum — source-verified); plan INF-004 deliberately pins FP32 → logits max abs diff 4.539e-01 vs the fp16 reference on the airway study, segmentation (softmax+argmax) 100.00000% voxel-identical. The plan's ~1e-6 logits tolerance is unreachable against an fp16 reference; the seg identity gate is controlling (and plan 05's DICOM-SEG pixel-exact gate is final).
- [Phase 1, Plan 03]: Resample stays on the reference CPU (scipy) path per the Phase 0 decision — bit-exactness additionally required replicating the reference's `torch.set_num_threads(default_num_processes)` scope around the CPU softmax (torch CPU softmax is not thread-reproducible: 2-ulp flips at ~70–80 voxels).
- [Phase 1, Plan 03]: No GPU skimage in the venv (no cupyx.scimage/cucim/cuml) → custom deterministic CuPy two-pass min-seed CC, full 26-connectivity (the effective connectivity of `skimage.measure.label(connectivity=None)`, verified empirically); voxel-identical to MONAI keep-largest (incl. highest-feature-id tie-break) and nnUNet acvl (keep-all-max-tied).
- [Phase 1, Plan 03]: DLPack transfers buffer ownership — `cp.from_dlpack` consumes the caller's torch tensor (observed: zombie handle read back in-place-written values). `postprocess_gpu` clones before the transfer; the DAG reuses upstream tensors.
- [Phase 1, Plan 03]: The bundle's `postprocessing.pkl` contains 0 rules (verified) — the effective reference CC step is the app's `KeepLargestConnectedComponentd(applied_labels=[1])`; the pkl-rules machinery is generic (interpreted, unknown rules raise) for future bundles.
- [Phase 1, Plan 03]: E2E residual vs the reference is the upstream fp16-vs-fp32 argmax boundary (Plan 02 INF-004) plus measured reference run-to-run nondeterminism (~1 voxel); post-inference math itself is bit-exact (per-stage + label-mask level). Final pixel-exactness = plan 05's DICOM-SEG gate vs a fresh reference.

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 0] Reference corpus gap (task 0.3): testdata/input is 1 MR series (62 dcm) with complete SC/SEG/SR ground truth — usable as a single-study dev corpus, but the ≥5 CT studies criterion is unmet and no CT data exists in the repo. Decision needed: relax criterion to the single MR study (document deviation from TEST-01) or obtain ≥5 CT studies.
- [Phase 0] Baseline benchmark script not written (task 0.4) — driver gate is cleared, 0.4/0.5 can proceed now against the single-study dev corpus.
- [Phase 0] Models present: testdata/models has 2d + 3d_fullres only (5-fold ensembles each + nnunet_checkpoint.pth, jsonpkls plans/dataset/postprocessing). 3d_lowres + 3d_cascade_fullres absent — Phase 2/3 multi-config work is gated on those checkpoints.
- [Phase 0, HAZARD] The new app's editable install (`cchmc-nnunet-fast` 0.1.0 in the venv) registers a meta-path finder that maps the package name `my_app` to `examples/apps/cchmc-nnunet-fast/my_app`. Any `python -m my_app` run from a cwd that does NOT contain a local `my_app/` package silently executes the NEW skeleton instead of the intended app (hit 2026-08-17 while benchmarking the reference app). Mitigation: always run apps from their own app-root directory so the local package wins resolution. Longer-term: give the new app a unique package name (e.g. `cchmc_nnunet_fast_app`) to remove the collision.
- [Phase 0] PROJECT.md previously marked Phase 0 "validated" (commit 19b4a94) before the driver regression surfaced; corrected 2026-08-14 to partial.

## Session Continuity

Last session: 2026-08-19 08:15 UTC
Stopped at: Completed 2-gpu-acceleration-03-PLAN.md (cascade operator plumbing + reference model-list semantics; PIPE-03/PIPE-04 operator level, 12-case unit suite, fullres E2E clean)
Resume file: None

## Next — Phase 1 (Core Pipeline)

Phase 1 (Core Pipeline) is now **PLANNED** — 5 plans across 4 waves in `.planning/phases/01-core-pipeline/`:

- **01** (wave 1): PreprocessOperator + GPU handoff contract (PREP-01..05, INF-005)
- **02** (wave 2): SlideWindowOperator / inference core — model load in setup, TTA order, FP32 accumulator (INF-001..008, INF-011)
- **03** (wave 2): PostResample + EnsembleAverage (in-memory, no disk) + Postprocess (CC on GPU) (POST-01..03, INF-009/010)
- **04** (wave 3): DAG assembly in app.py (replace NNUnetSegOperator) + NVTX + timing logs (PIPE-01/02/05, INFR-005/006)
- **05** (wave 4): validation tools (pixel-diff, GPU-residency) + pixel-exact E2E gate + SR comparison (TEST-01 dev-study, TEST-002..004)

Scope: single config `3D_fullres`; operators built config-driven so Phase 2 adds 2D/lowres/cascade. Gate = pixel-exact vs freshly regenerated reference (`testdata/current_output` == historical GT).

Next:

1. **`/gsd-execute-plan 1 05`** (validation tools + pixel-exact E2E gate + SR comparison, wave 4). CRITICAL gate input (Plan 04 finding): compare against a **3d_fullres-only reference run** (e.g. reference app with `model_list=['3d_fullres']`, as Plan 03 did in-harness) — `testdata/current_output` is the full-bundle cascade ensemble (2447 voxels) and is out of scope for Phase 1 (3655 voxels).
2. Carried env facts:
   - App runs: from `examples/apps/cchmc-nnunet-fast`, `ulimit -s unlimited`, `/tmp/monai-env/.venv/bin/python my_app -i <dicom> -m <bundle> -o <out>` (my_app name-collision + 32 MB stack hazards)
   - venv monai ptp patches (re-apply after monai reinstall); baseline to beat: 169.7 ± 7.3 s/study
   - Docker build + container test deferred to post-Phase-3 — validate pythonically in Phase 1
3. **Carried blocker (not a plan task):** TEST-01's ≥5-CT-study corpus is the final Phase 1 acceptance gate — blocked on CT data. The single airway study (dev) gate is what plans 01–05 verify now.

> Note: shazam LSP "client not started" noise seen on 2026-08-15 was stale in-process manager state after an external pyright-langserver kill — not a code issue; cleared by a fresh pi session.
