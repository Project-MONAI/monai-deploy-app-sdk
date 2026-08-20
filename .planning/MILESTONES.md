# Milestones

Completed and in-flight milestones, most recent last.

---

## v1.0 — Holoscan-native nnUNet inference

**Date:** 2026-08-14 → 2026-08-20 (COMPLETE, all phases verified)
**Phases:** 4 — 00-foundation, 01-core-pipeline, 02-gpu-acceleration, 03-optimization
**Plans:** 17 (5 + 6 + 5 + 1 foundation scaffold) — **all plans executed**, all phase gates passed
**Archive:** [v1.0-ROADMAP.md](./milestones/v1.0-ROADMAP.md) · [v1.0-REQUIREMENTS.md](./milestones/v1.0-REQUIREMENTS.md) · [v1.0-MILESTONE-AUDIT.md](./milestones/v1.0-MILESTONE-AUDIT.md)

### Headline Achievements

- **End-to-end DICOM → DICOM-SEG GPU pipeline** — `cchmc-nnunet-fast` replaces the monolithic `NNUnetSegOperator` (CPU + `.npz` disk I/O) with a Holoscan-native operator chain (Preprocess → SlideWindow → PostResample → EnsembleAverage → Postprocess) and zero SDK core edits; exactly one authorized CPU transfer at the pipeline boundary.
- **All 3D nnUNet configs + cascade + default bundle, pixel-exact vs reference oracles** — one Subgraph per config in a single multi-fragment DAG; cross-fragment cascade handoff (lowres seg → one-hot channels) with zero disk I/O; final gates: fullres 99.99986% (3 documented fp16↔fp32 boundary voxels), lowres/cascade/bundle 100.00000% (0 voxels), SR 0.0% delta, GPU residency static+runtime PASS.
- **RMM + buffer reuse** — RMM pluggable allocator (import-order hazard pinned), 4 GiB initial-pool pin, memory-budget calculator with defer strategy, and shape-keyed cross-study GPU buffer reuse proven by 3× same-study replay (data_ptr stable, cudaMalloc 0/0 on studies 2/3, byte-identical repeats); MEM-003 lowres weight release after the aux fragment.
- **Concurrent fragments** — `EventBasedScheduler(5)` default ON behind `HOLOSCAN_CONCURRENT_FRAGMENTS`; nsys evidence of 49.8 s fullres∥lowres inference overlap (vs Phase 2's single-stream back-to-back).
- **GPU resample (stock CuPy), default ON** — `HOLOSCAN_GPU_RESAMPLE` ships ON after the amended D-22b ≥99% per-tensor gate came back 100.0000% (max abs diff 0) and the ON gate was pixel-identical to baseline; resample-dominated spans collapsed 28.8 s → 9.65 s (bundle).
- **Two-bar latency, final shipping config:** **bundle 104,180 ms = 1.629× vs the 169,747 ms reference baseline (1.243× vs Phase 2)**; **same-scope fullres 49,673 ms = 1.244× vs Phase 1 61.8 s (1.150× vs Phase 2 57.14 s)**. No regressed matrix cell; zero pixel-regressions shipped across all three verification passes.

### Known Gaps / External Dependencies

Non-blocking by construction (D-26 pattern — recorded, re-opens as a gap plan if the dependency lands):

1. **≥5-CT corpus re-run (TEST-01 final gate)** — blocked on CT data. All pixel gates verified on the single airway dev study per the Phase 0/1 documented deviation; `pixel_diff.py` + `reference_fullres_run.py` are corpus-agnostic, so the re-run is operational, not code.
2. **ncu admin access (`ERR_NVGPUCTRPERM`)** — kernel-counter profiling requires `NVreg_RestrictProfilingToAdminUsers=0` (sudo). This is what keeps ACCEL-01/02/03 deferred; nsys `cuda_gpu_kern_sum` is the standing ceiling (inference kernels already 91–96% GPU-busy).
3. **2d model validation (blocked-on-model, D-01/D-03)** — the bundle has no 2d checkpoint; fragment wiring is config-generic (D-02), so a real 2d model is a test, not a code change (D-04).
4. **User's INFR-02 reference examples** — never arrived before the Phase 3 Plan-03 gate; INFR-02 shipped on the D-24(a)+(b) proof instead. Fold into the gate oracle if they land.
5. **Deferred (locked reasons, not gaps):** ACCEL-01/02/03 (ncu admin-blocked + hostile compile env), MEM-01 (not a measured bottleneck — models load once per fresh process), MEM-02 (8 GB target hardware-unverifiable on A100-40GB; measured data points shipped: 4 GiB RMM pin, ~0.8 GB pool release, 9.552 GiB peak).
