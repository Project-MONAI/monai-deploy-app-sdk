# Retrospective

Per-milestone retrospective. New sections appended at milestone close.

---

## v1.0 — Holoscan-native nnUNet inference (2026-08-14 → 2026-08-20)

4 phases / 17 plans, all executed; 3 verification passes with 74/74 must-haves across Phases 2–3 + 24/24 Phase 1 + 5/5 Phase 0; zero pixel-regressions shipped.

### What Worked

- **Pixel-gate-as-anchor (D-25).** Every Phase 3 plan shipped behind a re-run of the full `phase2_gate.py` suite — no optimization landed without reproducing the exact Phase 2 pixel baseline. It caught a real aliasing bug the unit tests missed (Phase 3 Plan 03: a cached per-fold `predicted_logits` view wiped the TTA running sum — 1363 differing voxels, IoU 0.56; fixed with the fold-1 clone + regression test). The gate turned "the optimization didn't break correctness" from a hope into a per-plan artifact.
- **Live-probing holoscan APIs before planning.** holoscan-cu13 4.2 lacked several assumed APIs (no `MemoryData`; app_driver rejects C++ `Fragment` + app-operator mixing; fragment→fragment `add_flow` has no port addressing). Probing live before locking plans produced documented, workable substitutions (DLPack `holoscan.core.Tensor` emit; `Subgraph` + interface ports) instead of plan rewrites mid-execution.
- **The D-26 blocked-on-external pattern.** ≥5-CT corpus, ncu admin access, and the user's INFR-02 reference examples were classified blocked-on-external (non-blocking, recorded, re-opens as a gap plan if the dependency lands) instead of being allowed to stall phases. All three stayed open through the milestone without gating a single plan.
- **The 99%-gate amendment (D-22b).** When the bounded custom-RawKernel attempt failed (o3 diverged 100% of voxels + real-shape crash), the user amended the arbiter from byte-identity to ≥99% per-tensor accuracy — recognizing the real requirement is a fast model that doesn't diverge significantly (ensemble averaging absorbs bit-level noise). This saved ~1.5 h of a byte-for-byte rabbit hole and let the stock `cupyx.scipy.ndimage` mirror ship default-ON with a 100.0000% measured gate and zero new pixel divergence.

### What Was Hard

- **holoscan 4.2 API gaps.** The multi-fragment story had to be built on `Subgraph` + interface ports with every port declared/wired (an unwired declared port hangs the run); timing/NVTX had to resolve the top-level app through sub-Fragments (`gpu_util._root`, Pitfall 9).
- **RMM/cudnn/torch interactions.** torch 2.13's cudnn benchmark mode calls the pluggable allocator's unsupported `cacheInfo` (RuntimeError on the first conv) — benchmark mode had to be disabled under RMM; `import rmm` before holoscan is load-bearing (undefined-symbol hazard, subprocess-pinned); rmm 26.2.0's default initial pool silently reserves ½ GPU (19.97 GiB) until explicitly pinned to 4 GiB; `torch.cuda.empty_cache()` is a silent driver-level no-op under RMM (Open Q2).
- **CuPy bit-identity.** CuPy reductions are not bit-exact vs numpy (all reductions stayed CPU), `cp.from_dlpack` consumes the caller's buffer (clone needed), and the GPU-resample kernel port hit a genuinely wrong spline-prefilter that only the real 256³ shape exposed (illegal address). Bit-exactness had to be verified empirically per operation, never assumed.

### Metrics

| Metric | Value |
|---|---|
| Plans | 17 (P0 scaffold + 5 + 6 + 5), all executed |
| Phases | 4/4 verified (5/5, 24/24, 38/38, 12/12 must-haves) |
| Pixel-regressions shipped | 0 (final gate pixel-identical to Phase 2/3 baseline) |
| Final latency | bundle 104,180 ms = 1.629× vs 169,747 ms reference; same-scope fullres 49,673 ms = 1.244× vs 61.8 s |
| Requirements | 33/36 Done (3 locked deferrals: ACCEL-01/02/03, MEM-01, MEM-02) |
| External dependencies open | 4 (≥5-CT corpus, ncu admin, 2d model, INFR-02 user examples) — all non-blocking |

### Cross-Milestone Trends

None — first milestone.
