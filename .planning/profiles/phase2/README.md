# Phase 2 profiling artifacts (bundle run, airway study)

Trace: full-process `nsys profile` of the **fast app's default BUNDLE** run
(`HOLOSCAN_MODEL_LIST` UNSET = 3d_fullres + 3d_lowres + 3d_cascade_fullres) on
`testdata/airway_input`, A100-SXM4-40GB, venv `/tmp/monai-env/.venv`,
2026-08-19. Harness: `.planning/scripts/nsight_profile_phase2.sh`
(`--trace=cuda,nvtx,osrt,cublas,cudnn`, nsys 2025.6.3; NO capture-range — the
setup/warmup span must be in the trace for the RMM churn check).

## Files

| File | Content |
|---|---|
| `phase2_bundle_20260819_071235.nsys-rep` | the trace (72.5 MB) — open with `nsys-ui` |
| `phase2_bundle_20260819_071235.sqlite` | `nsys export --type=sqlite` sidecar (SQL queries below) |
| `phase2_bundle_cuda_api_sum.txt` | `nsys stats -r cuda_api_sum` — RMM churn check |
| `phase2_bundle_nvtx_sum.txt` | `nsys stats -r nvtx_sum` — per-operator wall ranges (per-config legible) |
| `phase2_bundle_nvtx_kern_sum.txt` | `nsys stats -r nvtx_kern_sum` — kernels per NVTX range |
| `phase2_bundle_cuda_gpu_kern_sum.txt` | `nsys stats -r cuda_gpu_kern_sum` — kernel timeline summary |
| `ncu_status.txt` | ncu BLOCKED (`ERR_NVGPUCTRPERM`) — admin requirement documented |

## Findings (full analysis in `02-BENCHMARK-REPORT.md` §5)

**RMM churn check (INFR-005): PASS — no per-tile churn.** `cuda_api_sum` shows
9 `cudaMalloc` (each < 2 ms, ≈1.98 GB total) + 1 `cudaFree` in a 117.5 s
14-operator study window that launched **629,929 GPU kernels** (~209k launches
per inference fragment / ~4,000 sliding-window patches). One allocation in
setup (t−9.9 s: model load / RMM warm pool); the remaining 8 sit at the start
of later operator stages (t+1.8–2.1 s, +13.7 s, +79.1 s vs the first operator
range) — RMM pool *expansion* events, never per-tile.

**Per-config NVTX legibility (INFR-005): PASS.** All 14 operator ranges are
legible in the exports: `preprocess_/inference_/postresample_` ×
`{3d_fullres, 3d_lowres, 3d_cascade_fullres}` + `ensemble_average`,
`postprocess`, `write_seg/sr/sc` (see `phase2_bundle_nvtx_sum.txt`).

**Stream overlap (D-16, honest observation): NOT visible.** The three
inference fragments run strictly back-to-back — `inference_3d_fullres` ends at
t=39.3 s, `inference_3d_lowres` starts 0.2 ms later; every kernel in the whole
run executes on a **single CUDA stream (id 7)**. The per-fragment
`CudaStreamPool`s (`streams_<cfg>`) are never concurrently active because the
fragments themselves are scheduled serially by the DAG (cascade additionally
waits on the lowres `lowres_seg` edge). Likely reason = serial fragment
scheduling, not the stream pools. Overlap IS visible at the
postresample/fullres-inference boundary (postresample_3d_fullres runs while
lowres inference is in flight), but never between the heavy inference spans.

**Top CPU-bound / GPU-idle regions (Phase 3 shortlist, kernel timeline):**

1. **scipy resample spans** — `preprocess_*` wall time is 99.9% CPU
   (7.6 s / 5.0 s / 9.4 s vs 2–3 ms of kernels) — the D-13 CPU
   round-trips; `postresample_*` likewise (1.7–3.2 s, 0 ms kernels).
2. **postprocess** — 9.9 s wall NVTX (GPU 5 ms; includes the documented
   exactly-once D2H boundary + two-pass CC).
3. **Setup** (DICOM load + model load + RMM warm) ≈ 11 s of the 129.5 s E2E.
4. **Inference spans** are GPU-saturated (91–96% kernel-busy); further
   inference speedups need Phase 3 kernel-level work (blocked on ncu
   admin access for counter metrics).
