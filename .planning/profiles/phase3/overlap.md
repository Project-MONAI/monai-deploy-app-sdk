# D-21 overlap evidence — concurrent bundle run vs Phase 2 serial baseline

**Date:** 2026-08-19 · **Device:** NVIDIA A100-SXM4-40GB, `CUDA_VISIBLE_DEVICES=0`
(Pitfall 7 pin; devices 4–7 tenant-occupied) · **Config:** default bundle
(`HOLOSCAN_MODEL_LIST` unset), `HOLOSCAN_CONCURRENT_FRAGMENTS=1`
(= the shipping default after step 5 flip; see "Flag default" below).

**Trace:** `overlap_concurrent_20260819_104410.nsys-rep` (+ `.sqlite`,
`_nvtx_sum.txt`, `_nvtx_kern_sum.txt`, `_log`) — full-process nsys capture
(2025.6.3, `--trace=cuda,nvtx,osrt,cublas,cudnn`, no `--capture-range`).

## Per-config NVTX overlap table (concurrent run)

Timestamps relative to the first preprocess NVTX start (t=0). `tid` =
`globalTid` of the GXF worker thread executing the span — **five distinct
worker threads (…486–…490)** for `worker_thread_number=5`, vs the Phase 2
single serial scheduler thread.

| span | start (s) | end (s) | dur (s) | tid (worker) |
|---|---|---|---|---|
| preprocess_3d_fullres | 0.000 | 7.977 | 7.98 | …6563489 |
| preprocess_3d_lowres | 0.000 | 5.361 | 5.36 | …6563490 |
| inference_3d_lowres | 5.362 | 56.430 | 51.07 | …6563487 |
| inference_3d_fullres | 7.978 | 57.790 | 49.81 | …6563486 |
| postresample_3d_lowres | 56.430 | 59.827 | 3.40 | …6563489 |
| postresample_3d_fullres | 57.791 | 59.899 | 2.11 | …6563488 |
| preprocess_3d_cascade_fullres | 59.828 | 69.204 | 9.38 | …6563490 |
| inference_3d_cascade_fullres | 69.205 | 94.932 | 25.73 | …6563486 |
| postresample_3d_cascade_fullres | 94.932 | 96.619 | 1.69 | …6563487 |
| ensemble_average | 96.619 | 96.632 | 0.01 | …6563488 |
| postprocess | 96.633 | 107.026 | 10.39 | …6563489 |
| write_sr / write_sc / write_seg | 107.03 | 107.92 | ≤0.89 | …486/…487/…490 |

**Overlapping pairs (true time-window overlap):**
- `inference_3d_fullres` ∥ `inference_3d_lowres` — **49.8 s overlap** (7.98→56.43 s)
  on two different worker threads — the headliner; in Phase 2 these were
  strictly back-to-back on one stream.
- `preprocess_3d_fullres` ∥ `preprocess_3d_lowres` — 5.36 s overlap (0.00→5.36 s).
- `postresample_3d_fullres` ∥ `postresample_3d_lowres` — 2.04 s overlap.
- `write_sr` ∥ `write_sc` ∥ `write_seg` — all three writers overlap at the tail.

**Contrast — Phase 2 §5 (trace `.planning/profiles/phase2/phase2_bundle_20260819_071235.*`):**
every kernel executed on a single CUDA stream (id 7) and the three inference
fragments were strictly back-to-back with a 0.2 ms gap (one GreedyScheduler
thread); the only overlaps were light-boundary (postresample_3d_fullres ∥
lowres inference). The concurrent run above shows the inference spans
themselves now co-existing across distinct worker threads — the D-16 note's
"scheduling is the blocker, not the pools" is confirmed as the lever.

## Measured wall delta (single bundle reps, non-nsys, same session, device 0)

| | process wall (s) | in-study timing window (s) |
|---|---|---|
| serial (flag unset, pre-flip default) | 120.4 | 108.5 |
| concurrent (flag =1) | **110.4** | **100.9** |
| delta | **−10.1 s (−8.4%)** | **−7.6 s (−7.0%)** |

Phase 2 benchmark reference: bundle 129.54 ± 0.90 s (3-run mean). The
−10 s single-run delta is below the Phase 2 mean because this session's
serial rep also ran ~9 s faster than the Phase 2 mean (warmer machine + the
Task 1 RMM pin removing a 20 GiB bootstrap reservation); the concurrent-vs-
serial delta measured in the same session is the apples-to-apples number.

**Honest ceiling note (not the fallback clause — gates are fully green):**
the realized gain (~10 s) is far below the ~25 s potential because
(a) `inference_fullres ∥ inference_lowres` is two GPU-saturated fragments
time-sharing the GPU — each slowed ~26 s → ~50 s (total GPU work conserved,
exactly the research §D-21 GPU-probe prediction), and (b) the cascade
preprocess (9.4 s, the biggest hiddable CPU span) can only start after
`lowres_seg` is emitted (56.4 s), so it runs under nothing — it stays serial
after the parallel inference pair. The CPU spans that DO hide (fullres
preprocess ∥ lowres preprocess, the postresample pair, the writer tail) are
the source of the measured −10 s. A Plan 05 final benchmark (3-run mean,
D-18 convention) will report the shipping number.

## Gate results (D-25 anchor)

- **Serial (flag unset):** `03-GATE-serial.json` — fullres 99.99986%/3
  (the documented fp16↔fp32 boundary class), lowres/cascade/bundle
  100.00000%/0, SR 0.0% ×4, residency static+runtime PASS — exact Phase 2
  reproduction.
- **Concurrent (flag =1):** `03-GATE-concurrent.json` — **fully green,
  identical numbers** (the fullres 3-voxel boundary gate did NOT flip under
  the `set_num_threads` interleaving hazard the research flagged; sanity:
  bundle vs fullres-only 382 differing voxels — the bundle ensembles).

## Flag default (final shipped state)

**ON by default.** `app.py` condition:
`os.environ.get("HOLOSCAN_CONCURRENT_FRAGMENTS", "1") != "0"` — concurrency
ships default-ON (step 5 of the plan, taken because the concurrent gate run
is fully green); explicit `HOLOSCAN_CONCURRENT_FRAGMENTS=0` restores the
serial Phase 2 GreedyScheduler behavior (verified: serial gate suite green +
`scheduler: default GreedyScheduler (serial, Phase 2 behavior)` logged;
serial-vs-concurrent gate outputs are pixel-identical).
