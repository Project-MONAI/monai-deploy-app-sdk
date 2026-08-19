# INFR-02 / D-24 Proof Record: Cross-Study GPU Buffer Reuse (a)+(b)

**Date:** 2026-08-19 · **Device:** NVIDIA A100-SXM4-40GB, `CUDA_VISIBLE_DEVICES=0` (pinned, Pitfall 7)
**Shipping state:** concurrent fragments ON (Plan 01 default) + RMM `initial_pool_size` 4 GiB pin (Plan 01) + MEM-003 release hook (Plan 02, inert for this fullres-only chain) + INFR-02 shape caches (this plan)
**Replay path used:** the plan's PREFERRED path — **direct operator drive** (no GXF app run). One process, 3 sequential passes over the same airway study through the REAL operators: `PreprocessOperator` → `SlideWindowOperator` → `PostResampleOperator` → `EnsembleAverageOperator` (legacy single-probabilities mode) → `PostprocessOperator`, config `3d_fullres`, with the study loaded through the app's OWN SDK load path (`DICOMDataLoaderOperator` → `DICOMSeriesSelectorOperator` → `DICOMSeriesToVolumeOperator`). Every inter-operator handoff is the real DLPack `holoscan.core.Tensor` the DAG would pass.

Harness: `.planning/scripts/infr02_replay.py` (replay mode + `--analyze` sqlite mode). Run under full-process `nsys profile --trace=cuda,nvtx` (no capture-range restriction, so the RMM bootstrap span is in the trace): `.planning/profiles/phase3/infr02_replay_20260819_151533.{nsys-rep,sqlite,cuda_api_sum.txt}`.

## (b.1) Address stability — cached-buffer `data_ptr()` table, 3 studies

Snapshot taken after each study's preprocess + inference stages complete (plan spec). The table below is study 1's; **studies 2 and 3 are byte-identical to it — every cached buffer and the setup-time gaussian tensor reuse study 1's exact device addresses** (harness assertion `[PASS] study 2/3: data_ptr table IDENTICAL to study 1 (5 cached buffers + gaussian ptr identical)`).

| operator | (shape, dtype) | data_ptr (study 1 = 2 = 3) | size |
|---|---|---|---|
| preprocess (CuPy) | ((1, 256, 256, 256), fp32) — `vol` | 22709673132032 | 67.1 MB |
| preprocess (CuPy) | ((256, 256, 256), uint8) — `mask` | 22709740240896 | 16.8 MB |
| slidewindow (torch) | ((2, 255, 256, 255), torch.float32) — `predicted_logits` | 22705631426048 | 133.2 MB |
| slidewindow (torch) | ((255, 256, 255), torch.float32) — `n_predictions` | 22705764597248 | 66.6 MB |
| slidewindow (torch) | ((1, 1, 128, 128, 128), torch.float32) — per-patch `workon` | 22705831182848 | 8.4 MB |
| slidewindow | setup-time `gaussian` (computed once) | 22705556451840 | 8.4 MB |

`total_bytes`: preprocess 83,886,080 · slidewindow 208,145,408 — unchanged across studies.

**vol_c note (aliasing by key collision, safe):** the airway study's crop is the identity (256³ → 256³), so `vol_c`'s `(shape, dtype)` key collides with `vol`'s and both names refer to the same cached buffer. This is safe by the cache's own contract — `vol_c[...] = vol_c_src` fully overwrites before any read, and at that point `vol` has already served its purpose (mask computed, crop copied). For any non-identity crop the shapes differ and the keys are distinct (shape-key invalidation is unit-tested, `test_buffer_cache.py` case 2). The cascade-config sites (`one_hot`, `vol2`) are exercised by the D-25 gate (cascade row 100.00000%/0), not by this fullres-only replay.

## (b.2) cudaMalloc churn — flat across studies 2/3

Per-study counts from the nsys sqlite (`CUPTI_ACTIVITY_KIND_RUNTIME`, `cudaMalloc%`/`cudaFree%` names; study N = Nth `preprocess_3d_fullres` start → Nth `postprocess` end NVTX window):

| window | cudaMalloc | cudaFree | classification |
|---|---|---|---|
| bootstrap (before study 1) | 1 | 0 | RMM pool initial reservation (4 GiB pin, 1.51 ms) |
| study 1 | 8 | 1 | pool expansions / first-touch (each **< 1 ms**: 722 µs, 600 µs, 332 µs, 497 µs, 441 µs, 259 µs, 220 µs, 8 µs) — never per-tile (180,538 `cudaMemcpyAsync` + 43,398 `cudaMemsetAsync` + 627,124 kernel launches in the whole process) |
| **study 2** | **0** | **0** | **FLAT** |
| **study 3** | **0** | **0** | **FLAT** |
| total process | 9 | 1 | vs Plan-01 churn baseline: 10 cudaMalloc / 1 cudaFree per bundle study (3 configs); this replay is a single-config chain, so 9 total across the WHOLE process (bootstrap + 3 studies) is the same "pool expansions only" class |

**Verdict: studies 2/3 are exactly flat (0/0)** — the Nth study's allocations reuse the 1st study's buffers (address stability above) and the allocator sees no per-study traffic. Classification rule = Phase 2 Plan 06 / Plan 01 baseline (pool-expansion events < 2 ms at stage starts, never per-tile).

## (b.3) Pool occupancy + output integrity

- **Driver-level VRAM (pynvml, post each study, device 0):** after s1 = **5,469 MiB**, s2 = 5,469 MiB (**+0.00 %**), s3 = 5,469 MiB (**+0.00 %**) — flat across studies (±5 % bar). Replay start (post model load) 5,121 MiB. (Pool level read at the driver level via pynvml — rmm 26.2.0 exposes no Python pool-stats API, same method as Plan 02.)
- **Byte-identity of repeat outputs:** studies 2 and 3 final segmentation payloads are **byte-identical** to study 1's (`np.array_equal`, 256³ uint8 contour payload, 2,331 nonzero voxels — the baseline post-CC fullres count) and the SR result text is identical (`Airway Volume: 1 mL`). The reuse did not corrupt a data path.
- Per-study wall: s1 45.4 s → s2 41.3 s → s3 41.3 s (study 1 pays first-touch allocation/CuPy pool warm-up; **honest scope: single-study-per-run is the clinical model, so first-study latency is the shipping number — INFR-02 delivers provable reuse + reduced allocator traffic, not a first-study speed claim**).

## (a) Headless unit proof (synthetic sizes)

`examples/apps/cchmc-nnunet-fast/scripts/test_buffer_cache.py` — exit 0, covers: multi-call reuse (same `data_ptr`), shape-key invalidation (new shape → new buffer, both retained), dtype/contiguity invariants (no cross-dtype sharing; C-contiguous asserted at allocation), zero-on-borrow semantics (both families), `clear()`/`keys()`/`total_bytes()`/`shares_storage()`, and the SlideWindow site semantics incl. the **multi-fold accumulation aliasing regression** (fold-1 clone == fresh-allocation reference; without the clone the sum is corrupted — the Rule 1 bug caught during this plan's gate re-run).

## D-25 anchor

`gates/03-GATE-infr02.json` — full D-25 gate suite with the caches ACTIVE (all prior plans in shipping state): **ALL GATES PASS, pixel-identical to the Phase 2 baseline** — fullres 99.99986 %/3 (documented fp16↔fp32 boundary class), lowres/cascade/bundle 100.00000 %/0, SR 0.0 % delta ×4, residency static + runtime PASS, sanity (bundle ensembles: 382 differing voxels).

## D-26 external-dependency record

**INFR-02 user reference examples — blocked-on-external, non-blocking** (user adding during Phase 3); they did not arrive before the gate ran, so this plan ships with the D-24(a)+(b) proof strategy as locked — if the examples land before VERIFICATION.md, fold them into the gate oracle. Same external-dependency class as the 2d model (D-01/D-03) and the ≥5-CT corpus (TEST-01 final gate).

## Artifacts

- `.planning/profiles/phase3/infr02_replay_20260819_151533.nsys-rep` (70.5 MB) + `.sqlite` (172.4 MB) + `.cuda_api_sum.txt`
- `.planning/scripts/infr02_replay.py` (replay + `--analyze`)
- raw replay log: `/tmp/infr02_replay.log` (session scratch; key lines reproduced above)
