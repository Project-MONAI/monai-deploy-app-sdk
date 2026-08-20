# Phase 3 Plan 04 Summary — D-22 Gated GPU-Resample Experiment (AMENDED: D-22a/D-22b)

**Status: COMPLETE — flag default ON, gate green, phase 3 4/5 plans shipped.**

The plan executed under the **amended gate** (03-CONTEXT.md D-22a + D-22b,
direct user directive, 2026-08-19 — supersedes the plan's byte-identity
arbiter): the custom RawKernel got exactly one bounded final attempt; on
failure the shipping path is stock `cupyx.scipy.ndimage`; the arbiter is
**≥99% per-tensor accuracy** (not byte-identity) + the 4-config gate
(≥99.0% SEG byte-identity, IoU ≥0.99, SR ≤0.1%); if green, the flag ships
**ON**.

## Kernel verdict (Step A)

`scripts/test_gpu_zoom_verdict.py` — the one-and-only bounded attempt
(3 small synthetic 32×32×16 volumes, non-unity factors, orders 0/1/3 +
1 real 256³→(255,256,255) o3 bundle case; per-case `timeout 120` in a
child process, driver `timeout 1500`; a prior full-suite run had wedged
70+ min and was killed — this run finished in ~4 min):

- o0, o1: **byte-identical** on both levels (primitive + full chain).
- o3 small synthetic: **FAIL — 100.000000% of voxels differ, max_abs
  2.588608e+00** (full-range wrong spline-prefilter arithmetic, not ulp
  noise).
- o3 real 256³ bundle shape: **CRASH — `CUDA_ERROR_ILLEGAL_ADDRESS`**.

→ Custom RawKernel **discarded from the shipping path per D-22a**.
`gpu_zoom.py` (RawKernel: `gpu_zoom_grid_mode`/`gpu_zoom_resize`) + the
original arbiter `test_gpu_zoom.py` stay committed as provenance
(docstring marked "NOT WIRED — provenance only, discarded per D-22a").

## What shipped (Step B)

Stock `cupyx.scipy.ndimage` mirror — `stock_gpu_zoom` /
`stock_gpu_resize` in `gpu_zoom.py` — reproducing the exact scipy call the
flag-OFF path makes: `zoom(grid_mode=True, mode='nearest', cval=0,
prefilter defaults)` with skimage's float64 `in/out` factor derivation
(`zoom_factors_for`) and the OFF path's dtype behavior (exact fp64
widening of fp32 app data — the OFF path upcasts via `data.astype(float)`
before the skimage chain), then the reference's fp64 clip (channel
min/max; [0,1] for seg masks) + fp32 cast / `>= 0.5` tail. Wired at the 3
flag sites: preprocess image `_resample_to_shape` (non-sep-z), cascade seg
multihot `_resize_segmentation` (o0 + o>0), postresample
`resample_probabilities_to_shape`. Separate-z `map_coordinates` branches
stay scipy in both flag states (inactive in this bundle). The OFF path is
byte-for-byte the Phase 2/3 code; Plan 03 shape caches untouched (flag
branches placed next to the cached allocations, no disturbance).

Stock CuPy vs scipy, small-shape smoke (32×32×16→(36,30,22)): o0
100.0000% byte-identical; o1 99.9874% fp32-equal (fp64 max_abs 5.8e-15);
o3 100.0000% fp32-equal (fp64 max_abs 1.6e-14).

## Per-tensor accuracy vs scipy on the dev corpus (Step C.6, D-22b ≥99% bar)

`scripts/measure_resample_accuracy.py` — flag ON vs flag OFF per resampled
tensor, real corpus CT (256³ @ 0.704/0.7/0.7 mm), real config shapes and
orders (all 4 model configs reduce to these tensor pairs; bundle
re-ensambles the fullres pair):

| tensor | shape pair | order | equal | max abs diff |
| --- | --- | --- | --- | --- |
| image | (256,256,256)→(255,255,256) | 3 | **100.0000%** | 0 |
| image | (256,256,256)→(201,201,202) | 3 | **100.0000%** | 0 |
| seg multihot | (256,256,256)→(255,255,256) | 1 | **100.0000%** | 0 |
| seg nearest | (256,256,256)→(255,255,256) | 0 | **100.0000%** | 0 |
| prob 2ch | (2,201,201,202)→(2,256,256,256) | 1 | **100.0000%** | 0 |

(Prob channels: corpus-derived 0/1 at the real lowres geometry — no
softmax probabilities persisted by the pipeline; order-1 zoom is linear so
the arithmetic is identical. `evidence/step6_per_tensor_accuracy.json`.)

## Gate table (Step C.7/C.8 — shipping state: concurrency ON, RMM 4 GiB pin,
buffer caches, MEM-003 release; device 0; `phase2_gate.py` both ways)

| row | flag OFF: id% / diff / IoU / SR | flag ON: id% / diff / IoU / SR |
| --- | --- | --- |
| fullres_only | 99.99986% / 3 / 0.998714 / 0.0000% | **99.99986% / 3 / 0.998714 / 0.0000%** |
| lowres_only | 100.00000% / 0 / 1.0 / 0.0000% | **100.00000% / 0 / 1.0 / 0.0000%** |
| cascade_only | 100.00000% / 0 / 1.0 / 0.0000% | **100.00000% / 0 / 1.0 / 0.0000%** |
| bundle | 100.00000% / 0 / 1.0 / 0.0000% | **100.00000% / 0 / 1.0 / 0.0000%** |
| residency | static PASS + runtime PASS | **static PASS + runtime PASS** |

- OFF reproduces the Phase 2/3 baseline exactly (99.99986%/3 = the known
  fullres-only fp16↔fp32 reference boundary class — the ON row matches it
  voxel-for-voxel; the GPU resample adds zero divergence).
- ON: every row clears the D-22b bars (≥99.0% byte-identity, IoU ≥0.99,
  SR ≤0.1%) with wide margins. `ALL GATES: PASS` both ways.
- Sanity (bundle ensembles): 382 differing voxels vs fullres-only, both
  flag states. JSONs: `gates/03-GATE-resample-{off,on}.json`.

## Flag default decision (Step C.9)

**Default ON** (`gpu_resample_enabled()` = `HOLOSCAN_GPU_RESAMPLE` unset/"1"
→ ON; `=0` forces the scipy/skimage CPU reference — D-21-style
default-ON convention). Verified: flag resolution unit check (default ON /
=0 OFF / =1 ON) + default-config bundle E2E (no env override) exit 0 with
SC/SEG/SR and exact SR text. Timing glimpse (single bundle rep, default
ON): postresample spans 323.9 ms / 175.7 ms / 2,043.3 ms
(fullres/cascade/lowres) vs multi-second scipy CPU in Phase 2 — Plan 05's
2×2 matrix measures the ~28.8 s (22.2%) resample bottleneck properly.

## GPUP-01 / GPUP-02 status (amended D-22b — met-with-documented-tolerance, measured numbers)

- **GPUP-01: met-with-documented-tolerance — tolerance not exercised on the
  dev corpus.** The enforced bar is ≥99% of elements equal per resampled
  tensor + gate ≥99.0% byte-identity / IoU ≥0.99 / SR ≤0.1%. Measured:
  **100.0000%** per-tensor equality on all 5 tensors (max abs diff 0);
  ON-gate identity 99.99986%/100%/100%/100% (the 3 fullres differing
  voxels are the pre-existing boundary class, present identically on the
  OFF path). Theoretical stock-CuPy-vs-scipy fp64 divergence at order ≥1:
  ≤1.6e-14 (measured; below fp32 ulp at every measured corpus voxel).
- **GPUP-02: met for the resample span only; numpy reductions + ~8 MB mask
  round trip stay CPU per locked D-12/D-13.** Image span GPU end-to-end
  (span-boundary transfers + 2×4-byte scalar syncs for clip bounds);
  residuals: numpy mean/std reductions (D-12/D-13), ~8 MB mask round trip
  (scipy `binary_fill_holes`), per-label fp64 seg-mask D2H for the numpy
  `>=0.5` tail, postresample D2H before the reference torch CPU softmax
  (thread-scoped bit-exactness decision).

## Kernel implementation choices / per-call timing

The shipping path has no custom kernel (stock CuPy). The discarded
RawKernel (provenance) implemented the research §D-22 spec (fp64 splvals,
C-order 27-tap loop, `--fmad=false` + explicit RN double ops, fp64
spline-prefilter chain, `__double2float_rn` output); its o3 prefilter math
is what diverged (the o0/o1 paths — which share everything but the
prefilter — were byte-identical). Per-call kernel timing: N/A (discarded);
shipping-path per-call cost is visible in the default-ON E2E timing above
and will be quantified in Plan 05's 2×2 matrix.

## Commits

1. `d554519` perf(03-04): bounded kernel verdict — custom RawKernel discarded per D-22a, stock CuPy helpers shipped
2. `7df0de8` perf(03-04): rewire HOLOSCAN_GPU_RESAMPLE flag path to stock cupyx at the 3 call sites
3. `9de1838` perf(03-04): flag-OFF gate green (baseline reproduced) + Step 6 per-tensor corpus accuracy
4. `5aa14e0` perf(03-04): flip HOLOSCAN_GPU_RESAMPLE default to ON — D-22b gate green

## Deviations (from the original plan text — all driven by the D-22a/D-22b amendment)

- The plan's Task 1 arbiter was the full `test_gpu_zoom.py` byte-identity
  suite; per D-22a it was replaced by the bounded
  `test_gpu_zoom_verdict.py` (one attempt, hard caps) after a prior full
  run wedged 70+ min.
- The plan's must-have "flag defaults OFF in both outcomes" is superseded
  by D-22b: green gate → flag ships ON.
- The plan's `03-GATE-resample-on.json` "only valid if byte-identity held"
  constraint is superseded by the D-22b accuracy bars (≥99.0%/IoU 0.99/
  SR 0.1%) — all cleared.
- Plan-05 matrix note: the OFF column is now `HOLOSCAN_GPU_RESAMPLE=0`
  (no longer the default); expect the ~22.2 s resample spans to collapse
  to seconds on the ON column.
