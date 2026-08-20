# GPU Resample Verdict (D-22 / D-22a / D-22b) — Plan 04

**Verdict: SHIPPED-ON.** GPU resampling is the DEFAULT path
(`HOLOSCAN_GPU_RESAMPLE` default ON; `=0` forces the scipy/skimage CPU
reference). The shipping path is the **stock `cupyx.scipy.ndimage` mirror**
(D-22a fallback after the final custom-kernel attempt failed). The amended
D-22b gate is green with wide margins — in fact the ON path is
**pixel-identical to the OFF baseline on the dev corpus** (better than the
≥99% bar required). GPUP-01/02 land as met-with-documented-tolerance (see
below — the measured numbers, not a bare "met").

## 1. Kernel verdict (Step A — the one-and-only bounded attempt, D-22a)

`scripts/test_gpu_zoom_verdict.py` (3 small synthetic 32×32×16 volumes,
non-unity factors, orders 0/1/3 + 1 real bundle-shape 256³→(255,256,255)
o3; each case in its own subprocess under `timeout 120`, driver cap
`timeout 1500`):

| case | shape → target | order | primitive (scipy zoom) | full chain (skimage resize) |
| --- | --- | --- | --- | --- |
| syn-o0 | (32,32,16)→(40,28,24) | 0 | PASS (byte-identical) | PASS (byte-identical) |
| syn-o1 | (32,32,16)→(24,40,20) | 1 | PASS (byte-identical) | PASS (byte-identical) |
| syn-o3 | (32,32,16)→(36,30,22) | 3 | **FAIL — 23760/23760 voxels (100.000000%) differ, max_abs 2.588608e+00** | **FAIL — 23723/23760 (99.844276%), max_abs 2.524817e+00** |
| real | 256³→(255,256,255) | 3 | **CRASH — `CUDADriverError: CUDA_ERROR_ILLEGAL_ADDRESS`** | — (child died) |

The o3 divergence is full-range (max_abs ≈ 2.6 on a [−3,4] input) — wrong
spline-prefilter arithmetic, not ulp noise; the real-shape crash is an
illegal memory access in the custom kernel. Per D-22a the custom RawKernel
(`gpu_zoom_grid_mode` / `gpu_zoom_resize` in `gpu_zoom.py`) is **discarded
from the shipping path** — it stays committed as provenance (docstring
marked "NOT WIRED — provenance only, discarded per D-22a") together with
the original byte-identity arbiter `scripts/test_gpu_zoom.py`.

## 2. What shipped (Step B — stock CuPy, D-22a)

`stock_gpu_zoom` / `stock_gpu_resize` in `gpu_zoom.py`: the stock
`cupyx.scipy.ndimage.zoom(grid_mode=True, mode='nearest', cval=0)` call
with the OFF path's exact factor derivation (`zoom_factors_for`,
skimage's `in/out` float64) and the OFF path's dtype behavior (exact fp64
widening of the fp32 app data — the OFF path upcasts via
`data.astype(float)` before the skimage chain), then the reference's fp64
clip (channel min/max; `[0,1]` for seg masks) and the fp32 cast / `>= 0.5`
threshold tail. Wired at the 3 flag sites (preprocess image
`_resample_to_shape`, cascade seg multihot `_resize_segmentation` o0+o>0,
postresample `resample_probabilities_to_shape`); separate-z
`map_coordinates` branches stay scipy in both flag states (inactive in this
bundle). The OFF path is byte-for-byte the Phase 2/3 code.

Stock CuPy vs scipy, small-shape smoke (32×32×16→(36,30,22)): o0
100.0000% byte-identical; o1 99.9874% fp32-equal (max_abs 5.8e-15 at
fp64); o3 100.0000% fp32-equal (max_abs 1.6e-14 at fp64).

## 3. Per-tensor accuracy vs scipy on the dev corpus (Step C.6, D-22b bar ≥99%)

`scripts/measure_resample_accuracy.py` — flag ON (stock CuPy) vs flag OFF
(scipy reference) per resampled tensor, real corpus CT (256³ @ 0.704/0.7/0.7
mm), real config shapes/orders (all 4 model configs reduce to these
tensor pairs; bundle re-ensambles the fullres pair):

| tensor | shape pair | order | elements equal | max abs diff | ≥99% bar |
| --- | --- | --- | --- | --- | --- |
| image | (256,256,256)→(255,255,256) | 3 | **100.0000%** | 0 | MEETS |
| image | (256,256,256)→(201,201,202) | 3 | **100.0000%** | 0 | MEETS |
| seg multihot | (256,256,256)→(255,255,256) | 1 | **100.0000%** | 0 | MEETS |
| seg nearest | (256,256,256)→(255,255,256) | 0 | **100.0000%** | 0 | MEETS |
| prob 2ch | (2,201,201,202)→(2,256,256,256) | 1 | **100.0000%** | 0 | MEETS |

(Prob channels are corpus-derived 0/1 at the real lowres geometry — no
softmax probabilities are persisted by the pipeline; order-1 zoom is
linear, so the arithmetic exercised is identical. Raw table:
`evidence/step6_per_tensor_accuracy.json`.)

## 4. Gate suite, shipping state (concurrency ON, RMM 4 GiB pin, buffer
caches, MEM-003 release), device 0 (Step C.7/C.8, D-22b bar: every row
SEG byte-identity ≥99.0% AND IoU ≥0.99 AND SR delta ≤0.1%)

| row | flag OFF: id% / diff / IoU / SR | flag ON: id% / diff / IoU / SR | D-22b bars (ON) |
| --- | --- | --- | --- |
| fullres_only | 99.99986% / 3 / 0.998714 / 0.0000% | **99.99986% / 3 / 0.998714 / 0.0000%** | PASS (≥99.0 / ≥0.99 / ≤0.1) |
| lowres_only | 100.00000% / 0 / 1.0 / 0.0000% | **100.00000% / 0 / 1.0 / 0.0000%** | PASS |
| cascade_only | 100.00000% / 0 / 1.0 / 0.0000% | **100.00000% / 0 / 1.0 / 0.0000%** | PASS |
| bundle | 100.00000% / 0 / 1.0 / 0.0000% | **100.00000% / 0 / 1.0 / 0.0000%** | PASS |
| residency | static PASS + runtime PASS | **static PASS + runtime PASS** | — |

`gates/03-GATE-resample-off.json` reproduces the Phase 2/3 baseline
exactly (fullres 99.99986%/3 = the known fp16↔fp32 reference boundary
class; the ON row matches it voxel-for-voxel — the GPU resample adds zero
divergence on this corpus). Sanity (bundle ensembles): 382 differing
voxels vs fullres-only, both flag states.

## 5. GPUP-01 / GPUP-02 status (amended D-22b: met-with-documented-tolerance)

**GPUP-01 (GPU resampling of bottleneck #1): met-with-documented-tolerance
— tolerance not exercised on the dev corpus.** The requirement the gate
actually enforces is "≥99% of elements equal per resampled tensor + final
4-config gate ≥99.0% byte-identity / IoU ≥0.99 / SR ≤0.1%". Measured:
**100.0000%** per-tensor equality on all 5 measured tensors (max abs diff
0) and ON-gate identity **99.99986%/100%/100%/100%** (the 3 fullres
differing voxels are the pre-existing fp16↔fp32 reference boundary class,
present identically on the OFF path — not resample divergence). The
theoretical fp64-level divergence of stock CuPy vs scipy at order ≥1 is
≤1.6e-14 (measured small-shape; below fp32 ulp at every measured corpus
voxel). No "bare met": the guarantee is the measured ≥99% bar, and the
corpus evidence is 100.0000%.

**GPUP-02 (zero CPU-GPU transfers in the resample span): met for the
resample span only; numpy reductions + ~8 MB mask round trip stay CPU per
locked D-12/D-13.** Flag ON: the image resample span is GPU-end-to-end
(fine H2D of fp32 input + D2H of the fp32 result only at the span
boundaries; per-channel min/max clip bounds are 2×4-byte scalar syncs).
Residuals: the Z-score/CT mean-std reductions stay on numpy (Phase 1
bit-exactness decision, D-12/D-13), the nonzero-mask round trip (~8 MB
uint8) stays CPU (scipy `binary_fill_holes`), the seg path round-trips
per-label fp64 masks D2H for the numpy `>= 0.5` tail, and the postresample
result returns to CPU because the reference torch CPU softmax
(thread-scoped, bit-exactness decision) runs downstream.

## 6. Flag default + Plan 05 note

Flag default flipped **ON** (this commit) after the Step C gate was green;
`HOLOSCAN_GPU_RESAMPLE=0` is the documented scipy fallback (D-21-style
default-ON convention). Expected-latency note for Plan 05's 2×2 matrix:
the ~28.8 s scipy resample spans (22.2% of the 129.5 s Phase 2 bundle)
should collapse toward seconds on the ON path — the matrix measures it.
The matrix's OFF column = `HOLOSCAN_GPU_RESAMPLE=0` (no longer the
default).
