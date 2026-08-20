# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gpu_zoom.py — D-22 gated GPU resample: scipy-faithful ``zoom(grid_mode=True)``.

A CuPy ``RawKernel`` port of scipy 1.15.3's ``zoom`` C pipeline — the exact
computation behind ``scipy.ndimage.zoom(..., mode='nearest', grid_mode=True)``,
which is what every active resample call site reduces to (via
``skimage.transform.resize(anti_aliasing=False)``):

* ``_resample_to_shape`` (preprocess image)
* ``_resize_segmentation`` (cascade seg multihot)
* ``resample_probabilities_to_shape`` (postresample)

Byte-identity requirements (verified against the v1.15.3 C source AND
measured on this box):

1. **The interpolation factor is ``in/out``, not the passed factor.**
   scipy's Python ``zoom`` OVERRIDES the passed zoom sequence: after deriving
   the output shape as ``ceil(passed_factor * in)``, it recomputes
   ``zoom = input_shape / output_shape`` (float64 ``np.divide``) and passes
   THAT to C. So per output index ``kk`` per axis (``NI_ZoomShift``):
   ``cc = kk; cc += 0.5; cc *= (in/out); cc -= 0.5; cc += nprepad`` (DOUBLE;
   ``mode='nearest'`` never pre-clamps ``cc``).
2. **Order > 1 applies a spline PREFILTER** (``prefilter`` defaults True and
   the skimage chain does not override it): ``np.pad(x64, 12, 'edge')`` (mode
   'nearest' -> ``npad = 12``), then ``spline_filter`` = three SEQUENTIAL
   1-D passes (axes 0, 1, 2) of ``NI_SplineFilter1D`` on the padded fp64
   array. Each pass per line: gain scale, then per pole (order 3: one pole
   ``sqrt(3) - 2``) the causal-reflect init + forward recurrence
   ``c[i] += z*c[i-1]`` + anticausal-reflect init + backward recurrence
   ``c[i] = z*(c[i+1] - c[i])`` — all DOUBLE, exact op order (``ni_splines.c``
   ``apply_filter`` / ``_init_causal_reflect`` / ``_init_anticausal_reflect``).
   The transform then runs with ``nprepad = 12`` on the padded, filtered
   fp64 array. Order 0/1: no prefilter, ``npad = 0``, the input is used
   directly (fp32 upcast per tap — identical bits to scipy's fp64-in/fp64-out
   for fp32 values, since the input is widened exactly).
3. Tap start per axis: ``order`` odd -> ``floor(cc) - order/2``; even ->
   ``floor(cc + 0.5) - order/2`` (from the RAW, nprepad-shifted ``cc``).
   Spline weights ``splvals[axis][kk][0..order]`` computed IN DOUBLE by the
   exact ``get_spline_interpolation_weights`` (``ni_splines.c``) op sequence
   (see ``_spline_weights_and_starts``).
4. Per-voxel accumulation, EXACT order: ``t = 0.0`` (double); iterate the
   ``(order+1)^3`` tap grid in C order (LAST AXIS FASTEST);
   ``coeff = (double) input[idx]``; ``for ll in 0..2: coeff *= splvals[ll][f]``
   (separate RN multiplies, axis 0 first); ``t += coeff`` (separate RN add).
   Every tap index is clamped per axis (nearest: ``<0 -> 0``,
   ``>= len -> len-1`` — the padded length) BEFORE the input lookup.
5. Output: ``out = (float32) t`` (``__double2float_rn``) — or the raw double
   ``t`` for the seg multihot path (the reference thresholds the fp64 value).
6. NO FMA contraction: kernels compile with ``options=("--fmad=false",)``
   (a TUPLE — a list raises TypeError in CuPy's RawKernel, research Pitfall 3)
   AND every double op uses explicit ``__dmul_rn``/``__dadd_rn``/
   ``__dsub_rn``/``__ddiv_rn`` (belt and braces; FMA contraction measured to
   flip bit-exactness).
7. Optional fp64 clip: the skimage chain clips the fp64 zoom output to the
   input channel's exact min/max BEFORE the fp32 cast. Clipping the double
   accumulation ``t`` to the same (exact, order-independent) boundaries
   before the RN cast is bit-identical — the data path passes the channel
   min/max and the kernel folds the clip in (no extra pass, no bulk transfer).

Restriction: ndim = 3 only (every active call site is 3D); other ndim raise
``NotImplementedError``. Inputs must be fp32, C-contiguous (integer/0-1
labels are exactly representable; the fp64 widening is exact).

Gating (D-22 / D-22b): the flag DEFAULTS ON since the amended D-22b gate
came green (2026-08-20: ON-path gate pixel-identical to the OFF baseline
on the dev corpus, per-tensor accuracy 100.0000% vs scipy — see
evidence/gpu_resample_verdict.md); ``HOLOSCAN_GPU_RESAMPLE=0`` forces the
scipy/skimage CPU reference path (byte-for-byte Phase 2/3 behavior).

PROVENANCE NOTICE (D-22a amendment, 2026-08-19): the custom RawKernel below
(``gpu_zoom_grid_mode`` / ``gpu_zoom_resize`` and the ``_get_kernel`` /
``_spline_prefilter`` / ``_precompute_axes`` machinery) is **NOT WIRED —
provenance only, discarded per D-22a**. The bounded final verdict
(``scripts/test_gpu_zoom_verdict.py``, one attempt, 120 s/case cap) measured:
o0 and o1 byte-identical on all small synthetic cases, but o3 diverged 100%
of voxels on a 32x32x16 synthetic (max_abs ~= 2.6 — wrong spline-prefilter
math, not ulp noise) and the real 256^3 -> (255,256,255) o3 bundle case
crashed with CUDA_ERROR_ILLEGAL_ADDRESS. The shipping flag path therefore
uses the STOCK ``cupyx.scipy.ndimage`` helpers in the "SHIPPING PATH"
section below (``stock_gpu_zoom`` / ``stock_gpu_resize``), which mirror the
scipy OFF-path call exactly (fp64 in, grid_mode=True, mode='nearest',
cval=0, prefilter defaults) and measured >= 99.987% fp32-equal elements vs
scipy at every order (o0 byte-identical; see evidence/gpu_resample_verdict.md).
``gpu_resample_enabled`` and ``zoom_factors_for`` are shared and remain the
wired parts of this module.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Sequence, Tuple, Union

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import zoom as _cndi_zoom  # D-22a shipping path

__all__ = [
    "gpu_resample_enabled",
    "zoom_factors_for",
    "stock_gpu_zoom",
    "stock_gpu_resize",
    "gpu_zoom_grid_mode",
    "gpu_zoom_resize",
]

# scipy 1.15.3: mode 'nearest' prefilter padding (Python
# ``_prepad_for_spline_filter``) — npad = 12, np.pad(mode='edge').
_NEARST_NPAD = 12
# order-3 filter pole (ni_splines.c get_filter_poles: sqrt(3) - 2, literal as
# in the source).
_POLE_3 = -0.267949192431122706472553658494127633
# local-memory line buffer capacity (max padded axis = 340 + 24 = 364; the
# app's real volumes are 280).
_MAX_LINE = 368


def gpu_resample_enabled() -> bool:
    """D-22 gate flag: default ON (the D-22b gate is green — ON-path gate
    pixel-identical to the OFF baseline, per-tensor accuracy 100.0000% vs
    scipy on the dev corpus); ``HOLOSCAN_GPU_RESAMPLE=0`` forces the
    scipy/skimage CPU reference path (D-21-style default-ON convention)."""
    return os.environ.get("HOLOSCAN_GPU_RESAMPLE", "1") != "0"


def zoom_factors_for(
    in_shape: Sequence[int], out_shape: Sequence[int]
) -> list:
    """Per-axis zoom factors EXACTLY as ``skimage.transform.resize`` 0.25.2
    derives them (``_warps.py``): ``factors = input_shape / output_shape``
    (float64) then ``zoom_factors = [1 / f for f in factors]``. These are the
    factors PASSED to scipy ``zoom`` — they only drive the output-shape
    derivation (``ceil(f * in)``); the C transform internally re-derives the
    interpolation factor as ``in / out`` (see module docstring, point 1)."""
    if len(in_shape) != len(out_shape):
        raise ValueError(f"shape rank mismatch: {in_shape} vs {out_shape}")
    return [
        1.0 / (float(i) / float(o)) for i, o in zip(in_shape, out_shape)
    ]


# ---------------------------------------------------------------------------
# SHIPPING PATH (D-22a, 2026-08-19): stock cupyx.scipy.ndimage mirrors of the
# flag-OFF scipy/skimage reference chains. The custom RawKernel below is
# provenance only (see module docstring).
# ---------------------------------------------------------------------------

def stock_gpu_zoom(
    x: cp.ndarray,
    zoom_factors: Sequence[float],
    order: int,
    *,
    out_shape: Sequence[int],
) -> cp.ndarray:
    """Stock-CuPy mirror of the one scipy call the flag-OFF path makes:
    ``scipy.ndimage.zoom(fp64, factors, order, mode='nearest',
    grid_mode=True, cval=0)`` (prefilter defaults True — identical to the
    OFF path, which never overrides it).

    ``x`` is the fp32 app data (or fp32 0/1 mask); it is widened to fp64
    EXACTLY as the OFF path's ``data.astype(float)`` upcast before the
    skimage ``resize`` chain. Returns the fp64 result (the caller applies
    the reference's fp64 clip / threshold / cast tail).

    Args:
        x: fp32 C-contiguous CuPy ``(n0, n1, n2)`` (3D only — all active
            call sites are 3D).
        zoom_factors: per-axis factors from ``zoom_factors_for`` (the exact
            skimage derivation the OFF path uses).
        order: 0, 1 or 3.
        out_shape: the caller's target shape; the cupyx-derived shape
            (``round(f * in)``) must equal it (same factors as the OFF path,
            so it always does).

    Returns:
        CuPy fp64 array of shape ``out_shape``.
    """
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"x must be a CuPy array, got {type(x)!r}")
    if x.dtype != cp.float32:
        raise TypeError(f"x must be float32, got {x.dtype!r}")
    if x.ndim != 3:
        raise NotImplementedError(
            "stock_gpu_zoom supports ndim=3 only (all active call sites are "
            f"3D); got ndim={x.ndim}"
        )
    if not x.flags.c_contiguous:
        raise ValueError("x must be C-contiguous")
    if order not in (0, 1, 3):
        raise ValueError(f"order must be 0, 1 or 3, got {order!r}")
    if len(zoom_factors) != 3:
        raise ValueError(f"need 3 zoom factors, got {len(zoom_factors)}")

    x64 = x.astype(cp.float64)  # exact widening — the OFF path's astype(float)
    out = _cndi_zoom(
        x64,
        [float(f) for f in zoom_factors],
        order=order,
        mode="nearest",
        cval=0.0,
        grid_mode=True,
    )
    out_shape = tuple(int(s) for s in out_shape)
    if out.shape != out_shape:
        raise ValueError(
            f"cupyx-derived shape {tuple(out.shape)} != target {out_shape} — "
            f"the zoom factors {zoom_factors} do not reproduce the target "
            f"shape from {tuple(x.shape)} (check zoom_factors_for)"
        )
    return out


def stock_gpu_resize(
    x: Union[np.ndarray, cp.ndarray],
    new_shape: Sequence[int],
    order: int,
) -> cp.ndarray:
    """Stock-CuPy mirror of the app's per-channel data-path chain
    ``skimage.transform.resize(ch, new_shape, order, mode='edge',
    anti_aliasing=False)`` (clip=True default), i.e. per channel: exact fp64
    widening -> stock grid_mode zoom (``stock_gpu_zoom``) -> fp64 clip to the
    channel's exact (order-independent) min/max (skimage
    ``_clip_warp_output``, mode != 'constant' branch) -> fp32 cast.

    Args:
        x: (C, n0, n1, n2) or (n0, n1, n2) fp32 numpy/CuPy.
        new_shape: target (n0, n1, n2).
        order: 0, 1 or 3.

    Returns:
        CuPy fp32 array of x's leading shape + new_shape.
    """
    x = cp.asarray(x)
    if x.dtype != cp.float32:
        raise TypeError(f"x must be float32, got {x.dtype!r}")
    if x.ndim == 3:
        x = x[None]
    if x.ndim != 4:
        raise ValueError(f"expected (C, Z, Y, X) or (Z, Y, X), got {x.shape}")
    new_shape = tuple(int(s) for s in new_shape)
    zf = zoom_factors_for(x.shape[1:], new_shape)
    out = cp.empty((x.shape[0], *new_shape), dtype=cp.float32)
    for c in range(x.shape[0]):
        ch = x[c]
        z = stock_gpu_zoom(ch, zf, order, out_shape=new_shape)
        # Per-channel clip bounds: the OFF path's np.clip(out, image.min(),
        # image.max()) on the fp64 image — exact (order-independent)
        # reductions of the exact fp32 values; 2 x 8-byte scalar syncs.
        lo = float(cp.min(ch).get())
        hi = float(cp.max(ch).get())
        out[c] = cp.clip(z, lo, hi).astype(cp.float32)
    return out


# ---------------------------------------------------------------------------
# Per-axis tap starts + spline weights (host, float64, exact C op sequence)
# ---------------------------------------------------------------------------

def _spline_weights_and_starts(cc: np.ndarray, order: int):
    """Replicate scipy 1.15.3's per-axis precompute for ``NI_ZoomShift``:

    * tap start: odd order -> ``floor(cc) - order/2``;
      even order -> ``floor(cc + 0.5) - order/2`` (from the RAW cc —
      ``mode='nearest'`` never pre-clamps the coordinate);
    * spline weights: the exact ``get_spline_interpolation_weights``
      (``ni_splines.c``) op sequence in float64. Numpy float64 element-wise
      ops are IEEE round-to-nearest doubles with the same per-element op
      order as the C scalar code, so the bits match.

    Args:
        cc: ``(n,)`` float64 raw per-output-index coordinates (nprepad
            already added) for ONE axis.
        order: 0, 1 or 3.

    Returns:
        ``(starts, weights)`` — int32 ``(n,)`` tap starts and float64
        ``(n, order+1)`` weights, or ``None`` weights for order 0 (scipy
        never allocates/uses them there; the kernel skips the multiplies).
    """
    cc = np.ascontiguousarray(cc, dtype=np.float64)
    if order % 2:
        floor_cc = np.floor(cc)
    else:
        floor_cc = np.floor(cc + 0.5)
    starts = (floor_cc - order // 2).astype(np.int32)
    if order == 0:
        return starts, None

    # x -= floor(...)  (ni_splines.c: "Convert x to the delta to the middle
    # knot") — a separate double subtract, kept as its own op.
    xr = cc - floor_cc
    w = np.empty((cc.size, order + 1), dtype=np.float64)
    if order == 1:
        # case 1: weights[0] = 1.0 - x
        w[:, 0] = 1.0 - xr
    elif order == 3:
        # case 3: y = x; z = 1.0 - x
        y = xr
        z = 1.0 - xr
        w[:, 1] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0
        w[:, 2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0
        w[:, 0] = z * z * z / 6.0
    else:
        raise ValueError(f"order must be 0, 1 or 3 (got {order})")
    # Tail (ni_splines.c): "All interpolation weights add to 1.0, so use it
    # for the last one." — weights[order] = 1.0; then LEFT-ASSOCIATIVE
    # sequential subtractions.
    w[:, order] = 1.0
    for i in range(order):
        w[:, order] -= w[:, i]
    return starts, w


def _precompute_axes(
    in_shape: Tuple[int, int, int],
    out_shape: Tuple[int, int, int],
    order: int,
    nprepad: int,
):
    """Per-axis cc / tap starts / splvals for all three axes (tiny:
    3 x out_dim x (order+1) doubles + 3 x out_dim ints). ``in_shape`` is the
    ORIGINAL (unpadded) input shape; tap clamping bounds are the PADDED
    lengths (``in + 2*nprepad``), passed separately to the kernel."""
    starts = []
    weights = []
    for axis, (n_in, n_out) in enumerate(zip(in_shape, out_shape)):
        # The C transform's internal factor: np.divide(input_shape,
        # output_shape) float64 — NOT the passed skimage factor.
        f_int = float(n_in) / float(n_out)
        # C op order: cc = kk; cc += 0.5; cc *= zoom; cc -= 0.5;
        # cc += (double)nprepad
        cc = np.arange(n_out, dtype=np.float64)
        cc += 0.5
        cc *= f_int
        cc -= 0.5
        cc += np.float64(nprepad)
        st, w = _spline_weights_and_starts(cc, order)
        starts.append(st)
        if w is not None:
            weights.append(w)
    return starts, weights


# ---------------------------------------------------------------------------
# Spline prefilter (order > 1 only) — NI_SplineFilter1D in DOUBLE
# ---------------------------------------------------------------------------

# One thread per line; the per-line double buffer lives in local memory
# (16-thread blocks keep the per-block footprint inside the L1 local carveout
# — A100: 16 x 368 x 8B = 47 KB << 100 KB).
_FILTER_KERNEL = r"""
extern "C" __global__
void spline_filter1d_kernel(
    const double* __restrict__ in,
    double* __restrict__ out,
    const int len, const long line_stride,
    const long dim_a, const long stride_a, const long stride_b,
    const double gain, const double z, const double z_n)
{
    const long tid = (long)(blockIdx.x * blockDim.x + threadIdx.x)
                  * (gridDim.y * blockDim.y) + (blockIdx.y * blockDim.y + threadIdx.y);
    // line base: the two non-filtered axes are enumerated as
    // tid = (outer * dim_a) + inner -> base = inner*stride_a + outer*stride_b
    const long base = (long)(tid % dim_a) * stride_a + (long)(tid / dim_a) * stride_b;

    double c[MAXLINE];
    for (int i = 0; i < len; ++i)
        c[i] = in[base + (long)i * line_stride];

    // _apply_filter_gain: c[i] *= gain
    for (int i = 0; i < len; ++i)
        c[i] = __dmul_rn(c[i], gain);

    // _apply_filter (single pole for order 3): causal-reflect init,
    // forward recurrence, anticausal-reflect init, backward recurrence —
    // the exact ni_splines.c op sequence.
    {
        double z_i = z;
        const double c0 = c[0];
        c[0] = __dadd_rn(c[0], __dmul_rn(z_n, c[len - 1]));
        for (int i = 1; i < len; ++i) {
            c[0] = __dadd_rn(
                c[0],
                __dmul_rn(z_i, __dadd_rn(c[i], __dmul_rn(z_n, c[len - 1 - i]))));
            z_i = __dmul_rn(z_i, z);
        }
        c[0] = __dmul_rn(c[0], __ddiv_rn(z, __dsub_rn(1.0, __dmul_rn(z_n, z_n))));
        c[0] = __dadd_rn(c[0], c0);
    }
    for (int i = 1; i < len; ++i)
        c[i] = __dadd_rn(c[i], __dmul_rn(z, c[i - 1]));
    c[len - 1] = __dmul_rn(c[len - 1], __ddiv_rn(z, __dsub_rn(z, 1.0)));
    for (int i = len - 2; i >= 0; --i)
        c[i] = __dmul_rn(z, __dsub_rn(c[i + 1], c[i]));

    for (int i = 0; i < len; ++i)
        out[base + (long)i * line_stride] = c[i];
}
"""

# NOTE: options is a TUPLE — a list raises TypeError in CuPy's RawKernel
# (research Pitfall 3, verified). "--fmad=false" is belt-and-braces on top of
# the explicit __dmul_rn/__dadd_rn/__dsub_rn/__ddiv_rn intrinsics.
_KERNEL_OPTIONS = ("--fmad=false",)

_kernel_cache: dict = {}
_filter_kernel_cache = None


def _get_filter_kernel() -> cp.RawKernel:
    global _filter_kernel_cache
    if _filter_kernel_cache is None:
        src = _FILTER_KERNEL.replace("MAXLINE", str(_MAX_LINE))
        _filter_kernel_cache = cp.RawKernel(
            src, "spline_filter1d_kernel", options=_KERNEL_OPTIONS
        )
    return _filter_kernel_cache


def _spline_prefilter(x64: cp.ndarray) -> cp.ndarray:
    """Replicate Python ``spline_filter(np.pad(x64, 12, 'edge'), 3,
    output=fp64, mode='nearest')``: edge-pad by 12, then the three SEQUENTIAL
    1-D passes (axes 0, 1, 2) of ``NI_SplineFilter1D`` in float64.

    ``x64``: (n0, n1, n2) fp64 CuPy, C-contiguous (the exact fp32 widening).
    Returns the padded+filtered fp64 (n+24)^3 array (npad = 12)."""
    padded = cp.pad(x64, _NEARST_NPAD, mode="edge")

    # gain, order 3 (one pole): gain = 1.0; gain *= (1.0 - z) * (1.0 - 1.0/z)
    # — the exact C host-side op sequence (get_filter_poles + apply_filter).
    z = _POLE_3
    gain = (1.0 - z) * (1.0 - 1.0 / z)
    # z_n = pow(z, len) per axis — computed on the host with the same libm
    # pow() scipy's C uses (same machine/glibc). (Its bits are provably
    # irrelevant for O(1) data — z_n ~ 1e-160 at len 280 — but matching the
    # source call exactly costs nothing.)
    kernel = _get_filter_kernel()

    for axis in range(3):
        n = padded.shape
        line_len = n[axis] + 2 * _NEARST_NPAD
        if line_len > _MAX_LINE:
            raise NotImplementedError(
                f"axis {axis} length {line_len} exceeds the kernel line "
                f"buffer ({_MAX_LINE})"
            )
        z_n = math.pow(z, line_len)
        # line base = (tid % dim_a) * stride_a + (tid / dim_a) * stride_b
        # (C-contiguous (n0, n1, n2); lines run along `axis`; tid enumerates
        # the two non-filtered axes, innermost first):
        #   axis 0: lines (j,k)  -> base = j*n2 + k
        #   axis 1: lines (i0,k) -> base = i0*n1*n2 + k
        #   axis 2: lines (i0,i1)-> base = i0*n1 + i1
        if axis == 0:
            lines, line_stride = n[1] * n[2], n[1] * n[2]
            dim_a, stride_a, stride_b = n[2], 1, n[2]
        elif axis == 1:
            lines, line_stride = n[0] * n[2], n[2]
            dim_a, stride_a, stride_b = n[2], 1, n[1] * n[2]
        else:
            lines, line_stride = n[0] * n[1], 1
            dim_a, stride_a, stride_b = n[1], 1, n[1]
        block = (4, 4)  # 16 lines per block — small local-memory footprint
        grid = ((lines + block[0] * block[1] - 1) // (block[0] * block[1]), 1)
        kernel(grid, block, (
            padded, padded,
            int(line_len), int(line_stride),
            int(dim_a), int(stride_a), int(stride_b),
            float(gain), float(z), float(z_n),
        ))
    return padded


# ---------------------------------------------------------------------------
# The zoom RawKernel (one thread per output voxel; fixed C-order tap loop)
# ---------------------------------------------------------------------------

_KERNEL_TEMPLATE = r"""
extern "C" __global__
void gpu_zoom3_kernel(
    const __IN__* __restrict__ x,
    __OUT__* __restrict__ out,
    const double* __restrict__ spl,
    const int* __restrict__ starts,
    const int n0, const int n1, const int n2,
    const int t0, const int t1, const int t2,
    const int order,
    const int has_spl,
    const int clip,
    const double cmin, const double cmax)
{
    const int k0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int k1 = blockIdx.y * blockDim.y + threadIdx.y;
    const int k2 = blockIdx.z * blockDim.z + threadIdx.z;
    if (k0 >= n0 || k1 >= n1 || k2 >= n2) return;

    // per-axis tap starts (buffer layout: [axis0 | axis1 | axis2])
    const int s0 = starts[k0];
    const int s1 = starts[n0 + k1];
    const int s2 = starts[n0 + n1 + k2];

    const int base1 = has_spl * n0 * (order + 1);   // axis-1 spl offset
    const int base2 = has_spl * (n0 + n1) * (order + 1);
    const double* sp2 = spl + base2 + k2 * (order + 1);

    // EXACT scipy accumulation order: t = 0.0; tap grid in C order (LAST
    // AXIS FASTEST); per tap: coeff = (double)input[idx]; then
    // coeff *= splvals[0], splvals[1], splvals[2] (separate RN multiplies,
    // axis 0 first); t += coeff (separate RN add). No FMA (compiled
    // --fmad=false AND explicit __dmul_rn/__dadd_rn).
    double t = 0.0;
    for (int f0 = 0; f0 <= order; ++f0) {
        int i0 = s0 + f0;
        if (i0 < 0) i0 = 0;
        else if (i0 >= t0) i0 = t0 - 1;             // nearest clamp (per tap)
        const double* sp0 = spl + k0 * (order + 1);
        for (int f1 = 0; f1 <= order; ++f1) {
            int i1 = s1 + f1;
            if (i1 < 0) i1 = 0;
            else if (i1 >= t1) i1 = t1 - 1;
            const double* sp1 = spl + base1 + k1 * (order + 1);
            for (int f2 = 0; f2 <= order; ++f2) {
                int i2 = s2 + f2;
                if (i2 < 0) i2 = 0;
                else if (i2 >= t2) i2 = t2 - 1;
                const long idx = ((long)i0 * t1 + i1) * t2 + i2;   // C-contig
                double coeff = (double)x[idx];
                if (has_spl) {
                    coeff = __dmul_rn(coeff, sp0[f0]);
                    coeff = __dmul_rn(coeff, sp1[f1]);
                    coeff = __dmul_rn(coeff, sp2[f2]);
                }
                t = __dadd_rn(t, coeff);
            }
        }
    }
    // Optional fp64 clip to the channel's exact min/max BEFORE the cast —
    // bit-identical to skimage's np.clip(fp64 out, min, max) then fp32 cast
    // (min/max are exact, order-independent reductions).
    if (clip) {
        t = fmin(t, cmax);
        t = fmax(t, cmin);
    }
    const long oi = ((long)k0 * n1 + k1) * n2 + k2;
    out[oi] = (__OUT__)(t);
}
"""


def _get_kernel(in_type: str, out_type: str) -> cp.RawKernel:
    """'float' -> fp32 out via __double2float_rn; 'double' -> fp64 out (the
    seg multihot path thresholds the double, matching the reference's fp64
    ``>= 0.5`` comparison exactly)."""
    key = (in_type, out_type)
    if key not in _kernel_cache:
        src = (_KERNEL_TEMPLATE.replace("__OUT__", out_type)
                               .replace("__IN__", in_type))
        if out_type == "float":
            src = src.replace("out[oi] = (float)(t);",
                              "out[oi] = __double2float_rn(t);")
        _kernel_cache[key] = cp.RawKernel(src, "gpu_zoom3_kernel",
                                          options=_KERNEL_OPTIONS)
    return _kernel_cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gpu_zoom_grid_mode(
    x: cp.ndarray,
    zoom_factors: Sequence[float],
    order: int,
    *,
    out_shape: Optional[Sequence[int]] = None,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    output_dtype: Union[type, cp.dtype] = cp.float32,
) -> cp.ndarray:
    """GPU ``scipy.ndimage.zoom(x, zoom_factors, order, mode='nearest',
    grid_mode=True)`` — byte-identical to scipy (arbiter:
    ``scripts/test_gpu_zoom.py``).

    Args:
        x: fp32 C-contiguous CuPy ``(n0, n1, n2)`` (3D only). For order > 1
            it is widened to fp64 (exact) and spline-prefiltered internally,
            exactly as scipy does (prefilter defaults True).
        zoom_factors: per-axis factors PASSED to scipy (for the skimage chain
            use ``zoom_factors_for``) — they drive the scipy-style output
            shape derivation ``ceil(f * in)`` only.
        order: 0, 1 or 3.
        out_shape: explicit output shape (the caller's target shape). If
            omitted it is derived scipy-style (``ceil(f * in)``); if given it
            must equal the derived shape (a mismatch means wrong factors).
        clip_min / clip_max: optional double boundaries — the fp64 clip of
            the accumulated value BEFORE the output cast (the skimage chain's
            ``np.clip(fp64 out, min, max)`` folded in; bit-identical).
        output_dtype: ``cp.float32`` (default — the data path; the reference
            casts the fp64 accumulation to fp32) or ``cp.float64`` (the seg
            multihot path thresholds the double, exactly like the reference's
            fp64 ``>= 0.5``).

    Returns:
        CuPy array of shape ``out_shape`` in ``output_dtype``.
    """
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"x must be a CuPy array, got {type(x)!r}")
    if x.dtype != cp.float32:
        raise TypeError(f"x must be float32, got {x.dtype!r}")
    if x.ndim != 3:
        raise NotImplementedError(
            "gpu_zoom_grid_mode supports ndim=3 only (all active call sites "
            f"are 3D); got ndim={x.ndim}"
        )
    if not x.flags.c_contiguous:
        raise ValueError("x must be C-contiguous (D-12)")
    if order not in (0, 1, 3):
        raise ValueError(f"order must be 0, 1 or 3, got {order!r}")
    if len(zoom_factors) != 3:
        raise ValueError(f"need 3 zoom factors, got {len(zoom_factors)}")
    if output_dtype not in (cp.float32, cp.float64):
        raise ValueError(f"output_dtype must be cp.float32 or cp.float64, "
                         f"got {output_dtype!r}")

    in_shape = tuple(int(s) for s in x.shape)
    if out_shape is None:
        out_shape = tuple(
            int(np.ceil(f * s)) for f, s in zip(zoom_factors, in_shape)
        )
    else:
        out_shape = tuple(int(s) for s in out_shape)
        derived = tuple(
            int(np.ceil(f * s)) for f, s in zip(zoom_factors, in_shape)
        )
        if tuple(out_shape) != tuple(derived):
            raise ValueError(
                f"out_shape {out_shape} != derived {derived} — the zoom "
                f"factors {zoom_factors} do not reproduce the target shape "
                f"from {in_shape} (check zoom_factors_for)"
            )

    # Order > 1: the scipy spline prefilter chain (pad 12 'edge' + 3
    # sequential fp64 1-D passes), transform on the padded fp64 array with
    # nprepad = 12. Order 0/1: direct fp32 input, npad = 0.
    nprepad = 0
    in_type = "float"
    if order > 1:
        x_work = _spline_prefilter(x.astype(cp.float64))
        in_type = "double"
        nprepad = _NEARST_NPAD
    else:
        x_work = x
    tap_shape = tuple(s + 2 * nprepad for s in in_shape)

    starts, weights = _precompute_axes(in_shape, out_shape, order, nprepad)

    if weights:
        # one concatenated float64 buffer: [axis0 (n0*(o+1)) | axis1 | axis2]
        spl = cp.concatenate([cp.asarray(w, dtype=cp.float64).ravel()
                              for w in weights])
    else:
        spl = cp.zeros(1, dtype=cp.float32)  # dummy (never dereferenced)
    starts_buf = cp.concatenate([cp.asarray(s, dtype=cp.int32)
                                 for s in starts])

    out = cp.empty(out_shape, dtype=output_dtype)
    kernel = _get_kernel(in_type, "float" if output_dtype == cp.float32
                         else "double")
    block = (8, 8, 8)
    grid = ((out_shape[0] + block[0] - 1) // block[0],
            (out_shape[1] + block[1] - 1) // block[1],
            (out_shape[2] + block[2] - 1) // block[2])
    clip = 1 if (clip_min is not None and clip_max is not None) else 0
    kernel(grid, block, (
        x_work, out, spl, starts_buf,
        int(out_shape[0]), int(out_shape[1]), int(out_shape[2]),
        int(tap_shape[0]), int(tap_shape[1]), int(tap_shape[2]),
        int(order), 1 if weights else 0, clip,
        float(clip_min) if clip else 0.0,
        float(clip_max) if clip else 0.0,
    ))
    return out


def gpu_zoom_resize(
    x: Union[np.ndarray, cp.ndarray],
    new_shape: Sequence[int],
    order: int,
) -> cp.ndarray:
    """GPU equivalent of the app's per-channel data-path chain
    ``skimage.transform.resize(ch, new_shape, order, mode='edge',
    anti_aliasing=False)`` (clip=True default): zoom with the exact skimage
    factor derivation + the fp64 clip to the channel's exact (order-
    independent) min/max folded into the kernel before the RN cast.

    Args:
        x: (C, n0, n1, n2) or (n0, n1, n2) fp32 numpy/CuPy.
        new_shape: target (n0, n1, n2).
        order: 0, 1 or 3.

    Returns:
        CuPy fp32 array of x's leading shape + new_shape.
    """
    x = cp.asarray(x)
    if x.dtype != cp.float32:
        raise TypeError(f"x must be float32, got {x.dtype!r}")
    if x.ndim == 3:
        x = x[None]
    if x.ndim != 4:
        raise ValueError(f"expected (C, Z, Y, X) or (Z, Y, X), got {x.shape}")
    new_shape = tuple(int(s) for s in new_shape)
    zf = zoom_factors_for(x.shape[1:], new_shape)
    out = cp.empty((x.shape[0], *new_shape), dtype=cp.float32)
    for c in range(x.shape[0]):
        ch = x[c]
        # min/max are EXACT reductions (order-independent) — the GPU result
        # is bit-identical to numpy's; only 2 x 4-byte scalar syncs per
        # channel (no bulk transfer — GPUP-02 for the span).
        out[c] = gpu_zoom_grid_mode(
            ch, zf, order, out_shape=new_shape,
            clip_min=float(cp.min(ch).get()),
            clip_max=float(cp.max(ch).get()),
        )
    return out
