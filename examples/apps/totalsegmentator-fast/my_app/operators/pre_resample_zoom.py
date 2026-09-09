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

"""pre_resample_zoom.py — spacing-factor pre-resample zoom wrapper (Phase 8.1).

Pure-function replica of the oracle's ``ResampleToModelSpacingd`` resample
call (``ct-totalsegmentator-map/app_total/nnunet_seg_operator.py``, pinned
SHA 7676d95), re-stated here for MAP self-containment — the oracle is NOT
imported by app code or app tests:

    ndimage_zoom(img_np[c], zoom_factors, order=3, mode="nearest")   # per channel

with the oracle's exact semantics:

* ``zoom_factors = current_spacing / 1.5`` per RAS axis, where
  ``current_spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))`` (column
  norms of the RAS affine) — direction: original spacing ÷ target 1.5 mm.
* ``grid_mode`` is NOT passed → scipy default ``False`` (the v1.1
  ``gpu_zoom.py`` ``stock_gpu_zoom`` hardcodes the v1.1 grid-mode variant
  (grid_mode enabled) and consumes shape-derived factors — the WRONG
  signature here; it is NOT reused, Pitfall 1).
* ``prefilter`` default True, ``cval`` default 0.0 (irrelevant under
  ``mode="nearest"``).
* Skip branch: ``np.allclose(zoom_factors, 1.0, atol=1e-3)`` with the
  DEFAULT ``rtol=1e-5`` → per-axis window |f−1| ≤ 0.00101 ⇔ spacing
  ∈ [1.498505, 1.501495]; ALL three axes must pass. A skipped volume is
  returned untouched (no cast, no re-allocation).

Quantization chain (08-03 gate probe-verified on real corpus studies —
LOAD-BEARING; SUPERSEDES the 08-01 int16-round assumption):

* The oracle's LoadImaged (InMemImageReader) casts the native int16 CT
  volume to **float32** BEFORE ResampleToModelSpacingd runs — verified by
  dumping the oracle's own chain: the MetaTensor reaching the resample is
  f32 (exact int16 values). So the oracle's zoom is:

      f32 widen (exact) -> ndimage_zoom(f32, order=3, mode='nearest')
      (internally fp64 spline) -> **f32 output** (scipy writes the result
      in the input dtype, fp64->f32 round-to-even) -> `.astype(int32)`
      (oracle line 130 — a float->int **truncation toward zero**, NOT a
      no-op, NOT round-half-even).

* The 08-01 assumption (int16 zoom in, int16 round-half-even write, no-op
  int32 cast) DIVERGES from the oracle by 1 HU on ~40% of real-study
  voxels (08-03 gate, first run). The f32-widen chain is byte-identical
  (0 differing voxels) to the oracle dump.
* GPU/cupyx mirror: exact f64 widening (int→f64, then the same values as
  scipy's f32 input upcast) → ``cupyx.scipy.ndimage.zoom`` (order=3,
  mode='nearest', grid_mode=False, cval=0.0; f64 kernel, bit-matching
  scipy's f64 internal spline) → f32 cast (round-to-even) →
  ``.astype(int32)`` (truncation toward zero). A cupyx f32-input zoom or a
  ``cp.round``/int16 step is WRONG (08-03 probe: 155/57600 voxels Δ1 and
  ~40% Δ1 respectively).

Gating: the flag ``HOLOSCAN_GPU_PRERESAMPLE`` (default ON — flipped
2026-08-27 in Phase 11, after 3/3 in-chain parity A/B at the milestone
bars + the Phase 8 task 8.3 zero-tolerance stage proof: scipy == GPU
mirror == oracle dump, 0 differing voxels on all 3 pins). Semantics:
UNSET or exactly ``"1"`` → the GPU cupyx mirror; any other value
(e.g. ``"0"``) forces the scipy CPU kernel. Distinct from v1.1's
existing GPU-resample flag in gpu_zoom.py, which DEFAULTS ON with
``!= "0"`` semantics and keeps governing the nnUNet-chain call sites —
do NOT confuse the two.

Axis order: the volume argument is the RAS-ordered array — (R, A, S) per
channel, or (C, R, A, S) for a channel stack — as produced by the oracle's
``LoadImaged`` + ``Orientationd(axcodes="RAS")`` chain.
"""

from __future__ import annotations

import os

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import zoom as _cndi_zoom  # grid_mode=False mirror

__all__ = [
    "TARGET_SPACING",
    "gpu_preresample_enabled",
    "preresample_zoom_factors",
    "preresample_should_skip",
    "preresample_channel_cpu",
    "preresample_channel_gpu",
    "preresample_volume",
]

# Mirrors the oracle's _TARGET_SPACING (nnunet_seg_operator.py:52).
TARGET_SPACING = np.array([1.5, 1.5, 1.5], dtype=np.float64)


def gpu_preresample_enabled() -> bool:
    """Preresample kernel-selection flag: DEFAULT ON (flipped 2026-08-27,
    Phase 11 — 3/3 in-chain parity A/B at the milestone bars + the P8
    08-03 zero-tolerance stage proof).

    UNSET or exactly ``"1"`` → True (GPU cupyx mirror); any other value
    (e.g. ``"0"``) → False (forces the scipy CPU kernel).

    Deliberately distinct from v1.1's existing GPU-resample flag in
    gpu_zoom.py (default ON, ``!= "0"`` semantics, nnUNet-chain call
    sites) — do NOT confuse the two.
    """
    return os.environ.get("HOLOSCAN_GPU_PRERESAMPLE", "1") == "1"


def preresample_zoom_factors(affine: np.ndarray) -> np.ndarray:
    """Per-RAS-axis zoom factors EXACTLY as the oracle computes them
    (``nnunet_seg_operator.py:110,113``): column norms of the 4x4 RAS
    affine divided by the 1.5 mm target — direction original ÷ target.

    Args:
        affine: 4x4 affine (numpy, or torch-convertible).

    Returns:
        float64 (3,) array of zoom factors in RAS axis order.
    """
    affine_np = affine.numpy() if hasattr(affine, "numpy") else np.asarray(affine)
    current_spacing = np.sqrt((affine_np[:3, :3] ** 2).sum(axis=0))
    return (current_spacing / TARGET_SPACING).astype(np.float64)


def preresample_should_skip(zoom_factors: np.ndarray) -> bool:
    """The oracle's exact skip test (``nnunet_seg_operator.py:116``):
    ``np.allclose(zoom_factors, 1.0, atol=1e-3)`` — no rtol passed, so the
    default rtol=1e-5 applies.

    Per-axis skip window: |f−1| ≤ 0.00101 ⇔ spacing s ∈ [1.498505,
    1.501495]; ALL three axes must pass for the volume to be skipped.
    """
    return bool(np.allclose(zoom_factors, 1.0, atol=1e-3))


def preresample_channel_cpu(ch: np.ndarray, zoom_factors: np.ndarray, order: int = 3) -> np.ndarray:
    """CPU reference — the oracle's per-channel call re-stated with the
    oracle's ACTUAL input dtype (08-03 probe): the LoadImaged step casts
    the volume to float32, so the zoom runs on f32 (internally fp64)
    and WRITES f32; the oracle's line-130 ``astype(int32)`` is then a
    float->int TRUNCATION toward zero (exact int32 output).

    ``scipy.ndimage.zoom(ch_f32, zoom_factors, order=3, mode="nearest")``
    per channel (grid_mode default False, prefilter default True, cval
    irrelevant under mode='nearest').

    ``order`` defaults to 3 (oracle/MONAI semantics, byte-locked for
    total, Phase 8); order=1 matches the TS 2.18 CLI change_spacing
    input resample (Phase 19, 19-03: spec-driven via
    TaskSpec.resample_order).

    ``ch`` must be a 3D INTEGER array (int16 for CT — the widen to f32 is
    exact and reproduces the oracle's LoadImaged dtype). Float inputs
    raise TypeError — the oracle never receives float here.

    Returns int32.
    """
    ch = np.asarray(ch)
    if ch.ndim != 3:
        raise ValueError(f"expected a 3D channel, got ndim={ch.ndim}")
    if ch.dtype.kind not in "iu":
        raise TypeError(f"expected an integer channel (int16 for CT), got {ch.dtype!r}")
    from scipy.ndimage import zoom as ndimage_zoom

    x = ch.astype(np.float32)  # exact widen — the oracle's LoadImaged dtype
    out = ndimage_zoom(x, zoom_factors, order=order, mode="nearest")  # f32 out
    return out.astype(np.int32)  # oracle line 130: float->int truncation


def preresample_channel_gpu(ch: cp.ndarray, zoom_factors: np.ndarray, order: int = 3) -> cp.ndarray:
    """GPU mirror of ``preresample_channel_cpu`` — byte-identical on the
    oracle-faithful chain (08-03 gate-proven on real studies):

    exact f64 widen → ``cupyx.scipy.ndimage.zoom`` (order=3, mode='nearest',
    ``grid_mode=False``, cval=0.0) → f64 output → f32 cast (round-to-even,
    mirroring scipy's f32 write) → ``.astype(int32)`` (C truncation toward
    zero, matching the oracle's line 130 exactly).

    ``order`` defaults to 3 (oracle byte-locked, Phase 8); order=1 mirrors
    the TS 2.18 CLI input resample (Phase 19, 19-03) and is byte-identical
    to ``preresample_channel_cpu(..., order=1)`` (same f64 spline kernel
    family, same f32-write + truncation tail).

    08-03 probe (96^2x48, factors 0.462891^2/2.0): cupyx fed f32 input
    diverges from scipy on 155/57600 voxels (Δ1) — cupyx's spline kernel
    keeps f32 precision while scipy computes in f64. The f64 upcast (exact
    for integer inputs) is 0/57600 — so it is load-bearing. NO
    cp.round/int16 step: that was the 08-01 variant and diverges from the
    oracle's truncation.
    """
    if not isinstance(ch, cp.ndarray):
        raise TypeError(f"expected a CuPy array, got {type(ch)!r}")
    if ch.ndim != 3:
        raise ValueError(f"expected a 3D channel, got ndim={ch.ndim}")
    if ch.dtype.kind not in "iu":
        raise TypeError(f"expected an integer channel, got {ch.dtype!r}")
    if not ch.flags.c_contiguous:
        ch = cp.ascontiguousarray(ch)

    x64 = ch.astype(cp.float64)  # exact widen — cupyx spline must run in f64
    z = _cndi_zoom(
        x64,
        [float(f) for f in zoom_factors],
        order=order,
        mode="nearest",
        cval=0.0,
        grid_mode=False,
    )
    # Mirror the oracle: scipy WRITES f32 (f64 result rounded to f32), then
    # the oracle's line-130 astype(int32) truncates THAT f32 value.
    return z.astype(cp.float32).astype(cp.int32)


def preresample_volume(arr: np.ndarray, zoom_factors: np.ndarray, order: int = 3) -> np.ndarray:
    """Volume dispatch for the 08-02 operator.

    Contract (both paths): 3D in (R, A, S) -> 3D int32 out; 4D in
    (C, R, A, S) -> 4D int32 out. No channel dim is added or dropped.

    * SKIP: if ``preresample_should_skip(zoom_factors)`` returns the input
      UNCHANGED (same object — the oracle's skip is a full ``continue``
      before any cast or re-allocation). The skip branch is
      order-independent.
    * RESAMPLE, scipy CPU kernel (no longer the default — forced with
      ``HOLOSCAN_GPU_PRERESAMPLE=0``): ``preresample_channel_cpu`` per
      channel — oracle-identical by construction.
    * RESAMPLE, GPU mirror (default ON since 2026-08-27, Phase 11; force
      OFF with ``HOLOSCAN_GPU_PRERESAMPLE=0``): ``preresample_channel_gpu``
      per channel (C-contiguous), stacked on the GPU, ONE bulk D2H of the
      whole stack at the end (per-channel transfers would be acceptable at
      <=5 channels; the bulk form is strictly cheaper).

    ``order`` is the scipy interpolation order: default 3 = oracle/
    MONAI byte-locked semantics (total); 1 = TS 2.18 CLI parity
    (Phase 19, 19-03 — spec-driven via TaskSpec.resample_order, body=1).
    """
    arr = np.asarray(arr)
    was_3d = arr.ndim == 3
    if was_3d:
        channels = [arr]
    elif arr.ndim == 4:
        channels = [arr[c] for c in range(arr.shape[0])]
    else:
        raise ValueError(f"expected 3D (R,A,S) or 4D (C,R,A,S), got ndim={arr.ndim}")

    if preresample_should_skip(zoom_factors):
        return arr  # untouched — same object, dtype, and shape

    if gpu_preresample_enabled():
        chs_gpu = [cp.asarray(np.ascontiguousarray(ch)) for ch in channels]
        first = preresample_channel_gpu(chs_gpu[0], zoom_factors, order=order)
        out = cp.empty((len(chs_gpu),) + first.shape, dtype=cp.int32)
        out[0] = first
        for c in range(1, len(chs_gpu)):
            out[c] = preresample_channel_gpu(chs_gpu[c], zoom_factors, order=order)
        result = np.ascontiguousarray(out.get())
        return result[0] if was_3d else result

    result = np.stack([preresample_channel_cpu(ch, zoom_factors, order=order) for ch in channels], axis=0)
    return result[0] if was_3d else result
