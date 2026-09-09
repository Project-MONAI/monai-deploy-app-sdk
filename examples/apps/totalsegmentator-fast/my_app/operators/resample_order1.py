# Copyright 2021-2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

"""resample_order1.py — TS-semantic order-1 volume resample (Phase 18,
plan 18-02) for the `liver_segments` crop cascade.

Oracle: TotalSegmentator 2.18.0 `resampling.py::change_spacing` (the app
never imports the TS package — 18-RESEARCH Anti-patterns), called by
`nnunet.py:518-527` (task-res resample AFTER the crop) and the crop-model
input path (native -> 6.0 mm) with `order=resampling_order` (CLI default 1)
and `dtype=np.int32`:

    data   = img.get_fdata()                       # float64 (exact for int inputs)
    zoom   = old_spacing / new_spacing             # per axis (coarser target -> < 1)
    out    = ndimage.zoom(data, zoom, mode="nearest", order=1)
    out    = out.astype(np.float32).astype(np.int32)   # f32 boundary gate + TRUNCATION toward zero
    # exact-equal spacing input -> returned WITHOUT resampling (identity)

This is DELIBERATELY SEPARATE from pre_resample_zoom.py (order 3,
oracle/MONAI-semantic, byte-locked for the `total` task — Phase 17
regression lock; 18-RESEARCH Pitfall 4: reusing it would diverge from TS
near 6mm boundaries). The scipy CPU path is the reference; the cupyx GPU
mirror follows the Phase 8.1 proven pattern: exact f64 widening ->
`cupyx.scipy.ndimage.zoom` (order=1, mode="nearest", grid_mode=False) ->
direct f64->int32 truncation, with a load-bearing f32 round-to-even
boundary gate inserted before the truncation (scipy-vs-cupyx f64 outputs
1-ulp apart at integer boundaries — see resample_order1_cpu docstring).
Byte-identity scipy-vs-cupyx on the cascade's real spacing pairs is proven by
scripts/test_resample_order1.py (51-check-style suite).

Array convention: 3D (D, H, W) per-array-axis — the native DICOM array
order; spacings are per-array-axis (D, H, W) too.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

__all__ = [
    "order1_resample_enabled",
    "resample_order1_cpu",
    "resample_order1_gpu",
    "resample_order1",
]


def order1_resample_enabled() -> bool:
    """Kernel-selection flag for the cascade order-1 resample.

    UNSET or exactly ``"1"`` -> GPU cupyx mirror (default); any other
    value (e.g. ``"0"``) forces the scipy CPU reference. Same
    strictly-``"1"``-only semantics as HOLOSCAN_GPU_PRERESAMPLE.
    """
    return os.environ.get("HOLOSCAN_CASCADE_RESAMPLE", "1") == "1"


def _check_inputs(vol: np.ndarray, in_spacing: Sequence[float], out_spacing: Sequence[float]) -> np.ndarray:
    vol = np.asarray(vol)
    if vol.ndim != 3:
        raise ValueError(f"expected a 3D volume, got ndim={vol.ndim}")
    if vol.dtype.kind not in "iu":
        raise TypeError(f"expected an integer volume (int16 CT), got {vol.dtype!r}")
    in_sp = np.asarray(in_spacing, dtype=np.float64)
    out_sp = np.asarray(out_spacing, dtype=np.float64)
    if in_sp.shape != (3,) or out_sp.shape != (3,):
        raise ValueError("spacings must be 3-tuples (per-array-axis, mm)")
    if (in_sp <= 0).any() or (out_sp <= 0).any():
        raise ValueError("spacings must be positive")
    return vol


def resample_order1_cpu(vol: np.ndarray, in_spacing: Sequence[float], out_spacing: Sequence[float]) -> np.ndarray:
    """CPU reference — TS change_spacing(order=1, dtype=int32) re-stated:

    exact f64 widen -> `scipy.ndimage.zoom(order=1, mode="nearest")`
    (internally f64) -> f32 cast (round-to-even boundary gate — see below)
    -> `.astype(int32)` (float->int TRUNCATION toward zero). Exact-equal
    spacings return the volume unchanged (TS early return), still cast to
    int32 for the contract.

    Load-bearing f32 boundary gate (proven by scripts/test_resample_order1.py):
    scipy's and cupyx's f64 order-1 outputs differ by 1 ulp on ~5% of voxels
    (accumulation order); at voxels landing on an integer boundary that flips
    the int32 truncation by 1 (measured 1/103968 on the 64199 native->6mm
    extreme-range case: scipy 2262.0 vs cupyx 2261.9999999999995). The f32
    round-to-even step gates both kernels to the same representative before
    truncation (exact for CT-range magnitudes < 2^24; same pattern as
    pre_resample_zoom, 08-03) — after the gate the two paths are byte-
    identical on every suite case. TS's own GPU resample is cucim (never
    scipy), so CPU/GPU byte-identity inside the app is the binding contract.

    Args:
        vol: 3D integer volume (D, H, W) (int16 CT).
        in_spacing: current per-array-axis spacing in mm.
        out_spacing: target per-array-axis spacing in mm.

    Returns:
        int32 3D volume at the target spacing.
    """
    vol = _check_inputs(vol, in_spacing, out_spacing)
    in_sp = np.asarray(in_spacing, dtype=np.float64)
    out_sp = np.asarray(out_spacing, dtype=np.float64)
    if np.array_equal(in_sp, out_sp):
        # TS change_spacing early return: "Input spacing is equal to new
        # spacing. Return image without resampling." (cast for the int32
        # contract — no value change)
        return vol.astype(np.int32)

    from scipy.ndimage import zoom as ndi_zoom  # local import keeps module import light

    zoom = in_sp / out_sp
    data = vol.astype(np.float64)  # exact widen — get_fdata equivalent
    out = ndi_zoom(data, zoom, order=1, mode="nearest")
    return out.astype(np.float32).astype(np.int32)  # f32 boundary gate + truncation


def resample_order1_gpu(vol, in_spacing: Sequence[float], out_spacing: Sequence[float]):  # vol: cupy.ndarray
    """GPU mirror of `resample_order1_cpu` — byte-identical (proven in
    scripts/test_resample_order1.py): exact f64 widen (element-wise cast is
    bit-exact for integer values) -> `cupyx.scipy.ndimage.zoom`
    (order=1, mode="nearest", grid_mode=False, f64 kernel) -> f32 cast
    (round-to-even boundary gate — REQUIRED: scipy-vs-cupyx f64 outputs
    differ by 1 ulp on ~5% of voxels, which flips the int32 truncation at
    integer boundaries; see the CPU docstring) -> int32 truncation.

    Args:
        vol: 3D CuPy integer volume (D, H, W).
        in_spacing / out_spacing: per-array-axis spacing in mm.

    Returns:
        CuPy int32 3D volume at the target spacing.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import zoom as cndi_zoom  # grid_mode=False mirror

    if not isinstance(vol, cp.ndarray):
        raise TypeError(f"expected a CuPy array, got {type(vol)!r}")
    if vol.ndim != 3:
        raise ValueError(f"expected a 3D volume, got ndim={vol.ndim}")
    if vol.dtype.kind not in "iu":
        raise TypeError(f"expected an integer volume, got {vol.dtype!r}")
    in_sp = np.asarray(in_spacing, dtype=np.float64)
    out_sp = np.asarray(out_spacing, dtype=np.float64)
    if in_sp.shape != (3,) or out_sp.shape != (3,):
        raise ValueError("spacings must be 3-tuples (per-array-axis, mm)")
    if (in_sp <= 0).any() or (out_sp <= 0).any():
        raise ValueError("spacings must be positive")
    if not vol.flags.c_contiguous:
        vol = cp.ascontiguousarray(vol)

    if np.array_equal(in_sp, out_sp):
        return vol.astype(cp.int32)

    zoom = [float(v) for v in in_sp / out_sp]
    data = vol.astype(cp.float64)  # exact widen
    out = cndi_zoom(data, zoom, order=1, mode="nearest", grid_mode=False)
    return out.astype(cp.float32).astype(cp.int32)  # f32 boundary gate + truncation


def resample_order1(vol: np.ndarray, in_spacing: Sequence[float], out_spacing: Sequence[float]) -> np.ndarray:
    """Dispatcher: GPU cupyx mirror by default (HOLOSCAN_CASCADE_RESAMPLE
    strictly ``"1"``/unset); any other flag value forces the scipy CPU
    reference. Input must be a host numpy 3D integer volume; the GPU path
    round-trips through one H2D/D2H. Both paths are byte-identical (unit
    pinned)."""
    vol = _check_inputs(vol, in_spacing, out_spacing)
    if order1_resample_enabled():
        import cupy as cp

        out = resample_order1_gpu(cp.asarray(np.ascontiguousarray(vol)), in_spacing, out_spacing)
        return np.ascontiguousarray(out.get())
    return resample_order1_cpu(vol, in_spacing, out_spacing)
