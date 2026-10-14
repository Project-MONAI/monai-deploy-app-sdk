# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shape-aware per-study VRAM estimation.

A flat plans.json MEDIAN study size as the shape term is identical for
every study — so an oversized study (e.g. study-4, ~420 M resampled
fullres voxels) sails through admission and OOMs the daemon mid-pipeline.
An OOM-survival probe (2026-09-10) showed that failure mode bricks the
service: typed RMM bad_alloc at the
slidewindow, the RPC hangs to the client deadline, and the gRPC server stops
answering. The only reliable protection is rejecting studies that cannot fit
BEFORE the pipeline starts.

This module provides the admission-side estimator:

* :func:`study_geometry` — header-only DICOM walk (``stop_before_pixels``;
  no pixel data, no torch, no CUDA) yielding native shape + spacing.
* :func:`resampled_voxels_per_config` — the nnUNet resample target shape per
  config, replicating ``preprocess_operator._compute_new_shape`` exactly
  (``round(old_spacing / new_spacing * dim)`` per axis, with the same
  transpose_forward + reversed-spacing convention). Uses the FULL
  (pre-crop) volume: crop is always a subset, so this is a safe upper bound.
* :func:`estimate_study_peak_mib` — empirical peak working set in MiB from
  the calibrated model in ``study_vram_calibration.json`` (quadratic in
  Vsum = lowres + fullres resampled voxel count; fitted to the measured
  floor increments of three ascending soak studies).

The estimate is compared at admission against the device headroom
(device total - startup floor - slack, see ``grpc_ingress_operator``); the
decision is deliberately INDEPENDENT of current free VRAM because the RMM
pool is grow-only: after a large study, free VRAM is low yet smaller studies
still fit from pool reuse (measured across ascending studies; OOM
  probe). Free VRAM is logged
informationally only.

The module imports NO torch/cupy/cuda — it runs on the gRPC handler thread
(the handler stays ML-free).
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:  # package-style import (my_app.*)
    from my_app.config import load_preprocess_params
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from config import load_preprocess_params

__all__ = [
    "StudyShapeError",
    "nvml_device_index",
    "study_geometry",
    "resampled_voxels_per_config",
    "load_calibration",
    "estimate_study_peak_mib",
]


class StudyShapeError(RuntimeError):
    """The study's geometry cannot be derived from its DICOM headers.

    Raised on missing/unreadable headers, missing spacing, or empty series.
    The ingress catches this and falls back to the flat (plans-median)
    estimate with a warning — an estimation failure must never hard-fail an
    admissible study, and never let the pipeline start blind.
    """


def nvml_device_index() -> int:
    """Physical NVML index of the device this daemon should budget against.

    NVML indices are PHYSICAL and ignore CUDA_VISIBLE_DEVICES (observed on a
    multi-GPU host: NVML index 0 read the idle GPU 0). The daemon is
    launched with CUDA_VISIBLE_DEVICES=<gpu>, so the first visible-device
    entry is the physical index we want. Unset/empty -> 0 (production
    single-GPU container: the only device). UUID or MIG-form values -> 0
    (index form is all this deployment uses).
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = raw.split(",")[0].strip()
    try:
        idx = int(first)
        return idx if idx >= 0 else 0
    except ValueError:
        return 0


def _read_header(path: Path):
    import pydicom

    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True)
    except Exception as exc:  # noqa: BLE001 - fail loud: unreadable payloads must not silently undercount slices
        raise StudyShapeError(f"unreadable DICOM file {path.name}: {type(exc).__name__}: {exc}") from exc


def study_geometry(dicom_dir: Path) -> Dict[str, Any]:
    """Header-only study geometry.

    Returns ``{"n_slices", "rows", "cols", "sx", "sy", "sz"}`` where
    (n_slices, rows, cols) is the native (D, H, W) array order the
    DICOMSeriesToVolumeOperator produces, and (sx, sy, sz) the physical
    spacing in (x, y, z) LPS axes: x = PixelSpacing[1] (column spacing),
    y = PixelSpacing[0] (row spacing), z = median |delta ImagePositionPatient
    z| over sorted slice positions (SliceThickness fallback).

    Raises :class:`StudyShapeError` when the series is empty or spacing
    cannot be derived.
    """
    p = Path(dicom_dir)
    files = sorted(f for f in p.rglob("*") if f.suffix.lower() in (".dcm", ".dicom"))
    if not files:
        raise StudyShapeError(f"no DICOM instances under {p}")

    headers = [_read_header(f) for f in files]
    # The first file alphabetically may be a header-only / non-image instance
    # (no image dimensions); find the first real image instance. Tag-based
    # access for the dimension elements: pydicom >= 3.0 renamed the
    # ImageRows/ImageColumns keyword aliases (ds.ImageRows raises
    # AttributeError), while (0028,0010)/(0028,0011) tag access is stable.
    tag_rows = (0x0028, 0x0010)
    tag_cols = (0x0028, 0x0011)
    tag_pspacing = (0x0028, 0x0030)
    tag_ipp = (0x0020, 0x0032)
    tag_st = (0x0018, 0x0050)

    first = None
    zs: List[float] = []
    thicknesses: List[float] = []
    n_image_instances = 0
    for ds in headers:
        try:
            int(ds[tag_rows].value)
        except Exception:  # noqa: BLE001 - not an image instance
            continue
        n_image_instances += 1
        if first is None:
            try:
                float(ds[tag_pspacing].value[0])
            except Exception:  # noqa: BLE001
                pass  # image instance without PixelSpacing; keep looking for first
            else:
                first = ds
        try:
            zs.append(float(ds[tag_ipp].value[2]))
        except Exception:  # noqa: BLE001
            pass
        try:
            thicknesses.append(float(ds[tag_st].value))
        except Exception:  # noqa: BLE001
            pass
    if first is None:
        raise StudyShapeError(f"no image instance with rows/PixelSpacing under {p}")
    rows = int(first[tag_rows].value)
    cols = int(first[tag_cols].value)
    pixel_spacing = [float(v) for v in first[tag_pspacing].value]  # [row, col]

    sz: Optional[float] = None
    if len(zs) >= 2:
        z_sorted = sorted(zs)
        dz = [z_sorted[i + 1] - z_sorted[i] for i in range(len(zs) - 1)]
        dz = [d for d in dz if d > 0]
        if dz:
            sz = statistics.median(dz)
    if sz is None and thicknesses:
        sz = statistics.median(t for t in thicknesses if t > 0)
    if sz is None or sz <= 0:
        raise StudyShapeError("cannot derive slice spacing (no ImagePositionPatient z deltas or SliceThickness)")

    return {
        "n_slices": n_image_instances,
        "rows": rows,
        "cols": cols,
        "sx": pixel_spacing[1],
        "sy": pixel_spacing[0],
        "sz": float(sz),
    }


def resampled_voxels_per_config(
    geom: Mapping[str, Any], model_path: Any, run_list: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    """Per-config resampled target shape + voxel count (pre-crop upper bound).

    Replicates ``preprocess_operator`` exactly: native (D,H,W) -> channel-first
    -> ``transpose_forward`` -> ``_compute_new_shape(cropped_shape,
    original_spacing, params.spacing)`` with ``original_spacing =
    [reversed(spacing_xyz)][i] for i in transpose_forward``. The crop step
    (nonzero-mask bbox) is skipped deliberately: it only SHRINKS the volume,
    so the full-volume target is a safe upper bound for admission.
    """
    native = (int(geom["n_slices"]), int(geom["rows"]), int(geom["cols"]))  # (D, H, W)
    reversed_spacing = (float(geom["sz"]), float(geom["sy"]), float(geom["sx"]))  # (z, y, x)
    out: Dict[str, Dict[str, Any]] = {}
    for cfg in run_list:
        params = load_preprocess_params(model_path, cfg)
        tf = list(params.transpose_forward)
        vol_t_shape = tuple(native[i] for i in tf)
        orig_spacing = tuple(reversed_spacing[i] for i in tf)
        target = [float(s) for s in params.spacing]
        new_shape = tuple(int(round(o / t * d)) for o, t, d in zip(orig_spacing, target, vol_t_shape))
        voxels = 1
        for s in new_shape:
            voxels *= s
        out[str(cfg)] = {"new_shape": new_shape, "voxels": int(voxels)}
    return out


def _calibration_path() -> Path:
    override = os.environ.get("MAP_VRAM_CALIBRATION")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "study_vram_calibration.json"


def load_calibration(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the committed calibration (a/b/c coefficients + provenance)."""
    p = Path(path) if path is not None else _calibration_path()
    data = json.loads(p.read_text())
    for key in ("a", "b", "c"):
        float(data[key])  # validate presence/type
    return data


def vsum_mvox(per_config: Mapping[str, Mapping[str, Any]]) -> float:
    """Vsum in M-voxels: lowres + fullres resampled voxel counts.

    The cascading config is EXCLUDED: it resamples to the fullres space and
    runs after lowres completes (measured 32 s-8 min gap in the study
    logs);
    the quadratic model was fitted on exactly this Vsum definition, so the
    peak measurement already absorbs the (small) fullres/cascade overlap.
    """
    total = 0.0
    for cfg, info in per_config.items():
        name = str(cfg).lower()
        if "lowres" in name:
            total += float(info["voxels"])
        elif "fullres" in name and "cascade" not in name:
            total += float(info["voxels"])
    return total / 1e6


def estimate_study_peak_mib(
    dicom_dir: Path, model_path: Any, run_list: Sequence[str], calibration: Optional[Mapping[str, Any]] = None
) -> Tuple[float, Dict[str, Any]]:
    """Estimate the study's peak GPU working set (MiB above the startup floor).

    Model: ``peak_mib = a * vsum^2 + b * vsum + c`` with ``vsum`` in
    M-voxels (see :func:`vsum_mvox`) — the quadratic interpolates the three
    measured soak floor increments exactly and is convex (conservative)
    beyond them.

    Returns ``(peak_mib, detail)``; detail carries the per-config shapes for
    the admission log line. Raises :class:`StudyShapeError` on unreadable
    geometry (ingress falls back to the flat estimate).
    """
    geom = study_geometry(dicom_dir)
    per_config = resampled_voxels_per_config(geom, model_path, run_list)
    if not per_config:
        raise StudyShapeError(f"empty run_list for {dicom_dir}")
    cal = dict(calibration) if calibration is not None else load_calibration()
    vsum = vsum_mvox(per_config)
    peak = float(cal["a"]) * vsum * vsum + float(cal["b"]) * vsum + float(cal["c"])
    detail = {
        "geom": {k: geom[k] for k in ("n_slices", "rows", "cols", "sx", "sy", "sz")},
        "per_config": per_config,
        "vsum_mvox": round(vsum, 3),
        "model": f"peak_mib = {cal['a']}v^2 + {cal['b']}v + {cal['c']} (v in M-vox)",
    }
    return peak, detail
