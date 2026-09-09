# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure, device-agnostic SR volume-metrics core (Phase 13, WS-1).

This module replaces the legacy CPU per-label loop in
``my_app/segmentation_metrics_operator.py`` with a single-pass
``numpy.bincount`` formulation that is **bit-identical** to the oracle's
``_compute_volume_or_area`` for the gated ``vol`` values.

Design notes (see ``.planning/phases/12-optimization-research/12-RESEARCH.md``
Workstream 1):

* The SR gate (``p10_gate.py`` check 4) parses values to ``float`` and only
  consumes ``{label}.vol`` (the SR writer is constructed with
  ``included_fields=["vol"]``). Intensity stats and connected components are
  non-gated, so they are intentionally dropped here (the scipy
  ``ndimage.label`` x 117 pass is the dominant legacy CPU cost being removed).
* The per-label ``vol`` math mirrors the oracle EXACTLY, including float
  evaluation order, so a single-pass ``bincount`` yields bit-identical
  ``float64`` volumes (not merely ``allclose``):

      vpv_mm3 = float(spacing[0]) * float(spacing[1]) * float(spacing[2])
      volumes = (counts.astype(np.float64) * vpv_mm3) / 1000.0

  Do NOT compute ``vpv_mm3 / 1000.0`` first and then multiply -- reordering the
  float ops changes the last ULP for some labels. ``spacing`` must be sourced
  from the same place the legacy op used (the input-scan Image metadata) so the
  ``float`` inputs are identical.

* ``slice.range`` / ``num.slices`` are integer/None values derived from a
  per-slice presence matrix, so they match the oracle exactly by construction
  (no thresholds, no fp comparisons). The binding gate constraint is the *exact
  zero-row set*, which the integer count pass preserves exactly.

The module is pure: no holoscan import, no file I/O, no logging side effects,
and no hard torch dependency (torch is only needed to accept a torch tensor
input; numpy input works standalone). It is unit-testable headless on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # torch is optional at import time so the module stays headless-importable
    import torch
except Exception:  # pragma: no cover - torch is expected in the runtime venv
    torch = None  # type: ignore[assignment]

__all__ = [
    "VolumeMetricsResult",
    "compute_volume_metrics",
    "compute_slice_stats",
]

# Per-label metric keys. These MUST match the legacy
# ``segmentation_metrics_operator.py`` output dict exactly (the SR writer and
# the p10 SR-parity gate depend on the ``"{label}.vol"`` concept names).
KEY_VOL = "vol"
KEY_NUM_SLICES = "num.slices"
KEY_SLICE_RANGE = "slice.range"
KEY_PIXEL_COUNT = "pixel.count"


@dataclass
class VolumeMetricsResult:
    """Single-pass volume-metrics output, indexed by raw label value.

    All array fields have length ``num_labels + 1`` and are indexed by the raw
    label value (``0`` == background, ``1..num_labels`` == organs). The legacy
    op skips background, so consumers iterate ``1..num_labels``.
    """

    volumes: np.ndarray  # (num_labels+1,) float64, mL
    label_counts: np.ndarray  # (num_labels+1,) int64
    num_slices: np.ndarray  # (num_labels+1,) int64
    slice_range: List[Optional[Tuple[int, int]]]  # (num_labels+1,) (first,last) or None
    nonzero_labels: List[int] = field(default_factory=list)  # present label values (excl. bg)


def _to_numpy_uint8(labels: Union["np.ndarray", "torch.Tensor"]) -> "np.ndarray":
    """Coerce a 3-D labelmap to a C-contiguous ``uint8`` numpy array.

    Accepts a numpy array or a torch tensor (the merge op emits a 3-D uint8 CPU
    torch tensor under ``seg_merged``). Defensively drops a leading singleton
    batch dimension if a 4-D ``(1, D, H, W)`` tensor slips through.
    """
    if torch is not None and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    arr = np.asarray(labels)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected a 3-D (D,H,W) labelmap or a (1,D,H,W) batched one, got shape {arr.shape}")
        arr = arr[0]  # squeeze(0)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-D (D,H,W) labelmap, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _vpv_mm3(spacing: Sequence[float]) -> float:
    """Voxel volume in mm^3, computed in the oracle's exact float order."""
    if len(spacing) < 3:
        raise ValueError(f"spacing must have 3 elements (got {len(spacing)}): {spacing!r}")
    return float(spacing[0]) * float(spacing[1]) * float(spacing[2])


def compute_volume_metrics(
    labels: Union["np.ndarray", "torch.Tensor"],
    spacing: Sequence[float],
    num_labels: int = 117,
) -> VolumeMetricsResult:
    """Single-pass volume + slice metrics for a 3-D integer labelmap.

    Args:
        labels: ``(D, H, W)`` integer labelmap (values ``0..num_labels``), as a
            numpy array or a torch tensor (CPU or GPU). Label ``0`` is
            background and is excluded from :attr:`VolumeMetricsResult.nonzero_labels`.
        spacing: physical voxel spacing ``(sx, sy, sz)`` in mm, in the SAME
            source/order the legacy op used (input-scan Image metadata).
        num_labels: number of organ labels (background excluded), default 117.

    Returns:
        A :class:`VolumeMetricsResult` whose ``volumes``/``label_counts`` are
        bit-/value-identical to the oracle's per-label loop.
    """
    seg = _to_numpy_uint8(labels)
    n = num_labels + 1  # bins for label values 0..num_labels
    vpv = _vpv_mm3(spacing)

    flat = seg.ravel()
    counts = np.bincount(flat, minlength=n)[:n].astype(np.int64, copy=False)

    # BIT-IDENTICAL volume math (see module docstring for why order matters).
    volumes = (counts.astype(np.float64) * vpv) / 1000.0

    # Per-slice presence matrix via one bincount over (label, slice) pairs.
    # d = depth (slice) axis. slice_idx[i] = the slice of flattened voxel i.
    # Encoding: v = label*d + slice; reshape(n, d) in C order maps
    # flat1d[i*d + j] -> [i, j], so row l / col s counts label l in slice s.
    d, h, w = seg.shape
    slice_idx = np.arange(flat.size, dtype=np.int64) // (h * w)
    present = np.bincount(flat.astype(np.int64) * d + slice_idx, minlength=n * d).reshape(n, d) > 0
    num_slices = present.sum(axis=1).astype(np.int64, copy=False)
    first_idx = present.argmax(axis=1)  # first True, or 0 if absent
    last_idx = (d - 1) - present[:, ::-1].argmax(axis=1)  # last True, or d-1 if absent
    slice_range: List[Optional[Tuple[int, int]]] = [
        (int(first_idx[l]), int(last_idx[l])) if int(num_slices[l]) > 0 else None for l in range(n)
    ]
    nonzero_labels = [int(l) for l in range(1, n) if int(counts[l]) > 0]

    return VolumeMetricsResult(
        volumes=volumes,
        label_counts=counts,
        num_slices=num_slices,
        slice_range=slice_range,
        nonzero_labels=nonzero_labels,
    )


def compute_slice_stats(
    labels: Union["np.ndarray", "torch.Tensor"],
    spacing: Sequence[float],
    num_labels: int = 117,
) -> dict:
    """Per-label metric dict keyed by raw label value (``1..num_labels``).

    Returns a dict ``{label_value: {KEY_VOL, KEY_NUM_SLICES, KEY_SLICE_RANGE,
    KEY_PIXEL_COUNT}}`` using the EXACT inner key names the legacy op emits, so
    the thin operator can re-key by organ name and emit a drop-in
    ``metrics_dict``. Background (label 0) is intentionally omitted, matching
    the legacy op (its ``labels_dict`` excludes ``"background"``).
    """
    res = compute_volume_metrics(labels, spacing, num_labels)
    out: dict = {}
    for l in range(1, num_labels + 1):
        out[l] = {
            KEY_VOL: float(res.volumes[l]),
            KEY_NUM_SLICES: int(res.num_slices[l]),
            KEY_SLICE_RANGE: res.slice_range[l],
            KEY_PIXEL_COUNT: int(res.label_counts[l]),
        }
    return out
