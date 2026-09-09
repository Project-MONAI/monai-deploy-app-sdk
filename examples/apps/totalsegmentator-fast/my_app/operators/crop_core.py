# Copyright 2021-2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

"""crop_core.py — pure-numpy crop math for the `liver_segments` crop cascade
(Phase 18, plan 18-02). NO holoscan/torch/cupy imports (unit-testable
headless, same discipline as task_specs' stdlib-only rule).

Oracle: TotalSegmentator 2.18.0 (read-only venv /tmp/totalseg-venv)
  * `totalsegmentator/cropping.py` — get_bbox_from_mask / crop_to_bbox /
    crop_to_bbox_nifti / undo_crop. The arithmetic here is a 1:1 port of
    that source, re-stated on plain numpy arrays (the app never imports
    the TS package — 18-RESEARCH Anti-patterns).
  * `nnunet.py:479` — the working volume at the crop is the NATIVE
    input (no task-res resample before cropping).
  * `nnunet.py:493` — the empty-mask check (`mask.sum() == 0`) happens
    BEFORE the crop; the main model then NEVER runs.
  * `nnunet.py:512` — `crop_to_mask(native_vol, native_mask, addon,
    dtype=int32)` — crop at native resolution.
  * `nnunet.py:518-527` — the task-spacing resample happens AFTER the
    crop.
  * `python_api.py:362-411` — crop model 6.0 mm; addon forced to
    [20, 20, 20] (MM) when `crop_model is None`.

CROP-RESOLUTION OVERRIDE (recorded per 18-02 plan / 18-RESEARCH
§Critical Correction): the ROADMAP/REQUIREMENTS parenthetical claiming a
"task-resolution crop" is DISPROVEN (source + 2 live runs). This module
implements the native-resolution crop: addon (20,20,20) mm -> voxels by
TRUNCATED division (`.astype(int)`, never rounding) by the NATIVE
zooms; the 6mm crop-model labelmap is back-resampled to native (order 0)
before the mask is built; the task-res resample runs after the crop.
The ROADMAP text is corrected at phase closeout (orchestrator-owned).

Array convention: mask and volume arrive in the native DICOM array order
(D, H, W) — the SDK `Image.asnumpy()` order — with a 4x4
`nifti_affine_transform`. TS's nibabel (z, x, y) index structure maps
1:1 onto (D, H, W), so the ported index math is verbatim.

Bit-pinned by scripts/test_crop_core.py against a verbatim inline port
of TS cropping.py (truncation / bit-equality / clamp / empty /
round-trip / live-bbox checks).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "zooms_from_affine",
    "get_bbox_from_mask",
    "crop_to_mask",
    "undo_crop",
    "mask_from_labelmap_6mm",
    "resolve_crop_label_ids",
    "back_resample_6mm_labelmap",
    "select_crop_labels",
]

ArrayLike = Union[np.ndarray, "Any"]  # numpy on the CPU path; kept Any-typed


def zooms_from_affine(affine: np.ndarray) -> np.ndarray:
    """Per-array-axis voxel spacing from a 4x4 affine: column norms of
    ``affine[:3, :3]`` (sign-invariant — in-plane column signs flip with
    orientation). Matches nibabel ``header.get_zooms()[:3]`` for
    axis-aligned data, which is this corpus."""
    affine_np = np.asarray(affine, dtype=np.float64)
    if affine_np.shape != (4, 4):
        raise ValueError(f"expected a 4x4 affine, got shape {affine_np.shape}")
    return np.sqrt((affine_np[:3, :3] ** 2).sum(axis=0))


def get_bbox_from_mask(
    mask: np.ndarray, zooms: Sequence[float], addon_mm: Sequence[float]
) -> Tuple[Optional[list], np.ndarray]:
    """TS ``cropping.py::get_bbox_from_mask`` + the ``crop_to_mask`` addon
    conversion, ported verbatim onto a (D, H, W) array.

    * ``addon_vox = (addon_mm / zooms).astype(int)`` — TRUNCATED, never
      rounded (np.round would shift the crop margin by 1+ voxels/axis —
      18-RESEARCH Pitfall 2).
    * bbox from ``np.where(mask > 0)`` min/max ± addon_vox, clamped to
      ``[0, shape]`` per axis (TS "Avoid bbox to get out of image size").

    Args:
        mask: 3D array (D, H, W), any dtype; foreground = value > 0.
        zooms: per-array-axis spacing in mm, (D, H, W) order.
        addon_mm: addon in MM per axis (the crop spec: (20, 20, 20)).

    Returns:
        ``(bbox, addon_vox)`` — bbox = [[d0,d1],[h0,h1],[w0,w1]] half-open
        clamped intervals; EMPTY mask -> ``(None, addon_vox)`` sentinel so
        callers take the empty branch (TS performs the emptiness check at
        nnunet.py:493 BEFORE any crop — the app keeps that ordering).
    """
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(f"expected a 3D mask, got ndim={mask.ndim}")
    zooms_np = np.asarray(zooms, dtype=np.float64)
    addon_vox = (np.array(addon_mm, dtype=np.float64) / zooms_np).astype(int)  # mm to voxels — truncated

    if (mask > 0).sum() == 0:
        # TS prints a warning here, but in the TS pipeline this branch is
        # unreachable via crop_to_mask (nnunet.py:493 checks first). The
        # sentinel keeps the app's empty-path contract explicit.
        return None, addon_vox

    d, h, w = np.where(mask > 0)
    min_d = int(d.min()) - addon_vox[0]
    max_d = int(d.max()) + 1 + addon_vox[0]
    min_h = int(h.min()) - addon_vox[1]
    max_h = int(h.max()) + 1 + addon_vox[1]
    min_w = int(w.min()) - addon_vox[2]
    max_w = int(w.max()) + 1 + addon_vox[2]

    # Avoid bbox to get out of image size (verbatim TS clamp).
    s = mask.shape
    min_d = max(0, min_d)
    max_d = min(s[0], max_d)
    min_h = max(0, min_h)
    max_h = min(s[1], max_h)
    min_w = max(0, min_w)
    max_w = min(s[2], max_w)

    return [[int(min_d), int(max_d)], [int(min_h), int(max_h)], [int(min_w), int(max_w)]], addon_vox


def crop_to_mask(
    vol: np.ndarray,
    mask: np.ndarray,
    zooms: Sequence[float],
    affine: np.ndarray,
    addon_mm: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """TS ``cropping.py::crop_to_mask`` + ``crop_to_bbox_nifti`` ported
    onto arrays (nnunet.py:512 call site, dtype=int32 contract).

    Crops `vol` and `mask` with the SAME clamped bbox and updates the
    affine translation VERBATIM to the TS formula
    (``affine[:3,3] = dot(affine, [min_d, min_h, min_w, 1])[:3]`` — note
    the nibabel index order, preserved here for bit-parity).

    Args:
        vol: 3D volume (D, H, W) (int32 per the TS call site).
        mask: 3D binary/label mask, same shape as `vol`.
        zooms: per-array-axis spacing in mm, (D, H, W) order.
        affine: 4x4 native affine (NOT modified — a copy is returned in
            `bbox_info["affine"]`).
        addon_mm: addon in MM per axis.

    Returns:
        ``(cropped_vol, bbox_info)`` where `bbox_info` carries:
          "bbox"        — [[d0,d1],[h0,h1],[w0,w1]] half-open, clamped
          "addon_vox"   — the truncated mm->vox addon
          "shape"       — cropped shape
          "affine"      — the offset-updated affine (fresh 4x4 array)
          "affine_offset" — the updated translation column (3,)
          "cropped_mask"— the mask cropped with the same bbox
    Raises:
        ValueError: on an empty mask (callers must check first — the
            nnunet.py:493 ordering).
    """
    vol = np.asarray(vol)
    if vol.ndim != 3:
        raise ValueError(f"expected a 3D volume, got ndim={vol.ndim}")
    if np.asarray(mask).shape != vol.shape:
        raise ValueError(f"mask shape {np.asarray(mask).shape} != volume shape {vol.shape}")

    bbox, addon_vox = get_bbox_from_mask(mask, zooms, addon_mm)
    if bbox is None:
        raise ValueError(
            "crop_to_mask: empty mask — the emptiness check belongs upstream "
            "(TS nnunet.py:493); take the empty-canvas branch instead."
        )

    b = np.asarray(affine, dtype=np.float64)
    # TS crop_to_bbox (verbatim index structure: [d-slice, h-slice, w-slice]).
    cropped = vol[bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]]
    cropped_mask = np.asarray(mask)[bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]]

    # TS crop_to_bbox_nifti affine update (verbatim — copy, never mutate the
    # caller's affine).
    new_affine = np.copy(b)
    new_affine[:3, 3] = np.dot(b, np.array([bbox[0][0], bbox[1][0], bbox[2][0], 1]))[:3]

    bbox_info: Dict[str, Any] = {
        "bbox": [[int(v) for v in pair] for pair in bbox],
        "addon_vox": [int(v) for v in addon_vox],
        "shape": tuple(int(s) for s in cropped.shape),
        "affine": new_affine,
        "affine_offset": new_affine[:3, 3].copy(),
        "cropped_mask": cropped_mask,
    }
    return cropped, bbox_info


def undo_crop(canvas_shape: Sequence[int], cropped: np.ndarray, bbox_info: Dict[str, Any]) -> np.ndarray:
    """TS ``cropping.py::undo_crop`` ported: zero canvas at
    `canvas_shape`, paste `cropped` at the (clamped) bbox.

    Output dtype matches `cropped` (uint8 for segmentations).
    """
    cropped = np.asarray(cropped)
    bbox = bbox_info["bbox"]
    canvas = np.zeros(tuple(int(s) for s in canvas_shape), dtype=cropped.dtype)
    canvas[bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]] = cropped
    return canvas


def mask_from_labelmap_6mm(labelmap_6mm: np.ndarray, native_shape: Sequence[int], liver_id: int = 5) -> np.ndarray:
    """TS nnunet.py:758+ contract: the 6mm crop-model labelmap is
    BACK-RESAMPLED to the native shape with ORDER 0 (nearest) before the
    crop mask is built (``mask = (labelmap == liver_id)``, uint8).

    `labelmap_6mm` must already cover the FULL native grid in the native
    array order (D, H, W) — in the app this is PostResample(6mm)
    ``seg_argmax_dicom`` (zero-filled to `meta["shape_before_cropping"]`,
    the full 6mm grid, original DICOM orientation). The zoom factor per
    axis is `native_shape / labelmap_6mm.shape` (equivalent to TS's
    spacing-based change_spacing for axis-aligned grids).

    scipy ``ndimage.zoom`` order=0 is bit-exact for the binary mask
    (nearest-neighbor on exact 0/1 values), and TS itself uses nearest
    neighbor for this step.

    Args:
        labelmap_6mm: 3D uint8/int labelmap, full-coverage 6mm grid (D, H, W).
        native_shape: target (D, H, W).
        liver_id: the liver label in the crop model's local label table
            (resolved from the bundle dataset.json at operator setup —
            NOT a hard code in the operator; 5 is this bundle's value).

    Returns:
        uint8 (D, H, W) binary mask, values in {0, 1}.
    """
    labelmap_6mm = np.asarray(labelmap_6mm)
    if labelmap_6mm.ndim != 3:
        raise ValueError(f"expected a 3D labelmap, got ndim={labelmap_6mm.ndim}")
    native_shape = tuple(int(s) for s in native_shape)
    if labelmap_6mm.shape == native_shape:
        return (labelmap_6mm == int(liver_id)).astype(np.uint8)

    from scipy.ndimage import zoom as ndi_zoom  # local import: keeps module import pure-numpy

    factors = tuple(float(n) / float(c) for n, c in zip(native_shape, labelmap_6mm.shape))
    binary = (labelmap_6mm == int(liver_id)).astype(np.float64)
    out = ndi_zoom(binary, factors, order=0, mode="nearest")  # nearest: bit-exact for 0/1
    return (out >= 0.5).astype(np.uint8)


def resolve_crop_label_ids(label_table: Dict[str, int], names: Sequence[str]) -> List[int]:
    """Resolve crop label NAMES (from the task spec, e.g. `("liver",)` or the
    five lung-lobe names) to integer ids in the crop bundle's dataset.json
    label table. Case- and whitespace-insensitive on the names; raises
    ``KeyError`` naming the first unresolved entry and the table size.

    Names resolve in the given (spec) order, preserving any duplicates the
    caller passes. This is the data-driven replacement for the historical
    hard-coded single-label lookup — the source of the Phase 22-07 Class C
    bug, where the crop mask always resolved the liver id even for tasks
    whose TS-oracle crop set is the lung lobes.

    Args:
        label_table: the ``labels`` dict from the 6mm bundle dataset.json
            (name -> id; keys are strings, values may be int/str).
        names: ordered sequence of label names from the task spec.

    Returns:
        list of int ids in the same order as `names`.
    """
    table = {str(k).strip().lower(): int(v) for k, v in dict(label_table).items()}
    ids: List[int] = []
    for name in names:
        key = str(name).strip().lower()
        if key not in table:
            raise KeyError(
                f"crop label {name!r} not found in dataset.json label table "
                f"({len(table)} keys) — cannot build the crop mask"
            )
        ids.append(table[key])
    return ids


def back_resample_6mm_labelmap(labelmap_6mm: np.ndarray, native_shape: Sequence[int]) -> np.ndarray:
    """Back-resample the 6mm crop-model labelmap to the native shape with
    ORDER 0 (nearest) — the exact pipeline ``CropMaskOp.compute`` historically
    inlined: transpose (2,1,0) from the SAR/slice-major (z,y,x) layout to the
    (x,y,z) RAS frame, per-axis order-0 nearest zoom (factor native/6mm),
    round, uint8.

    Same-axis-order passthrough: if the input already matches `native_shape`
    it is returned unmodified (no zoom), bit-identical to the operator's
    `else` branch.

    Args:
        labelmap_6mm: 3D labelmap in (z, y, x) slice-major (SAR) order — the
            PostResample `seg_argmax_dicom` full 6mm grid.
        native_shape: target shape in the (x, y, z) RAS order, i.e.
            `meta["_pre_resample_spatial_shape"]`.

    Returns:
        uint8 labelmap in (x, y, z) order at `native_shape`.
    """
    lm = np.asarray(labelmap_6mm)
    if lm.ndim != 3:
        raise ValueError(f"expected a 3D labelmap, got ndim={lm.ndim}")
    native_shape = tuple(int(s) for s in native_shape)
    if lm.shape == native_shape:
        return lm.astype(np.uint8, copy=False)
    ras = np.ascontiguousarray(np.transpose(lm, (2, 1, 0)))  # (z,y,x) SAR -> (x,y,z) RAS
    factors = tuple(float(n) / float(c) for n, c in zip(native_shape, ras.shape))
    from scipy.ndimage import zoom as ndi_zoom  # local import: keeps module import pure-numpy

    return ndi_zoom(ras.astype(np.float64), factors, order=0, mode="nearest").round().astype(np.uint8)


def select_crop_labels(labelmap: np.ndarray, label_ids: Sequence[int]) -> np.ndarray:
    """Multi-label binary selection: ``np.isin(labelmap, label_ids)`` as a
    uint8 {0,1} mask. Bit-identical to the historical single-label
    ``(labelmap == id)`` comparison when `label_ids` holds one element.

    Mirrors TS python_api.py:396-409 (union over the crop config's labels:
    `crop_mask[seg == id] = 1` per label, then the mask is the union).

    Args:
        labelmap: the native-resolution labelmap (x, y, z) RAS frame.
        label_ids: int label ids to include in the crop mask (union).

    Returns:
        uint8 (x, y, z) binary mask, values in {0, 1}.
    """
    ids = [int(i) for i in (label_ids if isinstance(label_ids, (list, tuple, set)) else [label_ids])]
    return np.isin(np.asarray(labelmap), ids).astype(np.uint8)
