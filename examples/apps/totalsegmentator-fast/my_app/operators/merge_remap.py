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

"""merge_remap.py — oracle-exact 5-part merge + single post-merge back-resample + SAR→DHW flip (Phase 9, plan 09-01).

PURE module: numpy + scipy only. NO holoscan, NO torch, NO GPU — it must
import headless (unit-tested in scripts/test_merge_remap.py without a GPU).
Its only my_app dependency is ``config`` (headless-clean: json/os only).
Do NOT import from merge_operator.py (it pulls torch + monai.deploy.core).

Oracle anchors re-stated for MAP self-containment (the oracle is NOT
imported by app code or app tests — pinned SHA 7676d95,
``ct-totalsegmentator-map/app_total/nnunet_seg_operator.py``):

MERGE (``_run_multi_model_inference`` ~:672-748, offsets :68-74):

    _MODEL_PARTS = [organs(0), vertebrae(24), cardiac(50), muscles(68), ribs(91)]
    if combined_seg is None:
        combined_seg = np.zeros(part_seg.shape, dtype=np.uint8)
    for local_label in range(1, int(part_seg.max()) + 1):
        combined_seg[part_seg == local_label] = local_label + label_offset

* Fixed list order organs→vertebrae→cardiac→muscles→ribs; LATER PARTS
  OVERWRITE EARLIER at any overlapping voxel (later-part-wins).
* ONLY voxels with local label >= 1 are ever written — background (0) is
  NEVER shifted by a later part's offset. The vectorized form
  ``mask = part_seg > 0; combined[mask] = part_seg[mask] + offset`` is
  mathematically identical (both touch exactly the non-zero pixels, same
  values) and is what this module uses.
* Empty part (max == 0) -> ``range(1, 1)`` empty -> legal no-op.

BACK-RESAMPLE (``ResampleToOriginalSpacingd`` :154-222 — runs ONCE on the
MERGED seg, on CPU, at original spacing):

    orig_shape_ras = input_img.meta["_pre_resample_spatial_shape"].tolist()  # RAS
    target_shape_sar = list(reversed(orig_shape_ras))                        # SAR
    if current_spatial == [int(s) for s in target_shape_sar]: continue       # skip
    zoom_factors = [target_shape_sar[i] / current_spatial[i] for i in range(3)]
    resampled = ndimage_zoom(seg_np.astype(np.float64), zoom_factors,
                             order=0, mode="nearest").astype(uint8)

* ONE ``scipy.ndimage.zoom`` call, order=0 (nearest), mode="nearest",
  fp64 widen (exact for uint8), uint8 back-cast. Shape rounding property:
  out_dim = round(f*n) lands exactly on the target shape (asserted).

ORIENTATION (``compute_impl`` ~:785-792): the merged array is SAR from
argmax through back-resample; exactly ONE ``np.flip(axis=[1, 2])`` → DHW
for the SEG writer (the pre-flip SAR array is the metrics path).

P10 (44238 gate) writer-contract clarification — TWO distinct flips, do
not conflate them:

* ``flip_sar_to_writer`` (axis [1,2]) is the ORACLE-EXACT SEG-writer
  INPUT. The highdicom-based writer assigns each frame its IOP from the
  SOURCE CT series' instance order (d=0 = first series instance = most
  superior) and then position-sorts the frames (``plane_sort_index`` in
  ``highdicom.image.Image._init_multiframe_image`` — verified v0.28.1).
  Feeding it the all-three-axes flip D-mirrors the physical content
  (inferior slice parked at the superior IOP) — the P10 smoke 44238
  failure mode: decode(app_dcm) == flipD(seg_total_dhw.npy) at 100%.
* ``flip_sar_to_dhw`` (axis [0,1,2]) is the SPATIALLY-CORRECT DHW
  labelmap — the decoded SEG FRAME order of the (baseline) dcm, i.e. the
  array the P9 centroid verification and the P10 gate compare npys
  against. It is ``flipD`` of the writer input.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
from scipy.ndimage import zoom as ndimage_zoom

try:  # package-style import (my_app.*)
    from my_app.config import EXPECTED_MAX_LOCAL_LABEL, MODEL_PARTS
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from config import EXPECTED_MAX_LOCAL_LABEL, MODEL_PARTS

__all__ = ["merge_remap", "back_resample_merged", "flip_sar_to_dhw", "flip_sar_to_writer"]

# Unified label space: 5 parts x offsets 0/24/50/68/91 -> max 91+26 = 117.
_MAX_UNIFIED_LABEL = 117


def merge_remap(
    parts: Sequence[Tuple[str, np.ndarray]],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Oracle-exact 5-part merge (later-part-wins, non-background only).

    Args:
        parts: list of ``(name, seg)`` tuples, len 5, in EXACTLY the oracle
            ``MODEL_PARTS`` order (organs, vertebrae, cardiac, muscles,
            ribs). Each seg: 3D uint8, model-resolution (SAR), same shape.

    Returns:
        ``(combined, info)`` — combined is the unified uint8 SAR map
        (labels 0-117); info carries per-part stats, label list, and the
        invariant check counts (all must be 0).

    Raises:
        ValueError: part names/order != MODEL_PARTS (arrival-order bug
            killer — fires BEFORE any math); wrong dtype/ndim/shape/empty;
            per-part max local label > its model's head count; unified
            max > 117; per-part background-shift invariant violation
            (a background voxel of part p showing any value in
            (offset, offset + cap] — only a ``where(seg>0, seg+o, o)``-style
            background shift can produce that).
    """
    oracle_names = [p["name"] for p in MODEL_PARTS]
    got_names = [n for n, _ in parts]

    # GUARD FIRST (before any math): order-sensitive name match. This is
    # the arrival-order / half-wired-subset bug killer (Pitfall 3/5).
    if got_names != oracle_names:
        raise ValueError(
            f"merge_remap parts order {got_names} != oracle order {oracle_names} "
            f"(fixed _MODEL_PARTS order is required; a misordered or partial "
            f"part list is a wiring bug, not data)"
        )

    segs = [np.asarray(seg) for _, seg in parts]
    for (name, _), seg in zip(parts, segs):
        if seg.dtype != np.uint8:
            raise ValueError(f"merge_remap part {name!r}: expected uint8 (argmax path is " f"uint8), got {seg.dtype}")
        if seg.ndim != 3:
            raise ValueError(f"merge_remap part {name!r}: expected 3D, got shape {seg.shape}")
        if seg.size == 0:
            raise ValueError(f"merge_remap part {name!r}: received an empty segmentation")
    ref_shape = segs[0].shape
    for (name, _), seg in zip(parts, segs):
        if seg.shape != ref_shape:
            raise ValueError(
                f"merge_remap shape mismatch (wiring bug): part {name!r} "
                f"has shape {seg.shape}, expected {ref_shape}"
            )

    combined = np.zeros(ref_shape, dtype=np.uint8)

    # Fixed MODEL_PARTS order, later-part-wins, non-background-only writes
    # (the vectorized form of the oracle's local-label loop — identical).
    per_part_bg_voxels_in_range: Dict[str, int] = {}
    part_stats: Dict[str, Dict[str, int]] = {}
    for (name, _), seg in zip(parts, segs):
        part = next(p for p in MODEL_PARTS if p["name"] == name)
        offset = int(part["label_offset"])
        cap = EXPECTED_MAX_LOCAL_LABEL[name]

        max_local = int(seg.max())
        if max_local > cap:
            raise ValueError(
                f"Part {name!r} produced max local label {max_local} but its model "
                f"only has {cap} foreground heads (labels 1-{cap})."
            )

        mask = seg > 0
        combined[mask] = seg[mask] + offset

        # SOUND background-shift invariant for EVERY part (range check —
        # the old `combined == o` point-check was unsound: the
        # immediately-preceding part reaches exactly label 50/68/91). A
        # background voxel of part p showing ANY value in (o, o+cap] can
        # only come from a background shift: no earlier part can produce
        # a value above its own range and no later part has written yet.
        bg_violations = int(((seg == 0) & (combined > offset) & (combined <= offset + cap)).sum())
        per_part_bg_voxels_in_range[name] = bg_violations
        if bg_violations != 0:
            raise ValueError(
                f"Part {name!r}: {bg_violations} background voxels show a value in "
                f"({offset}, {offset + cap}] of the merged map — background-shift "
                f"bug (e.g. np.where(seg>0, seg+{offset}, {offset})-style offset on "
                f"background). A correct later-part-wins merge must keep 0."
            )

        part_stats[name] = {"max_local": max_local, "n_fg": int((seg > 0).sum())}

    max_label = int(combined.max())
    if max_label > _MAX_UNIFIED_LABEL:
        raise ValueError(
            f"merge_remap unified max label {max_label} exceeds {_MAX_UNIFIED_LABEL} "
            f"(5 parts, offsets 0/24/50/68/91, max heads 24/26/18/23/26)."
        )

    labels = np.unique(combined).tolist()
    info = {
        "parts": part_stats,
        "labels": labels,
        "labels_log": (labels[:50] + [f"... ({len(labels)} unique)"]) if len(labels) > 50 else labels,
        "max_label": max_label,
        "shape": list(ref_shape),
        "invariants": {
            "bg_shift_violations": 0,
            "per_part_bg_voxels_in_range": per_part_bg_voxels_in_range,
            "max_le_117": True,
        },
    }
    return combined, info


def back_resample_merged(
    merged_sar: np.ndarray,
    orig_shape_ras: Sequence[int],
    skipped: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Oracle-exact single post-merge back-resample (SAR, order-0 nearest).

    Args:
        merged_sar: the MERGED uint8 seg in SAR (nnUNet post-transpose)
            model space (one array — NOT per-part).
        orig_shape_ras: ``preprocessed_meta["_pre_resample_spatial_shape"]``
            (RAS spatial shape of the original study, e.g. [512, 512, 42]).
        skipped: ``preprocessed_meta["pre_resample_skipped"]`` — True for
            native-1.5mm studies (oracle ``continue`` branch).

    Returns:
        ``(result, info)`` — the back-resampled uint8 SAR seg at original
        spacing; info: ``{"skipped", "zoom_factors", "target_sar"}``.
        The skip branch returns the SAME object (no copy, no zoom call).
    """
    target_sar = [int(x) for x in reversed(orig_shape_ras)]

    # Skip branch (oracle :191 ``continue``): native-1.5mm study or the
    # model shape already IS the target shape -> identity, no zoom.
    if skipped or list(np.asarray(merged_sar).shape) == target_sar:
        return merged_sar, {"skipped": True, "zoom_factors": None, "target_sar": target_sar}

    factors = [target_sar[i] / merged_sar.shape[i] for i in range(3)]
    out = ndimage_zoom(merged_sar.astype(np.float64), factors, order=0, mode="nearest").astype(np.uint8)
    if out.shape != tuple(target_sar):
        raise RuntimeError(
            f"back_resample_merged: zoom produced shape {out.shape}, expected "
            f"target_sar {tuple(target_sar)} (factors={factors} — the round(f*n) "
            f"property must land exactly on the original shape)"
        )
    return out, {"skipped": False, "zoom_factors": factors, "target_sar": target_sar}


def flip_sar_to_writer(seg_sar: np.ndarray) -> np.ndarray:
    """The EXACT oracle SEG-writer input (oracle compute_impl ~:785-792 /
    nnunet_seg_operator.py:633): in-plane flip ONLY, axis [1,2] —
    ``[S,A,R] -> [S,P,L]``. Axis 0 (slice) is deliberately NOT touched.

    P10 (44238 gate) root cause — VERIFIED mechanism (highdicom 0.28.1,
    image.Image._prepare_spatial_metadata): the SDK's DICOMSeries hands
    the writer the source instances in Z-ASCENDING order (inferior first;
    on 44238 z runs -309.4 -> -227.4). highdicom pairs input plane i with
    source instance i, projects each plane onto the volume normal, then
    np.unique(origin_distances, return_index=True) re-orders frames by
    ASCENDING distance == DESCENDING z (superior first). Net effect for
    this corpus: the writer D-flips its input, so written frames are
    flipD(pixel_array) with a superior-first IOP grid (frame 0 IOP z is
    the highest z). Verified offline on 44238: writer(flip12(M)) decodes
    to the baseline at 99.9911%, writer(flip012(M)) decodes D-mirrored
    (73.36%) — the pre-fix smoke failure (inferior slice parked at the
    superior IOP). The oracle passes exactly this function's output
    (in-plane flip only); it is the byte-for-byte writer contract.
    """
    return np.ascontiguousarray(np.flip(np.asarray(seg_sar), axis=(1, 2)).astype(np.uint8, copy=False))


def flip_sar_to_dhw(seg_sar: np.ndarray) -> np.ndarray:
    """The SINGLE SAR→DHW flip (oracle compute_impl ~:785-792).

    P9 (09-04) fix: ``np.flip(axis=[0, 1, 2])`` — flip ALL THREE axes, not
    just in-plane. Axis 0 (S): the app's ascending slice order (SDK series
    sort, d=0 = inferior) is opposite to the oracle's SEG frame order
    (frame 0 = source instance 1 = most superior), so the slice axis must
    be flipped too. Axes 1,2 (A,R): A->P and R->L as before. Centroid
    verification on 44238 (label 21): oracle d=40.7 vs app raw d=0.3,
    41-0.3 = 40.7 exact, in-plane centroids unchanged by the added axis-0
    flip. Applied once, AFTER the back-resample. Returns a C-contiguous
    uint8 array.
    """
    return np.ascontiguousarray(np.flip(np.asarray(seg_sar), axis=(0, 1, 2)).astype(np.uint8, copy=False))
