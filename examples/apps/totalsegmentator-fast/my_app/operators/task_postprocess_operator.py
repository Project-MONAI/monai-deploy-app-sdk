# Copyright 2021-2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task-level postprocessing at model resolution (Phase 19, plan 19-05).

TS 2.18 applies per-task postprocessing in its CLI pipeline — for the
``body`` task, on the MODEL-resolution argmax label map, PRE-back-resample
(``totalsegmentator/nnunet.py:686-695``)::

    img_pred_pp = keep_largest_blob_multilabel(
        img_pred.get_fdata().astype(np.uint8), class_map[task_name],
        ["body_trunc"], debug=False, quiet=quiet)
    ...
    vox_vol = np.prod(img_pred.header.get_zooms())   # model spacing, mm^3
    img_pred_pp = remove_small_blobs_multilabel(
        img_pred.get_fdata().astype(np.uint8), class_map[task_name],
        ["body_extremities"],
        interval=[size_thr_mm3 / vox_vol, 1e10], debug=False, quiet=quiet)
    with size_thr_mm3 = 50000.

These rules live in the TS CLI package, NOT in the model bundle (the body
bundle ships no ``jsonpkls/postprocessing.pkl``), so the per-part pkl gate in
``PostResampleOperator`` cannot express them. This operator ports them:

* ``keep_largest_blob`` / ``remove_small_blobs`` are line-for-line ports of
  ``totalsegmentator/postprocessing.py`` (6-connectivity ``scipy.ndimage.label``,
  ``np.bincount`` blob sizes, the exact removal predicate
  ``(counts <= lo) | (counts > hi)``, the clear-then-write-back pattern).
* The rules themselves are DATA (``TaskSpec.postprocess``), not code — this
  operator carries no task-name or label-name literals; rule label names are
  resolved against the per-task label table (``label_map``, name -> id) at
  setup time. Rules are applied in the given (TS) order.
* Model-resolution arrays are small (body: 153x153x289), so the pass runs on
  CPU numpy/scipy. scipy is already the app's CPU resample kernel (zero new
  deps).

Named Inputs:
    seg: uint8 3D CUDA tensor, model resolution, original DICOM orientation
        (PostResample's ``seg_argmax_dicom`` — orientation is irrelevant to
        connected components and blob volumes).
Named Outputs:
    seg: uint8 3D CUDA tensor, same shape/dtype/order — the post-processed
        label map, ready for the back-resample op.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from holoscan.core import Operator, OperatorSpec
from scipy import ndimage
from scipy.ndimage import binary_dilation

try:  # package-style import (my_app.*)
    from my_app.operators.gpu_util import assert_on_gpu, get_study_id, nvtx_range
    from my_app.operators.preprocess_operator import to_holoscan_gpu_tensor
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from operators.gpu_util import assert_on_gpu, get_study_id, nvtx_range
    from operators.preprocess_operator import to_holoscan_gpu_tensor

__all__ = ["TaskPostprocessOperator", "apply_task_postprocess", "model_voxel_volume_mm3"]


def model_voxel_volume_mm3(model_spacing: Any) -> float:
    """Model-space voxel volume in mm^3 (TS: ``np.prod(img.header.get_zooms())``).

    ``model_spacing`` is the spec's ``target_spacing``: a scalar (isotropic)
    or a 3-tuple.
    """
    if np.isscalar(model_spacing):
        s = float(model_spacing)
        return s * s * s
    return float(np.prod(np.asarray(model_spacing, dtype=np.float64)))


def _as_spacing3(model_spacing: Any) -> Tuple[float, float, float]:
    """Normalize a spec target_spacing (scalar or 3-tuple) to a 3-tuple of mm."""
    if np.isscalar(model_spacing):
        s = float(model_spacing)
        return (s, s, s)
    t = tuple(float(x) for x in model_spacing)
    if len(t) != 3:
        raise ValueError(f"model_spacing must be a scalar or 3-tuple, got {model_spacing!r}")
    return t


def _labels_touch(data: np.ndarray) -> bool:
    """TS ``_multilabel_labels_touch`` (postprocessing.py:180-191), verbatim.

    True when two voxels of DIFFERENT non-background labels are 6-adjacent
    (axis-aligned neighbors in any axis).
    """
    for axis in range(data.ndim):
        sa = [slice(None)] * data.ndim
        sb = [slice(None)] * data.ndim
        sa[axis] = slice(1, None)
        sb[axis] = slice(None, -1)
        a = data[tuple(sa)]
        b = data[tuple(sb)]
        if np.any((a > 0) & (b > 0) & (a != b)):
            return True
    return False


def _ellipsoid_structuring_element(voxel_spacing: Tuple[float, ...], radius_mm: float) -> np.ndarray:
    """TS ``get_ellipsoid_structuring_element`` (postprocessing.py:193-208),
    verbatim — anisotropic 3D ellipsoid kernel in physical (mm) space."""
    if radius_mm <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    radii = [radius_mm / s for s in voxel_spacing]
    grids = np.ogrid[
        -radii[0] : radii[0] + 1,
        -radii[1] : radii[1] + 1,
        -radii[2] : radii[2] + 1,
    ]
    return sum((g * g) / (r * r) for g, r in zip(grids, radii)) <= 1


def _dilate_into_background(
    label_map: np.ndarray,
    label_ids: Tuple[int, ...],
    voxel_spacing: Tuple[float, float, float],
    dilation_mm: float,
) -> np.ndarray:
    """TS ``dilate_vertebrae_labels`` (postprocessing.py:210-247), verbatim.

    Each label dilates INDEPENDENTLY into background voxels only — the
    ``out == 0`` guard means an already-assigned label is never overwritten.
    ``dilation_mm <= 0`` is a no-op (TS parity).
    """
    if dilation_mm <= 0:
        return label_map
    struct_elem = _ellipsoid_structuring_element(voxel_spacing, dilation_mm)
    radius_vox = np.ceil([dilation_mm / s for s in voxel_spacing]).astype(int)
    out = label_map.copy()
    for label in sorted(label_ids):
        lc = np.where(label_map == label)
        if len(lc[0]) == 0:
            continue
        bbox_min = [max(int(c.min()) - r, 0) for c, r in zip(lc, radius_vox)]
        bbox_max = [min(int(c.max()) + r + 1, label_map.shape[ax]) for ax, (c, r) in enumerate(zip(lc, radius_vox))]
        bbox = tuple(slice(a, b) for a, b in zip(bbox_min, bbox_max))
        label_mask = label_map[bbox] == label
        dilated_mask = binary_dilation(label_mask, structure=struct_elem)
        out_bbox = out[bbox]
        out_bbox[(out_bbox == 0) & dilated_mask] = label
    return out


def _relabel_touching(
    label_map: np.ndarray,
    label_ids: Tuple[int, ...],
    voxel_spacing: Tuple[float, float, float],
    vox_vol_mm3: float,
    min_size_mm3: float,
) -> Tuple[np.ndarray, int]:
    """Data-driven port of the touching branch of TS
    ``postprocess_vertebrae_pp`` (postprocessing.py:290-357).

    Connected-component split of the foreground, min-size filter
    (``voxels * vox_vol >= min_size_mm3``), then anatomical relabel. All
    anchors are derived from the passed id table — NO name literals:
    top anchor = min non-background id, bottom anchor = max id (for the
    24-vertebra class map these are C1 and L5, which is exactly what TS
    resolves via ``label_map_inv["vertebrae_C1"/"vertebrae_L5"]``).
    count_from_top = bottom anchor absent AND top anchor present (TS 331-332).

    SI axis = the array axis of maximum physical extent (the vertebral-column
    axis — axis 0 in the app's DICOM-ordered model frame, axis 2 in TS's
    internal frame; the same axis, orientation-free). Sort direction encodes
    that the MOST INFERIOR component carries the LARGEST SI center index in
    this frame: count_from_top sorts ascending and assigns top..bottom ids;
    otherwise sorts descending and assigns max(present)..top ids (verbatim TS
    ordering, mirrored into this frame). Verified bit-equal against TS 2.18 on
    all 3 Phase-22 studies (22-class-a-diagnostics.md, verify_port.py).

    Returns (relabelled_map, n_components_kept).
    """
    ids = tuple(sorted(label_ids))
    top, bottom = ids[0], ids[-1]
    component_map, _ = ndimage.label(label_map > 0)
    component_sizes = np.bincount(component_map.ravel())
    keep = np.flatnonzero(component_sizes * vox_vol_mm3 >= min_size_mm3)
    keep = keep[keep != 0]
    if len(keep) == 0:
        return np.zeros_like(label_map, dtype=np.uint8), 0
    keep_lookup = np.zeros(component_sizes.shape, dtype=bool)
    keep_lookup[keep] = True
    keep_mask = keep_lookup[component_map]
    cleaned = label_map.copy()
    cleaned[~keep_mask] = 0
    id_set = set(ids)
    present = sorted(int(l) for l in np.unique(cleaned) if int(l) in id_set)
    if not present:
        return np.zeros_like(label_map, dtype=np.uint8), 0
    count_from_top = (bottom not in present) and (top in present)
    extents = np.array([label_map.shape[i] * voxel_spacing[i] for i in range(3)], dtype=np.float64)
    si = int(np.argmax(extents))
    centers = ndimage.center_of_mass(keep_mask, component_map, keep)
    if len(keep) == 1:
        centers = [centers]
    comp_centers = [(comp, ctr[si]) for comp, ctr in zip(keep, centers)]
    # In the app's model frame the largest SI center index is the MOST SUPERIOR
    # component, so we sort descending (superior first) and hand out increasing
    # labels top-down. Verified against real gate output (22-09): the naive
    # TS-frame direction reverses the spine top-to-bottom.
    if count_from_top:
        comp_centers.sort(key=lambda t: t[1], reverse=True)
        labels_to_assign = range(top, bottom + 1)
    else:
        comp_centers.sort(key=lambda t: t[1])
        labels_to_assign = range(max(present), top - 1, -1)
    out = np.zeros_like(label_map, dtype=np.uint8)
    for (comp, _), label in zip(comp_centers, labels_to_assign):
        out[component_map == comp] = label
    return out, len(keep)


def _keep_largest_blob(label_map: np.ndarray, idx: int) -> np.ndarray:
    """TS ``keep_largest_blob_multilabel`` per-ROI pass (postprocessing.py:13-45).

    Keeps only the single largest 6-connected component of ``label_map == idx``.
    A label absent or already single-blob returns the map unchanged.
    """
    data_roi = label_map == idx
    if not data_roi.any():
        return label_map
    blob_map, nr_of_blobs = ndimage.label(data_roi)
    if nr_of_blobs <= 1:
        return label_map
    counts = [np.sum(blob_map == i) for i in range(1, nr_of_blobs + 1)]
    largest_blob_label = int(np.argmax(counts)) + 1
    cleaned_roi = blob_map == largest_blob_label
    label_map[data_roi] = 0  # TS: clear the original ROI
    label_map[cleaned_roi] = idx  # TS: write back the cleaned ROI
    return label_map


def _remove_small_blobs(label_map: np.ndarray, idx: int, lo: float, hi: float = 1e10) -> np.ndarray:
    """TS ``remove_small_blobs`` per-ROI pass (postprocessing.py:46-105).

    Drops blobs of ``label_map == idx`` whose voxel count is outside
    ``(lo, hi]`` — TS removal predicate ``(counts <= lo) | (counts > hi)``
    applied to the ``np.bincount`` (background index 0 included, never
    removed: ``counts[0]`` is either huge (never ``<= lo``) or ``<= 1``...
    note background count is always > lo for any realistic volume, matching
    the TS behavior where the background index is passed through).
    """
    data_roi = (label_map == idx).astype(np.uint8)
    mask, number_of_blobs = ndimage.label(data_roi)
    counts = np.bincount(mask.flatten())
    if len(counts) <= 1:
        return label_map
    remove = np.where((counts <= lo) | (counts > hi), True, False)
    remove_idx = np.nonzero(remove)[0]
    mask[np.isin(mask, remove_idx)] = 0
    mask[mask > 0] = 1
    cleaned_roi = mask > 0.5
    label_map[label_map == idx] = 0  # TS: clear the original ROI
    label_map[cleaned_roi] = idx  # TS: write back the cleaned ROI
    return label_map


def apply_task_postprocess(
    label_map: np.ndarray,
    rules: Tuple[Tuple, ...],
    label_map_ids: Dict[str, int],
    vox_vol_mm3: float,
    voxel_spacing: Any = (1.0, 1.0, 1.0),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the TS task postprocessing rules in order to a uint8 label map.

    Args:
        label_map: HxWxD uint8 array (modified IN PLACE — callers pass a copy).
        rules: TaskSpec.postprocess — ``("keep_largest_blob", name)``,
            ``("remove_small_blobs", name, size_thr_mm3)``, and / or
            ``("dilate_labels", dilation_mm, min_size_mm3)`` (22-09: data-driven
            port of TS ``postprocess_vertebrae_pp`` — touching-relabel when
            triggered, then per-label dilation into background; operates on ALL
            non-background labels, rule[1] is a float, not a label name).
        label_map_ids: name -> id table (the spec's ``labels`` dict).
        vox_vol_mm3: model-space voxel volume (mm^3).
        voxel_spacing: model-space per-axis spacing (scalar or 3-tuple, mm) —
            needed by ``dilate_labels`` for the anisotropic ellipsoid geometry
            (TS semantics: radii = dilation_mm / spacing per axis).

    Returns:
        (label_map, per-rule stats dict)
    """
    spacing = _as_spacing3(voxel_spacing)
    non_bg_ids = tuple(sorted(int(v) for v in label_map_ids.values() if int(v) != 0))
    stats: Dict[str, Any] = []
    for rule in rules:
        kind = rule[0]
        if kind == "dilate_labels":
            # rule[1] is dilation_mm (float), rule[2] is min_size_mm3 (float).
            dilation_mm = float(rule[1])
            min_size_mm3 = float(rule[2])
            before = int(np.count_nonzero(label_map))
            data = label_map.astype(np.uint8, copy=False)
            extra: Dict[str, Any] = {}
            if _labels_touch(data):
                data, n_comp = _relabel_touching(data, non_bg_ids, spacing, vox_vol_mm3, min_size_mm3)
                extra = {"touching": True, "relabeled": True, "components_kept": n_comp}
            else:
                extra = {"touching": False, "relabeled": False, "components_kept": 0}
            data = _dilate_into_background(data, non_bg_ids, spacing, dilation_mm)
            label_map = data
            after = int(np.count_nonzero(label_map))
            rec: Dict[str, Any] = {
                "rule": kind,
                "dilation_mm": dilation_mm,
                "min_size_mm3": min_size_mm3,
                "voxels_before": before,
                "voxels_after": after,
            }
            rec.update(extra)
            stats.append(rec)
            continue
        # Per-label rules: rule[1] is a label name.
        if kind not in ("keep_largest_blob", "remove_small_blobs"):
            raise ValueError(f"unknown task postprocess rule kind {kind!r}")
        name = rule[1]
        idx = int(label_map_ids[name])
        before = int(np.sum(label_map == idx))
        if kind == "keep_largest_blob":
            label_map = _keep_largest_blob(label_map, idx)
        elif kind == "remove_small_blobs":
            size_thr_mm3 = float(rule[2])
            label_map = _remove_small_blobs(label_map, idx, size_thr_mm3 / vox_vol_mm3)
        after = int(np.sum(label_map == idx))
        stats.append(
            {
                "rule": kind,
                "label": name,
                "voxels_before": before,
                "voxels_after": after,
                "voxels_changed": before - after,
            }
        )
    return label_map, {"rules": stats}


class TaskPostprocessOperator(Operator):
    """Model-resolution TS task postprocessing (body: keep-largest +
    remove-small-blobs). See module docstring for the TS provenance.

    Args (constructor):
        fragment: the owning application.
        rules: TaskSpec.postprocess tuple (see TaskSpec doc).
        label_map: name -> id table (the spec's ``labels`` dict).
        model_spacing: spec target_spacing (float or 3-tuple, mm) — used for
            the mm^3 -> voxel-count threshold conversion (TS semantics) AND,
            as per-axis spacing, the ``dilate_labels`` ellipsoid geometry.
    """

    INPUT_SEG = "seg"
    OUTPUT_SEG = "seg"

    def __init__(
        self,
        fragment: Any,
        *args: Any,
        rules: Tuple[Tuple, ...] = (),
        label_map: Optional[Dict[str, int]] = None,
        model_spacing: Any = 1.0,
        **kwargs: Any,
    ):
        # NOTE: holoscan 4.2's Operator.__init__ invokes self.setup(spec)
        # before this constructor body finishes — initialize all state first
        # (Pitfall 7).
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._rules = tuple(rules)
        self._vox_vol_mm3 = model_voxel_volume_mm3(model_spacing)
        self._voxel_spacing = _as_spacing3(model_spacing)
        # Resolve label names -> ids NOW (setup runs from __init__).
        self._label_ids: Dict[str, int] = dict(label_map or {})
        if self._rules:
            # dilate_labels is all-labels: rule[1]/[2] are floats, not names.
            unknown = [r[1] for r in self._rules if r[0] != "dilate_labels" and r[1] not in self._label_ids]
            if unknown:
                raise ValueError(
                    f"TaskPostprocessOperator: rule label name(s) {unknown} not in label_map "
                    f"{sorted(self._label_ids)}"
                )
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input(self.INPUT_SEG)
        spec.output(self.OUTPUT_SEG)

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        with nvtx_range("task_postprocess"):
            t0 = time.time()
            holo = op_input.receive(self.INPUT_SEG)
            if holo is None:
                raise ValueError("TaskPostprocessOperator received no 'seg' input.")
            tensor = torch.utils.dlpack.from_dlpack(holo)
            assert_on_gpu(tensor)
            # D2H (model-resolution label map — small), CPU pass, H2D.
            seg = tensor.detach().cpu().numpy().astype(np.uint8, copy=True)
            seg, stats = apply_task_postprocess(
                seg,
                self._rules,
                self._label_ids,
                self._vox_vol_mm3,
                self._voxel_spacing,
            )
            out = torch.as_tensor(np.ascontiguousarray(seg), device=tensor.device)
            record = {
                "op": "task_postprocess",
                "study": get_study_id(self.fragment),
                "shape": list(seg.shape),
                "vox_vol_mm3": self._vox_vol_mm3,
                "rules": stats.get("rules", []),
                "wall_time_s": round(time.time() - t0, 4),
            }
            self._logger.info("task_postprocess record: %s", _json(record))
            op_output.emit(to_holoscan_gpu_tensor(out), self.OUTPUT_SEG)


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj)
