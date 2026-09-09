#!/usr/bin/env python
"""Phase 18 (plan 18-02, Task 1) unit suite for my_app/operators/crop_core.py.

House pattern (mirrors scripts/test_pre_resample_zoom.py): standalone
executable, NOT pytest. Exits 0 iff every check passes. Fully HEADLESS:
numpy/scipy only (crop_core.py is pure numpy by contract — no holoscan,
torch, or cupy may be imported by it).

Oracle: TotalSegmentator 2.18.0 (read-only venv /tmp/totalseg-venv) —
  * totalsegmentator/cropping.py  (get_bbox_from_mask / crop_to_bbox /
    crop_to_bbox_nifti / undo_crop semantics, ported VERBATIM below)
  * nnunet.py:479/493/512/518-527  (crop at NATIVE resolution; the
    empty-mask check precedes the crop; the task-res resample happens
    AFTER the crop)
  * python_api.py:411 (addon forced to [20, 20, 20] when crop_model is None)

Array convention (app side): the mask and volume arrive in the native
DICOM array order (D, H, W) = the SDK `Image.asnumpy()` order, with a
4x4 `nifti_affine_transform`. TS's nibabel (z, x, y) index structure
maps 1:1 onto (D, H, W), so the ported index math is verbatim.

Checks:
  A. TRUNCATION (plan behavior 1): addon 20 mm @ zooms (0.6, 0.7, 2.5)
     -> addon_vox EXACTLY (33, 28, 8). np.round would give (33, 29, 8)
     and must fail.
  B. BIT-EQUALITY (plan behavior 2): crop_core.get_bbox_from_mask /
     crop_to_mask vs a VERBATIM inline port of TS cropping.py on
     random masks/volumes/affines at 5 zoom sets, INCLUDING both live
     corpus sets — 64199 (2.0, 0.4492, 0.4492) and 31322
     (2.0, 0.899, 0.899) — cropped arrays, bboxes, AND the affine
     offset update must match exactly.
  C. CLAMP (plan behavior 3): mask touching every image border ->
     bbox clamped to [0, shape], no negative indices, crop == volume.
  D. EMPTY (plan behavior 4): all-zero mask -> sentinel (bbox None),
     no exception.
  E. ROUND-TRIP (plan behavior 5): undo_crop(crop_to_mask(x)) == x for
     interior masks (all 5 zoom sets).
  F. LIVE BBOX SHAPES (plan): synthetic masks shaped to reproduce the
     TS 2.18 live-run crop shapes — 64199 crop (433, 351, 81) and
     31322 crop (360, 293, 156) — are reachable with the native-zoom
     truncated-addon arithmetic. (TS prints shapes in its own order;
     the load-bearing assertion is the per-axis (D, H, W) equality
     against (z, x, y) = the nibabel reading of those logged triples.)
  G. MASK FROM 6MM LABELMAP: order-0 (nearest) zoom of the binary
     (labelmap == liver_id) mask to native shape, uint8, values in
     {0, 1}; exact 2x-upscale block mapping.

Run:
  cd examples/apps/totalsegmentator-fast
  /tmp/monai-env/.venv/bin/python scripts/test_crop_core.py
Exit: 0 iff every check passes; 1 otherwise (divergence tables printed).
No files are written; all data is in-memory synthetic.
"""

import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
MY_APP = APP_ROOT / "my_app"
sys.path.insert(0, str(MY_APP))

import numpy as np  # noqa: E402
from operators.crop_core import get_bbox_from_mask  # noqa: E402
from operators.crop_core import (
    back_resample_6mm_labelmap,
    crop_to_mask,
    mask_from_labelmap_6mm,
    resolve_crop_label_ids,
    select_crop_labels,
    undo_crop,
    zooms_from_affine,
)

FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# VERBATIM port of TS 2.18 cropping.py (numpy-only form). The index
# structure of get_bbox_from_mask, crop_to_bbox, the affine offset
# update of crop_to_bbox_nifti, and undo_crop are copied 1:1 — this is
# the oracle against which crop_core is bit-compared.
# ---------------------------------------------------------------------------


def ts_ref_get_bbox(mask: np.ndarray, zooms, addon) -> tuple:
    """TS cropping.get_bbox_from_mask (verbatim arithmetic; addon in mm).

    Returns (bbox, addon_vox) with bbox = [[d0,d1],[h0,h1],[w0,w1]]
    half-open; empty mask -> (None, addon_vox) sentinel (TS prints a
    warning and returns the FULL-image box, but TS handles the empty
    case at nnunet.py:493 BEFORE calling crop_to_mask — the app
    contract is the sentinel).
    """
    addon_vox = (np.array(addon, dtype=float) / np.array(zooms, dtype=float)).astype(int)
    if (mask > 0).sum() == 0:
        print("WARNING: Could not crop because no foreground detected")
        return None, addon_vox
    d, h, w = np.where(mask > 0)
    min_d = int(d.min()) - addon_vox[0]
    max_d = int(d.max()) + 1 + addon_vox[0]
    min_h = int(h.min()) - addon_vox[1]
    max_h = int(h.max()) + 1 + addon_vox[1]
    min_w = int(w.min()) - addon_vox[2]
    max_w = int(w.max()) + 1 + addon_vox[2]
    s = mask.shape
    min_d = max(0, min_d)
    max_d = min(s[0], max_d)
    min_h = max(0, min_h)
    max_h = min(s[1], max_h)
    min_w = max(0, min_w)
    max_w = min(s[2], max_w)
    return [[min_d, max_d], [min_h, max_h], [min_w, max_w]], addon_vox


def ts_ref_crop(data: np.ndarray, affine: np.ndarray, bbox):
    """TS cropping.crop_to_bbox + crop_to_bbox_nifti affine update (verbatim)."""
    cropped = data[bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]]
    new_affine = np.copy(affine)
    new_affine[:3, 3] = np.dot(affine, np.array([bbox[0][0], bbox[1][0], bbox[2][0], 1]))[:3]
    return cropped, new_affine


# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

ADDON_MM = [20, 20, 20]  # python_api.py:411 (crop_model is None)

# (name, zooms in app (D,H,W) axis order = nibabel (z,x,y) order)
ZOOM_SETS = [
    ("64199-live", (2.0, 0.4492, 0.4492)),
    ("31322-live", (2.0, 0.899, 0.899)),
    ("trunc-vs-round", (2.5, 0.7, 0.6)),
    ("iso-6mm", (6.0, 6.0, 6.0)),
    ("tiny", (1.0, 1.0, 1.0)),
]

AFFINE = np.array(
    [
        [-2.0, 0.0, 0.0, 113.5],
        [0.0, -0.4492, 0.0, -120.2],
        [0.0, 0.0, -0.4492, -300.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def make_mask_and_vol(shape, zooms, rng):
    """Interior liver-ish block: >= 21 voxels per axis so the bbox
    interior is non-degenerate; zoom-dependent but never clamped."""
    mask = np.zeros(shape, dtype=np.uint8)
    d0 = shape[0] // 4
    d1 = shape[0] - shape[0] // 4
    h0 = shape[1] // 4
    h1 = shape[1] - shape[1] // 4
    w0 = shape[2] // 4
    w1 = shape[2] - shape[2] // 4
    mask[d0:d1, h0:h1, w0:w1] = 1
    vol = (rng.integers(-1024, 300, size=shape, dtype=np.int64)).astype(np.int32)
    return vol, mask


def case_label(name: str, ref, got) -> str:
    if isinstance(ref, np.ndarray) and isinstance(got, np.ndarray):
        mask = ref != got
        return f"({name}): {int(mask.sum())}/{ref.size} voxels differ"
    return f"({name}): ref={ref!r} got={got!r}"


# ---------------------------------------------------------------------------
# A. truncation
# ---------------------------------------------------------------------------

print("A. addon truncation (mm -> vox, .astype(int) — never np.round)")
zooms_t = (0.6, 0.7, 2.5)  # 20/0.6=33.33->33, 20/0.7=28.57->28 (round: 29), 20/2.5=8
small = np.zeros((30, 30, 30), dtype=np.uint8)
small[5:25, 5:25, 5:25] = 1
_, addon_vox = get_bbox_from_mask(small, zooms_t, ADDON_MM)
check(
    "addon_vox == (33, 28, 8) exactly",
    list(np.asarray(addon_vox)) == [33, 28, 8],
    f"got {addon_vox} (np.round would give [33, 29, 8])",
)

# ---------------------------------------------------------------------------
# B. bit-equality vs the verbatim TS port
# ---------------------------------------------------------------------------

print("B. bit-equality vs verbatim TS cropping.py port")
rng = np.random.default_rng(1802)
for name, zooms in ZOOM_SETS:
    vol, mask = make_mask_and_vol((217, 512, 512), zooms, rng)
    bbox_ref, addon_ref = ts_ref_get_bbox(mask, zooms, ADDON_MM)
    bbox_got, addon_got = get_bbox_from_mask(mask, zooms, ADDON_MM)
    check(f"bbox {name}", bbox_got == bbox_ref, case_label("bbox", bbox_ref, bbox_got))
    check(
        f"addon_vox {name}",
        list(np.asarray(addon_got)) == list(np.asarray(addon_ref)),
        case_label("addon", addon_ref, addon_got),
    )
    cv_ref, aff_ref = ts_ref_crop(vol, AFFINE, bbox_ref)
    cv_got, bbox_info = crop_to_mask(vol, mask, zooms, AFFINE, ADDON_MM)
    aff_got = AFFINE.copy()
    aff_got[:3, 3] = bbox_info["affine_offset"]
    check(
        f"cropped volume {name}",
        np.array_equal(cv_ref, cv_got),
        case_label("vol", cv_ref.shape, cv_got.shape),
    )
    check(
        f"cropped bbox region {name}",
        tuple(cv_got.shape)
        == (
            bbox_ref[0][1] - bbox_ref[0][0],
            bbox_ref[1][1] - bbox_ref[1][0],
            bbox_ref[2][1] - bbox_ref[2][0],
        ),
        f"shape {cv_got.shape}",
    )
    check(
        f"affine offset {name}",
        np.array_equal(aff_ref, aff_got),
        case_label("aff", aff_ref[:3, 3].tolist(), aff_got[:3, 3].tolist()),
    )
    cm_ref, _ = ts_ref_crop(mask.astype(np.int32), AFFINE, bbox_ref)
    check(
        f"cropped mask {name}",
        np.array_equal(cm_ref, bbox_info["cropped_mask"]),
        case_label("mask", "diff", "diff"),
    )

# ---------------------------------------------------------------------------
# C. clamp (mask touching image borders)
# ---------------------------------------------------------------------------

print("C. border clamp")
for edge in ["all-borders", "d-edges", "h-edges", "w-edges"]:
    vol, mask = make_mask_and_vol((64, 128, 96), ZOOM_SETS[0][1], rng)
    if "all" in edge:
        mask[:] = 1
    if "d" in edge:
        mask[:1, :, :] = 1
        mask[-1:, :, :] = 1
    if "h" in edge:
        mask[:, :2, :] = 1
        mask[:, -2:, :] = 1
    if "w" in edge:
        mask[:, :, :3] = 1
        mask[:, :, -3:] = 1
    bbox_got, _ = get_bbox_from_mask(mask, ZOOM_SETS[0][1], ADDON_MM)
    s = mask.shape
    ok = (
        bbox_got[0][0] >= 0
        and bbox_got[0][1] <= s[0]
        and bbox_got[1][0] >= 0
        and bbox_got[1][1] <= s[1]
        and bbox_got[2][0] >= 0
        and bbox_got[2][1] <= s[2]
    )
    check(f"bbox within [0, shape] ({edge})", ok, f"got {bbox_got}")
    if edge == "all-borders":
        cv_got, info = crop_to_mask(vol, mask, ZOOM_SETS[0][1], AFFINE, ADDON_MM)
        check("full-mask crop == volume (all-borders)", np.array_equal(cv_got, vol), "")

# ---------------------------------------------------------------------------
# D. empty mask sentinel
# ---------------------------------------------------------------------------

print("D. empty mask -> sentinel, no exception")
empty = np.zeros((217, 512, 512), dtype=np.uint8)
bbox_got, addon_got = get_bbox_from_mask(empty, ZOOM_SETS[0][1], ADDON_MM)
check("empty -> bbox None", bbox_got is None, f"got {bbox_got}")
check(
    "empty -> addon_vox still computed",
    list(np.asarray(addon_got)) == [10, 44, 44],
    f"got {addon_got}",
)
try:
    crop_to_mask(
        np.zeros((10, 10, 10), dtype=np.int32),
        empty[:10, :10, :10],
        (2.0, 2.0, 2.0),
        AFFINE,
        ADDON_MM,
    )
    check("crop_to_mask on empty raises ValueError", False, "no exception raised")
except ValueError:
    check("crop_to_mask on empty raises ValueError", True)

# ---------------------------------------------------------------------------
# E. round-trip undo_crop(crop_to_mask(x)) == x
# ---------------------------------------------------------------------------

print("E. undo_crop o crop_to_mask == identity (interior masks)")
for name, zooms in ZOOM_SETS:
    shape = (217, 512, 512) if name.endswith("live") else (96, 120, 80)
    vol = (rng.integers(-1024, 300, size=shape, dtype=np.int64)).astype(np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    # interior block with >= addon margin from every border for ALL zoom sets
    mask[10 : shape[0] - 10, 15 : shape[1] - 15, 15 : shape[2] - 15] = 1
    cv, info = crop_to_mask(vol, mask, zooms, AFFINE, ADDON_MM)
    restored = undo_crop(shape, cv, info)
    bb = info["bbox"]
    inside = restored[bb[0][0] : bb[0][1], bb[1][0] : bb[1][1], bb[2][0] : bb[2][1]]
    outside = np.ones(shape, bool)
    outside[bb[0][0] : bb[0][1], bb[1][0] : bb[1][1], bb[2][0] : bb[2][1]] = False
    check(
        f"round-trip volume inside bbox {name}",
        np.array_equal(inside, vol[bb[0][0] : bb[0][1], bb[1][0] : bb[1][1], bb[2][0] : bb[2][1]]),
        "",
    )
    check(
        f"round-trip volume outside bbox zero {name}",
        int((restored[outside] != 0).sum()) == 0,
        f"{int((restored[outside] != 0).sum())} nonzeros",
    )
    # cropped mask round-trip: crop the mask itself with the same bbox info
    _, info_m = crop_to_mask(mask.astype(np.int32), mask, zooms, AFFINE, ADDON_MM)
    restored_m = undo_crop(shape, info_m["cropped_mask"], info)
    check(f"round-trip mask {name}", np.array_equal(restored_m, mask.astype(np.int32)), "")

# ---------------------------------------------------------------------------
# F. live bbox shapes (TS 2.18 live runs, 18-RESEARCH §Critical Correction)
# ---------------------------------------------------------------------------

print("F. live crop shapes reproducible at native resolution")
# 64199: native (D,H,W)=(217,512,512), zooms (2.0, 0.4492, 0.4492).
# TS log: crop (433, 351, 81) = (x, y, z) = (H, W, D) in our order.
# addon vox = (10, 44, 44) -> mask extents (D,H,W) = (61, 345, 263).
v, m = make_mask_and_vol((217, 512, 512), ZOOM_SETS[0][1], rng)
m = np.zeros_like(m)
m[50:111, 100:445, 100:363] = 1  # (61, 345, 263)
cv, info = crop_to_mask(v, m, ZOOM_SETS[0][1], AFFINE, ADDON_MM)
check(
    "64199 crop shape (D,H,W) == (81, 433, 351)  [TS log (433,351,81)]",
    tuple(cv.shape) == (81, 433, 351),
    f"got {cv.shape}",
)
# 31322: native (D,H,W)=(345,512,512), zooms (2.0, 0.899, 0.899).
# TS log: crop (360, 293, 156) = (H, W, D) -> (D,H,W) = (156, 360, 293).
# addon vox = (10, 22, 22) -> mask extents (136, 316, 249).
v2, m2 = make_mask_and_vol((345, 512, 512), ZOOM_SETS[1][1], rng)
m2 = np.zeros_like(m2)
m2[60:196, 100:416, 100:349] = 1
cv2, info2 = crop_to_mask(v2, m2, ZOOM_SETS[1][1], AFFINE, ADDON_MM)
check(
    "31322 crop shape (D,H,W) == (156, 360, 293)  [TS log (360,293,156)]",
    tuple(cv2.shape) == (156, 360, 293),
    f"got {cv2.shape}",
)

# ---------------------------------------------------------------------------
# G. mask_from_labelmap_6mm (order-0 zoom to native, uint8, binary)
# ---------------------------------------------------------------------------

print("G. mask_from_labelmap_6mm (order-0 nearest zoom, TS nnunet.py:758+ contract)")
# Exact 2x case: 6mm grid (10,12,14) -> native (20,24,28); liver (label 5)
# block (2:5, 3:8, 4:9) -> native block (4:10, 6:16, 8:18) exactly.
lm6 = np.zeros((10, 12, 14), dtype=np.uint8)
lm6[2:5, 3:8, 4:9] = 5
lm6[7:9, 9:11, 12:13] = 3  # another organ, must NOT leak into the mask
mask_native = mask_from_labelmap_6mm(lm6, (20, 24, 28), liver_id=5)
check("dtype uint8", mask_native.dtype == np.uint8, f"got {mask_native.dtype}")
check("shape == native", mask_native.shape == (20, 24, 28), f"got {mask_native.shape}")
check(
    "binary values",
    set(np.unique(mask_native).tolist()) <= {0, 1},
    f"got {np.unique(mask_native).tolist()}",
)
expected = np.zeros((20, 24, 28), dtype=np.uint8)
expected[4:10, 6:16, 8:18] = 1
check(
    "2x upscale block mapping exact",
    np.array_equal(mask_native, expected),
    f"got sum {int(mask_native.sum())}, expected {int(expected.sum())}",
)
# Non-integer factor (64199 geometry: 6mm grid 65x134x134 -> native 217x512x512)
grid = (65, 134, 134)
native = (217, 512, 512)
lm6b = np.zeros(grid, dtype=np.uint8)
lm6b[20:45, 30:100, 40:90] = 5
mn = mask_from_labelmap_6mm(lm6b, native, liver_id=5)
check(
    "non-integer factor: shape/dtype/binary",
    mn.shape == native and mn.dtype == np.uint8 and set(np.unique(mn).tolist()) <= {0, 1},
    f"got {mn.shape}/{mn.dtype}",
)
check(
    "non-integer factor: liver present, no all-zero collapse",
    int(mn.sum()) > 0,
    "sum=0",
)

# ---------------------------------------------------------------------------
# H. zooms_from_affine helper
# ---------------------------------------------------------------------------

print("H. zooms_from_affine (column norms, sign-invariant)")
got_zooms = zooms_from_affine(AFFINE)
check(
    "zooms == (2.0, 0.4492, 0.4492)",
    all(abs(a - b) < 1e-9 for a, b in zip(got_zooms, (2.0, 0.4492, 0.4492))),
    f"got {got_zooms}",
)

# ---------------------------------------------------------------------------
# I. 22-07: spec-driven crop-label mask helpers (CropMaskOp extraction)
# ---------------------------------------------------------------------------

print("I. 22-07 crop-label helpers (back_resample_6mm_labelmap / select_crop_labels / resolve_crop_label_ids)")

# --- I-1: back-resample + multi-label selection, bit-equal to a verbatim TS-port reference ---
# Geometry mirrors the 64199 live cell: 6mm grid (72,38,38) (z,y,x) SAR -> native (512,512,217) (x,y,z) RAS.
lm6_i = np.zeros((72, 38, 38), dtype=np.uint8)
lm6_i[45:68, 6:24, 5:27] = 10  # one lobe block (painted first)
lm6_i[48:60, 20:36, 18:37] = 13  # second lobe block, overlaps the first (painted last -> wins)
lm6_i[33:53, 12:31, 9:34] = 5  # a DIFFERENT organ (liver) — must not leak into the mask
NATIVE_I = (512, 512, 217)  # (x, y, z) RAS order — the helper's contract
LABEL_IDS_I = [10, 13]


# Oracle 1: the app's historical code path, re-implemented VERBATIM here
# (transpose (2,1,0) -> per-axis order-0 nearest zoom to the native (x,y,z) shape ->
#  label selection). This is exactly what CropMaskOp.compute used to inline.
def ts_ref_backresample_select(lm6, native_shape, ids):
    s = np.ascontiguousarray(np.transpose(lm6, (2, 1, 0)))
    factors = tuple(float(n) / float(c) for n, c in zip(native_shape, s.shape))
    from scipy.ndimage import zoom as _z

    lm_native = _z(s.astype(np.float64), factors, order=0, mode="nearest").round().astype(np.uint8)
    return np.isin(lm_native, list(ids)).astype(np.uint8)


ref_mask = ts_ref_backresample_select(lm6_i, NATIVE_I, LABEL_IDS_I)
got_mask = select_crop_labels(back_resample_6mm_labelmap(lm6_i, NATIVE_I), LABEL_IDS_I)
check(
    "I-1a bit-exact vs verbatim TS-port pipeline",
    np.array_equal(got_mask, ref_mask),
    f"xor voxels={int((got_mask ^ ref_mask).sum())}",
)
check(
    "I-1b mask binary uint8",
    got_mask.dtype == np.uint8 and set(np.unique(got_mask).tolist()) <= {0, 1},
    f"got {got_mask.dtype}",
)
# Oracle 2: TS cropping.py semantics — select labels on the 6mm grid FIRST, then
# nearest-upsample the binary union (order-0 zoom commutes with label selection).
from scipy.ndimage import zoom as _z2

union6 = np.isin(lm6_i, LABEL_IDS_I).astype(np.uint8)
ref2 = (
    _z2(
        union6.astype(np.float64),
        (217 / 72, 512 / 38, 512 / 38),
        order=0,
        mode="nearest",
    )
    .round()
    .astype(np.uint8)
)  # factors in (z,y,x) input order
ref2 = np.ascontiguousarray(np.transpose(ref2, (2, 1, 0)))  # back to (x,y,z) for comparison
check(
    "I-1c bit-exact vs union-then-upsample oracle",
    np.array_equal(got_mask, ref2),
    f"xor voxels={int((got_mask ^ ref2).sum())}",
)
# Oracle 3: explicit per-axis index mapping — output index i takes input
# index round(i * (in-1)/(out-1)) (the exact discrete mapping of
# scipy order-0 nearest zoom on an axis-aligned grid; verified empirically).
ix = np.clip(np.round(np.arange(NATIVE_I[0]) * 37 / 511), 0, 37).astype(int)
iy = np.clip(np.round(np.arange(NATIVE_I[1]) * 37 / 511), 0, 37).astype(int)
iz = np.clip(np.round(np.arange(NATIVE_I[2]) * 71 / 216), 0, 71).astype(int)
# out(x,y,z) == lm6_i[z, y, x] -> independent of scipy entirely.
exp_native = np.isin(lm6_i[np.ix_(iz, iy, ix)], LABEL_IDS_I).astype(np.uint8).transpose(2, 1, 0)
check(
    "I-1d bit-exact vs index-mapping reference (native)",
    np.array_equal(got_mask, exp_native),
    f"xor voxels={int((got_mask ^ exp_native).sum())}",
)
# The selected native voxels carry ONLY the selected label ids (liver id 5 never leaks).
lm_native_i = back_resample_6mm_labelmap(lm6_i, NATIVE_I)
sel_vals = set(np.unique(lm_native_i[got_mask == 1]).tolist())
check(
    "I-1e selected voxels carry only selected labels",
    sel_vals <= set(LABEL_IDS_I),
    f"got {sel_vals}",
)
# Same-shape passthrough: identity (no zoom), selection still correct.
pas = back_resample_6mm_labelmap(lm_native_i, NATIVE_I)
check("I-1f same-shape passthrough is a no-op", np.array_equal(pas, lm_native_i))

# --- I-2: bbox frame coherence (RAS mask + RAS zooms; SAR mirror symmetry) ---
# Reconstruct the 64199 liver mask RAS layout (shape (512,512,217), x,y,z) from the
# 22-05 recorded extent so the arithmetic is exercised at live-cell scale, headless.
rng2 = np.random.default_rng(2207)
live_mask = np.zeros((512, 512, 217), dtype=np.uint8)
live_mask[118:463, 159:422, 99:160] = 1  # x,y,z extents per the 22-05 ledger (64199 liver mask)
ZOOMS_RAS = (0.449219, 0.449219, 2.0)
ADDON_I2 = (20.0, 20.0, 20.0)
bb_i2, av_i2 = get_bbox_from_mask(live_mask, ZOOMS_RAS, ADDON_I2)
check("I-2a addon mm->vox TRUNCATED", list(av_i2) == [44, 44, 10], f"got {list(av_i2)}")
check(
    "I-2b bbox == extent +/- truncated addon (live 64199)",
    bb_i2 == [[74, 507], [115, 466], [89, 170]],
    f"got {bb_i2}",
)
# SAR mirror: flipping axes 0,1 of the mask must mirror the bbox indices exactly
# (half-open interval [lo,hi) of size N flips to [N-hi, N-lo)).
sar_mask = np.flip(live_mask, (0, 1))
bb_sar, _ = get_bbox_from_mask(sar_mask, ZOOMS_RAS, ADDON_I2)
N = live_mask.shape
expect_sar = [
    [N[0] - bb_i2[0][1], N[0] - bb_i2[0][0]],
    [N[1] - bb_i2[1][1], N[1] - bb_i2[1][0]],
    bb_i2[2],
]
check(
    "I-2c SAR-flipped mask -> mirrored bbox",
    bb_sar == expect_sar,
    f"got {bb_sar} want {expect_sar}",
)
# crop_to_mask on the same-frame (volume, mask) pair must produce the bbox dims.
vol_i2 = rng2.integers(-1000, 2000, size=N, dtype=np.int32)
cropped_i2, info_i2 = crop_to_mask(vol_i2, live_mask, ZOOMS_RAS, AFFINE, ADDON_I2)
check(
    "I-2d crop shape == bbox dims",
    tuple(cropped_i2.shape) == (433, 351, 81),
    f"got {cropped_i2.shape}",
)
check(
    "I-2e cropped volume is the exact frame slice",
    np.array_equal(cropped_i2, vol_i2[74:507, 115:466, 89:170]),
)

# --- I-3: paste confinement (PasteOp semantics: zoom to cropped_native_shape, assign into canvas[bbox]) ---
from scipy.ndimage import zoom as _z3

seg_ras = (rng2.integers(0, 4, size=(130, 105, 108)) - 1).clip(min=0).astype(np.uint8)
cropped_native_shape = (433, 351, 81)
factors_p = tuple(float(n) / float(c) for n, c in zip(cropped_native_shape, seg_ras.shape))
seg_native = _z3(seg_ras.astype(np.float64), factors_p, order=0, mode="nearest").round().astype(np.uint8)
canvas = np.zeros((512, 512, 217), dtype=np.uint8)
bb_i3 = [[74, 507], [115, 466], [89, 170]]
canvas[bb_i3[0][0] : bb_i3[0][1], bb_i3[1][0] : bb_i3[1][1], bb_i3[2][0] : bb_i3[2][1]] = seg_native
nz = np.argwhere(canvas > 0)
in_bbox = (
    (nz[:, 0] >= bb_i3[0][0])
    & (nz[:, 0] < bb_i3[0][1])
    & (nz[:, 1] >= bb_i3[1][0])
    & (nz[:, 1] < bb_i3[1][1])
    & (nz[:, 2] >= bb_i3[2][0])
    & (nz[:, 2] < bb_i3[2][1])
)
check(
    "I-3a pasted content fully inside crop bbox",
    bool(in_bbox.all()),
    f"{int((~in_bbox).sum())} voxels outside bbox",
)
# Reference semantics: crop_core.undo_crop must give the identical canvas.
canvas_ref = undo_crop(
    (512, 512, 217),
    seg_native,
    {
        "bbox": bb_i3,
        "shape": cropped_native_shape,
        "addon_vox": [44, 44, 10],
        "affine": None,
        "affine_offset": None,
    },
)
check("I-3b paste == undo_crop reference", np.array_equal(canvas, canvas_ref))

# --- I-4: liver regression guard (byte-identity of the resolved liver path) ---
# Label table mirroring the real total_6mm bundle ordering (liver=5, lobes 10-14);
# cross-checked against the real dataset.json when the model dir is present.
TABLE_I4 = {
    "background": 0,
    "spleen": 1,
    "kidney_right": 2,
    "kidney_left": 3,
    "gallbladder": 4,
    "liver": 5,
    "stomach": 6,
    "pancreas": 7,
    "lung_upper_lobe_left": 10,
    "lung_lower_lobe_left": 11,
    "lung_upper_lobe_right": 12,
    "lung_middle_lobe_right": 13,
    "lung_lower_lobe_right": 14,
}
_real_ds_path = (
    MY_APP.parents[3]
    / "ct-totalsegmentator-map"
    / "models"
    / "liver_segments"
    / "total_6mm"
    / "jsonpkls"
    / "dataset.json"
)
if _real_ds_path.exists():
    _real_labels = json.loads(_real_ds_path.read_text())["labels"]
    check(
        "I-4a real total_6mm table agrees (liver=5, lobes 10-14)",
        all(_real_labels.get(k) == v for k, v in TABLE_I4.items()),
        f"got liver={_real_labels.get('liver')} lobes="
        f"{[_real_labels.get(k) for k in TABLE_I4 if k.startswith('lung')]}",
    )
else:
    check(
        "I-4a real total_6mm table agrees (liver=5, lobes 10-14)",
        False,
        f"model dir absent at {_real_ds_path}",
    )
ids_liver = resolve_crop_label_ids(TABLE_I4, ("liver",))
check("I-4b ('liver',) resolves to [5]", ids_liver == [5], f"got {ids_liver}")
ids_lobes = resolve_crop_label_ids(
    TABLE_I4,
    (
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
    ),
)
check(
    "I-4c 5 lung lobes resolve to [10..14] in spec order",
    ids_lobes == [10, 11, 12, 13, 14],
    f"got {ids_lobes}",
)
check(
    "I-4d case-insensitive + whitespace-tolerant resolve",
    resolve_crop_label_ids(TABLE_I4, ("  Liver ",)) == [5],
)
try:
    resolve_crop_label_ids(TABLE_I4, ("liver", "not_an_organ"))
    check("I-4e unknown name raises KeyError", False, "no exception raised")
except KeyError:
    check("I-4e unknown name raises KeyError", True)
# Byte-identity: new single-id selection == legacy `(labelmap == 5)` on the 64199 geometry.
lm_i4 = np.zeros(NATIVE_I, dtype=np.uint8)
lm_i4[118:463, 159:422, 99:160] = 5
legacy = (lm_i4 == 5).astype(np.uint8)
new_path = select_crop_labels(lm_i4, ids_liver)
check(
    "I-4f select_crop_labels == legacy (lm == 5) bit-exact for liver",
    np.array_equal(legacy, new_path),
    f"xor voxels={int((legacy ^ new_path).sum())}",
)

# ---------------------------------------------------------------------------

print()
if FAILURES:
    print(f"FAILURES: {len(FAILURES)} -> {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED (test_crop_core)")
sys.exit(0)
