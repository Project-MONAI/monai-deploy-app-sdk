#!/usr/bin/env python
"""Phase 22 (plan 22-09) — TaskPostprocessOperator ``dilate_labels`` rule.

Standalone executable suite (NOT pytest), mirroring the house pattern in
``scripts/test_part_postprocess.py``.  CPU-only numpy/scipy — no CUDA needed,
no model bundle, no app pipeline.  Exits 0 iff every case passes.

Proves (plan 22-09 Task 2, per 22-class-a-diagnostics.md):
  1. ``dilate_labels`` rule reproduces a verbatim numpy port of TS
     ``dilate_vertebrae_labels`` (ellipsoid structuring element, radius_vox =
     ceil(dilation_mm/spacing), per-label dilation into BACKGROUND ONLY).
  2. No label clobber: two separated labels each dilate; neither overwrites
     the other's pre-existing voxels (the ``out == 0`` guard).
  3. Registry data: ``TASK_REGISTRY["vertebrae_pp"].postprocess`` ==
     ``(("dilate_labels", 3.0, 100.0),)``; ``body`` unchanged; every other
     task keeps ``postprocess=()``.
  4. ``apply_task_postprocess`` end-to-end with the rule returns the TS-ported
     result; ``dilation_mm=0`` is a no-op for the dilation (relabel branch
     still runs when triggered); an unknown rule kind still raises.
  5. Touching-label relabel branch: a synthetic merged vertebra stack is split
     by connected components, min-size filtered, and anatomically relabeled
     (data-driven min/max label anchors, no name literals).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

APP_ROOT = Path(__file__).resolve().parents[1]
MY_APP = APP_ROOT / "my_app"
sys.path.insert(0, str(MY_APP))

from operators.task_postprocess_operator import apply_task_postprocess  # noqa: E402
from task_specs import TASK_REGISTRY  # noqa: E402

FAILURES = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Verbatim TS reference (totalsegmentator/postprocessing.py, TS 2.18).
# These are the oracles the operator must match; they are DATA-FREE ports of
# the published TS source (kept here so the suite is self-contained and the
# operator itself never imports totalsegmentator).
# ---------------------------------------------------------------------------
def ref_ellipsoid(voxel_spacing, radius_mm):
    if radius_mm <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    radii = [radius_mm / s for s in voxel_spacing]
    grids = np.ogrid[-radii[0] : radii[0] + 1, -radii[1] : radii[1] + 1, -radii[2] : radii[2] + 1]
    return sum((g * g) / (r * r) for g, r in zip(grids, radii)) <= 1


def ref_dilate(data, label_ids, voxel_spacing=(1.0, 1.0, 1.0), dilation_mm=3):
    """TS dilate_vertebrae_labels (postprocessing.py:210-247), verbatim logic."""
    if dilation_mm <= 0:
        return data
    struct_elem = ref_ellipsoid(voxel_spacing, dilation_mm)
    radius_vox = np.ceil([dilation_mm / s for s in voxel_spacing]).astype(int)
    out = data.copy()
    for label in sorted(label_ids):
        lc = np.where(data == label)
        if len(lc[0]) == 0:
            continue
        bbox_min = [max(int(c.min()) - r, 0) for c, r in zip(lc, radius_vox)]
        bbox_max = [min(int(c.max()) + r + 1, data.shape[ax]) for ax, (c, r) in enumerate(zip(lc, radius_vox))]
        bbox = tuple(slice(a, b) for a, b in zip(bbox_min, bbox_max))
        lm = data[bbox] == label
        dm = binary_dilation(lm, structure=struct_elem)
        ob = out[bbox]
        ob[(ob == 0) & dm] = label
    return out


# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------
def build_separated_blobs():
    """48^3 with two well-separated blobs (labels 1 and 2), no contact."""
    seg = np.zeros((48, 48, 48), dtype=np.uint8)
    seg[8:14, 8:14, 8:14] = 1  # label 1: 6x6x6
    seg[34:40, 34:40, 34:40] = 2  # label 2: 6x6x6
    return seg


def build_touching_stack():
    """Two touching blobs sharing a face (a 'merged vertebra' situation).

    Label 1 (top, low z) and label 2 (bottom, high z) share a 4x4 face at the
    z boundary, so _multilabel_labels_touch(data) is True.  Sizes are well
    above min_size at 1.5mm voxel volume so both survive the filter.
    """
    seg = np.zeros((48, 48, 48), dtype=np.uint8)
    seg[4:20, 20:28, 20:28] = 1  # label 1: 16x8x8 = 1024 vox (top)
    seg[20:36, 20:28, 20:28] = 2  # label 2: 16x8x8 = 1024 vox (bottom, touches at z=20)
    return seg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test1_dilate_matches_ts():
    print("[1] dilate_labels == verbatim TS dilate_vertebrae_labels")
    seg = build_separated_blobs()
    ids = {1: 1, 2: 2}
    out, stats = apply_task_postprocess(
        seg.copy(),
        (("dilate_labels", 3.0, 100.0),),
        {"a": 1, "b": 2},
        vox_vol_mm3=3.375,
        voxel_spacing=(1.5, 1.5, 1.5),
    )
    ref = ref_dilate(seg, {1: "a", 2: "b"}, (1.5, 1.5, 1.5), 3.0)
    ok(
        "bit-equal to TS dilate reference",
        np.array_equal(out, ref),
        f"{int((out != ref).sum())} voxels differ",
    )
    ok("both labels present after dilation", set(np.unique(out)) >= {1, 2})
    ok(
        "rule stats recorded",
        len(stats["rules"]) == 1 and stats["rules"][0]["rule"] == "dilate_labels",
    )


def test2_no_clobber():
    print("[2] no label clobber (dilate into background only)")
    seg = build_separated_blobs()
    m1 = seg == 1
    m2 = seg == 2
    out, _ = apply_task_postprocess(
        seg.copy(),
        (("dilate_labels", 3.0, 100.0),),
        {"a": 1, "b": 2},
        vox_vol_mm3=3.375,
        voxel_spacing=(1.5, 1.5, 1.5),
    )
    # every original label-1 voxel is still label 1, and vice versa
    ok("original label-1 voxels preserved", bool((out[m1] == 1).all()))
    ok("original label-2 voxels preserved", bool((out[m2] == 2).all()))
    # dilation only grows into background (no label lost volume to another)
    ok("label-1 volume not reduced", int((out == 1).sum()) >= int((seg == 1).sum()))
    ok("label-2 volume not reduced", int((out == 2).sum()) >= int((seg == 2).sum()))


def test3_registry():
    print("[3] registry: vertebrae_pp dilate rule, others unchanged")
    ok(
        "vertebrae_pp postprocess == (('dilate_labels', 3.0, 100.0),)",
        TASK_REGISTRY["vertebrae_pp"].postprocess == (("dilate_labels", 3.0, 100.0),),
        f"got {TASK_REGISTRY['vertebrae_pp'].postprocess!r}",
    )
    ok(
        "body postprocess unchanged",
        TASK_REGISTRY["body"].postprocess
        == (
            ("keep_largest_blob", "body_trunc"),
            ("remove_small_blobs", "body_extremities", 50000.0),
        ),
        f"got {TASK_REGISTRY['body'].postprocess!r}",
    )
    other_nonempty = [k for k, v in TASK_REGISTRY.items() if v.postprocess != () and k not in ("body", "vertebrae_pp")]
    ok(
        "all other 41 tasks keep postprocess=()",
        other_nonempty == [],
        f"unexpected non-empty: {other_nonempty}",
    )


def test4_e2e_noop_and_errors():
    print("[4] e2e: dilation_mm=0 no-op (dilate) + unknown kind raises")
    seg = build_separated_blobs()
    out0, stats0 = apply_task_postprocess(
        seg.copy(),
        (("dilate_labels", 0.0, 100.0),),
        {"a": 1, "b": 2},
        vox_vol_mm3=3.375,
        voxel_spacing=(1.5, 1.5, 1.5),
    )
    # no touching -> only dilation, which at 0 is a no-op
    ok("dilation_mm=0 is a no-op on non-touching map", np.array_equal(out0, seg))

    try:
        apply_task_postprocess(
            seg.copy(),
            (("bogus_rule", 1),),
            {"a": 1},
            vox_vol_mm3=3.375,
            voxel_spacing=(1.5, 1.5, 1.5),
        )
        ok("unknown rule kind raises ValueError", False, "no exception")
    except ValueError as e:
        ok("unknown rule kind raises ValueError", True, str(e))
    except Exception as e:  # noqa: BLE001
        ok("unknown rule kind raises ValueError", False, f"{type(e).__name__}: {e}")


def test5_relabel_branch():
    print("[5] touching-label relabel branch (data-driven)")
    from operators.task_postprocess_operator import _labels_touch

    # In the app's DICOM-ordered model frame the SI axis (axis 0) has LARGER
    # index = MORE SUPERIOR (head). Verified against real 22-09 gate output:
    # most-inferior component carries the largest SI index and must receive the
    # largest label (L5), most-superior the smallest (C1).
    #
    # 5a: full-spine case (bottom anchor present -> count_from_top False).
    # A (z 4:20) label 3 + B (z 20:36) label 1 touch => merged component at
    # center 20 (smallest index = most inferior); C (z 44:60) label 2 separate
    # at center 52 (largest index = most superior). present={1,2,3}; bottom
    # anchor(3) present -> count_from_top False -> ascending sort, assign
    # [3,2,1]: most inferior (merged A+B) -> 3, most superior (C) -> 2.
    seg = np.zeros((96, 16, 16), dtype=np.uint8)
    seg[4:20, 4:12, 4:12] = 3  # A
    seg[20:36, 4:12, 4:12] = 1  # B: touches A at z=20
    seg[44:60, 4:12, 4:12] = 2  # C: separate (most superior)
    ok("5a fixture has touching labels", _labels_touch(seg))
    out, stats = apply_task_postprocess(
        seg.copy(),
        (("dilate_labels", 3.0, 100.0),),
        {"top": 1, "mid": 2, "bot": 3},
        vox_vol_mm3=3.375,
        voxel_spacing=(1.5, 1.5, 1.5),
    )
    r0 = stats["rules"][0]
    ok("5a relabel branch triggered", r0.get("relabeled") is True, f"got {r0}")
    ok("5a two components kept", r0.get("components_kept") == 2, f"got {r0}")
    ok(
        "5a inferior merged A+B region relabeled to 3",
        out[12, 8, 8] == 3 and out[28, 8, 8] == 3,
        f"got {out[12, 8, 8]},{out[28, 8, 8]}",
    )
    ok(
        "5a most-superior body C relabeled to 2",
        out[52, 8, 8] == 2,
        f"got {out[52, 8, 8]}",
    )
    ok("5a no voxel keeps label 1 (top anchor unused)", int(np.sum(out == 1)) == 0)

    # 5b: head-only case (bottom anchor absent, top anchor present ->
    # count_from_top True) => count DOWN from the top: most-superior body gets
    # the top anchor. ids {1,2,3} (bottom anchor=3 absent); A (z4:20) label 2
    # touches B (z20:36) label 1 -> merged at center 20 (most inferior); C
    # (z44:60) label 1 separate at center 52 (most superior). present={1,2};
    # bottom(3) absent, top(1) present -> count_from_top True -> descending
    # sort, assign [1,2,3]: most superior (C) -> 1 (C1), merged A+B -> 2.
    seg2 = np.zeros((96, 16, 16), dtype=np.uint8)
    seg2[4:20, 4:12, 4:12] = 2  # A
    seg2[20:36, 4:12, 4:12] = 1  # B: touches A at z=20
    seg2[44:60, 4:12, 4:12] = 1  # C: separate (most superior)
    ok("5b fixture has touching labels", _labels_touch(seg2))
    out2, stats2 = apply_task_postprocess(
        seg2.copy(),
        (("dilate_labels", 3.0, 100.0),),
        {"c1": 1, "c2": 2, "c3": 3},
        vox_vol_mm3=3.375,
        voxel_spacing=(1.5, 1.5, 1.5),
    )
    r2 = stats2["rules"][0]
    ok(
        "5b relabel triggered, 2 components",
        r2.get("relabeled") is True and r2.get("components_kept") == 2,
        f"got {r2}",
    )
    ok(
        "5b count-from-top: most-superior body C -> 1",
        out2[52, 8, 8] == 1,
        f"got {out2[52,8,8]}",
    )
    ok(
        "5b count-from-top: inferior merged A+B -> 2",
        out2[12, 8, 8] == 2 and out2[28, 8, 8] == 2,
        f"got {out2[12,8,8]},{out2[28,8,8]}",
    )
    ok("5b absent anchor id 3 unused", int(np.sum(out2 == 3)) == 0)


def main() -> int:
    print("test_task_postprocess (P22 22-09)")
    test1_dilate_matches_ts()
    test2_no_clobber()
    test3_registry()
    test4_e2e_noop_and_errors()
    test5_relabel_branch()
    print()
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}):")
        for n in FAILURES:
            print(f"  - {n}")
        return 1
    print("ALL PASS (5/5 cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
