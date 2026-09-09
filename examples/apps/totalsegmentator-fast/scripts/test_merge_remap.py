#!/usr/bin/env python
"""Phase 9.1 synthetic unit suite for merge_remap.py (plan 09-01, Task 3).

House pattern (mirrors scripts/test_pre_resample_zoom.py): standalone
executable, NOT pytest; exits 0 iff every case passes. Headless: CPU numpy
+ scipy only, no GPU, no holoscan, no model/corpus files.

Merge cases (roadmap 9.1 "synthetic 4-case unit suite"):
  1. overlap later-wins
  2. background-no-shift (incl. the np.where-with-offset trap) — the SOUND
     invariant must fire on a deliberate background-shifted part
  3. empty part (legal no-op)
  4. out-of-range per-part max + order guard (fires BEFORE any math)

Geometry cases (roadmap 9.4):
  5. back-resample skip (shape==target or skipped=True -> same object)
  6. back-resample zoom (exact target shape, uint8, == reference call)
  7. flip SAR->DHW (exact axis mapping, C-contiguous)

Run:
  cd examples/apps/totalsegmentator-fast
  /tmp/monai-env/.venv/bin/python scripts/test_merge_remap.py
Exit: 0 iff every case passes; 1 otherwise.
"""

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

import numpy as np
from my_app.operators.merge_remap import flip_sar_to_writer  # noqa: E402
from my_app.operators.merge_remap import (
    back_resample_merged,
    flip_sar_to_dhw,
    merge_remap,
)
from scipy.ndimage import zoom as ndi_zoom

FAILURES = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILURES
    status = "PASS" if ok else "FAIL"
    if not ok:
        FAILURES += 1
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not ok else ""))


def make_parts(shapes: dict, shape=(24, 24, 24)) -> list:
    """Build a valid 5-part list in oracle order from a name->max-label map."""
    parts = []
    for p in ("organs", "vertebrae", "cardiac", "muscles", "ribs"):
        seg = np.zeros(shape, dtype=np.uint8)
        if shapes.get(p, 0) > 0:
            seg[4, 4, 4] = shapes[p]
        parts.append((p, seg))
    return parts


def main() -> int:
    # ------------------------------------------------------------------ 1
    # overlap later-wins: organs label 2 and vertebrae label 3 at the SAME
    # voxel -> merged voxel == 24 + 3 == 27 (vertebrae, the later part, wins)
    def z():
        return np.zeros((24, 24, 24), dtype=np.uint8)

    organs, vertebrae = z(), z()
    organs[4, 4, 4] = 2
    vertebrae[4, 4, 4] = 3
    organs[10, 10, 10] = 1  # non-overlap voxel keeps its part's value
    parts = [
        ("organs", organs),
        ("vertebrae", vertebrae),
        ("cardiac", z()),
        ("muscles", z()),
        ("ribs", z()),
    ]
    c, info = merge_remap(parts)
    check(
        "1a overlap later-wins voxel == 27",
        int(c[4, 4, 4]) == 27,
        f"got {int(c[4, 4, 4])}",
    )
    check("1b non-overlap voxel keeps organs label 1", int(c[10, 10, 10]) == 1)
    check("1c info max_label == 27", info["max_label"] == 27)

    # ------------------------------------------------------------------ 2
    # background-no-shift: cardiac (offset 50) BACKGROUND at W where organs
    # has label 5 -> merged[W] == 5 (NOT 50+5); all-background voxel stays 0.
    organs2, cardiac = z(), z()
    organs2[7, 7, 7] = 5
    parts2 = [
        ("organs", organs2),
        ("vertebrae", z()),
        ("cardiac", cardiac),
        ("muscles", z()),
        ("ribs", z()),
    ]
    c2, _ = merge_remap(parts2)
    check(
        "2a background never shifted (merged[W] == 5)",
        int(c2[7, 7, 7]) == 5,
        f"got {int(c2[7, 7, 7])}",
    )
    check("2b all-background voxel stays 0", int(c2[0, 0, 0]) == 0)
    check(
        "2c per-part bg-in-range counts all zero",
        all(v == 0 for v in merge_remap(parts2)[1]["invariants"]["per_part_bg_voxels_in_range"].values()),
    )

    # The SOUND invariant must FIRE on a deliberate background-shifted part.
    # (a) the classic np.where(seg>0, seg+o, o) trap on cardiac (o=50):
    #     background -> 50, max 68 > cap 18 -> per-part max guard fires.
    organs3, vertebrae3 = z(), z()
    vertebrae3[5, 5, 5] = 26  # reaches exactly label 50 (24+26) pre-cardiac
    cardiac_corrupt = np.where(z() > 0, z() + 50, 50).astype(np.uint8)
    parts3 = [
        ("organs", organs3),
        ("vertebrae", vertebrae3),
        ("cardiac", cardiac_corrupt),
        ("muscles", z()),
        ("ribs", z()),
    ]
    try:
        merge_remap(parts3)
        check("2d np.where-with-offset corruption raises", False, "no error raised")
    except ValueError as e:
        check("2d np.where-with-offset corruption raises", True)
        print(f"       guard fired: {str(e)[:80]}")
    # (b) the old `combined == o` point-check was UNSOUND: the
    #     immediately-preceding part (vertebrae) reaches exactly label 50
    #     (24+26) at a voxel where cardiac (o=50) has BACKGROUND — a correct
    #     merge must NOT raise there (real data: 44238 has 8,514 voxels at
    #     label 68, i.e. muscles local 23 pre-ribs, same class).
    vertebrae_bg = z()
    vertebrae_bg[6, 6, 6] = 26  # reaches exactly label 50 (24+26) pre-cardiac
    parts4 = [
        ("organs", z()),
        ("vertebrae", vertebrae_bg),
        ("cardiac", z()),
        ("muscles", z()),
        ("ribs", z()),
    ]
    try:
        c4, _ = merge_remap(parts4)
        check(
            "2e sound invariant: preceding-part max at o does NOT raise",
            int(c4[6, 6, 6]) == 50,
            f"got {int(c4[6, 6, 6])}",
        )
    except ValueError as e:
        check(
            "2e sound invariant: preceding-part max at o does NOT raise",
            False,
            f"raised: {str(e)[:80]}",
        )

    # ------------------------------------------------------------------ 3
    # empty part: vertebrae all-zeros -> merged identical to the
    # no-vertebrae result (max==0 is a legal no-op, no error).
    parts5 = make_parts({"organs": 3, "ribs": 5})
    c5, info5 = merge_remap(parts5)
    check("3a empty part is a legal no-op", info5["parts"]["vertebrae"]["max_local"] == 0)
    check(
        "3b empty part contributes nothing (max == 96)",
        info5["max_label"] == 91 + 5,
        f"got {info5['max_label']}",
    )

    # ------------------------------------------------------------------ 4
    # out-of-range per-part max + order guard.
    parts6 = make_parts({"organs": 25})
    try:
        merge_remap(parts6)
        check("4a organs label 25 > cap 24 raises", False, "no error raised")
    except ValueError as e:
        check("4a organs label 25 > cap 24 raises", "organs" in str(e), str(e)[:80])
    parts7 = make_parts({"vertebrae": 27})
    try:
        merge_remap(parts7)
        check("4b vertebrae label 27 > cap 26 raises", False, "no error raised")
    except ValueError as e:
        check("4b vertebrae label 27 > cap 26 raises", "vertebrae" in str(e), str(e)[:80])
    # swapped names (muscles before cardiac) -> ValueError BEFORE any math.
    ok_parts = make_parts({"organs": 1, "ribs": 1})
    swapped = [ok_parts[0], ok_parts[1], ok_parts[4], ok_parts[3], ok_parts[2]]
    try:
        merge_remap(swapped)
        check("4c swapped order raises (before math)", False, "no error raised")
    except ValueError as e:
        check("4c swapped order raises (before math)", "order" in str(e), str(e)[:80])
    # subset (missing a part) -> ValueError.
    try:
        merge_remap(ok_parts[1:])
        check("4d subset part list raises", False, "no error raised")
    except ValueError as e:
        check("4d subset part list raises", True)

    # ------------------------------------------------------------------ 5
    # back-resample skip: merged shape == target (42,512,512) with orig
    # RAS [512,512,42] OR skipped=True -> SAME object, no zoom.
    m = np.zeros((42, 512, 512), dtype=np.uint8)
    m[1, 2, 3] = 7
    sk, sinfo = back_resample_merged(m, [512, 512, 42], False)
    check(
        "5a shape==target skip returns same object",
        sk is m and sinfo["skipped"] is True,
    )
    m2 = np.zeros((10, 10, 10), dtype=np.uint8)
    sk2, sinfo2 = back_resample_merged(m2, [512, 512, 42], True)
    check(
        "5b skipped=True returns same object",
        sk2 is m2 and sinfo2["zoom_factors"] is None,
    )

    # ------------------------------------------------------------------ 6
    # back-resample zoom: (56,158,158) SAR, orig RAS [512,512,42] ->
    # out.shape == (42,512,512) EXACT, uint8, max preserved, and out ==
    # the directly-computed reference scipy call (argument-wiring proof).
    rng = np.random.default_rng(7)
    in_sar = (rng.integers(0, 118, size=(56, 158, 158))).astype(np.uint8)
    out, oinfo = back_resample_merged(in_sar, [512, 512, 42], False)
    factors = [42 / 56, 512 / 158, 512 / 158]
    ref = ndi_zoom(in_sar.astype(np.float64), factors, order=0, mode="nearest").astype(np.uint8)
    check(
        "6a zoom lands exactly on target shape",
        out.shape == (42, 512, 512),
        str(out.shape),
    )
    check("6b dtype uint8", out.dtype == np.uint8)
    check(
        "6c max label preserved",
        int(out.max()) == int(in_sar.max()),
        f"in={int(in_sar.max())} out={int(out.max())}",
    )
    check(
        "6d byte-identical to reference scipy call",
        bool((out == ref).all()),
        f"{int((out != ref).sum())} voxels differ",
    )
    check(
        "6e info zoom_factors",
        oinfo["zoom_factors"] == factors and oinfo["skipped"] is False,
    )

    # ------------------------------------------------------------------ 7
    # flip: anisotropic probe (4,3,2) with distinct values -> d[i,j,k] =
    # a[3-i, 2-j, 1-k] (P9 09-04: ALL THREE axes flipped — S as well as
    # A,R — to match the oracle's SEG frame order; see flip_sar_to_dhw
    # docstring), C-contiguous uint8.
    a = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    d = flip_sar_to_dhw(a)
    ok = d.shape == (4, 3, 2) and d.flags["C_CONTIGUOUS"] and d.dtype == np.uint8
    ok = ok and bool(np.array_equal(d, np.flip(a, axis=(0, 1, 2))))
    ok = (
        ok
        and d[0, 0, 0] == 23
        and d[0, 0, 1] == 22
        and d[0, 1, 0] == 21
        and d[0, 1, 1] == 20
        and d[0, 2, 0] == 19
        and d[0, 2, 1] == 18
    )
    ok = ok and d[3, 2, 1] == 0 and d[3, 0, 1] == 4
    check("7 flip axis mapping + contiguity", ok)

    # ------------------------------------------------------------------ 8
    # P10: flip_sar_to_writer is the ORACLE-EXACT writer input (in-plane
    # only, axis [1,2]; depth axis untouched). Anisotropic probe (4,3,2):
    # w[i,j,k] = a[i, 2-j, 1-k]. flip_sar_to_writer == flipD(flip_sar_to_dhw)
    # by construction — the emit (dhw) and writer-input (seg_image) ports
    # differ by exactly the depth flip the highdicom writer applies itself.
    w = flip_sar_to_writer(a)
    ok = w.shape == (4, 3, 2) and w.flags["C_CONTIGUOUS"] and w.dtype == np.uint8
    ok = ok and bool(np.array_equal(w, np.flip(a, axis=(1, 2))))
    ok = ok and w[0, 0, 0] == 5 and w[0, 0, 1] == 4 and w[0, 1, 0] == 3 and w[3, 0, 0] == 23 and w[3, 2, 1] == 18
    ok = ok and bool(np.array_equal(w, np.flip(flip_sar_to_dhw(a), axis=0)))
    check("8 writer-input flip (in-plane only, P10)", ok)

    n = 23  # total atomic checks above
    print(f"{n - FAILURES} checks passed, {FAILURES} failures")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
