#!/usr/bin/env python
"""Headless bit-identity test for the GPU volume-metrics pure module (13-01).

Compares ``my_app/operators/volume_metrics.py`` against a VERBATIM port of the
oracle per-label loop (``ct-totalsegmentator-map/app_total/segmentation_metrics_operator.py``
-- ``_compute_volume_or_area`` + the per-label ``calculate_metrics`` loop),
evaluated on the pin-decoded 44238 labelmap. No GPU, no holoscan import.

Data sources (pinned, read-only):
  * Labelmap: /raid/tmp/p12_research/decode44238/oracle_total_dhw.npy
    (sha256-pinned; if missing, re-decoded from the pinned 44238 SEG dcm in
    /raid/tmp/ts_baseline using the P9 segment-major decode pattern).
  * Spacing: (0.462891, 0.462891, 2.0) -- the real physical spacing of the
    44238 corpus study (PixelSpacing + SliceThickness of first/last instance).

Pass criteria (any failure -> non-zero exit):
  1. np.array_equal on per-label volumes, labels 1..117 (BIT-IDENTICAL float64)
  2. exact zero-set equality (labels whose vol == 0.0)
  3. num.slices + slice.range equality for all 117 labels
  4. compute_slice_stats yields a 117-key dict with inner keys
     exactly {vol, num.slices, slice.range, pixel.count}
  5. all-background input -> all-zero volumes, empty nonzero set
  6. 4-D (1,D,H,W) torch input gives identical results to 3-D input

Exit 0 + "ALL PASS" when every check holds.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Constants (pinned)
# ---------------------------------------------------------------------------
NLY_PATH = "/raid/tmp/p12_research/decode44238/oracle_total_dhw.npy"
NLY_SHA256 = "ce2e2dd74811d348a743c3b227402ec6a42d2876380903ac3d82c4ff93a59c9c"
SEG_DCM = (
    "/raid/tmp/ts_baseline/06-20-2009-NA-CT-44238/SEG/"
    "1.2.826.0.1.3680043.10.511.3.9492230222357721157166297047794676.dcm"
)
SEG_SHA256 = "c2d29d208e665e3c0e90256f0cbff02f940ba5818eed1ac6cafe443d2117a2c2"
# Real physical spacing of corpus study 44238 (first+last instance:
# PixelSpacing (0.462891, 0.462891), SliceThickness 2.0) -- same values the
# legacy op would read from the input-scan Image metadata.
SPACING = (0.462891, 0.462891, 2.0)
NUM_LABELS = 117

MODULE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "my_app",
    "operators",
    "volume_metrics.py",
)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _decode_seg_dcm(path: str, n_segments: int = 117) -> np.ndarray:
    """P9/P10 decode pattern: pinned 1-bit multi-frame DICOM-SEG -> uint8 (D,H,W).

    Segment-major layout: segment k (0-based) owns frame block
    [k*nsl .. (k+1)*nsl); frame j within the block = slice j.
    """
    import pydicom

    if _sha256_file(path) != SEG_SHA256:
        raise RuntimeError(f"SEG pin sha256 mismatch for {path}")
    ds = pydicom.dcmread(path)
    seg_seq = ds.SegmentSequence  # (0x0062,0x0002)
    assert len(seg_seq) == n_segments, f"expected {n_segments} segments, got {len(seg_seq)}"
    frames = int(ds.NumberOfFrames)
    nsl = frames // n_segments
    assert frames == nsl * n_segments, f"frame count {frames} not divisible by {n_segments}"
    # pydicom 3.x removed the legacy `pixel_data` attribute; `pixel_array` is
    # the supported accessor (26-01 env-compat fix, test-only). Fallback keeps
    # pydicom 2.x interpreters working.
    pixel_data = getattr(ds, "pixel_array", None)
    if pixel_data is None:
        pixel_data = ds.pixel_data  # pydicom <3.0
    out = np.zeros((nsl, *pixel_data.shape[1:]), dtype=np.uint8)
    # P9 layout guard: verify the segment-major frame layout via the per-frame
    # functional groups WHEN PRESENT. NOTE (26-01): this pinned file (sha
    # c2d29d20, verified) carries no PerFrameFunctionalGroupsSequence — the
    # guard branch had never executed against it (the P12-era npy short-
    # circuited this decode until /raid/tmp cleanup). Keep the guard strict
    # where the metadata exists; warn where it does not.
    has_pf_groups = all(hasattr(seg_seq[k], "PerFrameFunctionalGroupsSequence") for k in range(n_segments))
    if not has_pf_groups:
        print(
            "[warn] no PerFrameFunctionalGroupsSequence in pinned SEG — "
            "segment-major layout guard skipped (structural frame-count asserts still active)"
        )
    for k in range(n_segments):
        block = pixel_data[k * nsl : (k + 1) * nsl]
        if has_pf_groups:
            for j in range(nsl):
                # (26-01) frame index per the segment-major contract above is
                # k*nsl + j; the pre-26-01 expression j*n_segments+k was
                # transposed and had never executed against this file.
                ff = seg_seq[k].PerFrameFunctionalGroupsSequence[k * nsl + j]
                dim_idx = ff.FrameContentSequence[0].DimensionIndexValues
                ref_seg = ff.SegmentIdentificationSequence[0].ReferencedSegmentNumber
                assert (
                    int(dim_idx[0]) == k + 1 and int(ref_seg) == k + 1
                ), f"segment-major layout guard failed at segment {k}, slice {j}"
        out = np.where(block.astype(bool), k + 1, out)
    return out


def load_labelmap() -> np.ndarray:
    if os.path.exists(NLY_PATH):
        if _sha256_file(NLY_PATH) != NLY_SHA256:
            raise RuntimeError(f"npy pin sha256 mismatch for {NLY_PATH}")
        return np.load(NLY_PATH)
    print(f"[info] {NLY_PATH} missing -- decoding from pinned SEG dcm (P9 pattern)")
    return _decode_seg_dcm(SEG_DCM)


def load_module():
    """Import the pure module directly from file path (no package init,
    so no holoscan/torch-operator imports are triggered)."""
    spec = importlib.util.spec_from_file_location("volume_metrics", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["volume_metrics"] = mod  # required: dataclass introspects sys.modules
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# VERBATIM port of the oracle per-label loop
# (ct-totalsegmentator-map/app_total/segmentation_metrics_operator.py)
# ---------------------------------------------------------------------------
def oracle_metrics(seg_array: np.ndarray, spacing: tuple, num_labels: int = 117) -> dict:
    """Verbatim port of the oracle per-label loop (vol + slice info only;
    intensity stats / connected components are intentionally omitted here
    because the SR gate consumes only {label}.vol).

    Note the oracle evaluates per label:
        volume_per_voxel_mm3 = spacing[0] * spacing[1] * spacing[2]
        volume_ml = float(pixel_count * volume_per_voxel_mm3) / 1000.0
    with pixel_count = np.sum(seg == label).
    """
    is_3d = True
    results = {}
    for label_idx in range(1, num_labels + 1):
        label_mask = seg_array == label_idx

        # Pixel count (oracle: xp.sum(label_mask), xp = np on CPU path)
        pixel_count = np.sum(label_mask)

        # Skip if label not present
        if pixel_count == 0:
            results[label_idx] = {
                "vol": 0.0,
                "num.slices": 0,
                "slice.range": None,
                "pixel.count": 0,
            }
            continue

        # Compute volume or area (oracle _compute_volume_or_area, is_3d branch)
        volume_per_voxel_mm3 = spacing[0] * spacing[1] * spacing[2]
        volume_ml = float(pixel_count * volume_per_voxel_mm3) / 1000.0

        # Slice information (oracle, is_3d branch)
        slices_with_label = np.any(label_mask, axis=(1, 2))
        slice_indices = np.where(slices_with_label)[0]
        num_slices = len(slice_indices)
        slice_range = (int(slice_indices[0]), int(slice_indices[-1])) if num_slices > 0 else None

        results[label_idx] = {
            "vol": float(volume_ml),
            "num.slices": int(num_slices),
            "slice.range": slice_range,
            "pixel.count": int(pixel_count),
        }
    return results


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
def main() -> int:
    failures = []

    def check(name: str, cond: bool, detail: str = ""):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    print("=" * 72)
    print("13-01 bit-identity test: volume_metrics vs oracle per-label loop")
    print("  data: 44238 pin (sha-pinned), spacing", SPACING)
    print("=" * 72)

    seg = load_labelmap()
    print(
        f"labelmap: shape={seg.shape} dtype={seg.dtype} "
        f"max={seg.max()} present={sorted(np.unique(seg)[1:].tolist())}"
    )

    vm = load_module()

    # -- oracle reference (verbatim port) -----------------------------------
    oracle = oracle_metrics(seg, SPACING, NUM_LABELS)

    # -- new module: single pass --------------------------------------------
    res = vm.compute_volume_metrics(seg, SPACING, NUM_LABELS)

    oracle_vol = np.array([oracle[l]["vol"] for l in range(1, NUM_LABELS + 1)], dtype=np.float64)
    new_vol = res.volumes[1 : NUM_LABELS + 1]

    # (1) bit-identical volumes
    check(
        "volumes bit-identical (np.array_equal, labels 1..117)",
        bool(np.array_equal(new_vol, oracle_vol)),
        (
            f"mismatches: {[l for l in range(1, NUM_LABELS + 1) if not new_vol[l - 1] == oracle[l]['vol']]}"
            if not np.array_equal(new_vol, oracle_vol)
            else ""
        ),
    )

    # (2) exact zero-set equality
    zero_new = {l for l in range(1, NUM_LABELS + 1) if res.volumes[l] == 0.0}
    zero_oracle = {l for l in range(1, NUM_LABELS + 1) if oracle[l]["vol"] == 0.0}
    check(
        "zero-set exact",
        zero_new == zero_oracle,
        f"only-new={sorted(zero_new - zero_oracle)} only-oracle={sorted(zero_oracle - zero_new)}",
    )

    # (3) num.slices + slice.range equality for all 117 labels
    ns_bad = [l for l in range(1, NUM_LABELS + 1) if int(res.num_slices[l]) != oracle[l]["num.slices"]]
    sr_bad = []
    for l in range(1, NUM_LABELS + 1):
        a, b = res.slice_range[l], oracle[l]["slice.range"]
        if (a is None) != (b is None) or (a is not None and tuple(a) != tuple(b)):
            sr_bad.append(l)
    check("num.slices equal (all 117)", not ns_bad, f"mismatches: {ns_bad[:10]}")
    check("slice.range equal (all 117)", not sr_bad, f"mismatches: {sr_bad[:10]}")

    # (4) compute_slice_stats: 117-key dict, exact inner keys
    stats = vm.compute_slice_stats(seg, SPACING, NUM_LABELS)
    expected_keys = {"vol", "num.slices", "slice.range", "pixel.count"}
    check(
        "stats dict has exactly 117 keys (labels 1..117)",
        set(stats.keys()) == set(range(1, NUM_LABELS + 1)),
    )
    key_ok = all(set(v.keys()) == expected_keys for v in stats.values())
    check("every entry has keys {vol, num.slices, slice.range, pixel.count}", key_ok)
    vol_match = all(np.float64(stats[l]["vol"]) == np.float64(oracle[l]["vol"]) for l in range(1, NUM_LABELS + 1))
    check("stats dict vols bit-identical to oracle", vol_match)
    pc_match = all(stats[l]["pixel.count"] == oracle[l]["pixel.count"] for l in range(1, NUM_LABELS + 1))
    check("stats dict pixel.count equal to oracle", pc_match)

    # (5) all-background input -> all 117 ORGAN volumes zero, empty nonzero set
    # (background bin 0 legitimately holds every voxel; the metrics contract
    #  covers only organ labels 1..117, matching the legacy op's labels_dict
    #  which excludes "background")
    bg = np.zeros_like(seg)
    res_bg = vm.compute_volume_metrics(bg, SPACING, NUM_LABELS)
    check(
        "all-background: 117 organ volumes zero",
        bool(np.all(res_bg.volumes[1:] == 0.0)),
    )
    check("all-background: nonzero_labels empty", res_bg.nonzero_labels == [])
    check(
        "all-background: organ num.slices zero",
        bool(np.all(res_bg.num_slices[1:] == 0)),
    )
    check(
        "all-background: organ slice.range None",
        all(r is None for r in res_bg.slice_range[1:]),
    )

    # (6) 4-D torch (1,D,H,W) == 3-D
    import torch

    seg_t = torch.from_numpy(seg)  # 3-D CPU tensor (merge_5part's actual contract)
    res_t = vm.compute_volume_metrics(seg_t, SPACING, NUM_LABELS)
    res4 = vm.compute_volume_metrics(seg_t[None, ...], SPACING, NUM_LABELS)  # (1,D,H,W)
    check(
        "4-D torch input == 3-D numpy input (volumes bit-identical)",
        bool(np.array_equal(res4.volumes, res.volumes)),
    )
    check(
        "3-D torch input == 3-D numpy input (volumes bit-identical)",
        bool(np.array_equal(res_t.volumes, res.volumes)),
    )

    # -- per-label table: 5 largest nonzero labels ---------------------------
    present = [l for l in range(1, NUM_LABELS + 1) if oracle[l]["vol"] > 0.0]
    top5 = sorted(present, key=lambda l: -oracle[l]["vol"])[:5]
    print()
    print(f"per-label bit-identity table (top 5 of {len(present)} present labels):")
    print(f"{'label':>6} | {'oracle mL':>16} | {'new mL':>16} | exact")
    print("-" * 62)
    for l in top5:
        ov, nv = oracle[l]["vol"], float(res.volumes[l])
        print(f"{l:>6} | {ov!r:>16} | {nv!r:>16} | {ov == nv}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s) failed: {failures}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
