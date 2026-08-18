#!/usr/bin/env python
"""pixel_diff.py — automated pixel-level DICOM-SEG comparison tool (TEST-003).

Compares two DICOM-SEG outputs (new app vs reference) at two levels:

1. **Raw pixel-data bytes** — byte-identity % over ``PixelData`` (SOP UIDs
   differ between runs, so file names/UIDs are reported but never compared).
2. **Decoded segment voxels** — the 1-bit segment mask is decoded
   (little-endian bit order, per the nnUNet/SDK SEG convention) and compared
   voxel by voxel: per-segment voxel counts, differing voxel count, IoU, and
   the first N differing coordinates ``(frame, row, col)``.

Geometry is checked before any pixel comparison: rows/cols/frames,
pixel representation, samples per pixel, photometric interpretation,
pixel spacing, image orientation/position, and the segment structure.

Pass/fail (exit 0 vs 1):
  * geometry (incl. PixelData length) must match exactly
  * byte-identity % >= ``--min-identity`` (default 99.9)
  * differing voxels  <= ``--max-diff-voxels`` (default 10000)
  * ``--exact``: zero differing voxels and 100% byte-identity

CI-friendly: stable JSON report via ``--json``; exit codes 0 (pass),
1 (divergence beyond tolerance), 2 (usage / input error).

Usage:
  python pixel_diff.py NEW_SEG_DIR REF_SEG_DIR [--min-identity 99.9]
                          [--max-diff-voxels 10000] [--exact]
                          [--json report.json] [--max-listed-diffs 20]

Each SEG path may be a ``.dcm`` file, a directory containing exactly one
SEG instance, or a MAP output dir containing a ``SEG/`` subfolder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fail_usage(msg: str) -> None:
    print(f"pixel_diff: USAGE ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def resolve_seg_file(path: str) -> Path:
    """Resolve a .dcm file / SEG dir / MAP output dir to a single .dcm."""
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() not in (".dcm", ".dicom"):
            fail_usage(f"not a DICOM file: {p}")
        return p
    if not p.is_dir():
        fail_usage(f"path not found: {p}")
    if (p / "SEG").is_dir():
        p = p / "SEG"
    dcms = sorted(x for x in p.iterdir() if x.suffix.lower() in (".dcm", ".dicom"))
    if not dcms:
        fail_usage(f"no .dcm files found under {p}")
    if len(dcms) > 1:
        print(f"pixel_diff: WARNING: {len(dcms)} .dcm files under {p}; "
              f"comparing {dcms[0].name}", file=sys.stderr)
    return dcms[0]


def read_seg(dcm_path: Path):
    """Read a binary (1-bit) multi-frame DICOM-SEG.

    Returns (ds, bits) where bits is a bool ndarray (frames, rows, cols).
    """
    import numpy as np
    from pydicom import dcmread

    try:
        ds = dcmread(str(dcm_path), force=True)
    except Exception as e:  # noqa: BLE001 - reported as usage error
        fail_usage(f"cannot read {dcm_path}: {e}")

    if int(ds.SamplesPerPixel) != 1:
        fail_usage(f"{dcm_path}: SamplesPerPixel={ds.SamplesPerPixel}; this tool "
                   f"compares binary (1-sample) segmentations")
    if int(ds.BitsAllocated) != 1:
        fail_usage(f"{dcm_path}: BitsAllocated={ds.BitsAllocated}; expected 1-bit SEG")

    raw = bytes(ds.PixelData)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
    rows, cols, frames = int(ds.Rows), int(ds.Columns), int(ds.NumberOfFrames)
    if bits.size != rows * cols * frames:
        # tolerate trailing padding to word boundary only
        if bits.size // (rows * cols) != frames:
            fail_usage(f"{dcm_path}: PixelData {len(raw)} bytes does not match "
                       f"{frames}x{rows}x{cols} 1-bit geometry")
        bits = bits[: rows * cols * frames]
    return ds, bits.reshape((frames, rows, cols))


def _lst(v):
    return [float(x) for x in v]


def geometry_report(ds: "object") -> dict:
    """Pixel-relevant geometry (UIDs deliberately excluded)."""
    g = {
        "Rows": int(ds.Rows),
        "Columns": int(ds.Columns),
        "NumberOfFrames": int(ds.NumberOfFrames),
        "BitsAllocated": int(ds.BitsAllocated),
        "BitsStored": int(ds.BitsStored),
        "HighBit": int(ds.HighBit),
        "PixelRepresentation": int(ds.PixelRepresentation),
        "SamplesPerPixel": int(ds.SamplesPerPixel),
        "PhotometricInterpretation": str(ds.PhotometricInterpretation),
        "TransferSyntaxUID": str(ds.file_meta.TransferSyntaxUID),
        "PixelDataLength": len(bytes(ds.PixelData)),
        "PixelSpacing": _lst(ds.PixelSpacing) if hasattr(ds, "PixelSpacing") else None,
        "ImageOrientationPatient": _lst(ds.ImageOrientationPatient)
        if hasattr(ds, "ImageOrientationPatient") else None,
        "ImagePositionPatient": _lst(ds.ImagePositionPatient)
        if hasattr(ds, "ImagePositionPatient") else None,
    }
    if hasattr(ds, "FramePositionSlider"):
        g["FramePositionSlider"] = _lst(ds.FramePositionSlider)
    segs = []
    try:
        for s in ds.SegmentSequence:
            segs.append({
                "SegmentNumber": int(s.SegmentNumber),
                "SegmentLabel": str(s.SegmentLabel),
                "SegmentType": str(s.SegmentType),
                "ReferencedSegmentSequence_frames": None,
            })
    except Exception:  # noqa: BLE001 - non-fatal: reported as geometry diff
        segs = None
    g["Segments"] = segs
    return g


def compare(name_a: str, path_a: str, name_b: str, path_b: str,
            min_identity: float, max_diff_voxels: int, exact: bool,
            max_listed: int):
    """Run the comparison. Returns (exit_code, report_dict)."""
    import numpy as np

    fa, fb = resolve_seg_file(path_a), resolve_seg_file(path_b)
    ds_a, bits_a = read_seg(fa)
    ds_b, bits_b = read_seg(fb)

    geo_a, geo_b = geometry_report(ds_a), geometry_report(ds_b)
    geo_fields = list(geo_a.keys())
    geo_diff = {k: (geo_a[k], geo_b[k]) for k in geo_fields if geo_a[k] != geo_b[k]}
    geo_ok = not geo_diff

    # raw byte identity
    raw_a = bytes(ds_a.PixelData)
    raw_b = bytes(ds_b.PixelData)
    n = min(len(raw_a), len(raw_b))
    same_bytes = int(sum(x == y for x, y in zip(raw_a[:n], raw_b[:n])))
    byte_identity_pct = 100.0 * same_bytes / n if n else 100.0

    # decoded voxel comparison
    total = int(bits_a.size)
    diff_mask = bits_a != bits_b
    n_diff = int(diff_mask.sum())
    n_a = int(bits_a.sum())
    n_b = int(bits_b.sum())
    inter = int((bits_a & bits_b).sum())
    union = int((bits_a | bits_b).sum())
    iou = (inter / union) if union else (1.0 if n_a == n_b == 0 else 0.0)

    diff_coords = []
    if n_diff:
        idx = np.nonzero(diff_mask.reshape(-1))
        flat = [int(i) for i in idx[0][:max_listed]]
        for f_ in flat:
            z = f_ // (bits_a.shape[1] * bits_a.shape[2])
            r = (f_ // bits_a.shape[2]) % bits_a.shape[1]
            c = f_ % bits_a.shape[2]
            diff_coords.append([z, r, c,
                                int(bits_a.reshape(-1)[f_]),
                                int(bits_b.reshape(-1)[f_])])

    reasons = []
    if not geo_ok:
        reasons.append(f"geometry mismatch: {geo_diff}")
    if exact and n_diff != 0:
        reasons.append(f"{n_diff} differing voxels (exact mode requires 0)")
    if exact and byte_identity_pct < 100.0:
        reasons.append(f"byte identity {byte_identity_pct:.5f}% < 100% (exact mode)")
    if not exact:
        if byte_identity_pct < min_identity:
            reasons.append(f"byte identity {byte_identity_pct:.5f}% < {min_identity}%")
        if n_diff > max_diff_voxels:
            reasons.append(f"{n_diff} differing voxels > {max_diff_voxels}")

    passed = not reasons
    report = {
        "a": {"name": name_a, "file": str(fa),
              "SOPInstanceUID": str(ds_a.SOPInstanceUID), "voxels": n_a},
        "b": {"name": name_b, "file": str(fb),
              "SOPInstanceUID": str(ds_b.SOPInstanceUID), "voxels": n_b},
        "geometry": {"match": geo_ok, "differences": geo_diff},
        "pixels": {
            "total_voxels": total,
            "differing_voxels": n_diff,
            "differing_voxel_pct": 100.0 * n_diff / total if total else 0.0,
            "iou": iou,
            "byte_identity_pct": round(byte_identity_pct, 5),
            "diff_coords_zrc_ab": diff_coords,
            "n_diff_coords_listed": len(diff_coords),
        },
        "tolerance": {"min_identity_pct": min_identity,
                      "max_diff_voxels": max_diff_voxels, "exact": exact},
        "pass": passed,
        "fail_reasons": reasons,
    }
    rc = 0 if passed else 1

    print("=" * 72)
    print(f"pixel_diff: {name_a}  vs  {name_b}")
    print("=" * 72)
    print(f"  A: {fa}  (voxels: {n_a})")
    print(f"  B: {fb}  (voxels: {n_b})")
    print(f"  geometry:            {'MATCH' if geo_ok else 'MISMATCH'}"
          + ("" if geo_ok else f"  {geo_diff}"))
    print(f"  byte-identity:       {byte_identity_pct:.5f}%  ({same_bytes}/{n} bytes)")
    print(f"  differing voxels:    {n_diff} / {total} "
          f"({100.0 * n_diff / total:.6f}%)" if total else
          f"  differing voxels:    {n_diff}")
    print(f"  IoU:                 {iou:.6f}")
    if diff_coords:
        print(f"  first {len(diff_coords)} diffs (z, row, col, valA, valB):")
        for d in diff_coords:
            print(f"    {d}")
    print(f"  tolerance:           min-identity {min_identity}%, "
          f"max-diff-voxels {max_diff_voxels}, exact={exact}")
    print(f"  RESULT:              {'PASS' if passed else 'FAIL'}")
    for r in reasons:
        print(f"    - {r}")
    return rc, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DICOM-SEG pixel-level diff tool")
    ap.add_argument("seg_a", help="new app SEG (file, SEG dir, or MAP output dir)")
    ap.add_argument("seg_b", help="reference SEG (file, SEG dir, or MAP output dir)")
    ap.add_argument("--min-identity", type=float, default=99.9,
                    help="min raw byte-identity %% to pass (default 99.9)")
    ap.add_argument("--max-diff-voxels", type=int, default=10000,
                    help="max differing voxels to pass (default 10000)")
    ap.add_argument("--exact", action="store_true",
                    help="require bit-for-bit identity (0 differing voxels)")
    ap.add_argument("--max-listed-diffs", type=int, default=20,
                    help="max differing coordinates to print (default 20)")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="also write the report as JSON to this path")
    args = ap.parse_args(argv)

    try:
        rc, report = compare(
            Path(args.seg_a).name or "A", args.seg_a,
            Path(args.seg_b).name or "B", args.seg_b,
            args.min_identity, args.max_diff_voxels, args.exact,
            args.max_listed_diffs)
    except SystemExit as e:
        return int(e.code)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  json report:           {args.json_out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
