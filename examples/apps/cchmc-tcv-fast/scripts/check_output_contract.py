#!/usr/bin/env python3
"""Output-contract check for cchmc-tcv-fast batch runs.

Standalone (stdlib + pydicom + numpy + SimpleITK; NO holoscan/torch): asserts
that a batch output root produced by this app carries the promised artifacts:

1. Presence: <out>/SEG and <out>/SR each hold exactly one .dcm;
   <out>/seg_tcv_dhw.npy and <out>/emit_meta.json exist (catch-all: a
   silent 0-series run produces no SEG/SR and fails here).
2. SEG structure: SOP 1.2.840.10008.5.1.4.1.1.66.4 (SDK highdicom writer
   emits Segmentation-with-Trackable-Segments; the same SOP this script
   asserts for the same stack), 3 segments with SegmentNumber {1,2,3}, SegmentLabel
   {Heart, Left Lung, Right Lung}, per-segment SegmentedPropertyType SCT code
   (80891009/26317004/8214007), (0062,0001) SegmentationType == BINARY (the
   SDK writer writes the representation token there — highdicom
   SegmentationTypeValues.BINARY), BitsAllocated == 1 (binary bitplanes; the
   SDK writer emits NO SegmentRepresentation element), ImageType contains
   PRIMARY (actual: DERIVED/PRIMARY), NumberOfFrames == 3 * slices,
   bitplane decode -> each structure has >= 1 foreground voxel in the
   ORIGINAL grid, and the decoded structures cover exactly {0,1,2}.
3. SR: SOP 1.2.840.10008.5.1.4.1.1.88.11, StudyInstanceUID == input study's,
   ContentSequence text lines are exactly the 3 volume lines.
4. Exact volume recompute: load seg_tcv_dhw.npy (uint8, labels subset of
   {0,1,2,3}); input geometry via SimpleITK (4x4 affine from
   GetDirection/GetSpacing, LPS, as the SDK builds nifti_affine_transform);
   per-label expected = int(round(count * |det(affine[:3,:3])| / 1000.0, 2))
   -- identical formula to postprocess_operator.calculate_volume_ml -- and
   assert EXACT equality with the SR lines. emit_meta.json histogram/shape
   cross-checked against the npy.
5. PASS/FAIL one-liner per check; optional --json report.

Exit 0 iff all checks pass, 1 otherwise, 2 on usage/environment errors.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

# SDK DICOMSegmentationWriterOperator (highdicom) emits Segmentation with
# Trackable Segments — same SOP this script.py asserts for the same stack.
SEG_SOP = "1.2.840.10008.5.1.4.1.1.66.4"
SR_SOP = "1.2.840.10008.5.1.4.1.1.88.11"
N_SEGMENTS = 3
EXPECTED_LABELS = {"Heart", "Left Lung", "Right Lung"}
# Heart code is pydicom's canonical codes.SCT.Heart (80891009 "Heart") — the
# plan literal 51230006 was a transcription error; the app uses
# codes.SCT.Heart, and the L/R lung codes are 26317004/8214007 per research Q1.
EXPECTED_SCT = {"80891009", "26317004", "8214007"}
VOLUME_LINE_RE = re.compile(r"^(Heart|Left lung|Right lung) Volume: [0-9.]+ mL$")
# SR line name -> (npy label value, human SEG segment label)
LINE_TO_LABEL = {
    "Heart": (1, "Heart"),
    "Left lung": (2, "Left Lung"),
    "Right lung": (3, "Right Lung"),
}


def _bitplane_frame(pixel_data: bytes, width: int, height: int) -> list:
    """Decode one 1-bit-per-pixel frame (DICOM little-endian bit order)."""
    import numpy as np

    raw = np.frombuffer(pixel_data, dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder="little")
    row_bytes = (width + 7) // 8
    return [list(bits[i * row_bytes * 8 : (i + 1) * row_bytes * 8][:width].astype(bool)) for i in range(height)]


def _iter_frame_pixels(seg_ds, n_frames: int):
    """Per-frame raw pixel bytes for a 1-bit SEG object.

    Handles both multi-frame encodings: a single PixelData blob (SDK writer
    path) and one FramePixelDataSequence item per frame.
    """
    import numpy as np

    fps = getattr(seg_ds, "FramePixelDataSequence", None)
    if fps is not None:
        return [bytes(it.value) for it in fps]
    raw = np.frombuffer(bytes(seg_ds.PixelData), dtype=np.uint8)
    per_frame = len(raw) // n_frames
    return [bytes(raw[i * per_frame : (i + 1) * per_frame]) for i in range(n_frames)]


def _decode_seg_masks(seg_ds, n_slices: int):
    """Segment-major bitplane decode -> per-slice column projections.

    Returns a dict keyed by 0-based structure index:
        {0: [slice0 flags, slice1 flags, ...], 1: [...], 2: [...]}
    where each slice entry is ONE width-sized bool list: entry[s][x] is True
    iff ANY row of slice s has its column-x bit set (a width-sized column
    projection per slice, not a full (H, W) mask — presence checks only need
    the projection).
    """
    import numpy as np

    width = int(seg_ds.Columns)
    height = int(seg_ds.Rows)
    n_frames = int(seg_ds.NumberOfFrames)
    acc = np.zeros((N_SEGMENTS, n_slices, width), dtype=bool)

    def _project(mask, f):
        seg_idx = f // n_slices
        if seg_idx < N_SEGMENTS:
            acc[seg_idx, f % n_slices] |= mask.any(axis=0)

    def _finish():
        return {i: [acc[i, sl].tolist() for sl in range(n_slices)] for i in range(N_SEGMENTS)}

    try:
        # pydicom decodes 1-bit packing and frame splitting correctly for
        # any width (the manual bitplane fallback below is the last-resort
        # path). Empirically (real 1-bit MONOCHROME1 SEG files from this
        # writer), pydicom returns the stored 1-bit payload as 0/1 with NO
        # MONOCHROME1 inversion for 1-bit data — "raw bit set == in-segment"
        # is preserved, so pixel_array is used directly. (A byte-level
        # inversion would show up as values outside {0,1} and fall through
        # to the raw-storage fallback.)
        frame_data = np.asarray(seg_ds.pixel_array)  # (nframes, h, w), 0/1
        if frame_data.ndim != 3 or frame_data.shape[1:] != (height, width):
            raise ValueError(f"unexpected pixel_array shape {frame_data.shape}")
        if not bool(np.all((frame_data == 0) | (frame_data == 1))):
            raise ValueError(f"unexpected pixel_array values ({frame_data.dtype})")
        for f in range(n_frames):
            _project(frame_data[f] != 0, f)
    except Exception:
        for f, raw in enumerate(_iter_frame_pixels(seg_ds, n_frames)):
            _project(np.asarray(_bitplane_frame(raw, width, height), dtype=bool), f)
    return _finish()


def _walk_sr(seq, text_lines: list) -> None:
    for it in seq:
        if hasattr(it, "TextValue") and it.TextValue:
            tv = it.TextValue
            for t in (tv if isinstance(tv, (list, tuple)) else [tv]):
                text_lines.extend(str(t).splitlines())
        if hasattr(it, "ContentSequence"):
            _walk_sr(it.ContentSequence, text_lines)


def run_contract(out: str, inp: str, require_npy: bool = True) -> tuple[bool, dict, dict]:
    """Run all contract checks. Returns (ok, checks, report_fields).

    require_npy=False (HOLOSCAN_EMIT_NPY=0 runs): skip npy/emit_meta presence,
    the exact volume recompute, and the emit_meta cross-check. The SEG/SR
    structure + SR text-format checks still run (the SR volume lines are
    written by the pipeline regardless of the emit gate).
    """
    import numpy as np
    import pydicom

    out = str(Path(out).resolve())
    inp = str(Path(inp).resolve())
    checks: dict = {}

    # ---------- 1. presence ----------
    seg_files = sorted(glob.glob(os.path.join(out, "SEG", "*.dcm")))
    sr_files = sorted(glob.glob(os.path.join(out, "SR", "*.dcm")))
    npy_path = os.path.join(out, "seg_tcv_dhw.npy")
    meta_path = os.path.join(out, "emit_meta.json")
    checks["seg_exactly_one"] = len(seg_files) == 1
    checks["sr_exactly_one"] = len(sr_files) == 1
    if require_npy:
        checks["npy_present"] = os.path.isfile(npy_path)
        checks["emit_meta_present"] = os.path.isfile(meta_path)
        if not all(
            [checks["seg_exactly_one"], checks["sr_exactly_one"], checks["npy_present"], checks["emit_meta_present"]]
        ):
            return False, checks, {}
    elif not (checks["seg_exactly_one"] and checks["sr_exactly_one"]):
        return False, checks, {}

    # ---------- 2. SEG structure ----------
    seg_ds = pydicom.dcmread(seg_files[0], force=True)
    checks["seg_sop_class"] = str(seg_ds.SOPClassUID) == SEG_SOP
    # The SDK writer sets segmentation_type=highdicom BINARY, which lands in
    # (0062,0001) SegmentationType (verified on real v60 canary output).
    checks["seg_type_BINARY"] = str(getattr(seg_ds, "SegmentationType", "")) == "BINARY"
    segs = list(getattr(seg_ds, "SegmentSequence", []))
    checks["seg_n_segments_eq_3"] = len(segs) == N_SEGMENTS
    checks["seg_numbers_1_2_3"] = sorted(int(s.SegmentNumber) for s in segs) == [1, 2, 3]
    checks["seg_labels_Heart_LeftRightLung"] = {str(s.SegmentLabel) for s in segs} == EXPECTED_LABELS
    sct_ok, sct_codes = True, set()
    for s in segs:
        seq = getattr(s, "SegmentedPropertyTypeCodeSequence", None)
        if not seq:
            sct_ok = False
            break
        code = seq[0]
        sct_codes.add(str(code.CodeValue))
        if str(code.CodingSchemeDesignator) != "SCT":
            sct_ok = False
    checks["seg_sct_codes"] = sct_ok and sct_codes == EXPECTED_SCT
    # Actual SDK ImageType: ['DERIVED', 'PRIMARY'] (no SEGMENTATION token).
    img_type = str(getattr(seg_ds, "ImageType", ""))
    checks["seg_imagetype_primary"] = "PRIMARY" in img_type
    # Binary bitplanes: the SDK writer emits no SegmentRepresentation element;
    # the 1-bit allocation is the representation guarantee (matches real output).
    checks["seg_bitsallocated_1"] = int(getattr(seg_ds, "BitsAllocated", -1)) == 1

    n_slices = len(sorted(glob.glob(os.path.join(inp, "*.dcm"))))
    n_frames = int(getattr(seg_ds, "NumberOfFrames", -1))
    # NOTE: n_frames / n_slices are INFORMATIONAL only — they must not be
    # stored in `checks` (the verdict is `all(v is True ...)` and ints would
    # poison it). They are reported in `fields` below.
    checks["seg_nframes_eq_3x_slices"] = n_frames == N_SEGMENTS * n_slices

    decoded = _decode_seg_masks(seg_ds, n_slices)
    for i, (_lab, _human) in enumerate(LINE_TO_LABEL.values()):
        any_fg = any(any(row) for row in decoded[i])
        checks[f"seg_fg_positive_structure{i + 1}"] = bool(any_fg)
    present = {i + 1 for i in range(N_SEGMENTS) if any(any(row) for row in decoded[i])}
    checks["seg_covers_exactly_3_structures"] = present == {1, 2, 3}

    # ---------- 3. SR ----------
    sr_ds = pydicom.dcmread(sr_files[0], force=True)
    checks["sr_sop_class"] = str(sr_ds.SOPClassUID) == SR_SOP
    in_dir_files = sorted(glob.glob(os.path.join(inp, "*.dcm")))
    in_ds = pydicom.dcmread(in_dir_files[0], force=True)
    study_uid = str(in_ds.StudyInstanceUID)
    checks["sr_study_uid_matches_input"] = str(sr_ds.StudyInstanceUID) == study_uid

    text_lines: list = []
    if hasattr(sr_ds, "ContentSequence"):
        _walk_sr(sr_ds.ContentSequence, text_lines)
    norm = [ln.strip() for ln in text_lines if ln.strip()]
    checks["sr_exactly_3_volume_lines"] = len(norm) == 3
    checks["sr_lines_match_format"] = all(VOLUME_LINE_RE.match(ln) for ln in norm)
    parsed = {}
    for ln in norm:
        m = VOLUME_LINE_RE.match(ln)
        if m:
            parsed[m.group(1)] = float(ln.split("Volume: ")[1].split(" mL")[0])
    checks["sr_lines_all_3_structures"] = set(parsed) == set(LINE_TO_LABEL)

    # ---------- 4. exact volume recompute (npy-gated) ----------
    voxel_volume_mm3 = None
    volumes = {}
    seg_shape = None
    if require_npy:
        seg = np.load(npy_path)
        checks["npy_uint8"] = seg.dtype == np.uint8
        checks["npy_labels_subset_0123"] = {int(v) for v in np.unique(seg)} <= {0, 1, 2, 3}
        checks["npy_shape_matches_input_grid"] = tuple(seg.shape) == (n_slices, int(seg_ds.Rows), int(seg_ds.Columns))
        seg_shape = list(seg.shape)

        img = _read_ct_series(inp, checks)
        if img is None:
            ok = all(v is True for v in checks.values() if isinstance(v, bool))
            return False, checks, {}
        direction = np.asarray(img.GetDirection(), dtype=float).reshape(3, 3)
        spacing = np.asarray(img.GetSpacing(), dtype=float)
        affine = np.eye(4)
        affine[:3, :3] = direction * spacing
        voxel_volume_mm3 = float(abs(np.linalg.det(affine[:3, :3])))

        for name, (lab, _human) in LINE_TO_LABEL.items():
            count = int(np.sum(seg == lab))
            expected = int(round(count * voxel_volume_mm3 / 1000.0, 2))
            volumes[name.lower()] = expected
            sr_val = parsed.get(name)
            if sr_val is None or float(sr_val) != float(expected):
                checks[f"sr_volume_exact_{name}"] = f"FAIL: SR={sr_val} expected={expected}"
            else:
                checks[f"sr_volume_exact_{name}"] = True

        try:
            meta = json.loads(open(meta_path).read())
            files0 = meta["files"][0]
            checks["emit_meta_shape_eq_npy"] = list(files0["shape"]) == list(seg.shape)
            hist = {int(k): int(v) for k, v in files0["label_histogram"].items()}
            actual_hist = {int(v): int(c) for v, c in zip(*np.unique(seg, return_counts=True))}
            checks["emit_meta_histogram_eq_npy"] = hist == actual_hist
        except Exception as e:  # noqa: BLE001
            checks["emit_meta_parseable"] = f"FAIL: {e!r}"

    ok = all(v is True for v in checks.values())
    fields = {
        "seg_sop": str(seg_ds.SOPClassUID),
        "sr_sop": str(sr_ds.SOPClassUID),
        "study_uid": study_uid,
        "voxel_volume_mm3": voxel_volume_mm3,
        "volumes_mL": volumes,
        "npy_shape": seg_shape,
        "seg_nframes": n_frames,
        "seg_n_slices_input": n_slices,
    }
    return ok, checks, fields


def _read_ct_series(inp: str, checks: dict):
    """Read the flat CT series via GDCM (sitk.ReadImage on a bare directory
    fails to select a reader for flat multi-file series)."""
    import SimpleITK as sitk  # noqa: N813

    reader = sitk.ImageSeriesReader()
    try:
        series_files = reader.GetGDCMSeriesFileNames(inp)
    except Exception as e:  # noqa: BLE001
        checks["input_series_readable"] = f"FAIL: GDCM series scan errored: {e!r}"
        return None
    if not series_files:
        checks["input_series_readable"] = "FAIL: no GDCM-readable CT series in input dir"
        return None
    reader.SetFileNames(series_files)
    try:
        return reader.Execute()
    except Exception as e:  # noqa: BLE001
        checks["input_series_readable"] = f"FAIL: GDCM series read errored: {e!r}"
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="cchmc-tcv-fast output contract")
    ap.add_argument("out_dir", help="batch output root (-o dir)")
    ap.add_argument("input_dcm_dir", help="input CT series directory (flat *.dcm)")
    ap.add_argument("--json", dest="report", default=None, help="write JSON report here")
    ap.add_argument(
        "--no-npy",
        action="store_true",
        help="skip npy/emit_meta presence + exact volume recompute (HOLOSCAN_EMIT_NPY=0 runs)",
    )
    args = ap.parse_args()

    # Canonicalize the CLI path arguments up front: resolve symlinks/``..``
    # and require both to be existing directories, so a malformed path
    # fails with a clear message instead of flowing into the file ops below.
    out_root = Path(args.out_dir).resolve()
    in_root = Path(args.input_dcm_dir).resolve()
    for label, p in (("out_dir", out_root), ("input_dcm_dir", in_root)):
        if not p.is_dir():
            ap.error(f"{label} is not an existing directory: {p}")

    t0 = time.monotonic()
    ok, checks, fields = run_contract(str(out_root), str(in_root), require_npy=not args.no_npy)
    for name, res in checks.items():
        if res is True:
            print(f"PASS: {name}")
        else:
            print(f"FAIL: {name}" + (f" -> {res}" if isinstance(res, str) else ""))
    print(f"{'CONTRACT PASS' if ok else 'CONTRACT FAIL'} (study={fields.get('study_uid')})")
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "out": str(out_root),
            "input": str(in_root),
            "checks": {k: ("PASS" if v is True else (v if isinstance(v, str) else "FAIL")) for k, v in checks.items()},
            **fields,
            "wall_time_s": round(time.monotonic() - t0, 3),
        }
        with open(report_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
