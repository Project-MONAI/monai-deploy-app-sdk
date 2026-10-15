"""CPU synthetic-fixture test for scripts/check_output_contract.py.

Proves the committed contract script RUNS and DISCRIMINATES without
any GPU or real E2E output:

* Builds a tiny synthetic study: 4-slice 32x32 CT input dir (flat .dcm,
  StudyInstanceUID 1.2.3.4.5.6.7.8.9, spacing 2x2x1 -> voxel volume 4 mm^3),
  a hand-packed 1-bit BINARY SEG matching the SDK highdicom writer output
  (SOP .66.4, SegmentationType=BINARY at (0062,0001), no SegmentRepresentation,
  ImageType DERIVED/PRIMARY, 3 segments, 12 frames = 3x4 slices, segment-major,
  correct descriptors/SCT codes), a Basic Text SR (SOP 88.11) whose 3 volume
  lines are computed from the npy with the exact
  postprocess_operator.calculate_volume_ml formula, plus seg_tcv_dhw.npy
  and emit_meta.json.
* Positive: contract script exits 0 on the valid fixture.
* Negative: SR text tampered by one volume value -> non-zero exit + FAIL.

This is a smoke-proof that the script runs and discriminates, NOT a SEG
writer test (that is an end-to-end job on real studies).

Run (venv, CPU):
    python -m pytest examples/apps/cchmc-tcv-fast/scripts/test_contract_synthetic.py -v
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from pydicom import valuerep
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian

APP = Path(__file__).resolve().parents[1]
SCRIPT = APP / "scripts" / "check_output_contract.py"
STUDY_UID = "1.2.3.4.5.6.7.8.9"
S, W = 4, 32  # slices, in-plane size (8x8 scaled up for non-zero integer mL)

# label -> rows occupied (per slice); rows tile [0,32) without overlap/gaps
LABEL_ROWS = {1: range(0, 5), 2: range(5, 16), 3: range(16, 32)}


def _expected_ml(count: int, voxel_mm3: float = 4.0) -> int:
    """Identical formula to postprocess_operator.calculate_volume_ml."""
    return int(round(count * voxel_mm3 / 1000.0, 2))


def _make_ct(inp: Path) -> None:
    inp.mkdir(parents=True, exist_ok=True)
    for i in range(S):
        ds = Dataset()
        ds.file_meta = Dataset()
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.file_meta.MediaStorageSOPInstanceUID = f"1.2.3.4.10.{i}"
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.SOPInstanceUID = f"1.2.3.4.10.{i}"
        ds.StudyInstanceUID = STUDY_UID
        ds.SeriesInstanceUID = "1.1.1"
        ds.InstanceNumber = i + 1
        ds.PatientID = "SYNTH"
        ds.PatientName = "SYNTH"
        ds.Modality = "CT"
        ds.Rows = W
        ds.Columns = W
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.PixelSpacing = MultiValue(valuerep.DS, ["2", "2"])
        ds.SliceThickness = valuerep.DS("1")
        ds.ImageOrientationPatient = MultiValue(valuerep.DS, [1, 0, 0, 0, 1, 0])
        ds.ImagePositionPatient = MultiValue(valuerep.DS, ["0", "0", str(i)])
        ds.PixelData = b"\x00" * (W * W * 2)
        ds.save_as(str(inp / f"ct_{i:03d}.dcm"), write_like_original=False)


def _make_npy(out: Path) -> np.ndarray:
    seg = np.zeros((S, W, W), dtype=np.uint8)
    for label, rows in LABEL_ROWS.items():
        seg[:, list(rows), :] = label
    np.save(out / "seg_tcv_dhw.npy", seg)
    return seg


def _pack_frame(rows_on) -> bytes:
    row_bytes = []
    for y in range(W):
        b = bytearray((W + 7) // 8)
        for x in range(W):
            if y in rows_on:
                b[x // 8] |= 1 << (x % 8)  # little-endian bit order
        row_bytes.append(bytes(b))
    return b"".join(row_bytes)


def _make_seg(out_seg: Path) -> None:
    # Matches the SDK DICOMSegmentationWriterOperator (highdicom) output:
    # SOP .66.4 (Trackable Segments), SegmentationType=BINARY at (0062,0001),
    # no SegmentRepresentation element, ImageType DERIVED/PRIMARY.
    ds = Dataset()
    ds.file_meta = Dataset()
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.66.4"
    ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.20"
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.66.4"
    ds.SOPInstanceUID = "1.2.3.4.20"
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = "2.1.1"
    ds.Modality = "SEG"
    ds.SegmentationType = "BINARY"  # (0062,0001) — SDK writes representation here
    ds.ImageType = ["DERIVED", "PRIMARY"]
    ds.Rows = W
    ds.Columns = W
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 1
    ds.BitsStored = 1
    ds.HighBit = 0
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.PixelRepresentation = 0
    ds.NumberOfFrames = 3 * S
    ds.SeriesDescription = "AI Generated DICOM SEG; Not for Clinical Use."
    descs = [
        (1, "Heart", "80891009"),  # pydicom codes.SCT.Heart (canonical)
        (2, "Left Lung", "26317004"),
        (3, "Right Lung", "8214007"),
    ]
    ds.SegmentSequence = Sequence()
    for num, label, sct in descs:
        s = Dataset()
        s.SegmentNumber = num
        s.SegmentLabel = label
        s.SegmentType = "SEGMENTATION"
        code = Dataset()
        code.CodeValue = sct
        code.CodingSchemeDesignator = "SCT"
        code.CodeMeaning = label
        s.SegmentedPropertyCategoryCodeSequence = Sequence([code])
        s.SegmentedPropertyTypeCodeSequence = Sequence([code])
        ds.SegmentSequence.append(s)
    # segment-major frames: for each segment, one bitplane frame per slice
    frames = bytearray()
    for s in range(3):
        rows_on = LABEL_ROWS[s + 1]
        for _ in range(S):
            frames += _pack_frame(rows_on)
    ds.PixelData = bytes(frames)
    out_seg.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_seg), write_like_original=False)


def _make_sr(out_sr: Path, seg: np.ndarray, tamper: str | None = None) -> None:
    lines = []
    for label, name in ((1, "Heart"), (2, "Left lung"), (3, "Right lung")):
        vol = _expected_ml(int(np.sum(seg == label)))
        line = f"{name} Volume: {vol} mL"
        if tamper and line.startswith(tamper):
            line = line.replace(f"{vol} mL", f"{vol + 1} mL")
        lines.append(line)
    ds = Dataset()
    ds.file_meta = Dataset()
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.88.11"
    ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.30"
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.88.11"
    ds.SOPInstanceUID = "1.2.3.4.30"
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = "2.2.2"
    ds.Modality = "SR"
    ds.ContentDate = "20260101"
    ds.ContentTime = "000000"
    ds.SeriesDescription = "AI Generated DICOM SR; Not for Clinical Use."
    texts = Sequence()
    for ln in lines:
        t = Dataset()
        t.ValueType = "TEXT"
        t.TextValue = [ln]
        texts.append(t)
    container = Dataset()
    container.ValueType = "CONTAINER"
    container.ContentSequence = texts
    ds.ContentSequence = Sequence([container])
    out_sr.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_sr), write_like_original=False)


def _build_fixture(root: Path, tamper: str | None = None):
    inp = root / "in"
    out = root / "out"
    _make_ct(inp)
    (out / "SEG").mkdir(parents=True, exist_ok=True)
    (out / "SR").mkdir(parents=True, exist_ok=True)
    seg = _make_npy(out)
    hist = {str(int(v)): int(c) for v, c in zip(*np.unique(seg, return_counts=True))}
    meta = {
        "part": "3d_fullres",
        "mode": "batch",
        "label_offset": 0,
        "files": [
            {
                "path": "seg_tcv_dhw.npy",
                "shape": list(seg.shape),
                "dtype": str(seg.dtype),
                "label_histogram": hist,
            }
        ],
        "study_id": STUDY_UID,
        "voxel_volume_mm3": 4.0,
        "wall_time_s": 0.001,
    }
    (out / "emit_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    _make_seg(out / "SEG" / "seg.dcm")
    _make_sr(out / "SR" / "sr.dcm", seg, tamper)
    return inp, out


def _run_contract(out: Path, inp: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(out), str(inp), "--json", str(out / "report.json")],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_contract_passes_on_valid_fixture(tmp_path):
    inp, out = _build_fixture(tmp_path)
    r = _run_contract(out, inp)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "CONTRACT PASS" in r.stdout
    assert "FAIL:" not in r.stdout
    report = json.loads((out / "report.json").read_text())
    assert report["study_uid"] == STUDY_UID
    assert abs(report["voxel_volume_mm3"] - 4.0) < 1e-9
    assert report["npy_shape"] == [S, W, W]


def test_contract_fails_on_tampered_sr(tmp_path):
    inp, out = _build_fixture(tmp_path, tamper="Right lung")
    r = _run_contract(out, inp)
    assert r.returncode != 0
    assert "FAIL" in r.stdout
    assert "CONTRACT PASS" not in r.stdout
