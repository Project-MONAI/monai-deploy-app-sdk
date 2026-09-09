#!/usr/bin/env python3
"""P14-02 Task 4: unit test — byte equality of serial vs parallel SEG save.

On the REAL 44238 labelmap (decoded from the sha256-pinned baseline SEG via
.planning/scripts/p9_seg_decode.py — never fabricated), build the SEG dataset ONCE
via the operator's build path, then:
  * save serial  -> path A (seg.save_as)
  * save parallel -> path B (save_seg_parallel(nproc=4))
and assert FULL-FILE sha256(A) == sha256(B).

Prints exactly:  BYTE-IDENTICAL PASS (<n> bytes)
Exits 0 on pass, 3 on any failure (size / sha mismatch).

Headless (CPU only). Read-only on the pin + corpus. Work in /raid/tmp/p14_writer.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pydicom

REPO = "/raid/monai-deploy-app-sdk"
STUDY_DIR = os.path.join(REPO, "examples/apps/totalsegmentator-fast/my_app")
sys.path.insert(0, STUDY_DIR)

PIN_SEG = (
    "/raid/tmp/ts_baseline/06-20-2009-NA-CT-44238/SEG/"
    "1.2.826.0.1.3680043.10.511.3.9492230222357721157166297047794676.dcm"
)
PIN_SHA256 = "c2d29d208e665e3c0e90256f0cbff02f940ba5818eed1ac6cafe443d2117a2c2"
DECODE_DIR = "/raid/tmp/p14_writer/decode44238"
STUDY = "/raid/map_outputs/ct_ts_validation/tcia-dcm/gpu/06-20-2009-NA-CT-44238"
WORK = "/raid/tmp/p14_writer"
NPNC = 4


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_seg(labelmap):
    """Same build path as the 44238 probe / operator create_dicom_seg."""
    import datetime

    import highdicom as hd
    from ai_segment_descriptions import ai_segment_descriptions
    from pydicom.valuerep import DA, TM

    from monai.deploy.operators.dicom_utils import ModelInfo

    dcms = []
    for f in sorted(
        os.listdir(STUDY),
        key=lambda x: int(pydicom.dcmread(os.path.join(STUDY, x), stop_before_pixels=True).InstanceNumber),
    ):
        dcms.append(pydicom.dcmread(os.path.join(STUDY, f), force=True))
    seruid = str(dcms[0].SeriesInstanceUID)
    seg = hd.seg.Segmentation(
        source_images=dcms,
        pixel_array=labelmap,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[sd.to_segment_description(i + 1) for i, sd in enumerate(ai_segment_descriptions)],
        series_instance_uid=hd.UID(),
        series_number=4321,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer="The MONAI Consortium",
        manufacturer_model_name="MONAI Deploy App SDK",
        software_versions="1.0+62",
        device_serial_number="0000",
        omit_empty_frames=False,
    )
    block = seg.private_block(0x0019, "CCHMC Private", create=True)
    block.add_new(0x01, "UI", seruid)
    dt_now = datetime.datetime.now()
    seg.SeriesDate = DA(dt_now.strftime("%Y%m%d"))
    seg.SeriesTime = TM(dt_now.strftime("%H%M%S"))
    seg.TimezoneOffsetFromUTC = dt_now.astimezone().isoformat()[-6:].replace(":", "")
    seg.update(
        {
            "SeriesDescription": "AI Generated DICOM SEG; Not for Clinical Use.",
            "AlgorithmName": "TotalSegmentator:1.0:0.1.0",
        }
    )
    mi = ModelInfo()
    seq_pr = pydicom.sequence.Sequence()
    d_pr = pydicom.dataset.Dataset()
    d_pr.CodeValue = "Newcode1"
    d_pr.CodingSchemeDesignator = "99IHE"
    d_pr.CodeMeaning = "Processing Algorithm"
    seq_pr.append(d_pr)
    seq_ce = pydicom.sequence.Sequence()
    d_ce = pydicom.dataset.Dataset()
    d_ce.PurposeOfReferenceCodeSequence = seq_pr
    d_ce.Manufacturer = mi.creator
    d_ce.ManufacturerModelName = mi.name
    d_ce.SoftwareVersions = mi.version
    d_ce.DeviceUID = mi.uid
    seq_ce.append(d_ce)
    seg.ContributingEquipmentSequence = seq_ce
    return seg


def main():
    os.makedirs(WORK, exist_ok=True)

    pin_sha = sha256_file(PIN_SEG)
    if pin_sha != PIN_SHA256:
        print(f"FAIL: pin sha mismatch {pin_sha}", file=sys.stderr)
        sys.exit(3)

    labelmap_path = os.path.join(DECODE_DIR, "oracle_total_dhw.npy")
    if not os.path.exists(labelmap_path):
        import subprocess

        r = subprocess.run(
            [
                sys.executable,
                os.path.join(REPO, ".planning/scripts/p9_seg_decode.py"),
                PIN_SEG,
                DECODE_DIR,
                "--expect-sha256",
                PIN_SHA256,
            ],
            capture_output=True,
            text=True,
        )
        print(r.stdout, file=sys.stderr)
        if r.returncode != 0:
            print("FAIL: pin decode failed", file=sys.stderr)
            sys.exit(3)
    labelmap = np.load(labelmap_path)

    seg = build_seg(labelmap)
    nframes = int(seg.NumberOfFrames)

    path_a = os.path.join(WORK, "test_A_serial.dcm")
    path_b = os.path.join(WORK, "test_B_parallel.dcm")

    seg.save_as(path_a)
    from dicom_seg_writer_parallel import save_seg_parallel

    info = save_seg_parallel(seg, path_b, nproc=NPNC, self_check=True)

    size_a = os.path.getsize(path_a)
    size_b = os.path.getsize(path_b)
    sha_a = sha256_file(path_a)
    sha_b = sha256_file(path_b)

    if size_a != size_b:
        print(f"FAIL: size mismatch A={size_a} B={size_b}", file=sys.stderr)
        sys.exit(3)
    if sha_a != sha_b:
        print(f"FAIL: sha mismatch A={sha_a} B={sha_b}", file=sys.stderr)
        sys.exit(3)

    with open(os.path.join(WORK, "test_writer_parallel_result.json"), "w") as f:
        json.dump(
            {
                "n_frames": nframes,
                "nproc": NPNC,
                "bytes": size_a,
                "sha256_A": sha_a,
                "sha256_B": sha_b,
                "self_check": "pass",
                "info": info,
            },
            f,
            indent=2,
        )

    print(f"BYTE-IDENTICAL PASS ({size_a} bytes)")


if __name__ == "__main__":
    main()
