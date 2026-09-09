#!/usr/bin/env python3
"""P14-02: 31322-class (production-scale) writer sweep — gate-decision input.

The Task-1 probe (seg_writer_parallel_probe.py) sweeps on 44238 (4914 frames),
which is BELOW the operator's 10k-frame threshold — the operator keeps that
study serial by policy, so its speedup is not the gate's decision input.
31322 (345 slices x 117 = 40,365 PFFG items, 1.341 GB SEG) is the study the
pool targets: the 45 s serial SEG write it removes is the dominant late
span of the whole app run.

Same method as the 44238 probe, same pure module (dicom_seg_writer_parallel
.save_seg_parallel), same faithful build path. Serial save x3 (median) vs
parallel n in {2,4,8} x3 (median), FULL-FILE sha256 byte-identity per config.

Emission: .planning/phases/14-concurrency-parallel-writer/p14_writer_sweep_31322.json

Headless (CPU only). Read-only on the pin + corpus. Work in /raid/tmp/p14_writer
(kept; record-first).
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pydicom

REPO = "/raid/monai-deploy-app-sdk"
STUDY_DIR = os.path.join(REPO, "examples/apps/totalsegmentator-fast/my_app")
sys.path.insert(0, STUDY_DIR)

PIN_SEG = (
    "/raid/tmp/ts_baseline/08-24-2007-NA-CT-31322/SEG/"
    "1.2.826.0.1.3680043.10.511.3.20806375679788734304513838207421022.dcm"
)
PIN_SHA256 = "219ae359c9346a3c41db9002818263b128369af741e2dd7c89fd603959f6abc1"
DECODE_DIR = "/raid/tmp/p14_writer/decode31322"
STUDY = "/raid/map_outputs/ct_ts_validation/tcia-dcm/cpu/08-24-2007-NA-CT-31322"
WORK = "/raid/tmp/p14_writer"
NS = [2, 4, 8]
REPS = 3
GATE = 1.5


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_seg(labelmap):
    """Faithful replica of the operator's create_dicom_seg build."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-json",
        default=os.path.join(
            REPO,
            ".planning/phases/14-concurrency-parallel-writer/" "p14_writer_sweep_31322.json",
        ),
    )
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)

    pin_sha = sha256_file(PIN_SEG)
    assert pin_sha == PIN_SHA256, f"pin sha mismatch: {pin_sha}"
    print("pin sha OK (31322)", flush=True)

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
        print(r.stdout, flush=True)
        print(r.stderr, file=sys.stderr, flush=True)
        assert r.returncode == 0, "pin decode failed"
    labelmap = np.load(labelmap_path)
    print(f"labelmap {labelmap.shape} max={int(labelmap.max())}", flush=True)

    from dicom_seg_writer_parallel import FRAME_THRESHOLD, save_seg_parallel

    t0 = time.perf_counter()
    seg = build_seg(labelmap)
    t_build = time.perf_counter() - t0
    nframes = int(seg.NumberOfFrames)
    print(
        f"built SEG in {t_build:.2f}s, frames={nframes} "
        f"(threshold={FRAME_THRESHOLD}, "
        f"{'parallel' if nframes >= FRAME_THRESHOLD else 'serial'} at default policy)",
        flush=True,
    )

    # serial reference
    serial_walls = []
    for rep in range(REPS):
        pth = os.path.join(WORK, f"S31322_rep{rep}.dcm")
        t0 = time.perf_counter()
        seg.save_as(pth)
        serial_walls.append(time.perf_counter() - t0)
        print(f"serial rep{rep}: {serial_walls[-1]:.3f}s", flush=True)
    serial_wall = sorted(serial_walls)[REPS // 2]
    serial_path = os.path.join(WORK, "S31322_rep2.dcm")
    serial_sha = sha256_file(serial_path)
    serial_bytes = os.path.getsize(serial_path)
    print(
        f"serial median {serial_wall:.3f}s size {serial_bytes} " f"sha256={serial_sha[:16]}",
        flush=True,
    )

    results = []
    for n in NS:
        walls, pools = [], []
        ok_all = True
        for rep in range(REPS):
            pth = os.path.join(WORK, f"P31322_n{n}_rep{rep}.dcm")
            t0 = time.perf_counter()
            info = save_seg_parallel(seg, pth, nproc=n)
            wall = time.perf_counter() - t0
            pools.append(info.get("pool_s", wall))
            sha = sha256_file(pth)
            ok = sha == serial_sha
            ok_all = ok_all and ok
            walls.append(wall)
            print(
                f"n={n} rep={rep}: wall {wall:.3f}s pool {info.get('pool_s')}s " f"bytes_ok={ok} {info}",
                flush=True,
            )
        med = sorted(walls)[REPS // 2]
        med_pool = sorted(pools)[REPS // 2]
        results.append(
            {
                "n": n,
                "wall_s_median": round(med, 3),
                "pool_s_median": round(med_pool, 3),
                "speedup_vs_serial": round(serial_wall / med, 3),
                "bytes_ok": ok_all,
            }
        )
        print(
            f"n={n}: median {med:.3f}s speedup {serial_wall/med:.3f}x " f"bytes_ok={ok_all}",
            flush=True,
        )

    if not all(r["bytes_ok"] for r in results):
        out = {
            "study": "cpu/08-24-2007-NA-CT-31322",
            "verdict": "BYTE_DIVERGENCE_STOP",
            "results": results,
            "serial_wall_s": round(serial_wall, 3),
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(
            "STOP: byte divergence at 31322 scale — investigate before gate",
            file=sys.stderr,
        )
        sys.exit(3)

    best = max(results, key=lambda r: r["speedup_vs_serial"])
    out = {
        "probe": "p14_writer_sweep_31322",
        "plan": "14-02",
        "study": "cpu/08-24-2007-NA-CT-31322",
        "pin_sha256": pin_sha,
        "pin_sha_ok": True,
        "n_frames": nframes,
        "frame_threshold": FRAME_THRESHOLD,
        "operator_policy_at_default": ("parallel" if nframes >= FRAME_THRESHOLD else "serial"),
        "build_s": round(t_build, 3),
        "reps": REPS,
        "serial_wall_s": round(serial_wall, 3),
        "serial_sha256": serial_sha,
        "serial_bytes": serial_bytes,
        "results": results,
        "recommended_workers": best["n"],
        "verdict": "GATE_PASS" if best["speedup_vs_serial"] >= GATE else "GATE_DEFER",
        "note": (
            "44238 probe (p14_writer_sweep.json) measured 1.386x at n=8 on "
            "4914 frames — below threshold, operator keeps that study serial. "
            "This 31322 sweep (40,365 frames) is the production-scale gate input: "
            "fixed overhead is a much smaller fraction, so the speedup is higher."
        ),
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n31322 recommended N = {best['n']} (speedup " f"{best['speedup_vs_serial']}x); verdict={out['verdict']}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
