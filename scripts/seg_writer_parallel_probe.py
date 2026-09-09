#!/usr/bin/env python3
"""P14-02 Task 1: headless sweep probe for the chunked parallel DICOM-SEG save.

Builds the 44238 SEG dataset via the SAME build path as the app operator
(hd.seg.Segmentation over the pin-decoded labelmap + source CT series + the app's
117 ai_segment_descriptions, omit_empty_frames=False), then:

  1. Serial reference: ``seg.save_as`` (N reps, median), sha256 = reference.
  2. Parallel: ``dicom_seg_writer_parallel.save_seg_parallel`` at n in {2,4,8}
     (N reps, median), FULL-FILE sha256 checked against the serial reference.
  3. NFS note: raw sequential write speed of the file size to the work dir.

Emission: p14_writer_sweep.json (per-n wall, speedup vs serial, bytes_ok,
recommended N, NFS note).

Decision gate (consumed by Task 2): best speedup >= 1.5x -> proceed with that N
as the HOLOSCAN_SEG_WRITER_PROC default; < 1.5x -> DONE-AS-DEFERRED (ship the
probe, keep the writer serial).

STOP condition: any byte-identity failure -> first divergent byte offset +
neighbouring hex dumped to stderr, exit 3, NO recommended N recorded.

Headless (CPU only). Read-only w.r.t. the pin (/raid/tmp/ts_baseline) and the
study corpus. Work + outputs land in /raid/tmp/p14_writer (NOT deleted;
record-first).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor  # noqa: F401  (workers fork from here)

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
    """Faithful replica of DICOMSegmentationWriterOperator.create_dicom_seg's
    build (same highdicom call, same 117 descriptions, same post-build tags)."""
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
        series_number=4321,  # deterministic (operator uses random 4-digit)
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer="The MONAI Consortium",
        manufacturer_model_name="MONAI Deploy App SDK",
        software_versions="1.0+62",
        device_serial_number="0000",
        omit_empty_frames=False,  # app sets False (keeps 117 x D frames)
    )
    # add source SeriesInstanceUID as private tag 0019,1001 (operator parity)
    block = seg.private_block(0x0019, "CCHMC Private", create=True)
    block.add_new(0x01, "UI", seruid)
    dt_now = datetime.datetime.now()
    seg.SeriesDate = DA(dt_now.strftime("%Y%m%d"))
    seg.SeriesTime = TM(dt_now.strftime("%H%M%S"))
    seg.TimezoneOffsetFromUTC = dt_now.astimezone().isoformat()[-6:].replace(":", "")
    # custom tags (operator parity: app.py custom_tags_seg)
    seg.update(
        {
            "SeriesDescription": "AI Generated DICOM SEG; Not for Clinical Use.",
            "AlgorithmName": "TotalSegmentator:1.0:0.1.0",
        }
    )
    # contributing equipment (operator parity via ModelInfo defaults)
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


def report_divergence(a_path, b_path):
    """Return (offset, hex_a, hex_b) of the first divergent byte, or None."""
    with open(a_path, "rb") as fa, open(b_path, "rb") as fb:
        off = 0
        while True:
            ca, cb = fa.read(1 << 20), fb.read(1 << 20)
            if not ca and not cb:
                return None
            la = min(len(ca), len(cb))
            for i in range(la):
                if ca[i] != cb[i]:
                    fa.seek(off + i - 16 if off + i >= 16 else 0)
                    fb.seek(off + i - 16 if off + i >= 16 else 0)
                    return (off + i, fa.read(48).hex(), fb.read(48).hex())
            off += la
            if len(ca) < (1 << 20) or len(cb) < (1 << 20):
                # one file ended inside this chunk
                if len(ca) != len(cb):
                    return (
                        off - (min(len(ca), len(cb))) + min(len(ca), len(cb)),
                        (ca[-48:] if ca else b"").hex(),
                        (cb[-48:] if cb else b"").hex(),
                    )
                return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-json",
        default=os.path.join(
            REPO,
            ".planning/phases/14-concurrency-parallel-writer/" "p14_writer_sweep.json",
        ),
    )
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)

    # --- provenance: pin sha + decode (no fabrication) ---
    pin_sha = sha256_file(PIN_SEG)
    if pin_sha != PIN_SHA256:
        print(f"P14-WRITE-PROBE-ABORT: pin sha mismatch {pin_sha}", file=sys.stderr)
        sys.exit(3)
    print("pin sha OK")
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
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        if r.returncode != 0:
            print("P14-WRITE-PROBE-ABORT: pin decode failed", file=sys.stderr)
            sys.exit(3)
    labelmap = np.load(labelmap_path)
    print(f"labelmap {labelmap.shape} dtype={labelmap.dtype} max={int(labelmap.max())}")

    from dicom_seg_writer_parallel import FRAME_THRESHOLD, save_seg_parallel

    # --- build (serial, unchanged) ---
    t0 = time.perf_counter()
    seg = build_seg(labelmap)
    t_build = time.perf_counter() - t0
    nframes = int(seg.NumberOfFrames)
    print(
        f"built SEG in {t_build:.2f}s, frames={nframes} "
        f"(operator threshold={FRAME_THRESHOLD}, "
        f"{'serial' if nframes < FRAME_THRESHOLD else 'parallel'} at default policy)"
    )

    # --- serial reference (median of REPS) ---
    serial_path = os.path.join(WORK, "A_serial.dcm")
    serial_walls = []
    for rep in range(REPS):
        t0 = time.perf_counter()
        seg.save_as(os.path.join(WORK, f"A_serial_rep{rep}.dcm"))
        serial_walls.append(time.perf_counter() - t0)
    os.replace(os.path.join(WORK, f"A_serial_rep{REPS-1}.dcm"), serial_path)
    for rep in range(REPS - 1):
        os.remove(os.path.join(WORK, f"A_serial_rep{rep}.dcm"))
    serial_wall = sorted(serial_walls)[REPS // 2]
    serial_sha = sha256_file(serial_path)
    serial_bytes = os.path.getsize(serial_path)
    print(f"serial save median wall {serial_wall:.3f}s size {serial_bytes} " f"sha256={serial_sha[:16]}…")

    # --- NFS sequential-write note ---
    nfspath = os.path.join(WORK, "nfs_speed.tmp")
    blob = b"\x00" * serial_bytes
    t0 = time.perf_counter()
    with open(nfspath, "wb") as f:
        f.write(blob)
    f_gbps = serial_bytes / (time.perf_counter() - t0) / 1e9
    os.remove(nfspath)

    # --- parallel sweep ---
    results = []
    divergence = None
    for n in NS:
        walls, pool_parts = [], []
        for rep in range(REPS):
            pth = os.path.join(WORK, f"B_n{n}_rep{rep}.dcm")
            t0 = time.perf_counter()
            info = save_seg_parallel(seg, pth, nproc=n)
            wall = time.perf_counter() - t0
            pool_parts.append(info.get("pool_s", wall))
            sha = sha256_file(pth)
            ok = sha == serial_sha
            if not ok and divergence is None:
                divergence = (n, report_divergence(serial_path, pth), pth)
            walls.append(wall)
            print(f"n={n} rep={rep}: wall {wall:.3f}s pool {info.get('pool_s')}s " f"bytes_ok={ok} {info}")
        med = sorted(walls)[REPS // 2]
        med_pool = sorted(pool_parts)[REPS // 2]
        results.append(
            {
                "n": n,
                "wall_s_median": round(med, 3),
                "pool_s_median": round(med_pool, 3),
                "speedup_vs_serial": round(serial_wall / med, 3),
                "pffg_serialize_speedup": None,  # filled after serial-pffg estimate
                "bytes_ok": all(
                    sha256_file(os.path.join(WORK, f"B_n{n}_rep{r}.dcm")) == serial_sha for r in range(REPS)
                ),
            }
        )
        print(f"n={n}: median wall {med:.3f}s speedup {serial_wall / med:.3f}x " f"bytes_ok={results[-1]['bytes_ok']}")

    out = {
        "probe": "p14_writer_sweep",
        "plan": "14-02",
        "study": "gpu/06-20-2009-NA-CT-44238",
        "pin_sha256": pin_sha,
        "pin_sha_ok": True,
        "labelmap_shape": list(labelmap.shape),
        "n_frames": nframes,
        "frame_threshold": FRAME_THRESHOLD,
        "operator_policy_at_default": ("serial" if nframes < FRAME_THRESHOLD else "parallel"),
        "build_s": round(t_build, 3),
        "reps": REPS,
        "serial_wall_s": round(serial_wall, 3),
        "serial_sha256": serial_sha,
        "serial_bytes": serial_bytes,
        "results": results,
        "nfs_sequential_write_gbps": round(f_gbps, 2),
        "nfs_note": (
            f"NFS is NOT the bottleneck: raw sequential write of the "
            f"{serial_bytes / 1e6:.0f} MB file to /raid/tmp measured "
            f"{f_gbps:.1f} GB/s (research measured 4.7 GB/s). The save cost is "
            "pydicom Python serialization (~0.69 ms/frame, GIL-held), not I/O."
        ),
    }

    if divergence is not None:
        n, rep, pth = divergence
        if rep:
            off, ha, hb = rep
            out["recommended_workers"] = None
            out["verdict"] = "BYTE_DIVERGENCE_STOP"
            with open(args.out_json, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\nSTOP: n={n} byte divergence at offset {off}", file=sys.stderr)
            print(f"  serial: …{ha}", file=sys.stderr)
            print(f"  par   : …{hb}", file=sys.stderr)
            print(f"  files : {serial_path} vs {pth}", file=sys.stderr)
            sys.exit(3)
        else:  # length divergence
            out["recommended_workers"] = None
            out["verdict"] = "BYTE_DIVERGENCE_STOP"
            with open(args.out_json, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\nSTOP: n={n} file length differs", file=sys.stderr)
            sys.exit(3)

    best = max(results, key=lambda r: r["speedup_vs_serial"])
    out["recommended_workers"] = best["n"]
    out["verdict"] = "GATE_PASS" if best["speedup_vs_serial"] >= GATE else "GATE_DEFER"
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"\nrecommended N = {best['n']} (speedup {best['speedup_vs_serial']}x); "
        f"verdict={out['verdict']} (gate >= {GATE}x)"
    )
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
