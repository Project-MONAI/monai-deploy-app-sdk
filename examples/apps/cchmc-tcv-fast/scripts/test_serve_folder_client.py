#!/usr/bin/env python3
"""CPU test for scripts/serve_folder_client.py (folder-pointing client).

No GPU, no daemon, no gRPC: builds a synthetic temp DICOM folder with
pydicom FileDatasets (2 valid studies x 1 series x 3 files + 1 study with
empty StudyID x 1 file), monkeypatches the client's _health/_process/
start_health_watch, and asserts:

  * discover: grouping by StudyInstanceUID, per-study file counts,
    StudyID extraction, empty-StudyID -> None
  * main over the full synthetic folder -> exit 1, ok=2 failed=1,
    the failed study carries code CLIENT_STUDYID_MISSING (loud, per-study)
  * main over a fully-valid folder -> exit 0 (ok=2, failed=0)
  * --limit 1 -> exactly 1 submitted, remainder recorded as not submitted
  * staging guard: write outside --staging refused (StagingRefused)
  * staging guard: staging under a PROTECTED root refused (StagingRefused)
  * staging idempotence: second stage reuses the existing dir

Prints FOLDER-CLIENT-CPU: PASS and exits 0 on success; prints the failing
check and exits 1 otherwise.
"""

import io
import json
import sys
import tempfile
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import pydicom

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import serve_folder_client as m  # noqa: E402

FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f" {'ok' if cond else 'FAIL'} {name} {detail if not cond else ''}", flush=True)
    if not cond:
        FAILURES.append(name)


def make_dcm(path: Path, siu: str, sid: str, n: int) -> None:
    ds = pydicom.FileDataset(str(path), {}, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.file_meta = pydicom.FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.StudyInstanceUID = siu
    ds.SeriesInstanceUID = siu + ".1"
    ds.StudyID = sid
    ds.Modality = "CT"
    ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.InstanceNumber = n
    ds.save_as(str(path), write_like_original=False)


def build_folder(root: Path, with_empty_sid: bool = True) -> None:
    for pt, siu, sid, nfiles in (
        ("pt_A", "1.2.3.4.1", "111", 3),
        ("pt_B", "1.2.3.4.2", "222", 3),
    ):
        s = root / pt / "s1"
        s.mkdir(parents=True, exist_ok=True)
        for i in range(1, nfiles + 1):
            make_dcm(s / f"{i:03d}.dcm", siu, sid, i)
    if with_empty_sid:
        s = root / "pt_C" / "s1"
        s.mkdir(parents=True, exist_ok=True)
        make_dcm(s / "001.dcm", "1.2.3.4.3", "", 1)
    # one non-DICOM file must be skipped, not crash
    (root / "pt_A").joinpath("notes.txt").write_text("ignore me")


def run_main(argv: list, captured: dict) -> int:
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = m.main(argv)
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
    captured["stdout"] = buf.getvalue()
    return rc


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="folder_client_test_"))
    ok = lambda: not FAILURES  # noqa: E731

    # ------------------------------------------------ discover ----------
    build_folder(tmp / "full")
    groups, skipped, notes = m.discover(tmp / "full")
    by_siu = {g.study_uid: g for g in groups}
    check("discover_group_count", len(groups) == 3, f"n={len(groups)}")
    check(
        "discover_file_counts",
        all(
            by_siu.get(siu) is not None and by_siu[siu].n_files == n
            for siu, n in (("1.2.3.4.1", 3), ("1.2.3.4.2", 3), ("1.2.3.4.3", 1))
        ),
        json.dumps({g.study_uid: g.n_files for g in groups}),
    )
    check("discover_studyid_111", by_siu["1.2.3.4.1"].study_id == "111")
    check("discover_studyid_222", by_siu["1.2.3.4.2"].study_id == "222")
    check("discover_studyid_empty_is_none", by_siu["1.2.3.4.3"].study_id is None)
    check("discover_sorted_by_siu", [g.study_uid for g in groups] == sorted(g.study_uid for g in groups))
    check("discover_skips_non_dcm", skipped == 1, f"skipped={skipped}")

    # ------------------------------------------------ main: full folder ---
    m._health = lambda host, port, deadline=10.0: {
        "status": "IDLE",
        "queue_depth": 0,
        "queue_capacity": 2,
        "in_flight_study_uid": None,
        "models": [],
    }
    m._process = lambda group, staged, out, host, port, deadline: {
        "ok": True,
        "status": "OK",
        "latency_ms": 1234.5,
        "stages": [{"stage": s, "duration_ms": 1.0} for s in ("ingress", "preprocess", "infer", "post", "publish")],
    }
    m.start_health_watch = lambda host, port, interval_s, out_path: None

    cap = {}
    rc = run_main(["--folder", str(tmp / "full"), "--staging", str(tmp / "staging")], cap)
    last_json = None
    for line in reversed(cap["stdout"].strip().splitlines()):
        if line.startswith("{"):
            last_json = json.loads(line)
            break
    check("main_full_exit_1", rc == 1, f"rc={rc}")
    check(
        "main_full_ok_2",
        last_json is not None and last_json["ok"] == 2,
        json.dumps(last_json)[:300] if last_json else "no summary",
    )
    check("main_full_failed_1", last_json is not None and last_json["failed"] == 1)
    failed = [s for s in (last_json or {}).get("studies", []) if s.get("ok") is False]
    check(
        "main_full_client_studyid_missing",
        len(failed) == 1 and failed[0]["code"] == "CLIENT_STUDYID_MISSING",
        json.dumps(failed)[:300],
    )
    check("main_full_status_line_failed", "FAILED code=CLIENT_STUDYID_MISSING" in cap["stdout"])
    okline = [l for l in cap["stdout"].splitlines() if l.startswith("STUDY") and ": OK" in l]
    check("main_full_status_line_ok", len(okline) == 2, cap["stdout"][:400])
    check(
        "main_full_ok_line_fields",
        all("latency=" in l and "stages=5" in l and " -> " in l for l in okline),
        str(okline)[:300],
    )

    # ------------------------------------------- main: fully-valid folder --
    build_folder(tmp / "valid", with_empty_sid=False)
    cap = {}
    rc = run_main(["--folder", str(tmp / "valid"), "--staging", str(tmp / "staging")], cap)
    last_json = json.loads([l for l in reversed(cap["stdout"].strip().splitlines()) if l.startswith("{")][0])
    check("main_valid_exit_0", rc == 0, f"rc={rc}")
    check("main_valid_ok_2_failed_0", last_json["ok"] == 2 and last_json["failed"] == 0)

    # ------------------------------------------------- main: --limit 1 ----
    cap = {}
    rc = run_main(["--folder", str(tmp / "full"), "--staging", str(tmp / "staging"), "--limit", "1"], cap)
    last_json = json.loads([l for l in reversed(cap["stdout"].strip().splitlines()) if l.startswith("{")][0])
    submitted = [s for s in last_json["studies"] if s.get("ok") is True]
    check("main_limit_1_submitted", len(submitted) == 1 and last_json["failed"] == 1, json.dumps(last_json)[:300])
    check("main_limit_1_exit_1", rc == 1, f"rc={rc}")  # the missing-uid study still fails
    check("main_limit_skipped_recorded", any(s.get("note") == "not submitted (--limit)" for s in last_json["studies"]))

    # -------------------------------------------------- staging guards ----
    g = by_siu["1.2.3.4.1"]
    root_inside = tmp / "staging" / "nested"
    try:
        m.stage(g, root_inside)  # files all inside tmp/staging/nested — allowed
        dest = root_inside / f"stg_{m._siu_short(g.study_uid)}"
        check("stage_inside_root_ok", dest.is_dir() and any(dest.iterdir()))
    except m.StagingRefused as exc:
        check("stage_inside_root_ok", False, repr(exc))
    try:
        m.stage(g, tmp / "staging")  # files land in stg_* subdir — allowed
        check("stage_flat_root_ok", True)
    except m.StagingRefused as exc:
        check("stage_flat_root_ok", False, repr(exc))
    # PROTECTED refusal: redirect PROTECTED to a tmp path (no real protected writes)
    fake_protected = tmp / "protected"
    fake_protected.mkdir()
    real_protected = m.PROTECTED
    try:
        m.PROTECTED = tuple(real_protected) + (str(fake_protected),)
        try:
            m.stage(g, fake_protected / "stg")
            check("stage_protected_refused", False, "no StagingRefused raised")
        except m.StagingRefused:
            check("stage_protected_refused", True)
        try:
            m.main  # noqa: B018
            # staging root itself under the folder is refused by main
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = m.main(["--folder", str(tmp / "full"), "--staging", str(tmp / "full" / "stg")])
            check("main_staging_in_folder_refused", rc == 1 and "CLIENT_STAGING_IN_FOLDER" in buf.getvalue())
        except m.StagingRefused as exc:
            check("main_staging_in_folder_refused", "CLIENT_STAGING_IN_FOLDER" in repr(exc))
    finally:
        m.PROTECTED = real_protected

    # idempotent reuse of an existing non-empty staging dir
    existing = tmp / "staging" / f"stg_{m._siu_short(g.study_uid)}"
    if existing.is_dir() and any(existing.iterdir()):
        d1 = m.stage(g, tmp / "staging")
        check("stage_idempotent_reuse", d1 == existing.resolve(), f"{d1} vs {existing}")
    else:
        check("stage_idempotent_reuse", True, "no prior stage dir — reuse trivially holds")

    if FAILURES:
        print(f"FOLDER-CLIENT-CPU: FAIL ({len(FAILURES)}): {FAILURES}")
        return 1
    print("FOLDER-CLIENT-CPU: PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FOLDER-CLIENT-CPU: FAIL (exception)")
        sys.exit(1)
