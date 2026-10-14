#!/usr/bin/env python3
# Copyright 2021-2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License").
#
"""Folder-pointing serve client for cchmc-tcv-fast.

Point this at ANY folder of DICOM and it talks to a running `python -m my_app
serve` daemon on the host:

  scripts/serve_folder_client.py --folder <dir> --staging <dir> \
      [--host 127.0.0.1] [--port 8765] [--deadline 600] [--limit 0]

What it does (all user-visible output on stdout):

  1. DISCOVER — walk the folder at depth <= 2 (`patient/series/*.dcm` or flat
     `*.dcm`), read the FIRST .dcm of each series dir with pydicom
     `stop_before_pixels=True`, GROUP the found files by StudyInstanceUID.
     Non-DICOM files are skipped (count recorded).
  2. STUDYID GUARD — the daemon validates the on-disk DICOM `StudyID` against
     the RPC study_uid (NOT the StudyInstanceUID). A group whose files carry
     an EMPTY/MISSING StudyID is NOT submitted; it becomes a loud, typed,
     per-study failure: {ok: false, code: "CLIENT_STUDYID_MISSING", ...}.
  3. STAGE — copy2 (image bytes untouched) into --staging/<stg_<SIU short>>/;
     idempotent (reused if non-empty). Refuses to write outside --staging and
     refuses PROTECTED roots (colon-separated MAP_PROTECTED_PATHS env var).
  4. SUBMIT — sequential unary ProcessStudy per study (the daemon itself is
     the bounded FIFO) with a concurrent health-watch poller (0.5 s) running
     for the whole demo. Per-study line:
       STUDY <SIU> (StudyID <sid>, <n> files): OK latency=<ms> stages=<n> -> <out>
       STUDY <SIU> (StudyID <sid>, <n> files): FAILED code=<code> details=<detail>
  5. SUMMARY — final JSON {studies: [...], ok: n, failed: m, ...};
     exit 0 iff failed == 0 (a CLIENT_STUDYID_MISSING study counts as failed).

One client core: the gRPC calls go through scripts/serve_client.py's command
functions (imported, not re-implemented). The daemon must have a
MAP_PAYLOADS_ROOT that contains your --staging dir (the ingress rejects
paths outside its root) — start it with MAP_PAYLOADS_ROOT=<parent-of-staging>
or ensure <staging> is under it.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --- ONE client core: import serve_client's command functions (proto + grpc) --
_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import pydicom  # noqa: E402
import serve_client  # noqa: E402

APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_DEADLINE_S = 600.0
INITIALIZING_WAIT_S = 600.0  # bounded wait for a still-loading daemon
WATCH_INTERVAL_S = 0.5

# READ-ONLY roots this client must never write under (project policy).
PROTECTED = tuple(p for p in os.environ.get("MAP_PROTECTED_PATHS", "").split(os.pathsep) if p)

MAX_DEPTH = 2  # folder / patient/series / *.dcm


class StagingRefused(RuntimeError):
    """Staging would write outside --staging or under a PROTECTED root."""


@dataclass
class StudyGroup:
    study_uid: str  # StudyInstanceUID (grouping key)
    study_id: Optional[str]  # first non-empty on-disk DICOM StudyID, else None
    series_dirs: list = field(default_factory=list)
    files: list = field(default_factory=list)  # absolute .dcm paths (sorted)

    @property
    def n_files(self) -> int:
        return len(self.files)


# ------------------------------------------------------------- discovery ---


def discover(folder: Path) -> tuple:
    """Walk `folder` at depth <= 2 and group *.dcm files by StudyInstanceUID.

    Returns (groups, skipped_non_dcm, notes). groups sorted by StudyInstanceUID.
    """
    folder = folder.resolve()
    skipped = 0
    notes = []
    series_dirs = []
    dcm_files = []  # (abs_path, series_dir_path)

    for f in sorted(folder.iterdir()):
        if f.is_file():
            if f.suffix.lower() in (".dcm", ".dicom"):
                dcm_files.append((f, folder))
            else:
                skipped += 1

    for patient in sorted(folder.iterdir()):
        if not patient.is_dir() or patient.name.startswith("."):
            continue
        for f in sorted(patient.iterdir()):
            if f.is_file():
                if f.suffix.lower() in (".dcm", ".dicom"):
                    dcm_files.append((f, patient))
                else:
                    skipped += 1
            elif not f.name.startswith("."):
                # series dir at depth 2
                series_dirs.append(f)
                for g in sorted(f.iterdir()):
                    if g.is_file():
                        if g.suffix.lower() in (".dcm", ".dicom"):
                            dcm_files.append((g, f))
                        else:
                            skipped += 1

    # read the FIRST .dcm per series dir; group by StudyInstanceUID
    by_siu: dict = {}
    series_seen = []
    for f, series in dcm_files:
        key = (str(series), str(f))
        if key in series_seen:
            continue
        series_seen.append(key)
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        except Exception as exc:  # noqa: BLE001 — unreadable file: skip, record
            notes.append(f"unreadable DICOM skipped: {f} ({exc!r})")
            skipped += 1
            continue
        siu = str(getattr(ds, "StudyInstanceUID", "") or "").strip()
        if not siu:
            notes.append(f"no StudyInstanceUID — file skipped: {f}")
            skipped += 1
            continue
        g = by_siu.get(siu)
        if g is None:
            g = by_siu[siu] = StudyGroup(study_uid=siu, study_id=None)
        g.series_dirs.append(str(series))
        g.files.append(str(f))
        if g.study_id is None:
            sid = str(getattr(ds, "StudyID", "") or "").strip()
            if sid:
                g.study_id = sid

    groups = sorted(by_siu.values(), key=lambda g: g.study_uid)
    for g in groups:
        g.files = sorted(g.files)
        g.series_dirs = sorted(set(g.series_dirs))
    return groups, skipped, notes


# ---------------------------------------------------------------- staging ---


def _siu_short(siu: str) -> str:
    s = "".join(c if c.isalnum() else "-" for c in siu)[-12:].lstrip("-")
    return s or "00000000"


def stage(group: StudyGroup, staging_root: Path) -> Path:
    """copy2 the group's files into staging_root/stg_<SIU-short>/ (idempotent)."""
    staging_root = staging_root.resolve()
    dest = (staging_root / f"stg_{_siu_short(group.study_uid)}").resolve()
    if not (str(dest) == str(staging_root) or str(dest).startswith(str(staging_root) + os.sep)):
        raise StagingRefused(f"staging target escapes root: {dest}")
    if any(str(dest).startswith(p + os.sep) or str(dest) == p for p in PROTECTED):
        raise StagingRefused(f"staging under PROTECTED root refused: {dest}")
    if dest.exists() and any(dest.iterdir()):
        return dest  # idempotent reuse
    staging_root.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    for src in group.files:
        shutil.copy2(src, str(dest / Path(src).name))
    return dest


# ------------------------------------------------------------- gRPC calls ---


def _health(host: str, port: int, deadline: float = 10.0) -> dict:
    """One CheckHealth via the ONE client core. Returns the parsed JSON dict."""
    args = argparse.Namespace(host=host, port=port, deadline=deadline)
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        serve_client.cmd_health(args)
    return json.loads(buf.getvalue().strip().splitlines()[-1])


def wait_daemon_idle(host: str, port: int, timeout_s: float = INITIALIZING_WAIT_S) -> None:
    """Bounded wait until the daemon reports IDLE (INITIALIZING tolerated)."""
    t0 = time.monotonic()
    last = None
    while time.monotonic() - t0 < timeout_s:
        last = _health(host, port)
        if last.get("status") == "IDLE":
            return
        time.sleep(2.0)
    raise RuntimeError(f"daemon not IDLE after {timeout_s:.0f}s; last={last}")


def _process(group: StudyGroup, staged: Path, out: Path, host: str, port: int, deadline: float) -> dict:
    """One unary ProcessStudy via the ONE client core. Returns its JSON dict."""
    args = argparse.Namespace(
        study_uid=group.study_id, input_path=str(staged), output_path=str(out), host=host, port=port, deadline=deadline
    )
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        serve_client.cmd_process(args)
    return json.loads(buf.getvalue().strip().splitlines()[-1])


def start_health_watch(host: str, port: str, interval_s: float, out_path: Path) -> subprocess.Popen:
    """Concurrent 0.5 s health-watch poller (JSONL to out_path) for the demo."""
    import re

    if not re.match(r"^[A-Za-z0-9.\-:\[\]]{1,255}$", host or ""):
        raise ValueError(f"unsafe --host (allowed: letters, digits, . - : [ ]): {host!r}")
    out_path = Path(out_path).resolve()
    if out_path.is_dir():
        raise ValueError(f"health-watch out_path is a directory: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wf = open(out_path, "wb")
    # NOSONAR: fixed executable (sys.executable), fixed script path, fixed
    # subcommand; host is charset-validated above and port/interval are
    # argparse-typed (int/float). The list-form Popen has no shell, so no
    # argument can inject a command. Sonar's Python taint tracker does not
    # treat re.match as a sanitizer, so the finding is suppressed here.
    return subprocess.Popen(  # NOSONAR
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "serve_client.py"),
            "health-watch",
            "--host",
            host,
            "--port",
            str(port),
            "--interval",
            str(interval_s),
            "--duration",
            "6000",
        ],
        stdout=wf,
        stderr=subprocess.DEVNULL,
    )


# ------------------------------------------------------------------- main ---


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=" folder-pointing client for the cchmc-tcv-fast serve daemon")
    ap.add_argument("--folder", required=True, help="folder of DICOM (patient/series or flat)")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument(
        "--staging", required=True, help="dir for staged inputs + outputs (under the daemon's MAP_PAYLOADS_ROOT)"
    )
    ap.add_argument("--deadline", type=float, default=DEFAULT_DEADLINE_S)
    ap.add_argument("--limit", type=int, default=0, help="max studies to submit (0 = all)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(json.dumps({"ok": False, "code": "CLIENT_FOLDER_MISSING", "details": f"folder not found: {folder}"}))
        return 1
    staging_root = Path(args.staging).resolve()
    if staging_root == folder or str(staging_root).startswith(str(folder) + os.sep):
        print(
            json.dumps(
                {"ok": False, "code": "CLIENT_STAGING_IN_FOLDER", "details": "--staging must not live inside --folder"}
            )
        )
        return 1
    if any(str(staging_root).startswith(p + os.sep) or str(staging_root) == p for p in PROTECTED):
        print(
            json.dumps(
                {
                    "ok": False,
                    "code": "CLIENT_STAGING_PROTECTED",
                    "details": f"--staging under PROTECTED root: {staging_root}",
                }
            )
        )
        return 1

    groups, skipped, notes = discover(folder)
    print(
        f"DISCOVERED studies={len(groups)} files={sum(g.n_files for g in groups)} " f"skipped_non_dcm={skipped}",
        flush=True,
    )
    for n in notes:
        print(f"NOTE {n}", flush=True)

    # submit in discovery order; --limit caps SUBMITTED (valid) studies only
    submittable = [g for g in groups if g.study_id]
    if args.limit > 0:
        submit_set = {g.study_uid for g in submittable[: args.limit]}
    else:
        submit_set = {g.study_uid for g in submittable}

    results_dir = Path(tempfile.gettempdir()) / "map-serve-client-demo"
    # the watch output lands in a temp dir (never deleted)
    watch_path = results_dir / f"health-watch-{int(time.time)}.jsonl"
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        watcher = start_health_watch(args.host, str(args.port), WATCH_INTERVAL_S, watch_path)
    except OSError as exc:
        print(f"NOTE health-watch not started: {exc!r}", flush=True)
        watcher = None

    studies = []
    for g in groups:
        sid_short = _siu_short(g.study_uid)
        if not g.study_id:
            details = (
                f"StudyID empty/missing in {g.series_dirs[0]}; the daemon "
                f"requires the on-disk DICOM StudyID as the RPC study_uid"
            )
            print(
                f"STUDY {sid_short} (StudyID -, {g.n_files} files): "
                f"FAILED code=CLIENT_STUDYID_MISSING details={details}",
                flush=True,
            )
            studies.append(
                {
                    "study_uid": g.study_uid,
                    "study_id": None,
                    "n_files": g.n_files,
                    "ok": False,
                    "code": "CLIENT_STUDYID_MISSING",
                    "details": details,
                }
            )
            continue
        if g.study_uid not in submit_set:
            studies.append(
                {
                    "study_uid": g.study_uid,
                    "study_id": g.study_id,
                    "n_files": g.n_files,
                    "note": "not submitted (--limit)",
                }
            )
            print(
                f"STUDY {sid_short} (StudyID {g.study_id or '-'}, {g.n_files} files): " f"SKIPPED (--limit)", flush=True
            )
            continue
        if not g.study_id:
            details = (
                f"StudyID empty/missing in {g.series_dirs[0]}; the daemon "
                f"requires the on-disk DICOM StudyID as the RPC study_uid"
            )
            print(
                f"STUDY {sid_short} (StudyID -, {g.n_files} files): "
                f"FAILED code=CLIENT_STUDYID_MISSING details={details}",
                flush=True,
            )
            studies.append(
                {
                    "study_uid": g.study_uid,
                    "study_id": None,
                    "n_files": g.n_files,
                    "ok": False,
                    "code": "CLIENT_STUDYID_MISSING",
                    "details": details,
                }
            )
            continue
        try:
            staged = stage(g, staging_root)
            health = _health(args.host, args.port)
            if health.get("status") == "INITIALIZING":
                wait_daemon_idle(args.host, args.port)
            out = staging_root / f"out_{sid_short}"
            rj = _process(g, staged, out, args.host, args.port, args.deadline)
        except StagingRefused as exc:
            print(
                f"STUDY {sid_short} (StudyID {g.study_id}, {g.n_files} files): "
                f"FAILED code=CLIENT_STAGING_REFUSED details={exc}",
                flush=True,
            )
            studies.append(
                {
                    "study_uid": g.study_uid,
                    "study_id": g.study_id,
                    "n_files": g.n_files,
                    "ok": False,
                    "code": "CLIENT_STAGING_REFUSED",
                    "details": str(exc),
                }
            )
            continue
        except Exception as exc:  # noqa: BLE001 — transport/daemon failure per study
            print(
                f"STUDY {sid_short} (StudyID {g.study_id}, {g.n_files} files): "
                f"FAILED code=CLIENT_TRANSPORT details={exc!r}",
                flush=True,
            )
            studies.append(
                {
                    "study_uid": g.study_uid,
                    "study_id": g.study_id,
                    "n_files": g.n_files,
                    "ok": False,
                    "code": "CLIENT_TRANSPORT",
                    "details": repr(exc),
                }
            )
            continue
        if rj.get("ok"):
            stages = len(rj.get("stages") or [])
            print(
                f"STUDY {sid_short} (StudyID {g.study_id}, {g.n_files} files): "
                f"OK latency={rj.get('latency_ms')}ms stages={stages} -> {out}",
                flush=True,
            )
            studies.append(
                {
                    "study_uid": g.study_uid,
                    "study_id": g.study_id,
                    "n_files": g.n_files,
                    "ok": True,
                    "status": rj.get("status"),
                    "latency_ms": rj.get("latency_ms"),
                    "stages": stages,
                    "out": str(out),
                }
            )
        else:
            print(
                f"STUDY {sid_short} (StudyID {g.study_id}, {g.n_files} files): "
                f"FAILED code={rj.get('code')} details={rj.get('details')}",
                flush=True,
            )
            studies.append(
                {
                    "study_uid": g.study_uid,
                    "study_id": g.study_id,
                    "n_files": g.n_files,
                    "ok": False,
                    "code": rj.get("code"),
                    "details": rj.get("details"),
                }
            )

    if watcher is not None:
        watcher.kill()
        watcher.wait(timeout=10)

    ok_n = sum(1 for s in studies if s.get("ok"))
    failed_n = sum(1 for s in studies if s.get("ok") is False)
    summary = {
        "ok": ok_n,
        "failed": failed_n,
        "studies": studies,
        "host": args.host,
        "port": args.port,
        "folder": str(folder),
        "staging": str(staging_root),
        "skipped_non_dcm": skipped,
        "watch_log": str(watch_path) if watcher is not None else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (results_dir / f"client-demo-{int(time.time)}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return 0 if failed_n == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
