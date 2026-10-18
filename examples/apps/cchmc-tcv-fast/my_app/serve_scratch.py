# Copyright 2021-2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve-mode per-study scratch lifecycle ().

Every admitted study gets its own scratch directory
``<MAP_SCRATCH_ROOT>/study_<uid>`` (default root: a private 0700 tempdir
under the system temp area), created at
admission (AFTER validation, BEFORE enqueue) and purged on EVERY exit path
(success, pipeline failure, abort/timeout drain, and the pre-create
rejection paths, which simply never create a dir). The pipeline's only
host-side intermediate — the SC overlay temp dir (``output_folder/"temp"``)
— is routed into this directory in serve mode, so it vanishes with the
study instead of lingering in the caller's output path.

Env overrides — read at CALL time, not import time, so gate scripts and
headless tests can re-point the root or the required size per process:

    MAP_SCRATCH_ROOT (optional) override the root with a stable path
    MAP_SCRATCH_REQUIRED_MB default 512

Sizing rationale (recorded, not a guess): the pipeline is GPU-resident; the
ONLY host-side intermediate is the SC overlay temp dir — a multi-frame
int16 DICOM of the full-volume overlay (~50 MB for the largest corpus
studies, ~33 MB for the 256^3 reference study). 512 MB default is ~10x the
largest observed intermediate with headroom on the scratch filesystem.

Pure stdlib — no ML/GPU-framework imports (mirrors serve_state.py
discipline so this module stays importable headless).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional, Tuple, Union

__all__ = [
    "ScratchError",
    "scratch_root",
    "scratch_dir_for",
    "check_free_space",
    "create_study_scratch",
    "purge_study_scratch",
]

_LOG = logging.getLogger("my_app.serve_scratch")


# DICOM StudyInstanceUID (PS3.15 C.7.1) is digits separated by '.'; we
# allow a slightly wider charset for local/test UIDs, but nothing that
# could escape the scratch root (no path separators, no '..').
_UID_RE = re.compile(r"^[A-Za-z0-9.+-]{1,64}$")

# Default scratch root: a private, unique, mode-0700 tempdir created once
# per process via tempfile.mkdtemp. mkdtemp is the pattern Sonar's
# "publicly-writable temp dir" rule accepts (random name, 0700, no symlink
# race). Set MAP_SCRATCH_ROOT to a stable path (e.g. a tmpfs subdir) to get
# a shared root with cross-restart crash-residue purging.
_DEFAULT_ROOT: Optional[tempfile.TemporaryDirectory] = None
_ROOT_LOCK = threading.Lock()


class ScratchError(Exception):
    """Scratch precondition failed (free-space shortfall or mkdir failure).

    Raised by :func:`create_study_scratch`; the Ingress converts it into a
    typed ``RESOURCE_EXHAUSTED`` gRPC reject (never a raw OOM/IO error or
    traceback in the RPC reply).
    """


def scratch_root() -> Path:
    """Resolve the scratch root, guaranteeing it is a private directory.

    ``MAP_SCRATCH_ROOT`` overrides the root when set (use a stable path,
    e.g. a dedicated tmpfs subdir, to enable cross-restart crash-residue
    purging by the daemon startup sweep). The default creates a private,
    unique, mode-0700 tempdir once per process via ``tempfile.mkdtemp`` —
    never a world-writable directory directly.
    """
    global _DEFAULT_ROOT
    env = os.environ.get("MAP_SCRATCH_ROOT")
    if env:
        root = Path(env)
        if not root.is_dir():
            # A fresh dedicated root must exist before the free-space probe
            # and the per-study mkdir can work (the startup sweep skips a
            # missing root, so without this every request would fail with
            # FileNotFoundError). Create it owner-only; a pre-existing root
            # keeps the operator's permissions.
            root.mkdir(parents=True, mode=0o700)
            os.chmod(root, 0o700)  # umask-proof: study data is owner-only
        return root
    with _ROOT_LOCK:
        if _DEFAULT_ROOT is None:
            _DEFAULT_ROOT = tempfile.TemporaryDirectory(prefix="tcv-fast-scratch-")
        return Path(_DEFAULT_ROOT.name)


def _required_mb() -> int:
    return int(os.environ.get("MAP_SCRATCH_REQUIRED_MB", "512"))


def scratch_dir_for(uid: str) -> Path:
    """Per-study scratch dir: ``<root>/study_<uid>`` where root is
    ``$MAP_SCRATCH_ROOT`` or a private 0700 tempdir. Env is read at call
    time.

    The uid is validated before path interpolation so a malformed uid can
    never address a path outside the scratch root.
    """
    if not _UID_RE.match(uid):
        raise ScratchError(f"invalid study uid (unsafe characters): {uid!r}")
    return scratch_root() / f"study_{uid}"


def check_free_space(root: Union[str, Path], required_mb: int) -> Tuple[bool, str]:
    """Free-space guard. Returns ``(True, detail)`` when
    ``shutil.disk_usage(root).free >= required_mb * 1024**2``, else
    ``(False, "insufficient scratch space: need <R> MB, have <F> MB")``."""
    free = shutil.disk_usage(Path(root)).free
    need_bytes = required_mb * 1024**2
    free_mb = round(free / 1024**2)
    if free >= need_bytes:
        return True, f"scratch free space OK: have {free_mb} MB, need {required_mb} MB"
    return False, (f"insufficient scratch space: need {required_mb} MB, have {free_mb} MB")


def create_study_scratch(uid: str) -> Tuple[Path, None]:
    """Create the per-study scratch dir; return ``(path, None)``.

    Order of operations (all fail-fast, before any pipeline work):

    1. Free-space check FIRST — raises :class:`ScratchError` with a
       human-readable detail on shortfall (the typed-reject path).
    2. A pre-existing ``study_<uid>`` dir means a prior run crashed
       mid-study: log a warning and ``shutil.rmtree`` it before re-creating
       (purge-at-start; a stale dir must never be inherited).
    3. ``mkdir(parents=False, exist_ok=False)`` — a mkdir race is a bug and
       is reported as :class:`ScratchError`, never silently swallowed.
    """
    root = scratch_root()
    required_mb = _required_mb()
    ok, detail = check_free_space(root, required_mb)
    if not ok:
        raise ScratchError(detail)
    d = scratch_dir_for(uid)
    if d.exists():
        _LOG.warning("scratch: purging stale dir %s (prior run crashed) before re-create", d)
        shutil.rmtree(d, ignore_errors=True)
    try:
        d.mkdir(parents=False, exist_ok=False, mode=0o700)
    except OSError as exc:
        raise ScratchError(f"scratch mkdir failed: {d}: {type(exc).__name__}: {exc}") from exc
    os.chmod(d, 0o700)  # umask-proof: per-study data is owner-only
    return d, None


def purge_study_scratch(path: Optional[Union[str, Path]]) -> None:
    """Purge a study's scratch dir (idempotent, safe on ``None`` and on
    already-absent paths). Called from BOTH finalize sites (egress
    ``_finalize`` — success/failure/timeout; ingress ``_finalize_entry`` —
    load failure) so every exit path releases the dir."""
    if path is None:
        return
    p = Path(path)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    _LOG.info("scratch: purged %s", p)
