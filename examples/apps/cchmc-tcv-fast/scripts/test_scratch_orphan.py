"""CPU tests for the cross-restart orphan scratch-root sweep (no GPU).

The default-mode scratch root is a fresh private mkdtemp dir per process, so
a hard-killed daemon's root (with its study_* residue) is invisible to the
next process's startup sweep. purge_orphan_scratch_roots() reclaims those
abandoned roots; these tests pin the behavior (stale purged, fresh spared,
foreign-uid spared).
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))  # make my_app importable from any cwd

from my_app import serve_scratch  # noqa: E402


def _plant_root(base: Path, fresh: bool) -> Path:
    """Emulate a (killed) daemon's leftover default-mode scratch root."""
    root = Path(tempfile.mkdtemp(prefix=serve_scratch._DEFAULT_PREFIX, dir=base))
    study = root / "study_deadbeef01"
    (study / "temp").mkdir(parents=True)
    (study / "temp" / "sc_overlay.dcm").write_bytes(b"x")
    if not fresh:
        old = time.time() - (serve_scratch._ORPHAN_AGE_S + 60.0)
        os.utime(root, (old, old))
    return root


def test_orphan_sweep_removes_stale_root(tmp_path):
    root = _plant_root(tmp_path, fresh=False)
    purged = serve_scratch.purge_orphan_scratch_roots(base=tmp_path)
    assert purged == 1
    assert not root.exists()


def test_orphan_sweep_spares_fresh_root(tmp_path):
    root = _plant_root(tmp_path, fresh=True)
    purged = serve_scratch.purge_orphan_scratch_roots(base=tmp_path)
    assert purged == 0
    assert root.exists()
    shutil.rmtree(root, ignore_errors=True)


def test_orphan_sweep_spares_foreign_uid(tmp_path, monkeypatch):
    root = _plant_root(tmp_path, fresh=False)
    # A root owned by another user must never be touched.
    monkeypatch.setattr(os, "getuid", lambda: 999999)
    purged = serve_scratch.purge_orphan_scratch_roots(base=tmp_path)
    assert purged == 0
    assert root.exists()
    shutil.rmtree(root, ignore_errors=True)
