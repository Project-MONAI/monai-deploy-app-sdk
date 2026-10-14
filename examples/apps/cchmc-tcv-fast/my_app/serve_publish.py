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

"""Serve-mode atomic output publish + stale state purge.

The final output path is written ONLY by publish_study; mid-pipeline work
goes to the per-study staging dir.

Crash-safety by construction: under the stock no-trap PID-1 tools.sh
entrypoint a docker kill/stop is a hard kill — the daemon never receives
SIGTERM, so it cannot clean up. Therefore:

1. Writers write into ``<output_path>.tmp-<study_uid>`` (a same-filesystem
   SIBLING of the final output path). A kill at ANY point before finalize
   leaves zero files at the final output path — the partial work lives in
   the staging sibling, never a partial tree in the published location.
2. Egress ``_finalize`` publishes on success via per-subdir ``os.rename``
   (POSIX-atomic on the same filesystem, which siblings guarantee).
3. Purge-at-start: daemon startup sweeps ``<MAP_SCRATCH_ROOT>/study_*``
   (a fresh daemon owns no study state); admission additionally purges a
   stale ``<out>.tmp-<uid>`` staging dir left by a killed run.
4. Failure semantics: on ``ctx.failure`` the stage is discarded AND the
   final output path is rmtree'd, preserving the "failed run leaves no
   outputs" contract.

Pure stdlib — mirrors serve_scratch.py discipline (importable headless, no
ML/GPU-framework imports).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

__all__ = [
    "stage_dir_for",
    "sweep_stale_scratch",
    "purge_stale_stage",
    "publish_study",
    "discard_stage",
]

_OUTPUT_SUBDIRS = ("SEG", "SR", "SC")

# Module logger with its OWN stderr handler (serve_scratch.py uses bare
# getLogger and relies on the root logger — which in the daemon is only
# configured by holoscan INSIDE app.run, AFTER the startup sweep in
# serve_daemon.run_serve fires). A dedicated handler guarantees the
# `lifecycle: startup scratch sweep purged=<N>` line (and the publish /
# stale-stage lines) reach docker logs unconditionally; propagate=False
# keeps them from double-printing once holoscan configures the root.
_LOG = logging.getLogger("my_app.serve_publish")
if not _LOG.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] (%(name)s) - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    _LOG.addHandler(_h)
    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False


# Strict name pattern for per-study dirs (mirrors serve_scratch._UID_RE).
# The startup sweep only removes real directories whose names it fully
# controls — never symlinks, never unexpected names.
_STUDY_DIR_RE = re.compile(r"^study_[A-Za-z0-9.+-]{1,64}$")


def stage_dir_for(output_path: Path, study_uid: str) -> Path:
    """Per-study staging dir: ``<output_path>.tmp-<study_uid>``, a SIBLING of
    the final output path on the SAME filesystem, so ``os.rename`` publish is
    atomic (cross-device rename is impossible by construction)."""
    return output_path.with_name(output_path.name + ".tmp-" + study_uid)


def sweep_stale_scratch() -> int:
    """Daemon-startup purge : remove EVERY ``<MAP_SCRATCH_ROOT>/study_*``
    dir. A fresh daemon owns no study state; any leftover is crash residue
    from a prior (killed) process. Returns the count purged. Logs
    unconditionally (even for 0 purged — it is the startup evidence line)."""
    from my_app.serve_scratch import scratch_root

    root = scratch_root()
    purged = 0
    if root.exists():
        for p in sorted(root.iterdir()):
            if p.is_dir() and not p.is_symlink() and _STUDY_DIR_RE.match(p.name):
                shutil.rmtree(p, ignore_errors=True)
                purged += 1
    _LOG.info("lifecycle: startup scratch sweep purged=%d (root=%s)", purged, root)
    return purged


def purge_stale_stage(output_path: Path, study_uid: str) -> bool:
    """Admission-time purge : remove a leftover ``<out>.tmp-<uid>``
    from a killed run BEFORE the pipeline writes (idempotent re-submit).
    Returns True if one was purged."""
    stage = stage_dir_for(output_path, study_uid)
    if stage.is_symlink():
        _LOG.warning("serve_publish: stale stage is a symlink, removing link only: %s", stage)
        stage.unlink()
        return True
    if stage.exists():
        shutil.rmtree(stage, ignore_errors=True)
        _LOG.warning("serve_publish: purged stale stage dir %s (prior run killed)", stage)
        return True
    return False


def publish_study(stage: Path, final: Path) -> None:
    """Atomic publish: per output subdir, ``os.rename(<stage>/<SUB> -> <final>/<SUB>)``.

    POSIX directory rename is atomic on the same filesystem (sibling dirs
    are guaranteed same-fs). A pre-existing final subdir (a prior completed
    run — the idempotent re-submit case) is removed and replaced. Each
    rename is all-or-nothing, so the final path never holds a partial tree.
    The stage dir is removed afterwards. Raises OSError if the stage root
    is missing (a success-path bug — let it propagate to the egress
    failure handling).
    """
    stage = Path(stage)
    final = Path(final)
    if not stage.is_dir():
        raise OSError(f"publish_study: stage dir missing: {stage}")
    final.mkdir(parents=True, exist_ok=True)
    for sub in _OUTPUT_SUBDIRS:
        s = stage / sub
        if not s.is_dir():
            continue  # success path implies all three exist; skip is defensive
        f = final / sub
        if f.exists():
            shutil.rmtree(f, ignore_errors=True)
        os.rename(str(s), str(f))
    shutil.rmtree(stage, ignore_errors=True)
    _LOG.info(
        "serve_publish: published %s -> %s (subdirs: %s)",
        stage,
        final,
        ", ".join(_OUTPUT_SUBDIRS),
    )


def discard_stage(stage: Path) -> None:
    """Failure path: drop the staging dir; the final output path is NOT
    touched here ( final-path clear is the caller's, unchanged)."""
    stage = Path(stage)
    if stage.exists():
        shutil.rmtree(stage, ignore_errors=True)
        _LOG.info("serve_publish: discarded stage dir %s (study failed)", stage)
