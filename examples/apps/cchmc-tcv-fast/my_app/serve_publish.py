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
    """Admission-time purge : remove leftover ``<out>.tmp-<uid>`` /
    ``<out>.prev-<uid>`` dirs from a killed run BEFORE the pipeline writes
    (idempotent re-submit). A kill between the two renames of a whole-tree
    publish can leave the ``.prev-`` tree behind; readers never see it (it
    is not the final path), but it must not accumulate. Returns True if
    anything was purged."""
    purged = False
    for stale in (
        stage_dir_for(output_path, study_uid),
        output_path.with_name(output_path.name + ".prev-" + study_uid),
    ):
        if stale.is_symlink():
            _LOG.warning("serve_publish: stale path is a symlink, removing link only: %s", stale)
            stale.unlink()
            purged = True
        elif stale.exists():
            shutil.rmtree(stale, ignore_errors=True)
            _LOG.warning("serve_publish: purged stale path %s (prior run killed)", stale)
            purged = True
    return purged


def publish_study(stage: Path, final: Path) -> None:
    """Atomic publish: ONE rename of the entire staged study tree.

    POSIX directory rename is atomic on the same filesystem (stage is a
    sibling of final by construction, so same-fs). A pre-existing final
    tree (a prior completed run — the idempotent re-submit case) is moved
    aside to ``<final>.prev-<uid>`` first, then the new tree is renamed
    into place, then the old tree is removed. Readers therefore observe
    either the complete old tree or the complete new one — never a
    partial mix. A kill between the two renames briefly shows NO tree at
    the final path; the next admission's purge_stale_stage removes the
    ``.tmp-``/``.prev-`` residue. Raises OSError if the stage root is
    missing (a success-path bug — let it propagate to the egress failure
    handling).
    """
    stage = Path(stage)
    final = Path(final)
    if not stage.is_dir():
        raise OSError(f"publish_study: stage dir missing: {stage}")
    prefix = final.name + ".tmp-"
    if not stage.name.startswith(prefix):
        raise OSError(f"publish_study: stage {stage.name!r} does not match final {final.name!r}")
    prev = final.with_name(final.name + ".prev-" + stage.name[len(prefix) :])
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        if final.is_symlink() or not final.is_dir():
            raise OSError(f"publish_study: final path exists and is not a directory: {final}")
        os.rename(str(final), str(prev))
    try:
        os.rename(str(stage), str(final))
    except OSError:
        # Swap failed: restore the old tree (if we moved it aside) so the
        # final path is left as found, then propagate to egress handling.
        if prev.exists():
            os.rename(str(prev), str(final))
        raise
    if prev.exists():
        shutil.rmtree(prev, ignore_errors=True)
    _LOG.info("serve_publish: published %s -> %s (whole-tree swap)", stage, final)


def discard_stage(stage: Path) -> None:
    """Failure path: drop the staging dir; the final output path is NOT
    touched here ( final-path clear is the caller's, unchanged)."""
    stage = Path(stage)
    if stage.exists():
        shutil.rmtree(stage, ignore_errors=True)
        _LOG.info("serve_publish: discarded stage dir %s (study failed)", stage)
