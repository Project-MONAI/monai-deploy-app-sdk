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

"""Serve-mode single-slot current-study registry ().

Why a module-level slot instead of graph state: holoscan 4.2's
``NetworkContext`` exposes NO arbitrary user dict keys (live-verified),
so per-study state cannot ride the compute context. Per-study state
therefore lives in this module's locked single slot.

This is safe ONLY because the Ingress admits EXACTLY ONE study at a time
(a serial admitter blocks on the study's done event until Egress
finalizes it). The slot setter is a fail-fast guard: setting a slot that
is already occupied raises ``RuntimeError`` rather than silently
overwriting another study's context (per-study I/O
boundaries).

Lifecycle (one study):

    Ingress.admitter: set_current(ctx) -> wake graph
    writers: get_current (read-only)
    Egress._finalize: clear (after slot consumers are done)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

__all__ = [
    "StudyContext",
    "set_current",
    "get_current",
    "clear",
    "reset_for_tests",
]


@dataclass
class StudyContext:
    """Per-study state for the serve pipeline (one live instance at a time).

    Attributes:
        study_uid: the validated DICOM StudyID (== on-disk StudyID).
        input_path: canonicalized input folder (under MAP_PAYLOADS_ROOT).
        output_path: canonicalized output folder; SEG/ SR/ SC/ written here.
        t_start: ``time.monotonic`` at admission — the latency_ms origin.
        failure: ``(stage, detail)`` of the FIRST failure seen; ``None`` while
            healthy. First failure wins: later stages may not overwrite it.
        serve_seen_stages: distinct writer stages Egress has counted for this
            study . Lives on the per-RPC context rather than in
            Egress keyed by uid string, so reusing the same StudyID across
            studies can never leave stale finalize state behind.
        scratch_path: per-study scratch dir () —
            ``<MAP_SCRATCH_ROOT>/study_<uid>`` (default: a private 0700
            tempdir),
            created by the Ingress at admission; purged by BOTH finalize
            sites on every exit path. ``None`` for studies rejected before
            creation (nothing to purge).
    """

    study_uid: str
    input_path: Path
    output_path: Path
    t_start: float = field(default_factory=time.monotonic)
    failure: Optional[Tuple[str, str]] = None
    serve_seen_stages: set = field(default_factory=set)
    scratch_path: Optional[Path] = None

    def record_failure(self, stage: str, detail: str) -> None:
        """Record the first failure only (first-failure-wins)."""
        if self.failure is None:
            self.failure = (stage, detail)


_SLOT_LOCK = threading.Lock()
_CURRENT: Optional[StudyContext] = None


def set_current(ctx: StudyContext) -> None:
    """Install ``ctx`` as the in-flight study.

    Raises:
        RuntimeError: if another study already occupies the slot. This is a
            bug (admission must be serialized) and must crash loudly, never
            overwrite silently.
    """
    global _CURRENT
    with _SLOT_LOCK:
        if _CURRENT is not None:
            raise RuntimeError(f"study slot occupied by {_CURRENT.study_uid!r}; " f"refusing {ctx.study_uid!r}")
        _CURRENT = ctx


def get_current() -> Optional[StudyContext]:
    """Return the in-flight study context, or ``None`` when idle."""
    with _SLOT_LOCK:
        return _CURRENT


def clear() -> None:
    """Release the slot (idempotent-safe: clearing an empty slot is a no-op)."""
    global _CURRENT
    with _SLOT_LOCK:
        _CURRENT = None


def reset_for_tests() -> None:
    """Clear the slot. Used ONLY by test/gate scripts — never at runtime."""
    clear()
