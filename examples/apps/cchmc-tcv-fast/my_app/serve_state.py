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

"""Serve-mode daemon state shared between gRPC threads (writers) and the
graph (mutators).

Read by ``CheckHealth`` from gRPC handler threads; mutated by the Ingress
admitter (begin) and the Egress finalize (end). All access is under one
lock; ``snapshot`` returns a plain-dict copy so readers never hold the
lock across a gRPC response.

No ML/GPU-framework imports in this module — CheckHealth must answer fast
even while a study is BUSY and the accelerator is under load
(the architecture doc rule 1).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

__all__ = ["DaemonState", "STATE"]

INITIALIZING = "INITIALIZING"
IDLE = "IDLE"
BUSY = "BUSY"


class DaemonState:
    """Thread-safe daemon lifecycle state.

    State machine: ``INITIALIZING -> IDLE -> BUSY -> IDLE -> ...``

    Illegal transitions raise ``RuntimeError`` (fail fast: admission must
    never race).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status: str = INITIALIZING
        self.in_flight_study_uid: Optional[str] = None
        self.queue_depth: int = 0
        self.queue_capacity: int = 4  # overwritten from MAP_GRPC_QUEUE_DEPTH at Ingress init

    # -- transitions (mutators) ------------------------------------------------

    def mark_ready(self) -> None:
        """INITIALIZING -> IDLE. Called once at the END of ``_compose_serve``
        (models loaded, graph built). Any other transition is a bug."""
        with self._lock:
            if self.status != INITIALIZING:
                raise RuntimeError(f"mark_ready from status {self.status!r}; expected {INITIALIZING!r}")
            self.status = IDLE

    def begin_study(self, uid: str) -> None:
        """IDLE -> BUSY, recording the in-flight uid.

        Called by the Ingress admitter after ``set_current`` and BEFORE the
        scheduler wake. Raises when not IDLE (admission must never race).
        """
        with self._lock:
            if self.status != IDLE:
                raise RuntimeError(f"begin_study({uid!r}) from status {self.status!r}; expected {IDLE!r}")
            self.status = BUSY
            self.in_flight_study_uid = uid

    def end_study(self) -> None:
        """BUSY -> IDLE, clearing the in-flight uid.

        Called by Egress ``_finalize`` (and the Ingress-side load-failure
        path) AFTER the study slot is cleared — the slot and STATE must be
        released before the admitter is unblocked.
        """
        with self._lock:
            if self.status != BUSY:
                raise RuntimeError(f"end_study from status {self.status!r}; expected {BUSY!r}")
            self.status = IDLE
            self.in_flight_study_uid = None

    def reset(self) -> None:
        """Return to INITIALIZING so a re-compose can run mark_ready again.

        Called by the serve_daemon in-process supervisor before it
        composes a FRESH app instance after a graph error. Without this, the
        module singleton is already past INITIALIZING and the second compose's
        mark_ready raises.
        """
        with self._lock:
            self.status = INITIALIZING
            self.in_flight_study_uid = None
            self.queue_depth = 0

    def set_queue_depth(self, n: int) -> None:
        """Update the waiting-queue depth (admitter/handler bookkeeping)."""
        with self._lock:
            self.queue_depth = int(n)

    def set_queue_capacity(self, n: int) -> None:
        """Set the queue capacity (from MAP_GRPC_QUEUE_DEPTH at Ingress init)."""
        with self._lock:
            self.queue_capacity = int(n)

    # -- readers ----------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Plain-dict copy for CheckHealth (status, uid, depth, capacity)."""
        with self._lock:
            return {
                "status": self.status,
                "in_flight_study_uid": self.in_flight_study_uid,
                "queue_depth": self.queue_depth,
                "queue_capacity": self.queue_capacity,
            }


# Module singleton: one daemon per process.
STATE = DaemonState()
