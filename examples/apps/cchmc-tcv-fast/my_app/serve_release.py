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

"""Serve-mode condition-release discipline.

Every event_state/notify_scheduler touch in serve mode goes through
ReleaseGuard.release. Set-once per study entry: a second release for the
same entry is a logged NO-OP (guards against double execution). The
transport (direct wake vs dispatcher thread — direct wake was verified
safe across 10k/10k wakes) is an injected callable, so this module itself
contains NO raw condition calls.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = ["ReleaseGuard", "GUARD"]


class ReleaseGuard:
    """Set-once guard around a condition-wake transport.

    ``transport`` is a callable(condition) that applies EVENT_DONE +
    notify (or enqueues it onto a dispatcher, per fact B). The guard owns
    the set-once registry; the transport owns the wake mechanics.
    ``releases``/``noops`` are plain counters for tests/observability.
    """

    def __init__(self, transport: Callable[[Any], None]):
        self._transport = transport
        self._lock = threading.Lock()
        self._released = set()  # id(entry) — set-once registry
        self.releases = 0  # counter for tests/observability
        self.noops = 0

    def release(self, entry: Any, condition: Any) -> bool:
        """Release ``entry``'s condition exactly once.

        Returns True when the transport fired, False (logged NO-OP) when
        this entry was already released.
        """
        with self._lock:
            key = id(entry)
            if key in self._released:
                self.noops += 1
                logger.warning("release guard: duplicate release for entry %s — NO-OP ", key)
                return False
            self._released.add(key)
            self.releases += 1
        self._transport(condition)  # outside the lock — the wake itself
        return True

    def forget(self, entry: Any) -> None:
        """Clear the entry's set-once slot (called after the study's done
        event fires, so a future entry object can be released normally)."""
        with self._lock:
            self._released.discard(id(entry))


# Module-level singleton: the Ingress operator constructs the real guard at
# startup (it owns the transport choice per fact B) and assigns it here.
GUARD: Optional[ReleaseGuard] = None
