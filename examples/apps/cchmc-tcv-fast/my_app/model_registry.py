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

"""Loaded-model residency registry for CheckHealth.

Populated via an OPTIONAL hook on SlideWindowOperator (set in the serve
compose branch only). Batch mode never sets the hook → the registry stays
empty and batch behavior is unchanged. Readers are gRPC handler threads —
this module must import nothing from any ML or GPU framework (pure
threading + stdlib only)."""

from __future__ import annotations

import threading
from typing import Any, Dict, List

__all__ = ["register", "models", "reset"]

_LOCK = threading.Lock()
_MODELS: Dict[str, Dict[str, Any]] = {}

# The only keys CheckHealth reports (proto ModelInfo has no mirror_axes).
_FIELDS = ("config", "trainer", "folds", "patch_size", "mirror_axes")


def register(record: Dict[str, Any]) -> None:
    """Record one loaded model, keyed by config name.

    First load wins — models load exactly once per daemon , so a
    duplicate register for the same config is silently dropped."""
    with _LOCK:
        cfg = str(record["config"])
        if cfg in _MODELS:
            return
        _MODELS[cfg] = {k: record.get(k) for k in _FIELDS}


def models() -> List[Dict[str, Any]]:
    """All recorded models, sorted by config name (deterministic for tests)."""
    with _LOCK:
        return [dict(_MODELS[cfg]) for cfg in sorted(_MODELS)]


def reset() -> None:
    """Clear the registry (negative tests / fresh daemons)."""
    with _LOCK:
        _MODELS.clear()
