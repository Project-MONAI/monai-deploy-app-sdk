# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared GPU primitives for the cchmc-nnunet-fast pipeline.

Every operator in the pipeline uses these helpers so that:

* CPU fallback is never silent — device guards raise ``RuntimeError``
  (INF-005: silent-fallback guard);
* NVTX ranges are visible in Nsight Systems traces (INFR-006);
* timing records are structured and JSON-serializable for logs.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import json
import time
from typing import Any, Dict, Iterator, List, Optional

import torch

__all__ = [
    "assert_cuda_available",
    "assert_on_gpu",
    "nvtx_range",
    "GpuTiming",
    "set_study_id",
    "get_study_id",
    "StudyTimingCollector",
]

# Per-fragment study identity. The DAG carries no explicit study-uid input on
# every edge, and holoscan 4.2's Python API exposes no compute-context user
# data, so the operator that sees the DICOM Image first (PreprocessOperator,
# which runs before all downstream GPU operators by DAG order) registers the
# study uid here; the other operators of the same fragment read it for their
# timing records. Valid for this app's single-study-per-run scope.
_study_by_fragment: Dict[int, str] = {}


def _root(fragment: Any) -> Any:
    """Top-level Application for a (sub-)fragment: in Phase 1 fragment == app;
    with real sub-Fragments, Fragment.application (verified present in 4.2)
    resolves the root; with Subgraphs (the 4.2 supported multi-fragment
    mechanism), ``Subgraph.fragment`` is the owning (top-level) Application.
    Keying by the root keeps one per-study aggregate across fragments
    (RESEARCH Pitfall 9)."""
    return getattr(fragment, "application", None) or getattr(fragment, "fragment", None) or fragment


def set_study_id(fragment: Any, study_id: str) -> None:
    """Register the current study identifier for a fragment (see module note).

    Keyed by the TOP-LEVEL application (``_root``) so sub-fragment operators
    share the study identity (RESEARCH Pitfall 9)."""
    _study_by_fragment[id(_root(fragment))] = str(study_id)


def get_study_id(fragment: Any, default: str = "unknown") -> str:
    """Return the study identifier registered for ``fragment`` (or ``default``),
    looking up the top-level application key (see :func:`_root`)."""
    return _study_by_fragment.get(id(_root(fragment)), default)


def assert_cuda_available() -> None:
    """Raise ``RuntimeError`` when no CUDA device is available.

    The pipeline is GPU-resident by contract (PREP-05); we never silently
    fall back to CPU. Use this at operator entry before any GPU handoff.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device is not available. cchmc-nnunet-fast requires a GPU for "
            "the zero-copy handoff contract; refusing to silently fall back to CPU."
        )


def assert_on_gpu(tensor: torch.Tensor) -> torch.Tensor:
    """Assert that ``tensor`` is resident on a CUDA device.

    Raises ``RuntimeError`` (never swallows) when the tensor is on CPU or any
    non-CUDA device, so a silent CPU fallback is impossible (INF-005).

    Args:
        tensor: the torch tensor to validate.

    Returns:
        The same tensor when the check passes (so call sites can chain).
    """
    if tensor is None:
        raise RuntimeError("assert_on_gpu: tensor is None.")
    if not torch.is_tensor(tensor):
        raise RuntimeError(f"assert_on_gpu: expected a torch.Tensor, got {type(tensor)!r}.")
    if tensor.device.type != "cuda":
        raise RuntimeError(
            f"assert_on_gpu: tensor is on {tensor.device} (expected a CUDA device). "
            "Zero-copy GPU handoff contract violated."
        )
    return tensor


@contextlib.contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Push/pop an NVTX range so the span shows up in Nsight traces.

    When no CUDA device is present the range is a no-op (the device guards
    above raise at the boundaries where it matters).
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


class GpuTiming:
    """Structured timing helper producing JSON-serializable records.

    Usage::

        timing = GpuTiming("preprocess")
        timing.start()
        ... work ...
        record = timing.stop()          # {"operator", "study", "start", "end",
                                        #  "start_ns", "end_ns", "duration_ms"}
        log.info("timing: %s", json.dumps(record))

    ``start``/``end`` are ISO-8601 UTC wall-clock timestamps; ``start_ns``/
    ``end_ns`` are monotonic nanoseconds; ``duration_ms`` is derived from the
    monotonic pair. ``study`` defaults to ``"unknown"`` — set it from the
    record's owner (e.g. the DICOM Image metadata) before logging.
    """

    def __init__(self, label: str = "stage"):
        self.label = label
        self._start_ns: Optional[int] = None
        self.record: Optional[Dict[str, Any]] = None

    @staticmethod
    def _iso(ns: int) -> str:
        return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()

    def start(self) -> "GpuTiming":
        """Mark the start instant (monotonic, nanoseconds)."""
        self._start_ns = time.time_ns()
        return self

    def stop(self) -> Dict[str, Any]:
        """Mark the end instant and return the JSON-serializable record."""
        end_ns = time.time_ns()
        if self._start_ns is None:
            raise RuntimeError("GpuTiming.stop() called before start().")
        self.record = {
            "operator": self.label,
            "label": self.label,
            "study": "unknown",
            "start": self._iso(self._start_ns),
            "end": self._iso(end_ns),
            "start_ns": self._start_ns,
            "end_ns": end_ns,
            "duration_ms": round((end_ns - self._start_ns) / 1e6, 6),
        }
        return self.record

    def to_json(self) -> str:
        """Return the current record as a JSON string (computing it if needed)."""
        if self.record is None:
            self.stop()
        return json.dumps(self.record)


class StudyTimingCollector:
    """Accumulates per-operator timing records per TOP-LEVEL application for
    the end-of-run per-study latency aggregate (INFR-006).

    Records are JSON-serializable dicts (``GpuTiming.stop()`` output plus an
    ``operator``/``study`` field). Records are keyed by ``_root(fragment)`` —
    the top-level Application — so operators in sub-Fragments of Phase 2's
    multi-fragment DAG land in the same per-study aggregate instead of
    silently fragmenting it (RESEARCH Pitfall 9). The Application logs one
    aggregate per study after the run completes.
    """

    _records: Dict[int, List[Dict[str, Any]]] = {}

    @classmethod
    def record(cls, fragment: Any, record: Dict[str, Any]) -> None:
        cls._records.setdefault(id(_root(fragment)), []).append(dict(record))

    @classmethod
    def studies(cls, fragment: Any) -> Dict[str, List[Dict[str, Any]]]:
        """Group the root application's records by their ``study`` field."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in cls._records.get(id(_root(fragment)), []):
            grouped.setdefault(r.get("study", "unknown"), []).append(r)
        return grouped

    @classmethod
    def clear(cls, fragment: Any = None) -> None:
        if fragment is None:
            cls._records.clear()
        else:
            cls._records.pop(id(_root(fragment)), None)
