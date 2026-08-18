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
import json
import time
from typing import Any, Dict, Iterator, Optional

import torch

__all__ = ["assert_cuda_available", "assert_on_gpu", "nvtx_range", "GpuTiming"]


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
        record = timing.stop()          # {"label", "start_ns", "end_ns", "duration_ms"}
        log.info("timing: %s", json.dumps(record))
    """

    def __init__(self, label: str = "stage"):
        self.label = label
        self._start_ns: Optional[int] = None
        self.record: Optional[Dict[str, Any]] = None

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
            "label": self.label,
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
