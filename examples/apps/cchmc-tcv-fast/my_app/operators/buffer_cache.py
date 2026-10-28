# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shape-keyed GPU buffer cache .

Operators pre-allocate/reuse GPU buffers across ``compute`` calls keyed on
``(shape, dtype)`` so the Nth study reuses the 1st study's buffers:

* the **CuPy family** closes the substantive gap — CuPy's LRU pool is
  INDEPENDENT of RMM, so its blocks bypass the RMM pool entirely;
* the **torch family** reduces allocator traffic and gives address stability
  on the big fixed-shape allocations (RMM already made torch allocations
  cheap — the cache only ever grows the pool, never per-tile).

Granularity/eviction (heuristic; operator discretion): in BATCH mode
(`max_entries=None`, the default) an unbounded dict is fine — the key set is
config-determined (few shapes per config), so there is no LRU/eviction
policy and no cap. In SERVE mode the daemon is long-lived
and the shape stream is study-driven, so callers may pass `max_entries` to
bound the cache with LRU eviction (least-recently-used entry dropped at the
cap); a hit re-touches its key. `clear` exists for explicit peak-VRAM
accounting if a future configuration ever needs it.

Safety invariants (non-negotiable):

* the key is ``(tuple(shape), str(dtype))`` — a different shape gets a
  different buffer (invalidation on shape change) and buffers are NEVER
  shared across dtypes;
* every allocation is C-contiguous by construction (``torch.empty`` /
  ``cp.empty``), preserving the fp32/C-contiguous invariant;
* a cached buffer must NEVER be handed to a DLPack consumer that retains it
  past the study (DLPack ownership lesson — ``cp.from_dlpack`` consumes
  the caller's buffer). Operators that cache must copy at the emit
  boundary when a cached buffer could end up as the emitted tensor
  (``shares_storage`` exists for exactly that check).

Zero semantics: ``get(shape, dtype, zero=True)`` applies ``zero_`` on
borrow — use it ONLY where the reference semantics allocate fresh zeroed
memory (e.g. the reference's ``torch.zeros`` accumulators). Callers must
never rely on stale contents of a ``zero=False`` borrow: every such site
fully overwrites the buffer before reading (justified per site in the
operator's cache-creation comment block).

Thread safety: one cache per operator INSTANCE; a single operator's
``compute`` is never re-entered concurrently, so no locking is needed
(the concurrent scheduler runs DIFFERENT operator instances in parallel).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Union

import cupy as cp
import torch

_LOG = logging.getLogger(__name__)


class _ShapeCache:
    """(shape, dtype) -> device buffer, for one allocator family.

    Args:
        device: CUDA device string (torch family) / default device (cupy).
        family: ``"torch"`` or ``"cupy"`` — which allocator backs the
            buffers. One instance serves exactly one family; the dtype key
            is ``str(dtype)``, which is unique within a family
            (``torch.float32`` vs ``float32`` can never collide because a
            family never sees the other family's dtype objects).
        max_entries: optional entry cap . ``None`` (default)
            = unbounded (batch behavior unchanged). An int >= 1 bounds the
            cache with LRU eviction: a hit moves the key to most-recent;
            on a miss the least-recently-used entries are evicted until the
            cache is below the cap, then the new entry is inserted.
    """

    def __init__(
        self,
        device: Union[str, "torch.device"] = "cuda",
        family: str = "torch",
        max_entries: Optional[int] = None,
    ):
        if family not in ("torch", "cupy"):
            raise ValueError(f"_ShapeCache: family must be 'torch' or 'cupy', got {family!r}")
        if max_entries is not None and (not isinstance(max_entries, int) or max_entries < 1):
            raise ValueError(f"_ShapeCache: max_entries must be None or an int >= 1, got {max_entries!r}")
        self._family = family
        self._device = device
        self._max_entries = max_entries
        # OrderedDict = insertion-ordered plain dict + popitem(last=False)
        # (CPython's built-in dict.popitem takes no keyword arguments) = the LRU
        # ordering primitive (oldest first item = least recently used; hits re-insert).
        self._d: "OrderedDict[Tuple[Tuple[int, ...], str], Any]" = OrderedDict()

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def get(self, shape, dtype, zero: bool = False):
        """Return the cached buffer for ``(shape, dtype)`` (allocating on
        first borrow). ``zero=True`` re-zeros on every borrow (reference
        fresh-``zeros`` semantics); ``zero=False`` leaves prior contents —
        the caller must fully overwrite before reading."""
        key = (tuple(int(s) for s in shape), str(dtype))
        buf = self._d.get(key)
        if buf is None:
            # / : LRU eviction on the MISS path — drop
            # least-recently-used entries (dict preserves insertion order;
            # hits re-insert, so oldest first-item = LRU) until below cap.
            while self._max_entries is not None and len(self._d) >= self._max_entries:
                old_key, _old_buf = self._d.popitem(last=False)
                _LOG.info(
                    "_ShapeCache(%s): LRU evict %s (cap %d)",
                    self._family,
                    old_key,
                    self._max_entries,
                )
            # Release the last-evicted buffer BEFORE allocating the
            # replacement: at a full cache the peak must be `cap` buffers,
            # not cap + 1 (an evict-then-grow shape change could otherwise
            # OOM exactly when eviction was supposed to make room).
            _old_buf = None
            if self._family == "torch":
                buf = torch.empty(key[0], dtype=dtype, device=self._device)
            else:
                buf = cp.empty(key[0], dtype=dtype)
            # invariant: C-contiguous (both allocators default to
            # C order — assert so a future change cannot silently break it).
            if self._family == "torch":
                assert buf.is_contiguous(), f"_ShapeCache: torch buffer not C-contiguous {key}"
            else:
                assert buf.flags.c_contiguous, f"_ShapeCache: cupy buffer not C-contiguous {key}"
            self._d[key] = buf
        else:
            # / : LRU touch — a hit becomes most-recently-used
            # (pop+re-insert moves the key to the end of insertion order).
            self._d[key] = self._d.pop(key)
        if zero:
            # torch: zero_; CuPy ndarrays use the numpy API (no zero_).
            if self._family == "torch":
                buf.zero_()
            else:
                buf.fill(0)
        return buf

    def clear(self) -> None:
        """Drop every cached buffer (peak-VRAM accounting hook)."""
        self._d.clear()

    def keys(self) -> List[Tuple[Tuple[int, ...], str]]:
        return list(self._d)

    def items(self):
        """((shape, dtype_str), buffer) pairs (proof-record helper)."""
        return list(self._d.items())

    def total_bytes(self) -> int:
        """Total device bytes currently held by the cache (proof record)."""
        return int(sum(buf.nbytes for buf in self._d.values()))

    def shares_storage(self, tensor: Any) -> bool:
        """True if ``tensor`` (or a view of it) aliases any cached buffer."""
        if not self._d:
            return False
        if self._family == "torch":
            # Views share the base storage at an OFFSET — data_ptr alone
            # is not a reliable cache-membership test, so compare the
            # storage base address instead.
            try:
                ptr = tensor.untyped_storage().data_ptr()
            except Exception:
                return False
            return any(b.untyped_storage().data_ptr() == ptr for b in self._d.values())
        # CuPy: a sliced view's .data.ptr is the OFFSET pointer, so pointer
        # comparison would miss views — use cp.shares_memory instead.
        try:
            return any(cp.shares_memory(b, tensor) for b in self._d.values())
        except Exception:
            return False
