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

"""RMM bootstrap — MUST be imported before any holoscan / MONAI Deploy import.

Verified hazard (2026-08-19, live-reproduced): loading ``rmm`` AFTER the
holoscan package has been loaded raises ``ImportError: undefined symbol:
__cxa_call_terminate``. Importing this module first installs RMM as torch's
CUDA allocator (pool allocator, INFR-01/D-14).

This module deliberately contains no holoscan or MONAI Deploy imports —
only ``rmm`` at import time and (lazily) ``torch`` inside the helpers.

rmm 26.x API note: the legacy ``rmm.mr`` pool-allocator class is gone in
26.x — the working sequence is the ``reinitialize`` + torch-allocator call
below (re-verified 2026-08-19).

Do NOT also set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` — RMM
and expandable_segments are alternative torch allocation strategies
(RESEARCH Pitfall 6); RMM is primary, expandable_segments is the documented
fallback only if instability appears.
"""

import rmm  # noqa: F401  (must precede any holoscan import)

rmm.reinitialize(pool_allocator=True, managed_memory=False)
from rmm.allocators.torch import rmm_torch_allocator  # noqa: E402


def install_torch_allocator() -> str:
    """Make RMM torch's current CUDA allocator; return the backend name.

    The backend reports ``"pluggable"`` when the RMM allocator is active
    (INFR-01). The swap is skipped if CUDA is unavailable (headless test
    imports); the allocator is registered either way.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
    return torch.cuda.memory.get_allocator_backend()


def warm_pool(num_bytes: int) -> None:
    """Pre-allocate then release so the RMM pool retains the memory before
    study 1's tiles start (pool pre-allocation in setup, INFR-01/D-14).

    Allocating one large fp32 tensor and releasing it grows the RMM pool to
    at least ``num_bytes``; subsequent per-tile allocations then draw from
    the pool instead of hitting ``cudaMalloc``.
    """
    import torch

    if not torch.cuda.is_available():
        return
    n = max(1, int(num_bytes) // 4)
    buf = torch.empty(n, dtype=torch.float32, device="cuda")
    del buf
