# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RMM bootstrap — MUST be imported before any holoscan / MONAI Deploy import.

Verified hazard (2026-08-19, live-reproduced): loading ``rmm`` AFTER the
holoscan package has been loaded raises ``ImportError: undefined symbol:
__cxa_call_terminate``. Importing this module first installs RMM as torch's
CUDA allocator (pool allocator, /).

This module deliberately contains no holoscan or MONAI Deploy imports —
only ``rmm`` at import time and (lazily) ``torch`` inside the helpers.

rmm 26.x API note: the legacy ``rmm.mr`` pool-allocator class is gone in
26.x — the working sequence is the ``reinitialize`` + torch-allocator call
below (re-verified 2026-08-19).

Do NOT also set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` — RMM
and expandable_segments are alternative torch allocation strategies. RMM
is primary; expandable_segments is the documented fallback only if
instability appears.
"""

import logging

import rmm  # noqa: F401 (must precede any holoscan import)

# Why this pin (live-re-verified 2026-08-19 in the dev environment):
# rmm 26.x's DEFAULT initial_pool_size is half of total GPU memory —
# ~20.0 GiB on the A100-SXM4-40GB, reserved immediately at reinit (pynvml
# probe: 19.97 GiB used before any torch allocation). The reference
# bundle's memory_budget total is 1,038,502,513 bytes (~0.97 GiB — from a
# fresh bundle run log, 2026-08-19), so pin the initial pool to 4 GiB: 4x
# headroom over the budget total while removing the wasteful default
# reservation. gpu_bootstrap.warm_pool(plan.total_bytes) at compose end
# still grows the pool to the per-bundle budget (the RMM warm-up), so the
# pin only trims the default, never the warm-up.
_INITIAL_POOL_SIZE = 4 * 1024**3
rmm.reinitialize(
    pool_allocator=True,
    managed_memory=False,
    initial_pool_size=_INITIAL_POOL_SIZE,
)
logging.info("rmm initial_pool_size: %d bytes (pinned initial pool, see module comment)", _INITIAL_POOL_SIZE)
from rmm.allocators.torch import rmm_torch_allocator  # noqa: E402


def install_torch_allocator() -> str:
    """Make RMM torch's current CUDA allocator; return the backend name.

    The backend reports ``"pluggable"`` when the RMM allocator is active
    . The swap is skipped if CUDA is unavailable (headless test
    imports); the allocator is registered either way. Idempotent (
    fix): if RMM is already the active allocator — e.g. app.py executed
    twice under 'app' and 'my_app.app' module names — skip the swap
    instead of raising "Can't swap an already initialized allocator".
    """
    import torch

    if torch.cuda.is_available():
        if torch.cuda.memory.get_allocator_backend() == "pluggable":
            return "pluggable"
        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
    return torch.cuda.memory.get_allocator_backend()


def warm_pool(num_bytes: int) -> None:
    """Pre-allocate then release so the RMM pool retains the memory before
    study 1's tiles start (pool pre-allocation in setup, /).

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
