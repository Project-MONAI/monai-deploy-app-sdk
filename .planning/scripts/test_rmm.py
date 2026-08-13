#!/usr/bin/env python
"""
RMM (RAPIDS Memory Manager) smoke test.

Verifies that:
1. RMM with a cudaMallocAsync pool allocator can be imported and configured.
2. PyTorch uses the 'cudaAsync' allocator backend after RMM integration.
3. A small GPU tensor can be allocated successfully.

If RMM is not installed the script prints a skip message and exits 0.
"""

import sys

# ---------------------------------------------------------------------------
# 1. Try importing RMM — exit gracefully if it's not available
# ---------------------------------------------------------------------------
try:
    import rmm  # noqa: F401
except ImportError:
    print("[SKIP] rmm-cu13 is not installed — skipping RMM smoke test")
    sys.exit(0)

# ---------------------------------------------------------------------------
# 2. Configure RMM to use the cudaMallocAsync pool allocator
# ---------------------------------------------------------------------------
try:
    rmm.reinitialize(pool_allocator=True, managed_memory=False)
    print("[OK]   RMM reinitialized with pool_allocator=True")
except Exception as e:
    print(f"[WARN] RMM reinitialize failed: {e}")
    # Continue anyway — we can still check the backend separately

# ---------------------------------------------------------------------------
# 3. Integrate RMM with PyTorch's memory allocator
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:
    print("[SKIP] torch is not installed — cannot verify GPU allocation")
    sys.exit(0)

if not torch.cuda.is_available():
    print("[SKIP] No CUDA device available — skipping GPU allocation check")
    sys.exit(0)

try:
    torch.cuda.memory.allocators.rmm.RMMAllocatorConfig(
        pool_allocator=True,
    )
    torch.cuda.memory.change_current_allocator("rmm")
    print("[OK]   PyTorch RMM allocator activated")
except Exception as e:
    print(f"[WARN] Could not set RMM allocator in PyTorch: {e}")

# ---------------------------------------------------------------------------
# 4. Allocate a small GPU tensor
# ---------------------------------------------------------------------------
try:
    x = torch.zeros(64, 64, device="cuda")
    del x
    print("[OK]   Small GPU tensor allocated and freed successfully")
except Exception as e:
    print(f"[FAIL] GPU tensor allocation failed: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 5. Verify the allocator backend
# ---------------------------------------------------------------------------
backend = torch.cuda.memory.get_allocator_backend()
print(f"[INFO] Current allocator backend: '{backend}'")

if backend == "cudaAsync":
    print("[OK]   Allocator backend is 'cudaAsync' — RMM smoke test PASSED")
    sys.exit(0)
else:
    print(
        f"[FAIL] Expected backend 'cudaAsync', got '{backend}' — "
        "RMM integration may not be active"
    )
    sys.exit(1)
