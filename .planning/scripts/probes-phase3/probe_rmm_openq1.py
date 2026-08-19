#!/usr/bin/env python3
"""probe_rmm_openq1.py — Open Q1: measure the RMM initial-pool reservation.

Runs in a FRESH process from examples/apps/cchmc-nnunet-fast with
``ulimit -s unlimited`` and a PINNED, known-free device
(CUDA_VISIBLE_DEVICES=<id> — research Pitfall 7). Samples the driver-level
memory of that device BEFORE and immediately AFTER importing
``my_app.gpu_bootstrap`` (which calls ``rmm.reinitialize(pool_allocator=True,
managed_memory=False)`` at import — the very site under question).

The delta S = used_after - used_before is the immediate post-reinit
reservation. Compare against:
  * Phase 2 trace: ~1.98 GB TOTAL cudaMalloc for the whole bundle run
    (implying a small initial pool at that time);
  * research measurement 2026-08-19: rmm 26.2.0 default initial pool =
    half total GPU memory = 20.0 GiB on the A100-40GB.

Decision rule (plan 3-optimization-01, task 1):
  S small (Phase 2-like, <= ~3 GiB)  -> no code change, document;
  S large (>= 10 GiB)                -> pin initial_pool_size in gpu_bootstrap.

Usage:
  cd examples/apps/cchmc-nnunet-fast
  ulimit -s unlimited
  CUDA_VISIBLE_DEVICES=<id> /tmp/monai-env/.venv/bin/python \
      /path/to/.planning/scripts/probes-phase3/probe_rmm_openq1.py
"""

import json
import os
import sys

# Running as a standalone script: put the app root (cwd) on sys.path so the
# `my_app` package resolves (sys.path[0] is the script's own directory).
sys.path.insert(0, os.path.abspath(os.getcwd()))

import pynvml

GiB = 1024.0**3


def main() -> int:
    pynvml.nvmlInit()
    # Device index 0 of the PINNED view (CUDA_VISIBLE_DEVICES set externally).
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
    before = pynvml.nvmlDeviceGetMemoryInfo(handle).used

    # The import runs rmm.reinitialize() (no torch allocation yet).
    import my_app.gpu_bootstrap  # noqa: F401

    after = pynvml.nvmlDeviceGetMemoryInfo(handle).used

    result = {
        "device": name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "total_gib": total / GiB,
        "used_before_gib": before / GiB,
        "used_after_gib": after / GiB,
        "post_reinit_reservation_gib": (after - before) / GiB,
        "rmm_version": None,
        "initial_pool_size_in_call": None,
    }
    import my_app.gpu_bootstrap as gb  # already imported above; same module

    import rmm

    result["rmm_version"] = rmm.__version__
    src = open(gb.__file__).read()
    result["initial_pool_size_in_call"] = (
        "initial_pool_size" in src and "reinitialize" in src
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
