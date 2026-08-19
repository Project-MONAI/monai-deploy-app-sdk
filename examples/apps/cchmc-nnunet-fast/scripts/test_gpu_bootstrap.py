#!/usr/bin/env python
"""Import-order self-test for the RMM bootstrap (INFR-01/D-14).

Spawns two SUBPROCESSES (each a fresh process — import order only makes sense
per process) with the project venv's python:

  (a) rmm-first:    ``import gpu_bootstrap`` (RMM before anything else), then
      ``import monai.deploy.core``, then print the torch allocator backend.
      Expect exit 0 and stdout containing ``pluggable``.
  (b) holoscan-first: ``import monai.deploy.core`` FIRST, then ``import rmm``.
      Expect a NON-ZERO exit and stderr containing ``undefined symbol`` —
      this documents the live hazard (reproduced 2026-08-19:
      ``ImportError: undefined symbol: __cxa_call_terminate``). If the
      failure mode ever changes, the test reports it explicitly instead of
      silently passing.

Run:  /tmp/monai-env/.venv/bin/python scripts/test_gpu_bootstrap.py
"""

import os
import resource
import subprocess
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
MY_APP = APP_ROOT / "my_app"
PYTHON = sys.executable  # must be /tmp/monai-env/.venv/bin/python

# Holoscan needs a >=32 MB stack; make the spawned subprocesses inherit an
# unlimited stack regardless of the caller's shell (the app normally relies
# on `ulimit -s unlimited`).
try:
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
except (ValueError, OSError):
    pass

RMM_FIRST = r"""
import sys
sys.path.insert(0, {my_app!r})
import gpu_bootstrap  # rmm + RMM pool + torch allocator, BEFORE holoscan
backend = gpu_bootstrap.install_torch_allocator()
import monai.deploy.core  # heavy holoscan/monai.deploy import path
print("allocator_backend:", backend)
"""

HOLOSCAN_FIRST = r"""
import monai.deploy.core  # holoscan FIRST (the anti-order)
import rmm  # expected to fail: undefined symbol
print("UNEXPECTED: rmm imported after holoscan succeeded")
"""


def run_case(label, snippet, expect_ok, expect_marker, marker_stream):
    proc = subprocess.run(
        [PYTHON, "-c", snippet],
        capture_output=True,
        text=True,
        cwd=str(APP_ROOT),
    )
    stream = proc.stdout if marker_stream == "stdout" else proc.stderr
    got_marker = expect_marker in stream
    if expect_ok:
        ok = proc.returncode == 0 and got_marker
        print(f"[{label}] exit={proc.returncode} marker({expect_marker!r}) in {marker_stream}: {got_marker}")
    else:
        # case (b): any crash is acceptable ONLY if it shows the documented
        # undefined-symbol hazard; a different failure mode is reported
        # explicitly (the test does not silently pass over it).
        ok = proc.returncode != 0 and got_marker
        if proc.returncode != 0 and not got_marker:
            print(f"[{label}] UNEXPECTED failure mode (exit={proc.returncode}): "
                  f"no {expect_marker!r} in {marker_stream}:\n{stream[-2000:]}")
        print(f"[{label}] exit={proc.returncode} marker({expect_marker!r}) in {marker_stream}: {got_marker}")
    if proc.stdout.strip():
        print(f"[{label}] stdout: {proc.stdout.strip()[:500]}")
    if not ok:
        print(f"[{label}] stderr: {proc.stderr.strip()[-2000:]}")
    print(f"[{label}] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    failures = []
    print("== case (a): rmm-first (gpu_bootstrap, then monai.deploy.core) ==")
    failures.append(not run_case(
        "a", RMM_FIRST.format(my_app=str(MY_APP)),
        expect_ok=True, expect_marker="pluggable", marker_stream="stdout",
    ))
    print("== case (b): holoscan-first (monai.deploy.core, then import rmm) ==")
    failures.append(not run_case(
        "b", HOLOSCAN_FIRST,
        expect_ok=False, expect_marker="undefined symbol", marker_stream="stderr",
    ))
    if any(failures):
        print("RESULT: FAIL")
        sys.exit(1)
    print("RESULT: PASS (rmm-first -> pluggable; holoscan-first -> undefined symbol)")


if __name__ == "__main__":
    main()
