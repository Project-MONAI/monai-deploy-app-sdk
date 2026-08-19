#!/usr/bin/env python
"""pynvml driver-level VRAM sampler (MEM-003 measurement, Phase 3 Plan 02).

Samples ``nvmlDeviceGetMemoryInfo`` (used/total bytes) for one device at a
fixed rate and writes a CSV with epoch-nanosecond timestamps:

    ts_ns,device,used_bytes,total_bytes

Stop conditions (whichever comes first): the ``--until-file`` sentinel
appears, or Ctrl-C. Prints the CSV path on exit.

Measurement discipline (03-RESEARCH Pitfall 2): torch's CUDA memory
counters RAISE under the RMM pluggable allocator — this sampler is
driver-level by design (pynvml only).

Run:
    python vram_sampler.py --device 0 --hz 2 --out /tmp/vram.csv \
        [--until-file /tmp/sentinel]
"""

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

import pynvml

_STOP = False


def _on_signal(signum, frame):
    global _STOP
    _STOP = True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, required=True, help="physical GPU index")
    ap.add_argument("--hz", type=float, default=2.0, help="sample rate (1-5 Hz)")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--until-file", default=None, help="stop when this file appears")
    args = ap.parse_args()

    if not 0.5 <= args.hz <= 10:
        print(f"warning: --hz {args.hz} outside the intended 1-5 Hz range", file=sys.stderr)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.device)
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode()
    print(f"vram_sampler: device {args.device} ({name}) at {args.hz} Hz -> {args.out}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    until = Path(args.until_file) if args.until_file else None

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    interval = 1.0 / args.hz
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_ns", "device", "used_bytes", "total_bytes"])
        f.flush()
        while not _STOP and not (until is not None and until.exists()):
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            w.writerow([time.time_ns(), args.device, mem.used, mem.total])
            f.flush()
            t0 = time.monotonic()
            # sleep the remainder of the interval (monotonic, not time.sleep
            # on the wall clock) for a stable rate
            elapsed = time.monotonic() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    print(f"vram_sampler: stopped; rows written to {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
