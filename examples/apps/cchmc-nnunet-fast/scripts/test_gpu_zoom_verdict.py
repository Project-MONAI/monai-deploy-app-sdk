#!/usr/bin/env python
"""D-22a FAST kernel verdict (amended gate) — bounded, no unbounded runs.

A prior full-suite run of test_gpu_zoom.py wedged for 70+ min and was killed.
This script is the ONE-AND-ONLY remaining bounded kernel-attempt check
(amended D-22a): 3 small synthetic volumes (32x32x16, non-unity zoom factors,
orders 0/1/3) + 1 real bundle-shape volume (256^3 -> (255,256,255) o3), each
case byte-compared (np.array_equal) against the scipy/skimage reference
chains. Each case runs in its OWN subprocess under `timeout 120` (a wedged
kernel hangs the child, never the driver); the whole driver must finish
under `timeout 1500`.

Comparison levels (mirroring what the flag-ON wiring computes):
  * primitive — scipy.ndimage.zoom(grid_mode=True, mode='nearest') with
    skimage's exact zoom-factor derivation vs gpu_zoom_grid_mode (fp32).
  * full chain — skimage.transform.resize(mode='edge', anti_aliasing=False)
    vs the flag-ON data chain (gpu_zoom_resize / per-channel clip-fold).

Verdict:
  exit 0  -> every case byte-identical on BOTH levels (keep the custom-kernel
             wiring — it exceeds the amended D-22b gate)
  exit 1  -> at least one case diverges OR times out (kernel is DISCARDED
             from the shipping path per D-22a; rewire to stock CuPy)

Run:
  cd examples/apps/cchmc-nnunet-fast
  ulimit -s unlimited && timeout 1500 \
    /tmp/monai-env/.venv/bin/python scripts/test_gpu_zoom_verdict.py
"""

import os
import subprocess
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
MY_APP = APP_ROOT / "my_app"
sys.path.insert(0, str(MY_APP))

CASE_TIMEOUT_S = 120

# (name, in_shape, out_shape, order) — non-unity factors on all axes.
CASES = [
    ("syn-o0", (32, 32, 16), (40, 28, 24), 0),
    ("syn-o1", (32, 32, 16), (24, 40, 20), 1),
    ("syn-o3", (32, 32, 16), (36, 30, 22), 3),
    ("real", (256, 256, 256), (255, 256, 255), 3),
]


def run_case(name: str, in_shape, out_shape, order: int) -> int:
    """Run one case in this process. Returns 0 iff byte-identical."""
    import cupy as cp
    import gpu_bootstrap  # RMM first — shipping import order
    import numpy as np
    from operators.gpu_zoom import gpu_zoom_grid_mode, zoom_factors_for
    from scipy.ndimage import zoom as ndi_zoom
    from skimage.transform import resize as sk_resize

    seed = int(np.random.RandomState(20260819).randint(2**31)) ^ (order * 7919 + sum(in_shape))
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 4.0, size=(1, *in_shape)).astype(np.float32)
    new_shape = tuple(int(s) for s in out_shape)
    failures = []

    # 1. primitive: scipy ndi.zoom vs gpu_zoom_grid_mode (fp32)
    zf = zoom_factors_for(in_shape, new_shape)
    ref = ndi_zoom(x[0], zf, order=order, mode="nearest", grid_mode=True).astype(np.float32)
    got = np.ascontiguousarray(
        gpu_zoom_grid_mode(cp.asarray(np.ascontiguousarray(x[0])), zf, order,
                           out_shape=new_shape).get())
    if not (ref.shape == got.shape and ref.dtype == got.dtype
            and np.array_equal(ref, got)):
        failures.append(("primitive", ref, got))

    # 2. full chain: skimage resize vs gpu_zoom_resize
    ref_c = sk_resize(x[0], new_shape, order, mode="edge", anti_aliasing=False)
    from operators.gpu_zoom import gpu_zoom_resize
    got_c = np.ascontiguousarray(gpu_zoom_resize(cp.asarray(x[0]), new_shape, order).get())
    if got_c.ndim == 4 and got_c.shape[0] == 1:  # gpu_zoom_resize adds a channel axis
        got_c = got_c[0]
    if not (ref_c.shape == got_c.shape and ref_c.dtype == got_c.dtype
            and np.array_equal(ref_c, got_c)):
        failures.append(("full-chain", ref_c, got_c))

    if failures:
        for label, r, g in failures:
            mask = r != g
            n = int(mask.sum())
            max_abs = float(np.abs((r - g).astype(np.float64)).max())
            print(f"  [FAIL] {label}: {n}/{r.size} voxels differ "
                  f"({100.0 * n / r.size:.6f}%), max_abs={max_abs:.6e}")
        print(f"CASE {name}: FAIL ({in_shape} -> {new_shape} o{order})")
        return 1
    print(f"CASE {name}: PASS ({in_shape} -> {new_shape} o{order}) "
          f"byte-identical on both levels")
    return 0


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--case":
        # child mode: run exactly one case
        name = sys.argv[2]
        for c in CASES:
            if c[0] == name:
                return run_case(*c)
        print(f"unknown case: {name}")
        return 2

    # driver mode: one subprocess per case, each under `timeout 120`
    py = sys.executable
    me = str(Path(__file__).resolve())
    print(f"device: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')} | "
          f"{len(CASES)} cases, {CASE_TIMEOUT_S}s timeout each")
    results = {}
    for name, in_shape, out_shape, order in CASES:
        print(f"-- case {name}: {in_shape} -> {out_shape} o{order}")
        try:
            p = subprocess.run(
                ["/usr/bin/timeout", str(CASE_TIMEOUT_S), py, me, "--case", name],
                capture_output=True, text=True,
            )
            out = (p.stdout or "").strip()
            err = (p.stderr or "").strip().splitlines()[-3:]
            if out:
                print("\n".join("  " + line for line in out.splitlines()))
            if p.returncode == 0:
                results[name] = "PASS"
            elif p.returncode == 1:
                results[name] = "FAIL"
            elif p.returncode in (124, 137):
                results[name] = "TIMEOUT"
                print(f"  [TIMEOUT] case killed after {CASE_TIMEOUT_S}s")
            else:
                results[name] = "ERROR"
                for line in err:
                    print(f"  [stderr] {line}")
        except Exception as e:  # noqa: BLE001
            results[name] = f"ERROR ({e})"

    print()
    for name, res in results.items():
        print(f"  {name}: {res}")
    bad = [n for n, r in results.items() if r != "PASS"]
    if bad:
        print(f"VERDICT: DISCARD — {bad} not byte-identical (D-22a: the custom "
              f"kernel is dropped from the shipping path; rewire to stock CuPy)")
        return 1
    print("VERDICT: KEEP — all cases byte-identical vs the scipy/skimage "
          "reference (the custom-kernel wiring exceeds the amended D-22b gate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
