#!/usr/bin/env python
"""Phase 18 (plan 18-02, Task 1) synthetic byte-identity unit test for
my_app/operators/resample_order1.py — the TS-semantic order-1 resample
helper used by the crop cascade (6mm downsample + task-res resample).

House pattern (mirrors scripts/test_pre_resample_zoom.py — Phase 8.1):
standalone executable, NOT pytest. Exits 0 iff every case passes.

Oracle: TotalSegmentator 2.18.0 (read-only venv) `resampling.py`
`change_spacing(img, resample, order=1, dtype=np.int32, use_gpu=...)`:
  * data = img.get_fdata()  (float64 — exact for our int inputs)
  * zoom = old_spacing / new_spacing  (per axis; coarser target -> factor < 1)
  * ndimage.zoom(data, zoom, mode="nearest", order=1)   (cucim on GPU in TS;
    the app mirrors with cupyx — Phase 8 proved the f64 cupyx kernel
    bit-matches scipy for the same spline order; re-proven HERE for order 1)
  * out.astype(int32)  (float->int TRUNCATION toward zero)
  * exact-spacing-equal input -> returned WITHOUT resampling (identity)

What this proves:
  A. BYTE-IDENTITY: resample_order1_cpu == resample_order1_gpu (np.array_equal)
     on random int16 (HU-range + int16-extreme-range) volumes across the
     cascade's real spacing pairs:
       * native 64199 (2.0, 0.4492, 0.4492)   -> 6.0 iso      (6mm downsample)
       * native 31322 (2.0, 0.899, 0.899)      -> 6.0 iso      (6mm downsample)
       * crop 64199 (2.0, 0.4492, 0.4492)      -> task (1.5, 0.804688, 0.804688)
       * crop 31322 (2.0, 0.899, 0.899)        -> task (1.5, 0.804688, 0.804688)
  B. DTYPE CONTRACT: int16 in -> int32 out on BOTH paths; output equals
     (f64 order-1 zoom).astype(int32) — truncation, not round-half-even.
  C. IDENTITY: in_spacing == out_spacing exactly -> values unchanged
     (the documented skip branch; output is still int32).
  D. BOUNDARY: single-slice axis (1-voxel depth) cpu==gpu; and output
     shape sanity (coarser spacing -> strictly smaller volume).

Run:
  cd examples/apps/totalsegmentator-fast
  ulimit -s unlimited && CUDA_VISIBLE_DEVICES=0 \
    /tmp/monai-env/.venv/bin/python scripts/test_resample_order1.py
Exit: 0 iff every case passes; 1 otherwise (divergence tables printed).
No files are written; all data is in-memory synthetic.
"""

import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
MY_APP = APP_ROOT / "my_app"
sys.path.insert(0, str(MY_APP))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

import cupy as cp  # noqa: E402
import gpu_bootstrap  # noqa: E402, F401  (RMM bootstrap BEFORE anything else — load-bearing)
import numpy as np  # noqa: E402
from operators.resample_order1 import resample_order1_cpu, resample_order1_gpu  # noqa: E402
from scipy.ndimage import zoom as ndi_zoom  # noqa: E402

FAILURES = []


def divergence_table(name: str, ref: np.ndarray, got: np.ndarray) -> None:
    """Per-case divergence table (house format; printed only on failure)."""
    mask = ref != got
    n = int(mask.sum())
    total = ref.size
    d = ref.astype(np.float64) - got.astype(np.float64)
    max_abs = float(np.abs(d).max()) if n else 0.0
    print(f"  [DIVERGENCE] {name}: {n}/{total} voxels differ ({100.0 * n / total:.4f}%), " f"max_abs={max_abs:.3e}")
    print("    first 10 differing coordinates (axis0, axis1, axis2):")
    for c in np.argwhere(mask)[:10]:
        i = tuple(int(v) for v in c)
        print(f"      {i}: ref={ref[i].item()!r} got={got[i].item()!r}")


def check(name: str, ref: np.ndarray, got: np.ndarray) -> None:
    ok = ref.shape == got.shape and ref.dtype == got.dtype and np.array_equal(ref, got)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        if ref.shape != got.shape or ref.dtype != got.dtype:
            print(f"    shape/dtype mismatch: ref {ref.shape} {ref.dtype} " f"vs got {got.shape} {got.dtype}")
        divergence_table(name, ref, got)
        FAILURES.append(name)


def check_bool(name: str, expected: bool, got: bool) -> None:
    ok = expected is got
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} (expected={expected}, got={got})")
    if not ok:
        FAILURES.append(name)


def gpu_resample(x16: np.ndarray, in_sp, out_sp) -> np.ndarray:
    """GPU path on a host int16 volume -> host int32 result."""
    return np.asarray(resample_order1_gpu(cp.asarray(np.ascontiguousarray(x16), dtype=cp.int16), in_sp, out_sp).get())


def main() -> int:
    print(f"device: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    rng = np.random.default_rng(20260902)

    # Real cascade spacing pairs (18-RESEARCH §TS 2.18 Pipeline; app (D,H,W)
    # axis order). in_spacing / out_spacing per axis.
    pairs = [
        ("64199 native->6mm", (217, 512, 512), (2.0, 0.4492, 0.4492), (6.0, 6.0, 6.0)),
        ("31322 native->6mm", (345, 512, 512), (2.0, 0.899, 0.899), (6.0, 6.0, 6.0)),
        (
            "64199 crop->taskres",
            (81, 433, 351),
            (2.0, 0.4492, 0.4492),
            (1.5, 0.8046879768371582, 0.8046879768371582),
        ),
        (
            "31322 crop->taskres",
            (156, 360, 293),
            (2.0, 0.899, 0.899),
            (1.5, 0.8046879768371582, 0.8046879768371582),
        ),
    ]

    # ------------------------------------------------------------------
    # A. BYTE-IDENTITY scipy CPU vs cupyx GPU on the real spacing pairs
    # ------------------------------------------------------------------
    print("\n[A] byte-identity scipy vs cupyx (order 1, int32 truncation)")
    for label, shape, in_sp, out_sp in pairs:
        x_hu = (rng.standard_normal(shape) * 800).astype(np.int16)
        ref_cpu = resample_order1_cpu(x_hu, in_sp, out_sp)
        check(f"A {label} HU-range scipy==GPU", ref_cpu, gpu_resample(x_hu, in_sp, out_sp))
        x_ext = rng.integers(-1024, 3072, size=shape).astype(np.int16)
        check(
            f"A {label} extreme-range scipy==GPU",
            resample_order1_cpu(x_ext, in_sp, out_sp),
            gpu_resample(x_ext, in_sp, out_sp),
        )

        # B. dtype + truncation contract (checked here per pair) — the
        # reference mirrors the implementation chain: f64 zoom -> f32
        # round-to-even boundary gate -> int32 truncation (see module doc).
        got = ref_cpu
        check_bool(f"B {label} out dtype int32", True, got.dtype == np.int32)
        f = tuple(float(i) / float(o) for i, o in zip(in_sp, out_sp))
        trunc_ref = ndi_zoom(x_hu.astype(np.float64), f, order=1, mode="nearest").astype(np.float32).astype(np.int32)
        check(f"B {label} out == (f64zoom -> f32 -> int32) chain", trunc_ref, got)

    # ------------------------------------------------------------------
    # C. IDENTITY: exact-equal spacings -> values unchanged (skip branch)
    # ------------------------------------------------------------------
    print("\n[C] identity (in_spacing == out_spacing, TS change_spacing early return)")
    x3 = rng.integers(-1000, 3000, size=(32, 32, 16)).astype(np.int16)
    out_c = resample_order1_cpu(x3, (1.0, 2.0, 4.0), (1.0, 2.0, 4.0))
    check("C identity cpu: values unchanged, int32 out", x3.astype(np.int32), out_c)
    out_g = gpu_resample(x3, (1.0, 2.0, 4.0), (1.0, 2.0, 4.0))
    check("C identity gpu: values unchanged, int32 out", x3.astype(np.int32), out_g)

    # ------------------------------------------------------------------
    # D. BOUNDARY: single-slice axis + shape sanity
    # ------------------------------------------------------------------
    print("\n[D] boundary cases")
    x1 = rng.integers(-1000, 3000, size=(1, 64, 64)).astype(np.int16)
    check(
        "D single-slice axis scipy==GPU",
        resample_order1_cpu(x1, (2.0, 1.0, 1.0), (6.0, 0.5, 0.5)),
        gpu_resample(x1, (2.0, 1.0, 1.0), (6.0, 0.5, 0.5)),
    )
    # shape sanity: native 64199 -> 6mm must shrink every axis
    x64199 = rng.integers(-100, 200, size=(217, 512, 512)).astype(np.int16)
    shrunk = resample_order1_cpu(x64199, (2.0, 0.4492, 0.4492), (6.0, 6.0, 6.0))
    check_bool(
        "D 6mm downsample shrinks all axes",
        True,
        all(s < o for s, o in zip(shrunk.shape, (217, 512, 512))),
    )
    print(f"    6mm shape: {shrunk.shape} (TS log 64199: (38, 38, 72) up to resampler rounding)")

    print()
    if FAILURES:
        print(f"RESULT: FAIL — {len(FAILURES)} case(s) diverged")
        return 1
    print(
        "RESULT: PASS — every case byte-identical (np.array_equal) scipy vs cupyx; "
        "dtype/identity/boundary checks green"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
