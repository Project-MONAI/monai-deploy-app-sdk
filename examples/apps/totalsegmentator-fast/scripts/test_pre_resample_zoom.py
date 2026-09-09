#!/usr/bin/env python
"""Phase 8.1 synthetic byte-identity unit test for pre_resample_zoom.py.

House pattern (mirrors v1.1 cchmc-nnunet-fast/scripts/test_gpu_zoom.py):
standalone executable, NOT pytest. Exits 0 iff every case passes.

What this proves (BEFORE any corpus work):
  A. BYTE-IDENTITY: scipy CPU path (oracle-exact call) vs cupyx GPU mirror on
     the SAME int16 input — 3 shapes x 4 real-corpus factor sets, HU-range
     data plus int16-extreme-range data, plus the f32-chain invariant
     (08-03 corrected: the oracle's LoadImaged casts to f32, so
     wrapper out == astype(int32) of the f32 zoom output — a float->int
     TRUNCATION, not the 08-01 int16 round-half-even variant).
  B. SKIP BRANCH: allclose(factors, 1.0, atol=1e-3) skip returns the input
     object untouched, including the four boundary pins that arbitrate the
     allclose window (1.499/1.501 -> skip, 1.498/1.502 -> resample).
  C. FACTOR MATH: column norms of the RAS affine / 1.5 (diag + rotated).
  D. FLAG SEMANTICS (post-flip P11 eea5ac7, default ON): HOLOSCAN_GPU_PRERESAMPLE
     unset -> on, "0" -> off, "1" -> on (strictly "1"-only; anything else is off).

Run:
  cd examples/apps/totalsegmentator-fast
  ulimit -s unlimited && CUDA_VISIBLE_DEVICES=0 \
    /tmp/monai-env/.venv/bin/python scripts/test_pre_resample_zoom.py
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
from operators.pre_resample_zoom import preresample_channel_cpu  # noqa: E402
from operators.pre_resample_zoom import (
    TARGET_SPACING,
    gpu_preresample_enabled,
    preresample_channel_gpu,
    preresample_should_skip,
    preresample_volume,
    preresample_zoom_factors,
)
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
        print(f"      {i}: ref={ref[c].item()!r} got={got[c].item()!r}")


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


def gpu_channel_ref(x16: np.ndarray, f: np.ndarray, order: int = 3) -> np.ndarray:
    """GPU path on a host int16 channel -> host int32 result."""
    return np.asarray(
        preresample_channel_gpu(cp.asarray(np.ascontiguousarray(x16), dtype=cp.int16), f, order=order).get()
    )


def main() -> int:
    print(f"device: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    print(f"target spacing: {TARGET_SPACING.tolist()}")
    rng = np.random.default_rng(20260826)

    # Real corpus spacings (research §4/§6); factors = s / 1.5 (oracle direction).
    factor_sets = {
        "s=0.462891^2 z=2.0": np.array([0.462891 / 1.5, 0.462891 / 1.5, 2.0 / 1.5]),
        "s=0.898438^2 z=2.0": np.array([0.898438 / 1.5, 0.898438 / 1.5, 2.0 / 1.5]),
        "s=0.742188^2 z=2.0": np.array([0.742188 / 1.5, 0.742188 / 1.5, 2.0 / 1.5]),
        "s=1.5^2       z=2.0": np.array([1.5 / 1.5, 1.5 / 1.5, 2.0 / 1.5]),
    }
    shapes = [(96, 96, 48), (327, 257, 169), (128, 64, 32)]

    # ------------------------------------------------------------------
    # A. BYTE-IDENTITY: scipy CPU (oracle call) vs cupyx GPU, same input
    # ------------------------------------------------------------------
    print("\n[A] byte-identity scipy vs GPU (same int16 input)")
    for shape in shapes:
        for fname, f in factor_sets.items():
            # HU-range-like data (generated ONCE, used for BOTH paths)
            x_hu = (rng.standard_normal(shape) * 800).astype(np.int16)
            ref = preresample_channel_cpu(x_hu, f)
            check(f"A {shape} {fname} HU-range scipy==GPU", ref, gpu_channel_ref(x_hu, f))
            # f32-chain invariant (08-03, oracle-faithful): the oracle's
            # LoadImaged casts int16 -> f32, scipy zoom writes f32
            # (internally fp64, f64->f32 round-to-even), then the oracle's
            # line-130 astype(int32) TRUNCATES toward zero.
            ref_f32 = ndi_zoom(x_hu.astype(np.float32), f, order=3, mode="nearest")
            check(
                f"A {shape} {fname} wrapper == f32zoom.astype(int32)",
                ref,
                ref_f32.astype(np.int32),
            )
            # int16-extreme-range data
            x_ext = rng.integers(-1024, 3072, size=shape).astype(np.int16)
            check(
                f"A {shape} {fname} extreme-range scipy==GPU",
                preresample_channel_cpu(x_ext, f),
                gpu_channel_ref(x_ext, f),
            )

    # ------------------------------------------------------------------
    # B. SKIP BRANCH + boundary pins
    # ------------------------------------------------------------------
    print("\n[B] skip branch + boundary pins")
    x3 = rng.integers(-1000, 3000, size=(32, 32, 16)).astype(np.int16)
    ones = np.array([1.0, 1.0, 1.0])
    check_bool("B identity factors 3D: out is x3", True, preresample_volume(x3, ones) is x3)
    x4 = x3[np.newaxis]
    check_bool("B identity factors 4D: out is x4", True, preresample_volume(x4, ones) is x4)

    # Pins arbitrate the allclose window (rtol=1e-5 default + atol=1e-3):
    # per-axis |f-1| <= 1.001e-3 skips. The test does NOT "fix" itself to
    # observed behavior — the oracle's np.allclose(atol=1e-3) is the arbiter.
    pins = [
        (1.499, True),  # |f-1| = 6.67e-4 <= 1.01e-3 -> skip
        (1.498, False),  # 1.33e-3 > 1.01e-3          -> resample
        (1.501, True),
        (1.502, False),
    ]
    for s, should_skip in pins:
        f = np.array([s / 1.5, 1.0, 1.0])
        check_bool(
            f"B pin s={s}: should_skip={should_skip}",
            should_skip,
            preresample_should_skip(f),
        )
        if not should_skip:
            # resampling pins must also be byte-identical scipy vs GPU
            check(
                f"B pin s={s} scipy==GPU",
                preresample_channel_cpu(x3, f),
                gpu_channel_ref(x3, f),
            )

    # ------------------------------------------------------------------
    # C. FACTOR MATH
    # ------------------------------------------------------------------
    print("\n[C] factor math")
    ident = np.eye(4)  # column norms = 1.0 mm -> factors 1.0 / 1.5 (NOT [1,1,1])
    check(
        "C identity affine -> 1/1.5 per axis",
        np.array([1.0 / 1.5, 1.0 / 1.5, 1.0 / 1.5]),
        preresample_zoom_factors(ident),
    )

    aff = np.eye(4)
    aff[0, 0], aff[1, 1], aff[2, 2] = 0.462891, 0.462891, 2.0
    expected = np.array([0.462891, 0.462891, 2.0]) / 1.5
    got = preresample_zoom_factors(aff)
    check("C diagonal affine", np.array([0.462891, 0.462891, 2.0]) / 1.5, got)
    check_bool(
        "C diagonal affine exact within 1e-12",
        True,
        bool(np.allclose(got, expected, rtol=0.0, atol=1e-12)),
    )

    # 90-degree rotation about z: column norms still equal the spacings.
    rot = np.eye(4)
    rot[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rot[0, 0] = 0.0
    rot[1, 0] = 0.449219  # col0 = (0, 0.449219, 0)
    rot[1, 1] = 0.0
    rot[0, 1] = -0.449219  # col1 = (-0.449219, 0, 0)
    rot[2, 2] = 2.0  # col2 = (0, 0, 2.0)
    check(
        "C 90-deg rotated columns",
        np.array([0.449219, 0.449219, 2.0]) / 1.5,
        preresample_zoom_factors(rot),
    )

    # ------------------------------------------------------------------
    # D. FLAG SEMANTICS
    # ------------------------------------------------------------------
    print("\n[D] HOLOSCAN_GPU_PRERESAMPLE flag semantics (default ON since P11 flip eea5ac7)")
    saved = os.environ.pop("HOLOSCAN_GPU_PRERESAMPLE", None)
    try:
        check_bool("D unset -> on (default-ON post-flip)", True, gpu_preresample_enabled())
        os.environ["HOLOSCAN_GPU_PRERESAMPLE"] = "0"
        check_bool("D '0' -> off", False, gpu_preresample_enabled())
        os.environ["HOLOSCAN_GPU_PRERESAMPLE"] = "1"
        check_bool("D '1' -> on", True, gpu_preresample_enabled())
        os.environ["HOLOSCAN_GPU_PRERESAMPLE"] = "true"
        check_bool("D 'true' -> off (strictly '1'-only)", False, gpu_preresample_enabled())
    finally:
        if saved is None:
            os.environ.pop("HOLOSCAN_GPU_PRERESAMPLE", None)
        else:
            os.environ["HOLOSCAN_GPU_PRERESAMPLE"] = saved

    # ------------------------------------------------------------------
    # E. VOLUME DISPATCH CONTRACT (3D in -> 3D out, 4D in -> 4D int32 out)
    # ------------------------------------------------------------------
    print("\n[E] volume dispatch contract")
    f = np.array([0.462891 / 1.5, 0.462891 / 1.5, 2.0 / 1.5])
    v3 = preresample_volume(x3, f)
    check_bool("E 3D in -> 3D int32 out", True, v3.ndim == 3 and v3.dtype == np.int32)
    v4 = preresample_volume(x3[np.newaxis], f)
    check_bool(
        "E 4D in -> 4D int32 out",
        True,
        v4.ndim == 4 and v4.dtype == np.int32 and v4.shape[0] == 1,
    )

    # ------------------------------------------------------------------
    # F. ORDER PARAMETERIZATION (Phase 19, 19-03: spec-driven input
    #    pre-resample order — body runs order 1 (TS 2.18 CLI parity),
    #    total keeps the default-3 byte-locked oracle semantics)
    # ------------------------------------------------------------------
    print("\n[F] order parameterization (19-03)")
    # (i) default == explicit order 3 (byte-exact identity on a
    #     non-skip factor pair) — the behavior lock for total.
    fx = np.array([0.462891 / 1.5, 0.462891 / 1.5, 2.0 / 1.5])
    x_o16 = (rng.standard_normal((48, 48, 24)) * 800).astype(np.int16)
    d3 = preresample_channel_cpu(x_o16, fx)
    check(
        "F(i) default == explicit order=3 (CPU, byte-exact)",
        d3,
        preresample_channel_cpu(x_o16, fx, order=3),
    )
    check(
        "F(i) default == explicit order=3 (GPU, byte-exact)",
        gpu_channel_ref(x_o16, fx),
        gpu_channel_ref(x_o16, fx, order=3),
    )
    # volume dispatch: default and order=3 byte-identical (non-skip factors)
    check(
        "F(i) volume default == order=3 (byte-exact)",
        preresample_volume(x_o16[np.newaxis], fx),
        preresample_volume(x_o16[np.newaxis], fx, order=3),
    )
    # (ii) order=1 CPU == inline reference recomputation (the existing
    #      dtype chain with order 1: f32 widen -> scipy zoom order=1
    #      mode=nearest -> f32 output -> int32 truncation).
    ref_o1 = ndi_zoom(x_o16.astype(np.float32), fx, order=1, mode="nearest")
    check(
        "F(ii) order=1 CPU == f32-zoom-order1.astype(int32)",
        preresample_channel_cpu(x_o16, fx, order=1),
        ref_o1.astype(np.int32),
    )
    # (iii) order=1 GPU mirror == order=1 CPU (byte-exact, same input).
    check(
        "F(iii) order=1 GPU == order=1 CPU (byte-exact)",
        preresample_channel_cpu(x_o16, fx, order=1),
        gpu_channel_ref(x_o16, fx, order=1),
    )
    # (iv) skip-branch unchanged: factor~1 returns the SAME object
    #      regardless of the order argument.
    check_bool(
        "F(iv) skip branch order-agnostic (3D, order=1)",
        True,
        preresample_volume(x3, ones, order=1) is x3,
    )
    check_bool(
        "F(iv) skip branch order-agnostic (4D, order=3)",
        True,
        preresample_volume(x4, ones, order=3) is x4,
    )

    print()
    if FAILURES:
        print(f"RESULT: FAIL — {len(FAILURES)} case(s) diverged")
        return 1
    print(
        "RESULT: PASS — every case byte-identical (np.array_equal) "
        "scipy vs GPU; skip/boundary/factor/flag checks green"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
