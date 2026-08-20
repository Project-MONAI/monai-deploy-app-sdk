#!/usr/bin/env python
"""D-22 byte-identity arbiter: the CuPy RawKernel zoom (gpu_zoom.py) vs the
scipy/skimage reference chain, per tensor, on ALL real call-site shapes plus
randomized shapes/orders (0/1/3, up and down zoom).

This is the D-22 gate for the gated GPU-resample experiment: ANY case that is
not byte-identical (np.array_equal) fails the run (nonzero exit + a printed
divergence table per case) and keeps the flag OFF (the measured divergence is
recorded in evidence/gpu_resample_verdict.md).

Two comparison levels (mirroring the two call-site chains):

  * PRIMITIVE — ``scipy.ndimage.zoom(x, zoom_factors, order,
    mode='nearest', grid_mode=True)`` (the one primitive all three call sites
    reduce to; the zoom_factors are derived EXACTLY as
    skimage.transform.resize 0.25.2 does: ``factors = in/out`` float64,
    ``zoom_factors = [1/f for f in factors]``) vs ``gpu_zoom_grid_mode``
    (fp32 out for the data path, fp64 out for the seg multihot path).
  * FULL CHAIN — the app's actual chains: skimage ``resize(mode='edge',
    anti_aliasing=False)`` (which is zoom + the fp64 clip to the input
    min/max + fp32 cast) vs the operator's flag-ON chain (kernel with the
    fp64 clip folded in before the RN cast, per channel); and the
    ``_resize_segmentation`` multihot replica (per-label fp64 0/1 mask zoom +
    ``>= 0.5`` threshold) vs the flag-ON GPU multihot path.

Case list (plan Task 1):
  REAL:  (a) 256^3 -> (255,256,255) order 3          (preprocess image, fullres/cascade)
         (b) 256^3 -> (201,202,201) order 3          (preprocess image, lowres)
         (c) 256^3 -> (255,256,255) order 1          (seg multihot 0/1 masks)
         (d) (255,256,255) -> 256^3 order 1, 2 ch    (postresample probabilities)
  RANDOM: 8 seeded cases — axes 128-320, per-axis scale 0.7-1.3 (up AND down),
         orders 0/1/3, fp32 uniform data; full chain + primitive.
  EDGE:   (e) zoom factor exactly 1.0 (200^3 -> 200^3, o1 and o3 — integer
             output coordinates, x=0 spline path)
          (f) non-integer output dim mix (257,259,263) -> (200,300,130) o3
  ORDER-0 SANITY: byte-equality vs scipy is the same level stock CuPy meets at
  order 0 (research §D-22 measured stock cupyx map_coordinates byte-identical
  at order 0 only) — one o0 case is additionally cross-checked against
  cupyx.scipy.ndimage.map_coordinates on the same zoom grid, proving the test
  harness itself is sound before the o1/o3 verdicts are trusted.

Timing note (NOT a gate): per-call kernel ms for case (a) — expected << 1 s
(fp64 A100 budget from the research: 256^3 x 27 taps ~ 2 GFLOP).

Run:  cd examples/apps/cchmc-nnunet-fast
      ulimit -s unlimited && /tmp/monai-env/.venv/bin/python scripts/test_gpu_zoom.py
Exit: 0 iff every case is byte-identical; 1 otherwise.
"""

import os
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
MY_APP = APP_ROOT / "my_app"
sys.path.insert(0, str(MY_APP))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

import cupy as cp  # noqa: E402
import gpu_bootstrap  # noqa: E402  (RMM first — shipping import order)
import numpy as np  # noqa: E402

from operators.gpu_zoom import (  # noqa: E402
    gpu_zoom_grid_mode,
    zoom_factors_for,
)
from scipy.ndimage import zoom as ndi_zoom  # noqa: E402
from skimage.transform import resize as sk_resize  # noqa: E402

FAILURES = []


def divergence_table(name: str, ref: np.ndarray, got: np.ndarray) -> None:
    """Print the per-case divergence table (required output on failure)."""
    diff = ~np.array_equal(ref, got)
    if not diff:
        return
    mask = ref != got
    n = int(mask.sum())
    total = ref.size
    d = (ref - got).astype(np.float64)
    max_abs = float(np.abs(d).max())
    coords = np.argwhere(mask)[:10]
    print(f"  [DIVERGENCE] {name}: {n}/{total} voxels differ ({100.0 * n / total:.4f}%), "
          f"max_abs={max_abs:.3e}")
    print("    first 10 differing coordinates (axis0, axis1, axis2):")
    for c in coords:
        i = tuple(int(v) for v in c)
        print(f"      {i}: ref={ref[c].item()!r} got={got[c].item()!r}")


def check(name: str, ref: np.ndarray, got: np.ndarray) -> None:
    ok = ref.shape == got.shape and ref.dtype == got.dtype and np.array_equal(ref, got)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        if ref.shape != got.shape or ref.dtype != got.dtype:
            print(f"    shape/dtype mismatch: ref {ref.shape} {ref.dtype} vs got {got.shape} {got.dtype}")
        divergence_table(name, ref, got)
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Reference chains (bit-exact replicas of the app call sites)
# ---------------------------------------------------------------------------

def ref_scipy_zoom(x: np.ndarray, new_shape, order: int) -> np.ndarray:
    """The one primitive: scipy.ndimage.zoom(grid_mode=True, mode='nearest')
    with skimage's exact zoom-factor derivation."""
    zf = zoom_factors_for(x.shape, tuple(int(s) for s in new_shape))
    return ndi_zoom(x, zf, order=order, mode="nearest", grid_mode=True)


def ref_data_chain(x: np.ndarray, new_shape, order: int) -> np.ndarray:
    """_resample_to_shape / resample_probabilities_to_shape per channel:
    skimage resize (mode='edge', anti_aliasing=False, clip default True)."""
    return sk_resize(x, tuple(int(s) for s in new_shape), order,
                     mode="edge", anti_aliasing=False)


def ref_resize_segmentation(segmentation: np.ndarray, new_shape, order: int) -> np.ndarray:
    """_resize_segmentation replica (vendored resize_segmentation semantics):
    per-label fp64 multihot resize + >= 0.5 threshold + label assignment."""
    tpe = segmentation.dtype
    if order == 0:
        return sk_resize(segmentation.astype(float), tuple(int(s) for s in new_shape),
                         order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    reshaped = np.zeros(tuple(int(s) for s in new_shape), dtype=tpe)
    for c in np.unique(segmentation.ravel()):
        mask = segmentation == c
        reshaped_multihot = sk_resize(mask.astype(float), tuple(int(s) for s in new_shape),
                                      order, mode="edge", clip=True, anti_aliasing=False)
        reshaped[reshaped_multihot >= 0.5] = c
    return reshaped


# ---------------------------------------------------------------------------
# Flag-ON chains (exactly what the operators do with HOLOSCAN_GPU_RESAMPLE=1)
# ---------------------------------------------------------------------------

def gpu_data_chain(x: np.ndarray, new_shape, order: int) -> np.ndarray:
    """Data path: per channel — exact (order-independent) fp32 min/max on GPU
    (4-byte scalar syncs), kernel zoom with the fp64 clip folded in before
    the RN cast. Byte-equivalent to the skimage chain (clip in fp64 then
    fp32 cast == fp64 clip of the double accumulation then cast)."""
    new_shape = tuple(int(s) for s in new_shape)
    xc = cp.asarray(np.ascontiguousarray(x, dtype=np.float32))
    out = cp.empty((xc.shape[0], *new_shape), dtype=cp.float32)
    zf = zoom_factors_for(xc.shape[1:], new_shape)
    for c in range(xc.shape[0]):
        ch = xc[c]
        out[c] = gpu_zoom_grid_mode(
            ch, zf, order,
            out_shape=new_shape,
            clip_min=float(cp.min(ch).get()),
            clip_max=float(cp.max(ch).get()),
        )
    return np.ascontiguousarray(out.get())


def gpu_seg_multihot(segmentation: np.ndarray, new_shape, order: int) -> np.ndarray:
    """Seg path: per-label fp32 0/1 mask -> GPU double zoom -> fp64 D2H -> the
    SAME >= 0.5 threshold / cast tail the scipy path has (fp64 double
    comparison — exact reference semantics)."""
    new_shape = tuple(int(s) for s in new_shape)
    tpe = segmentation.dtype
    base = cp.asarray(np.ascontiguousarray(segmentation), dtype=cp.float32)
    zf = zoom_factors_for(segmentation.shape, new_shape)
    if order == 0:
        out = gpu_zoom_grid_mode(base, zf, 0, out_shape=new_shape).get()
        return np.asarray(out, dtype=tpe)
    reshaped = np.zeros(new_shape, dtype=tpe)
    for c in np.unique(segmentation.ravel()):
        mask = cp.equal(base, np.float32(c)).astype(cp.float32)
        m = gpu_zoom_grid_mode(mask, zf, order, out_shape=new_shape,
                               output_dtype=cp.float64).get()
        reshaped[m >= 0.5] = c
    return reshaped


def cupy_o0_sanity(x: np.ndarray, new_shape, name: str) -> None:
    """ORDER-0 harness sanity: cross-check the o0 result against an INDEPENDENT
    implementation (stock cupyx.scipy.ndimage.map_coordinates on the same
    zoom grid) — stock CuPy is byte-identical to scipy at order 0 only
    (research §D-22, measured), so a three-way match proves the harness is
    sound before the o1/o3 verdicts are trusted."""
    import cupyx.scipy.ndimage as cndi
    new_shape = tuple(int(s) for s in new_shape)
    zf = zoom_factors_for(x.shape, new_shape)
    grid = []
    for i, f in enumerate(zf):
        cc = np.arange(x.shape[i], dtype=np.float64)
        cc += 0.5
        cc *= f
        cc -= 0.5
        grid.append(cc)
    grid3 = np.stack(np.meshgrid(*[np.broadcast_to(g, new_shape).ravel() for g in
                                   [np.arange(s, dtype=np.float64) for s in new_shape]]),
                     axis=-1)
    # build per-output-voxel coordinates the same way scipy zoom does
    coords = []
    for i, f in enumerate(zf):
        kk = np.arange(new_shape[i], dtype=np.float64)
        kk = kk + 0.5
        kk = kk * f
        kk = kk - 0.5
        shape = [1] * 3
        shape[i] = new_shape[i]
        coords.append(kk.reshape(shape))
    coord_map = np.stack(coords)  # (3, *new_shape)
    got = cndi.map_coordinates(np.ascontiguousarray(x), coord_map, order=0,
                               mode="nearest").astype(np.float32)
    del grid3, grid
    check(f"{name} [order-0 stock-CuPy cross-check]",
          ref_scipy_zoom(np.ascontiguousarray(x, dtype=np.float32), new_shape, 0), got)


def main() -> int:
    print(f"device: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    rng = np.random.default_rng(20260819)
    t0 = None

    # ------------------------------------------------------------------
    # REAL call-site shapes (research §D-22 call-site table)
    # ------------------------------------------------------------------
    # (a) 256^3 -> (255,256,255) order 3 — preprocess image (fullres/cascade)
    xa = rng.uniform(-3.0, 4.0, size=(1, 256, 256, 256)).astype(np.float32)
    new_a = (255, 256, 255)
    t0 = time.perf_counter()
    ref_a = ref_data_chain(xa, new_a, 3)
    got_a = gpu_data_chain(xa, new_a, 3)
    t_kernel_a = (time.perf_counter() - t0) * 1000.0
    check("REAL(a) 256^3->(255,256,255) o3 full chain", ref_a, got_a)
    check("REAL(a) 256^3->(255,256,255) o3 primitive (scipy zoom fp32)",
          ref_scipy_zoom(xa[0], new_a, 3),
          np.ascontiguousarray(gpu_zoom_grid_mode(
              cp.asarray(np.ascontiguousarray(xa[0])),
              zoom_factors_for(xa[0].shape, new_a), 3,
              out_shape=new_a).get()))
    print(f"  [timing] case (a) kernel+D2H per call: {t_kernel_a:.1f} ms "
          f"(gate: none — expected << 1 s)")

    # (b) 256^3 -> (201,202,201) order 3 — preprocess image (lowres)
    xb = rng.uniform(-3.0, 4.0, size=(1, 256, 256, 256)).astype(np.float32)
    new_b = (201, 202, 201)
    check("REAL(b) 256^3->(201,202,201) o3 full chain",
          ref_data_chain(xb, new_b, 3), gpu_data_chain(xb, new_b, 3))
    check("REAL(b) 256^3->(201,202,201) o3 primitive (scipy zoom fp32)",
          ref_scipy_zoom(xb[0], new_b, 3),
          np.ascontiguousarray(gpu_zoom_grid_mode(
              cp.asarray(np.ascontiguousarray(xb[0])),
              zoom_factors_for(xb[0].shape, new_b), 3,
              out_shape=new_b).get()))

    # (c) 256^3 -> (255,256,255) order 1 — seg multihot 0/1 masks
    seg = (rng.random((1, 256, 256, 256)) < 0.25).astype(np.uint8) * 1
    new_c = (255, 256, 255)
    mask64 = (seg[0] == 1).astype(np.float64)
    check("REAL(c) 256^3->(255,256,255) o1 multihot fp64 primitive (scipy zoom)",
          ref_scipy_zoom(mask64, new_c, 1),
          gpu_zoom_grid_mode(
              cp.asarray(np.ascontiguousarray(mask64, dtype=np.float32)),
              zoom_factors_for(mask64.shape, new_c), 1,
              out_shape=new_c, output_dtype=cp.float64).get())
    check("REAL(c) seg multihot labels o1 (>=0.5 threshold tail)",
          ref_resize_segmentation(seg[0], new_c, 1),
          gpu_seg_multihot(seg[0].astype(np.float64), new_c, 1).astype(np.float64))
    check("REAL(c) seg multihot labels o0 (early-return path)",
          ref_resize_segmentation(seg[0], new_c, 0),
          gpu_seg_multihot(seg[0].astype(np.float64), new_c, 0).astype(np.float64))

    # (d) (255,256,255) -> 256^3 order 1, 2 channels — postresample probabilities
    xd = rng.uniform(-2.0, 2.0, size=(2, 255, 256, 255)).astype(np.float32)
    new_d = (256, 256, 256)
    check("REAL(d) (255,256,255)->256^3 o1 2ch full chain",
          ref_data_chain(xd, new_d, 1), gpu_data_chain(xd, new_d, 1))
    for ci in range(2):
        check(f"REAL(d) ch{ci} primitive (scipy zoom fp32)",
              ref_scipy_zoom(xd[ci], new_d, 1),
              np.ascontiguousarray(gpu_zoom_grid_mode(
                  cp.asarray(np.ascontiguousarray(xd[ci])),
                  zoom_factors_for(xd[ci].shape, new_d), 1,
                  out_shape=new_d).get()))

    # ------------------------------------------------------------------
    # RANDOMIZED — 8 seeded cases (axes 128-320, scale 0.7-1.3, orders 0/1/3)
    # ------------------------------------------------------------------
    orders = [0, 1, 3, 3, 1, 0, 1, 3]
    for i, order in enumerate(orders):
        in_shape = tuple(int(rng.integers(128, 321)) for _ in range(3))
        scales = [float(rng.uniform(0.7, 1.3)) for _ in range(3)]
        out_shape = tuple(max(16, min(340, int(round(s * sc))))
                          for s, sc in zip(in_shape, scales))
        x = rng.uniform(-1.5, 2.5, size=(1, *in_shape)).astype(np.float32)
        check(f"RANDOM[{i}] {in_shape}->{out_shape} o{order} full chain",
              ref_data_chain(x, out_shape, order), gpu_data_chain(x, out_shape, order))
        check(f"RANDOM[{i}] {in_shape}->{out_shape} o{order} primitive (scipy zoom)",
              ref_scipy_zoom(x[0], out_shape, order),
              np.ascontiguousarray(gpu_zoom_grid_mode(
                  cp.asarray(np.ascontiguousarray(x[0])),
                  zoom_factors_for(in_shape, out_shape), order,
                  out_shape=out_shape).get()))
        if i == 0:
            cupy_o0_sanity(x[0], out_shape, f"RANDOM[0] o0 sanity {in_shape}->{out_shape}")

    # ------------------------------------------------------------------
    # EDGE cases
    # ------------------------------------------------------------------
    # (e) zoom factor exactly 1.0 — integer output coordinates (x=0 spline path)
    xe = rng.uniform(-2.0, 2.0, size=(1, 200, 200, 200)).astype(np.float32)
    for order in (1, 3):
        check(f"EDGE(e) 200^3->200^3 (zoom 1.0) o{order} full chain",
              ref_data_chain(xe, (200, 200, 200), order),
              gpu_data_chain(xe, (200, 200, 200), order))

    # (f) non-integer output dim mix
    xf = rng.uniform(-2.0, 2.0, size=(1, 257, 259, 263)).astype(np.float32)
    new_f = (200, 300, 130)
    check("EDGE(f) (257,259,263)->(200,300,130) o3 full chain",
          ref_data_chain(xf, new_f, 3), gpu_data_chain(xf, new_f, 3))
    check("EDGE(f) (257,259,263)->(200,300,130) o1 full chain",
          ref_data_chain(xf, new_f, 1), gpu_data_chain(xf, new_f, 1))

    # ------------------------------------------------------------------
    print()
    if FAILURES:
        print(f"RESULT: FAIL — {len(FAILURES)} case(s) diverge (D-22 fallback: "
              f"flag stays OFF, divergence recorded in evidence/gpu_resample_verdict.md)")
        return 1
    print("RESULT: PASS — every case byte-identical (np.array_equal) vs the "
          "scipy/skimage reference chain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
