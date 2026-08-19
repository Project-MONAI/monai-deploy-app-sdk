"""Byte-identity probe: scipy.ndimage.zoom/map_coordinates vs CuPy replicas.

Questions (for the D-22 gated GPU-resample experiment):
Q1: does scipy compute map_coordinates/zoom internally in float64 even for
    float32 input? (compare fp32-input output vs explicit-fp64-compute output)
Q2: is cupyx.scipy.ndimage.map_coordinates bit-identical to scipy at
    order 0 / 1 / 3 in fp32 and in fp64, mode='nearest' (skimage 'edge')?
Q3: full _resample_to_shape chain (skimage resize == ndi.zoom grid_mode)
    scipy-vs-CuPy replica at the actual bundle shapes (order 3, order 1,
    order 0-seg path, separate-z map_coordinates order_z 0/3).
"""
import numpy as np

np.random.seed(0)


def zoom_scipy(x, new_shape, order, mode="nearest"):
    from scipy.ndimage import zoom

    shape = x.shape
    zf = [float(s) / float(n) for s, n in zip(shape, new_shape)]
    return zoom(x, zf, order=order, mode=mode, grid_mode=True)


def mapc_scipy(x, coords, order, mode="nearest"):
    from scipy.ndimage import map_coordinates

    return map_coordinates(x, coords, order=order, mode=mode)


def mapc_cupy(x, coords, order, mode="nearest", dtype=None):
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates

    xf = cp.asarray(x.astype(dtype) if dtype is not None else x)
    cf = cp.asarray(np.stack([np.asarray(c) for c in coords]))
    return map_coordinates(xf, cf, order=order, mode=mode).get()


def coords_for(old, new):
    rows, cols, dim = (int(s) for s in new)
    orig = [int(s) for s in old]
    mr, mc, md = np.mgrid[:rows, :cols, :dim]
    cr = float(orig[0]) / rows * (mr + 0.5) - 0.5
    cc = float(orig[1]) / cols * (mc + 0.5) - 0.5
    cd = float(orig[2]) / dim * (md + 0.5) - 0.5
    return np.array([cr, cc, cd])


def cmp(tag, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        print(f"{tag}: SHAPE/DTYPE MISMATCH {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
        return
    eq = np.array_equal(a, b)
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    n = int((d > 0).sum())
    print(f"{tag}: byte_equal={eq}  n_diff={n}/{a.size}  max_abs={d.max():.3e}")


x3 = (np.random.rand(96, 80, 100) * 2000 - 1000).astype(np.float32)  # CT-like intensity
xseg = np.random.randint(0, 3, size=(64, 64, 64)).astype(np.uint8)

new = (84, 70, 88)

# Q1: scipy fp32-input vs explicit fp64 compute (is scipy upcasting internally?)
s32 = zoom_scipy(x3, new, 3)
s64 = zoom_scipy(x3.astype(np.float64), new, 3).astype(np.float32)
cmp("Q1 zoom o3: scipy(f32) vs scipy(f64).astype(f32)", s32, s64)
s32_1 = zoom_scipy(x3, new, 1)
s64_1 = zoom_scipy(x3.astype(np.float64), new, 1).astype(np.float32)
cmp("Q1 zoom o1: scipy(f32) vs scipy(f64).astype(f32)", s32_1, s64_1)

# Q2: CuPy map_coordinates (order 0/1/3) vs scipy, via the zoom coordinate map
# (reproduce zoom's grid: scipy zoom grid_mode uses the same scale*(c+0.5)-0.5 map)
co = coords_for(x3.shape, new)
for order in (0, 1, 3):
    ref = mapc_scipy(x3, co, order)
    for dt in (np.float32, np.float64):
        try:
            got = mapc_cupy(x3, co, order, dtype=dt)
        except Exception as e:  # noqa: BLE001
            print(f"Q2 mapc o{order} {np.dtype(dt).name}: FAILED {type(e).__name__}: {e}")
            continue
        tag = f"Q2 mapc o{order} {np.dtype(dt).name}: scipy vs cupy"
        cmp(tag, ref if dt is np.float32 else ref.astype(np.float64),
            got if dt is np.float32 else got)

# Q3: seg multihot path (order 1 on 0/1 mask, >=0.5 threshold) + order 0
mask = (xseg == 1).astype(np.float32)
ref1 = zoom_scipy(mask, new, 1)
got1 = mapc_cupy(mask, coords_for(mask.shape, new), 1)
cmp("Q3 seg multihot o1: scipy-zoom vs cupy-mapc f32", ref1, got1)
ref0 = zoom_scipy(mask, new, 0)
got0 = mapc_cupy(mask, coords_for(mask.shape, new), 0)
cmp("Q3 seg multihot o0: scipy-zoom vs cupy-mapc f32", ref0, got0)

# Q3b: separate-z map_coordinates order_z=0 on the 2D-resized stack (seg path)
stack = np.stack([zoom_scipy((xseg == c).astype(np.float32), new, 1) for c in (1, 2)], axis=0)
st64 = stack.astype(np.float64)
old3 = np.array(stack[0].shape, dtype=float)
coz = coords_for(tuple(int(v) for v in old3), (96, 90, 110))
r0 = mapc_scipy(stack, coz, 0)
g0 = mapc_cupy(stack, coz, 0)
cmp("Q3b sep-z mapc o0: scipy vs cupy f32", r0, g0)
r0d = mapc_scipy(stack, coz, 0).astype(np.float64)
g0d = mapc_cupy(stack, coz, 0, dtype=np.float64)
cmp("Q3b sep-z mapc o0: scipy(f32) vs cupy(f64)", r0d, g0d)
