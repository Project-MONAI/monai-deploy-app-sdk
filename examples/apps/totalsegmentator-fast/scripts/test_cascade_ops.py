#!/usr/bin/env python
"""Phase 22 (plan 22-11, Task 2) unit suite for the resample-None
(`plans_semantic`) cascade main-model I/O path.

Root cause fixed here (22-stage-frame-audit.md): for crop-cascade tasks with
`spec.resample is None` (e.g. pleural_pericard_effusion), TS 2.18 passes the
native crop UNRESAMPLED into nnUNet_predict_image and the nnUNet
preprocessor does: f32 -> crop_to_nonzero -> CTNormalization (BEFORE
resample) -> resampling_fn_data (plans orders 3/0, separate-z auto). The
app used an order-1 int32 world resample followed by in-swin CT
normalization, and back-resampled the LABEL with order-0 zoom instead of
TS's logits order-1/0-sepz resample to crop-native + argmax.

House pattern (mirrors scripts/test_crop_core.py): standalone executable,
NOT pytest. Exits 0 iff every check passes. HEADLESS math (numpy/scipy/
skimage + torch CPU); CUDA visible only because cascade_ops imports cupy.

Checks:
  A. cascade_ops.taskres_plans_semantic: bit-exact vs an INDEPENDENT
     inline replica of the TS chain (layout (1,z,y,x), f32,
     crop_to_nonzero, normalize-BEFORE-resample, sep-z 2D-cubic + nearest
     map_coordinates), real pleural main bundle plans properties.
  B. taskres_plans_semantic props: correct meta2 bookkeeping (layout-order
     shapes, spacings, full-extent bbox on a fully-nonzero crop).
  C. postresample_operator.seg_argmax_to_original_reference: bit-exact vs
     an independent inline replica (logits resample sep-z order-1/0 ->
     softmax -> argmax -> zero-fill revert + transpose-back), including a
     non-full crop bbox revert.
  D. Regression guards: CascadePrepOp.plans_semantic and
     PostResampleOperator.resample_seg_to_original default False
     (explicit-resample tasks take the byte-identical pre-fix paths).

Run:
  cd examples/apps/totalsegmentator-fast
  ulimit -s unlimited && CUDA_VISIBLE_DEVICES=0 \
    /tmp/monai-env/.venv/bin/python scripts/test_cascade_ops.py
Exit: 0 iff all assertions pass.
"""

import inspect
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[2]
MY_APP = APP_ROOT / "my_app"
sys.path.insert(0, str(MY_APP))  # flat import (my_app dir on sys.path)

import numpy as np
from config import load_preprocess_params
from operators import cascade_ops
from operators import postresample_operator as post
from scipy.ndimage import binary_fill_holes, map_coordinates
from skimage.transform import resize

MAIN_BUNDLE = str(
    REPO_ROOT / "ct-totalsegmentator-map" / "models" / "pleural_pericard_effusion" / "pleural_pericard_effusion"
)

PASS = []


def ok(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASS.append(name)
    print(f"  ok: {name}")


def _bbox_full_extent(shape):
    return [[0, int(s)] for s in shape]


def _sep_z_axis(current_sp, new_sp):
    """Independent mirror of _determine_do_sep_z_and_axis (force_sep=None).
    Returns the anisotropic axis or None."""
    if max(current_sp) / min(current_sp) > 3:
        cand = np.where(np.max(current_sp) / np.array(current_sp) == 1)[0]
        return int(cand[0]) if len(cand) == 1 else None
    if max(new_sp) / min(new_sp) > 3:
        cand = np.where(np.max(new_sp) / np.array(new_sp) == 1)[0]
        return int(cand[0]) if len(cand) == 1 else None
    return None


def _sep_z_resample(data, new_shape, current_sp, new_sp, order, order_z):
    """Independent sep-z replica: 2D in-plane resize (skimage) + nearest z
    (scipy map_coordinates, align_corners=False). data: (C, A, B, D)."""
    shape = np.array(data.shape[1:])
    new_shape = np.array(new_shape)
    axis = _sep_z_axis(current_sp, new_sp)
    if axis is None:
        out = np.zeros((data.shape[0], *new_shape))
        for c in range(data.shape[0]):
            out[c] = resize(data[c], new_shape, order, mode="edge", anti_aliasing=False)
        return out
    if axis == 0:
        new_2d = new_shape[1:]
    elif axis == 1:
        new_2d = new_shape[[0, 2]]
    else:
        new_2d = new_shape[:-1]
    out = np.zeros((data.shape[0], *new_shape))
    for c in range(data.shape[0]):
        tmp = new_shape.copy()
        tmp[axis] = shape[axis]
        here = np.zeros(tmp)
        for i in range(int(shape[axis])):
            if axis == 0:
                here[i] = resize(data[c, i], new_2d, order, mode="edge", anti_aliasing=False)
            elif axis == 1:
                here[:, i] = resize(data[c, :, i], new_2d, order, mode="edge", anti_aliasing=False)
            else:
                here[:, :, i] = resize(data[c, :, :, i], new_2d, order, mode="edge", anti_aliasing=False)
        if shape[axis] != new_shape[axis]:
            rows, cols, dim = (int(s) for s in new_shape)
            orows, ocols, odim = here.shape
            r = np.mgrid[:rows, :cols, :dim]
            mr = float(orows) / rows * (r[0] + 0.5) - 0.5
            mc = float(ocols) / cols * (r[1] + 0.5) - 0.5
            md = float(odim) / dim * (r[2] + 0.5) - 0.5
            out[c] = map_coordinates(here, [mr, mc, md], order=order_z, mode="nearest")
        else:
            out[c] = here
    return out


def _independent_ts_chain(vol_xyz, sp_xyz, params):
    """Independent replica of TS: layout (1,z,y,x) -> crop_to_nonzero ->
    normalize (f32) -> resample with plans orders (sep-z when
    2.0/0.5 > 3). Returns (vol_out, props)."""
    data = vol_xyz.astype(np.float32)[None].transpose(0, 3, 2, 1)
    sp = tuple(reversed(sp_xyz))  # (z, y, x) layout order
    tf = [int(i) for i in params.transpose_forward]
    orig_sp = [sp[i] for i in tf]
    # crop_to_nonzero (independent: scipy fill holes + extents)
    mask = data[0] != 0
    mask = binary_fill_holes(mask)
    zmax = mask.shape[0] - int(np.argmax(mask[::-1].any(axis=(1, 2))))
    ymax = mask.shape[1] - int(np.argmax(mask[:, ::-1].any(axis=(0, 2))))
    xmax = mask.shape[2] - int(np.argmax(mask[:, :, ::-1].any(axis=(0, 1))))
    bbox = [[0, zmax], [0, ymax], [0, xmax]]
    data = data[(slice(None),) + tuple(slice(lo, hi) for lo, hi in bbox)]
    shape_after = data.shape[1:]
    new_shape = tuple(int(round(i / j * k)) for i, j, k in zip(orig_sp, params.spacing, shape_after))
    # normalize BEFORE resample (CTNormalization, f32)
    props_c = params.intensity_properties["0"]
    np.clip(data, props_c["percentile_00_5"], props_c["percentile_99_5"], out=data)
    data -= props_c["mean"]
    data /= max(props_c["std"], 1e-8)
    # resample (plans orders)
    out = _sep_z_resample(
        data,
        new_shape,
        orig_sp,
        [float(s) for s in params.spacing],
        params.resample_order,
        params.resample_order_z,
    )
    props = {
        "bbox_used_for_cropping": bbox,
        "shape_before_cropping": [int(s) for s in vol_xyz.shape[::-1]],
        "shape_after_cropping_and_before_resampling": [int(s) for s in shape_after],
        "new_shape": [int(s) for s in new_shape],
        "original_spacing": [float(s) for s in orig_sp],
        "target_spacing": [float(s) for s in params.spacing],
        "transpose_forward": tf,
    }
    return out, props


def main():
    params = load_preprocess_params(MAIN_BUNDLE, "3d_fullres")
    assert list(params.normalization_schemes) == ["CTNormalization"], params.normalization_schemes
    assert tuple(params.spacing) == (1.5, 1.5, 1.5)

    # Synthetic anisotropic crop in (x, y, z) RAS order: 64 x 48 x 24 @
    # (0.5, 0.5, 2.0) mm. Values span the CT window so the percentile clip
    # actually fires (normalize-before-resample is order-sensitive through
    # the non-linear clip).
    rng = np.random.default_rng(2211)
    vol = rng.integers(-1100, 300, size=(64, 48, 24)).astype(np.int32)
    vol[:8, :, :] = -1024  # air band (exercises clip at -962)
    vol[56:, :, :] = 200  # soft-tissue slab
    sp_xyz = (0.5, 0.5, 2.0)

    print("A. taskres_plans_semantic bit-exactness")
    fn = getattr(cascade_ops, "taskres_plans_semantic", None)
    ok("A0 function exists", callable(fn), repr(fn))
    vol_out, props = fn(np.ascontiguousarray(vol), sp_xyz, params)
    exp_out, exp_props = _independent_ts_chain(vol, sp_xyz, params)
    ok(
        "A1 output shape (1,32,16,21)",
        tuple(vol_out.shape) == (1, 32, 16, 21),
        str(vol_out.shape),
    )
    ok("A2 dtype float32", vol_out.dtype == np.float32, str(vol_out.dtype))
    ad = np.abs(vol_out.astype(np.float64) - exp_out.astype(np.float64))
    ok(
        "A3 bit-exact vs independent TS chain",
        bool((vol_out == exp_out).all()),
        f"maxdiff={ad.max():.3e} ndiff={int((vol_out != exp_out).sum())}",
    )

    print("B. meta2 bookkeeping")
    ok(
        "B1 shape_before_cropping (24,48,64) layout",
        tuple(props["shape_before_cropping"]) == (24, 48, 64),
        str(props["shape_before_cropping"]),
    )
    ok(
        "B2 bbox full extent",
        props["bbox_used_for_cropping"] == [[0, 24], [0, 48], [0, 64]],
        str(props["bbox_used_for_cropping"]),
    )
    ok(
        "B3 original_spacing layout order",
        tuple(props["original_spacing"]) == (2.0, 0.5, 0.5),
        str(props["original_spacing"]),
    )
    ok(
        "B4 target_spacing plans",
        tuple(props["target_spacing"]) == (1.5, 1.5, 1.5),
        str(props["target_spacing"]),
    )
    ok(
        "B5 new_shape (32,16,21)",
        tuple(props["new_shape"]) == (32, 16, 21),
        str(props["new_shape"]),
    )
    ok(
        "B6 shape_after == layout crop shape",
        tuple(props["shape_after_cropping_and_before_resampling"]) == (24, 48, 64),
        str(props["shape_after_cropping_and_before_resampling"]),
    )

    print("C. seg_argmax_to_original_reference")
    fn2 = getattr(post, "seg_argmax_to_original_reference", None)
    ok("C0 function exists", callable(fn2), repr(fn2))
    # plans-space logits (3, 32, 16, 21) for the same geometry, tf identity:
    logits = rng.normal(size=(3, 32, 16, 21)).astype(np.float32)
    # make label 2 win a slab so the argmax structure is non-trivial
    logits[2, 10:22, :, :] += 4.0
    meta = {k: list(v) for k, v in exp_props.items()}
    seg_model, seg_dicom = fn2(logits, meta)
    # independent: sep-z resample of LOGITS (order 1 / order_z 0) -> softmax -> argmax
    lr = _sep_z_resample(
        logits,
        exp_props["shape_after_cropping_and_before_resampling"],
        exp_props["original_spacing"],
        exp_props["target_spacing"],
        1,
        0,
    )
    e = np.exp(lr - lr.max(axis=0, keepdims=True))
    probs = e / e.sum(axis=0, keepdims=True)
    exp_seg = probs.argmax(axis=0).astype(np.uint8)
    ok(
        "C1 seg shape crop-native (24,48,64)",
        tuple(seg_dicom.shape) == (24, 48, 64),
        str(seg_dicom.shape),
    )
    ok(
        "C2 bit-exact vs independent chain",
        bool((seg_dicom == exp_seg).all()),
        f"ndiff={int((seg_dicom != exp_seg).sum())}",
    )
    ok(
        "C3 seg_model is plans-space argmax",
        tuple(seg_model.shape) == (32, 16, 21) and bool((seg_model == np.argmax(logits, axis=0)).all()),
    )

    print("C4. non-full crop bbox revert (zero-fill)")
    meta_box = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in exp_props.items()}
    meta_box["shape_before_cropping"] = [30, 48, 64]
    meta_box["bbox_used_for_cropping"] = [[3, 27], [0, 48], [0, 64]]
    _, seg_box = fn2(logits, meta_box)
    ok(
        "C4a outer shape (30,48,64)",
        tuple(seg_box.shape) == (30, 48, 64),
        str(seg_box.shape),
    )
    ok(
        "C4b zero fill outside bbox",
        bool(seg_box[:3].sum() == 0) and bool(seg_box[27:].sum() == 0),
    )
    ok(
        "C4c content inside bbox equals no-crop seg",
        bool((seg_box[3:27] == exp_seg).all()),
    )

    print("D. regression guards (defaults off)")
    sig1 = inspect.signature(cascade_ops.CascadePrepOp.__init__)
    ok(
        "D1 CascadePrepOp.plans_semantic default False",
        sig1.parameters.get("plans_semantic") is not None and sig1.parameters["plans_semantic"].default is False,
    )
    sig2 = inspect.signature(post.PostResampleOperator.__init__)
    ok(
        "D2 PostResampleOperator.resample_seg_to_original default False",
        sig2.parameters.get("resample_seg_to_original") is not None
        and sig2.parameters["resample_seg_to_original"].default is False,
    )

    print(f"\nALL {len(PASS)} CHECKS PASSED")


if __name__ == "__main__":
    main()
