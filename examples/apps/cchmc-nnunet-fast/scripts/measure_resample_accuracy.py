#!/usr/bin/env python
"""D-22b Step C criterion 1 — per-tensor resample accuracy on the dev corpus.

Compares the flag-ON (stock cupyx.scipy.ndimage) resample path against the
flag-OFF scipy/skimage reference, PER RESAMPLED TENSOR, on the REAL
dev-corpus CT (testdata/airway_input) at the REAL shape pairs / orders of
all four model configs (3d_fullres, 3d_lowres, 3d_cascade_fullres, bundle
= the fullres pair re-ensambled — the resample tensors are config-pairs,
not per-config: image o3 at the fullres and lowres targets, seg o1,
probabilities o1).

Tensors measured (flag OFF = scipy reference, flag ON = stock CuPy):
  * image   — the real cropped+normalized CT through _resample_to_shape,
              at the fullres and lowres target shapes (order 3).
  * seg     — the real lowres SEG argmax labelmap (corpus model output),
              cropped with the image bbox, through _resize_segmentation
              (order 1 multihot + order 0).
  * prob    — corpus-derived 2-channel float volume at the real lowres
              geometry (real label pattern downsampled o0 to the lowres
              shape; no softmax probabilities are persisted by the
              pipeline — order-1 zoom is linear, so 0/1 inputs exercise
              the identical arithmetic), resampled to the real crop shape
              (order 1) through resample_probabilities_to_shape.

D-22b bar: >= 99% of elements equal per resampled tensor (small tolerance
explicitly acceptable; significant divergence is not).

Run:
  cd examples/apps/cchmc-nnunet-fast
  ulimit -s unlimited && CUDA_VISIBLE_DEVICES=0 timeout 1800 \
    /tmp/monai-env/.venv/bin/python scripts/measure_resample_accuracy.py \
    [--json <out>]
"""

import argparse
import json
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent.parent.parent
sys.path.insert(0, str(APP_ROOT / "my_app"))
sys.path.insert(0, str(APP_ROOT / "my_app" / "operators"))

MODEL_ROOT = (REPO_ROOT / "examples" / "apps"
              / "cchmc_nnunet_fifteen_ckpt_app" / "models")
INPUT_DIR = REPO_ROOT / "testdata" / "airway_input"
LOWRES_SEG_DIR = REPO_ROOT / "testdata" / "ref_lowres_only" / "SEG"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import cupy as cp  # noqa: E402
import gpu_bootstrap  # noqa: E402  (RMM first — shipping import order)
import numpy as np  # noqa: E402

from config import load_preprocess_params  # noqa: E402
from preprocess_operator import (  # noqa: E402
    _compute_new_shape,
    _create_nonzero_mask,
    _get_bbox_from_mask,
    _normalize_channel,
    _resample_to_shape,
    _resize_segmentation,
)
from postresample_operator import (  # noqa: E402
    _determine_do_sep_z_and_axis,
    resample_probabilities_to_shape,
)


def set_flag(on: bool) -> None:
    if on:
        os.environ["HOLOSCAN_GPU_RESAMPLE"] = "1"
    else:
        os.environ.pop("HOLOSCAN_GPU_RESAMPLE", None)


def compare(name: str, off, on) -> dict:
    off = np.asarray(off)
    on = np.asarray(on)
    shape_ok = off.shape == on.shape
    if not shape_ok:
        print(f"  [SHAPE MISMATCH] {name}: off {off.shape} vs on {on.shape}")
        return {"tensor": name, "shape_ok": False,
                "elements_equal_pct": 0.0, "max_abs_diff": None,
                "meets_99pct_bar": False}
    eq = float(np.mean(off == on)) * 100.0
    mad = float(np.abs(off.astype(np.float64) - on.astype(np.float64)).max())
    verdict = "MEETS" if eq >= 99.0 else "BELOW BAR"
    print(f"  {name}: shape {off.shape} dtype {off.dtype}  "
          f"equal {eq:.4f}%  max_abs={mad:.3e}  [D-22b >=99%: {verdict}]")
    return {"tensor": name, "shape_ok": True, "shape": list(off.shape),
            "dtype": str(off.dtype), "elements_equal_pct": eq,
            "max_abs_diff": mad, "meets_99pct_bar": eq >= 99.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="write the table to this path")
    args = ap.parse_args()

    from scipy.ndimage import zoom as ndi_zoom  # noqa: E402  (prob surrogate only)

    import pydicom  # noqa: E402

    def load_ct(dcm_dir: Path):
        """(K, I, J) volume + (x, y, z) LPS spacing, the reference
        derivation (affine column norms) built from the DICOM geometry.
        Array axis 0 = slice order (InstanceNumber), 1 = rows, 2 = cols.
        """
        files = sorted(dcm_dir.glob("*.dcm"))
        datasets = [pydicom.dcmread(p) for p in files]
        datasets.sort(key=lambda d: d.InstanceNumber)
        arr = np.stack([d.pixel_array for d in datasets])  # (K, I, J)
        d0 = datasets[0]
        iop = np.asarray(d0.ImageOrientationPatient, dtype=np.float64)
        row_dir, col_dir = iop[:3], iop[3:]
        row_sp, col_sp = (float(v) for v in d0.PixelSpacing)
        pos = [np.asarray(d.ImagePositionPatient, dtype=np.float64)
               for d in datasets]
        slice_vec = pos[1] - pos[0] if len(pos) > 1 else np.array([0, 0, 1.0])
        # affine columns for array axes (K, I, J) -> LPS
        c0 = slice_vec
        c1 = row_dir * row_sp
        c2 = col_dir * col_sp
        spacing_xyz = (float(np.linalg.norm(c0)),
                       float(np.linalg.norm(c1)),
                       float(np.linalg.norm(c2)))
        return np.ascontiguousarray(arr), spacing_xyz

    def load_seg_labelmap(seg_dir: Path) -> np.ndarray:
        """Single-segment BINARY segmentation storage: pixel_array is the
        (K, I, J) 0/1 labelmap directly.
        """
        files = sorted(seg_dir.glob("*.dcm"))
        ds = pydicom.dcmread(files[0])
        arr = ds.pixel_array
        assert arr.ndim == 3 and set(np.unique(arr)) <= {0, 1}, \
            f"unexpected labelmap {arr.shape} {np.unique(arr)[:8]}"
        return np.ascontiguousarray(arr.astype(np.uint8))

    # 1. load the real corpus CT (pydicom; spacing = affine column norms,
    #    LPS — the reference derivation).
    print(f"loading corpus CT from {INPUT_DIR}")
    arr, spacing_xyz = load_ct(INPUT_DIR)
    print(f"  CT array {arr.shape} {arr.dtype}, spacing_xyz={spacing_xyz}")

    # 2. shared layout (mirrors preprocess_reference / preprocess_image):
    #    data (C,X,Y,Z) channel-first; tf from the bundle.
    data0 = np.ascontiguousarray(arr[None, ...])
    spacing = tuple(reversed(spacing_xyz))

    p_full = load_preprocess_params(MODEL_ROOT, "3d_fullres")
    p_low = load_preprocess_params(MODEL_ROOT, "3d_lowres")
    tf = p_full.transpose_forward
    original_spacing = [spacing[i] for i in tf]

    def prepared(params):
        """Crop + normalize exactly like preprocess_reference (CPU fp32)."""
        d = data0.astype(np.float32)
        d = np.ascontiguousarray(d.transpose([0, *[i + 1 for i in tf]]))
        nz = _create_nonzero_mask(d)
        bbox = _get_bbox_from_mask(nz)
        slicer = tuple(slice(*i) for i in bbox)
        d = np.ascontiguousarray(d[(slice(None),) + slicer])
        ns = _compute_new_shape(d.shape[1:], original_spacing, params.spacing)
        nm = nz[slicer][None] if any(params.use_mask_for_norm) else None
        for c in range(d.shape[0]):
            cm = nm[c][0] if nm is not None else None
            d[c] = _normalize_channel(d[c], params, c, cm)
        return np.ascontiguousarray(d), ns

    results = []

    # 3. IMAGE tensor — real CT, both target shapes (o3).
    for label, params in (("fullres/cascade", p_full), ("lowres", p_low)):
        d, ns = prepared(params)
        print(f"image {label}: crop {d.shape[1:]} -> {ns} (order 3)")
        set_flag(False)
        off = _resample_to_shape(d, ns, original_spacing, params.spacing, params)
        set_flag(True)
        on = _resample_to_shape(cp.asarray(d), ns, original_spacing,
                                params.spacing, params)
        if isinstance(on, cp.ndarray):
            on = on.get()
        set_flag(False)
        results.append(compare(
            f"image o3 {label} {d.shape[1:]}->{tuple(ns)}", off, on))

    # 4. SEG tensor — real lowres SEG argmax labelmap (corpus model output),
    #    cropped with the image-derived bbox, through _resize_segmentation.
    seg3 = load_seg_labelmap(LOWRES_SEG_DIR)
    print(f"  lowres SEG labelmap {seg3.shape}, labels {np.unique(seg3)}")
    d_full, ns_full = prepared(p_full)
    bbox = _get_bbox_from_mask(_create_nonzero_mask(data0.astype(np.float32)))
    slicer = tuple(slice(*i) for i in bbox)
    seg_crop = np.ascontiguousarray(seg3[slicer])
    print(f"seg: crop {seg_crop.shape} -> {tuple(ns_full)} (order 1 + 0)")
    set_flag(False)
    off1 = _resize_segmentation(seg_crop, ns_full, 1)
    off0 = _resize_segmentation(seg_crop, ns_full, 0)
    set_flag(True)
    on1 = _resize_segmentation(seg_crop, ns_full, 1)
    on0 = _resize_segmentation(seg_crop, ns_full, 0)
    set_flag(False)
    results.append(compare(
        f"seg o1 multihot {seg_crop.shape}->{tuple(ns_full)}", off1, on1))
    results.append(compare(
        f"seg o0 nearest {seg_crop.shape}->{tuple(ns_full)}", off0, on0))

    # 5. PROB tensor — corpus-derived 2-channel float at the real lowres
    #    geometry -> real crop shape, order 1 (linear zoom: 0/1 channels
    #    exercise the identical arithmetic as softmax probabilities).
    _, ns_low = prepared(p_low)
    prob_shape = tuple(int(s) for s in ns_low)
    seg_low = ndi_zoom(seg3, tuple(prob_shape[i] / seg3.shape[i]
                                   for i in range(3)),
                       order=0, mode="nearest").astype(np.float32)
    prob = np.ascontiguousarray(
        np.stack([1.0 - seg_low, seg_low]))  # (2, *lowres) bg/fg
    crop_shape = d_full.shape[1:]  # real crop (pre-resample) geometry
    cur = list(p_low.spacing)
    tgt = list(p_full.spacing)
    do_sep, _axis = _determine_do_sep_z_and_axis(None, cur, tgt)
    set_flag(False)
    offp = resample_probabilities_to_shape(prob, crop_shape, cur, tgt, order=1)
    set_flag(True)
    onp = resample_probabilities_to_shape(prob, crop_shape, cur, tgt, order=1)
    set_flag(False)
    print(f"prob: (2, {prob_shape}) -> {tuple(crop_shape)} (order 1), "
          f"do_separate_z={do_sep}")
    results.append(compare(
        f"prob o1 2ch (2,{prob_shape})->{tuple(crop_shape)}", offp, onp))

    print()
    all_ok = all(r.get("meets_99pct_bar") for r in results)
    print(f"STEP 6 per-tensor bar (>=99% equal): "
          f"{'ALL MEET' if all_ok else 'NOT MET'}")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"corpus": str(INPUT_DIR), "results": results,
             "all_meet_99pct": all_ok}, indent=2))
        print(f"json: {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
