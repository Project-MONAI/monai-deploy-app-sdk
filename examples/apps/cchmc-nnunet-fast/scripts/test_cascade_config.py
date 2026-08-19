#!/usr/bin/env python
"""Unit tests for the cascade plumbing (PIPE-03/PIPE-04, Phase 2 Plan 03).

Covers, against the REAL bundle (the reference semantics live in
``plans.json``/``dataset.json``, not in test fixtures):

* ``resolve_run_model_list`` — the reference app's model-list semantics
  (default = plans.json configs with model dirs, plans.json order; lowres
  reordered immediately before cascade; ensemble = run list minus 3d_lowres),
  plus the data-driven previous-stage auto-insertion (the documented fast-app
  divergence from the reference's crash-on-cascade-only behavior).
* ``load_preprocess_params`` — the new cascade fields (previous_stage,
  resample_seg_* kwargs resolved through PlansManager, foreground_labels).
* ``load_inference_params`` — ``num_input_channels == 2`` for the cascade
  config (SlideWindow is already config-driven off this value).

Later tasks in this plan extend the file with the seg-resample replica vs
the vendored nnunetv2 2.8.1, the GPU one-hot vs the vendored
``convert_labelmap_to_one_hot``, and ``revert_crop_gpu``.

Run:  /tmp/monai-env/.venv/bin/python scripts/test_cascade_config.py
(exits non-zero on the first failure; prints one PASS line per case).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "my_app"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "my_app" / "operators"))

from config import (  # noqa: E402
    load_inference_params,
    load_preprocess_params,
    resolve_run_model_list,
)

MODEL_ROOT = Path(
    "/users/srv-mde/projects/monai-deploy-app-sdk/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models"
)


def load_plans():
    return json.loads((MODEL_ROOT / "jsonpkls" / "plans.json").read_text())


# ---------------------------------------------------------------------------
# resolve_run_model_list (PIPE-03): reference semantics + auto-insertion
# ---------------------------------------------------------------------------


def test_default_model_list():
    plans = load_plans()
    run, ensemble = resolve_run_model_list(None, plans, MODEL_ROOT)
    # 2d is filtered out (no model dir); lowres reordered before cascade;
    # ensemble = run minus 3d_lowres.
    assert run == ("3d_fullres", "3d_lowres", "3d_cascade_fullres"), f"got {run}"
    assert ensemble == ("3d_fullres", "3d_cascade_fullres"), f"got {ensemble}"
    print("PASS test_default_model_list")


def test_cascade_only_auto_inserts_previous_stage():
    plans = load_plans()
    run, ensemble = resolve_run_model_list(["3d_cascade_fullres"], plans, MODEL_ROOT)
    # Documented divergence: the reference CRASHES on cascade-only (missing
    # tmp/3d_lowres/Img_in_context.nii.gz); the in-memory cascade requires
    # the previous stage to be auto-inserted (data-driven off plans.json).
    assert run == ("3d_lowres", "3d_cascade_fullres"), f"got {run}"
    assert ensemble == ("3d_cascade_fullres",), f"got {ensemble}"
    print("PASS test_cascade_only_auto_inserts_previous_stage")


def test_lowres_cascade_explicit():
    plans = load_plans()
    run, ensemble = resolve_run_model_list(["3d_lowres", "3d_cascade_fullres"], plans, MODEL_ROOT)
    assert run == ("3d_lowres", "3d_cascade_fullres"), f"got {run}"
    assert ensemble == ("3d_cascade_fullres",), f"got {ensemble}"
    print("PASS test_lowres_cascade_explicit")


def test_standalone_lowres_runs_with_self_ensemble():
    plans = load_plans()
    # Documented fast-app extension (Phase 2 Plan 04): lowres-only must be
    # RUNNABLE (HOLOSCAN_MODEL_LIST=3d_lowres -> run=[3d_lowres],
    # ensemble=[3d_lowres], E2E exit 0 per the plan's 4-configuration table,
    # D-07 standalone gate). The reference app raises ValueError here instead
    # (nnunet_seg_operator.py:96-98 — it cannot read a lowres-only
    # probability map back from disk); the fast app's in-memory DAG can.
    # A truly empty list still raises the reference's exact error (below).
    run, ensemble = resolve_run_model_list(["3d_lowres"], plans, MODEL_ROOT)
    assert run == ("3d_lowres",), f"got {run}"
    assert ensemble == ("3d_lowres",), f"got {ensemble}"
    try:
        resolve_run_model_list([], plans, MODEL_ROOT)
    except ValueError as e:
        assert str(e) == (
            "At least one non-auxiliary model configuration is required "
            "for ensemble inference."
        ), f"got {e!r}"
    else:
        raise AssertionError("expected ValueError for empty run list")
    print("PASS test_standalone_lowres_runs_with_self_ensemble (plan 04 extension; empty list still raises reference error)")


def test_standalone_fullres():
    plans = load_plans()
    run, ensemble = resolve_run_model_list(["3d_fullres"], plans, MODEL_ROOT)
    assert run == ("3d_fullres",), f"got {run}"
    assert ensemble == ("3d_fullres",), f"got {ensemble}"
    print("PASS test_standalone_fullres")


def test_reference_reorder_with_reversed_pair():
    plans = load_plans()
    run, ensemble = resolve_run_model_list(
        ["3d_cascade_fullres", "3d_lowres"], plans, MODEL_ROOT
    )
    # Reference reorder (nnunet_seg_operator.py:92-95): lowres must end up
    # immediately before cascade.
    assert run == ("3d_lowres", "3d_cascade_fullres"), f"got {run}"
    assert ensemble == ("3d_cascade_fullres",), f"got {ensemble}"
    print("PASS test_reference_reorder_with_reversed_pair")


# ---------------------------------------------------------------------------
# load_preprocess_params: cascade fields (PIPE-04)
# ---------------------------------------------------------------------------


def test_preprocess_params_cascade():
    p = load_preprocess_params(MODEL_ROOT, "3d_cascade_fullres")
    assert p.previous_stage == "3d_lowres", f"got {p.previous_stage!r}"
    # The raw cascade entry has resampling_fn_seg_kwargs null (inherits_from
    # 3d_fullres) — these values must come from the PlansManager-resolved
    # configuration (venv-verified: is_seg=True, order=1, order_z=0,
    # force_separate_z=None).
    assert p.resample_seg_order == 1, f"got {p.resample_seg_order}"
    assert p.resample_seg_order_z == 0, f"got {p.resample_seg_order_z}"
    assert p.resample_seg_force_separate_z is None, f"got {p.resample_seg_force_separate_z}"
    # dataset.json {'airway': 1, 'background': 0} -> (1,)
    assert p.foreground_labels == (1,), f"got {p.foreground_labels}"
    print("PASS test_preprocess_params_cascade")


def test_preprocess_params_fullres_not_cascade():
    p = load_preprocess_params(MODEL_ROOT, "3d_fullres")
    assert p.previous_stage is None, f"got {p.previous_stage!r}"
    print("PASS test_preprocess_params_fullres_not_cascade")


def test_inference_params_cascade_two_input_channels():
    ip = load_inference_params(MODEL_ROOT, "3d_cascade_fullres")
    assert ip.num_input_channels == 2, f"got {ip.num_input_channels}"
    print("PASS test_inference_params_cascade_two_input_channels")


# ---------------------------------------------------------------------------
# Task 2: CPU seg-resample replica vs the VENDORED nnunetv2 2.8.1 reference,
# and the GPU one-hot vs the vendored convert_labelmap_to_one_hot.
# (CPU-reference replicas are validated against the vendored reference here;
# D-11's final-gate-only rule applies to the CuPy port as a whole.)
# ---------------------------------------------------------------------------


def test_seg_resample_replica():
    import numpy as np

    from nnunetv2.preprocessing.resampling.default_resampling import (
        compute_new_shape,
        resample_data_or_seg_to_shape,
    )

    from preprocess_operator import _resample_seg_to_shape

    params = load_preprocess_params(MODEL_ROOT, "3d_cascade_fullres")
    assert (params.resample_seg_order, params.resample_seg_order_z, params.resample_seg_force_separate_z) == (
        1,
        0,
        None,
    )

    rng = np.random.default_rng(0)
    cases = [
        # (shape, current_spacing, new_spacing, note)
        # 1) new_shape == shape -> the "no resampling necessary" short-circuit
        ((64, 64, 64), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), "short-circuit"),
        # 2) plain anisotropy-free resize (no separate-z branch)
        ((48, 96, 128), (1.0, 1.0, 1.0), (1.0, 1.0, 1.5), "plain resize"),
        # 3) current spacing anisotropy 1.2/0.3 = 4 > 3 -> separate-z branch,
        #    axis 2 only (1.2/1.2 == 1), and new_shape[axis] != shape[axis]
        #    (107 != 64) -> also exercises the map_coordinates order_z=0 pass
        ((100, 64, 64), (0.3, 1.0, 1.2), (1.0, 1.0, 2.0), "separate-z + map_coordinates"),
    ]
    for i, (shape, cur, new, note) in enumerate(cases):
        seg = np.ascontiguousarray(rng.integers(0, 2, size=(1, *shape), dtype=np.uint8))
        new_shape = compute_new_shape(shape, cur, new)
        got = _resample_seg_to_shape(seg, new_shape, cur, new, params)
        ref = resample_data_or_seg_to_shape(
            seg, new_shape, cur, new, is_seg=True, order=1, order_z=0, force_separate_z=None
        )
        assert got.dtype == np.uint8, f"case {i} ({note}): dtype {got.dtype}"
        assert np.array_equal(got, ref), f"case {i} ({note}): replica != vendored reference"
    print("PASS test_seg_resample_replica (3 shape regimes, np.array_equal vs vendored nnunetv2)")


def test_one_hot_vs_reference():
    import cupy as cp
    import numpy as np

    from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot

    rng = np.random.default_rng(1)
    seg = rng.integers(0, 2, size=(16, 40, 48), dtype=np.uint8)
    ref = convert_labelmap_to_one_hot(seg, [1], np.float32)
    assert ref.shape == (1, *seg.shape)

    # CPU twin of the GPU op: (seg == 1).astype(float32), channel axis 0.
    got_np = (seg == 1).astype(np.float32)[None, ...]
    assert np.array_equal(got_np, ref), "CPU one-hot != vendored convert_labelmap_to_one_hot"

    # The GPU op itself: same expression in CuPy, round-tripped.
    got_cp = ((cp.array(seg) == 1).astype(cp.float32)).get()[None, ...]
    assert np.array_equal(got_cp, ref), "CuPy one-hot != vendored convert_labelmap_to_one_hot"
    print("PASS test_one_hot_vs_reference (CPU + CuPy, np.array_equal vs vendored)")


def test_revert_crop_gpu():
    import numpy as np
    import torch

    from postresample_operator import revert_crop_gpu

    rng = np.random.default_rng(2)
    shape_before = (30, 40, 50)  # post-transpose order
    tf = (1, 2, 0)  # non-identity transpose_forward
    # bbox not touching any border, in post-transpose axis order
    bbox = [[4, 24], [6, 34], [8, 42]]
    crop_shape = tuple(hi - lo for lo, hi in bbox)
    seg_crop_np = rng.integers(0, 2, size=crop_shape, dtype=np.uint8)
    meta = {
        "shape_before_cropping": shape_before,
        "bbox_used_for_cropping": bbox,
        "transpose_forward": list(tf),
    }

    got = revert_crop_gpu(torch.as_tensor(seg_crop_np), dict(meta)).cpu().numpy()

    # Hand-rolled numpy reference of the same fill/insert/permute steps
    # (reference export_prediction.py: zeros fill -> insert at bbox ->
    # .transpose(transpose_backward)).
    tb = [list(tf).index(i) for i in range(len(tf))]
    full = np.zeros(shape_before, dtype=np.uint8)
    slicer = tuple(slice(lo, hi) for lo, hi in bbox)
    full[slicer] = seg_crop_np
    # outside the bbox (pre-permute) is background; inside equals the crop
    outside_mask = np.ones(shape_before, dtype=bool)
    outside_mask[slicer] = False
    assert int(full[outside_mask].sum()) == 0, "background fill must be 0 outside the bbox"
    assert np.array_equal(full[slicer], seg_crop_np), "insert must place the crop at the bbox"
    ref = full.transpose(tb)

    # The revert returns the ORIGINAL (pre-transpose) array order — the same
    # order as image.asnumpy() (Task 2's orientation contract) — so the
    # output shape is shape_before_cropping permuted by transpose_backward
    # (identical to the 4D probability revert's shape behavior; ==
    # shape_before_cropping only for the identity transpose_forward).
    assert got.shape == ref.shape, f"got {got.shape}, expected {ref.shape}"
    assert got.dtype == np.uint8, f"got {got.dtype}"
    assert np.array_equal(got, ref), "revert_crop_gpu != hand-rolled numpy reference"
    print("PASS test_revert_crop_gpu (fill/insert/permute, np.array_equal)")


def test_ensemble_order():
    """Phase 2 Plan 04 (INF-009/D-19): the ensemble mean is
    bit-deterministic in ensemble_model_list ORDER — the Phase 1 code
    path (in-place ``+=`` accumulation in list order + CuPy exact final
    division) is UNCHANGED; this test pins the order semantics and the
    list-order reconstruction compute() uses (arrival order is not
    guaranteed by GXF).

    IEEE float32 addition is commutative (a+b == b+a bitwise), so the
    order observable here is: (1) the result is bit-identical to the
    manual reference-order accumulation ``t0 + t1`` then CuPy ``/ 2``,
    and (2) reconstructing the list from a dict of REVERSED arrival
    positions in ensemble order yields the same bit-exact result.
    """
    import torch

    from ensemble_average_operator import _divide_refparity, average_probabilities

    assert torch.cuda.is_available(), "test_ensemble_order requires CUDA"
    rng = torch.Generator(device="cuda").manual_seed(7)
    t0 = torch.rand((2, 64, 64, 64), generator=rng, device="cuda", dtype=torch.float32)
    t1 = torch.rand((2, 64, 64, 64), generator=rng, device="cuda", dtype=torch.float32)

    # (1) list-order result == manual reference-order accumulation
    #     (first volume is the base; += in order; CuPy exact final /2).
    got = average_probabilities([t0.clone(), t1.clone()])
    ref = t0.clone()
    ref += t1
    ref = _divide_refparity(ref, 2)
    assert torch.equal(got, ref), "ensemble result != manual reference-order accumulation"

    # (2) list-order reconstruction under a REVERSED arrival order: the
    #     receive-mapping logic compute() uses (label by config, rebuild
    #     in ensemble order) must produce the same bit-exact tensor.
    ensemble_order = ["3d_fullres", "3d_cascade_fullres"]
    received = {
        "3d_fullres": t1.clone(),  # arrived first — WRONG position
        "3d_cascade_fullres": t0.clone(),  # arrived second
    }
    tensors = [received[cfg] for cfg in ensemble_order]  # compute() reconstruction
    got_reordered = average_probabilities([v.clone() for v in tensors])
    assert torch.equal(got_reordered, got), (
        "list-order reconstruction did not yield the ensemble-order result"
    )
    print("PASS test_ensemble_order (reference-order accumulation + list-order reconstruction, torch.equal)")


def test_preprocess_image_cascade_two_channel():
    """Phase 2 Plan 04 (PIPE-04): the cascade 2-channel path of
    ``preprocess_image`` — regression guard for the 5D one-hot bug
    (the resampled seg is 4D (1, *spatial); one-hotting it directly
    stacked to 5D and broke the channel concatenate with the image)."""
    import numpy as np
    import torch

    from preprocess_operator import PreprocessOperator

    class FakeImage:
        def __init__(self, arr):
            self._arr = arr

        def asnumpy(self):
            return self._arr

        def metadata(self):
            aff = np.eye(4, dtype=np.float64)
            return {"nifti_affine_transform": aff}

    from monai.deploy.core import Application

    app = Application(["prog"])
    op = PreprocessOperator(
        app,
        model_path=MODEL_ROOT,
        config_name="3d_cascade_fullres",
    )
    rng = np.random.default_rng(3)
    img = rng.integers(0, 300, size=(40, 48, 56)).astype(np.uint8)
    seg = rng.integers(0, 2, size=img.shape).astype(np.uint8)  # same array order as the image (orientation contract)
    vol, props = op.preprocess_image(FakeImage(img), torch.as_tensor(seg).to("cuda"))

    assert vol.ndim == 4, f"expected 4D (2, *spatial) cascade input, got ndim {vol.ndim}"
    assert vol.shape[0] == 2, f"expected 2 channels (image + one-hot), got {vol.shape[0]}"
    assert tuple(vol.shape[1:]) == tuple(int(s) for s in props["new_shape"]), (
        f"spatial shape {vol.shape[1:]} != new_shape {props['new_shape']}"
    )
    one_hot = vol[1]
    oh_np = one_hot.get()
    assert oh_np.dtype == np.float32, f"one-hot dtype {oh_np.dtype}"
    assert set(np.unique(oh_np).tolist()) <= {0.0, 1.0}, "one-hot must be 0/1"
    print("PASS test_preprocess_image_cascade_two_channel (2-channel (image, one-hot) fp32, 4D)")


if __name__ == "__main__":
    test_default_model_list()
    test_cascade_only_auto_inserts_previous_stage()
    test_lowres_cascade_explicit()
    test_standalone_lowres_runs_with_self_ensemble()
    test_standalone_fullres()
    test_reference_reorder_with_reversed_pair()
    test_preprocess_params_cascade()
    test_preprocess_params_fullres_not_cascade()
    test_inference_params_cascade_two_input_channels()
    test_seg_resample_replica()
    test_one_hot_vs_reference()
    test_revert_crop_gpu()
    test_ensemble_order()
    test_preprocess_image_cascade_two_channel()
    print("ALL PASS (plan 03: model-list, cascade params, seg-resample replica, one-hot, revert_crop_gpu; plan 04: ensemble order)")
