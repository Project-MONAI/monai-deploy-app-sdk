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


def test_standalone_lowres_raises_reference_error():
    plans = load_plans()
    # Reference semantics (nnunet_seg_operator.py:96-98, replicated exactly
    # per PIPE-03 / the plan's must-have "ensemble = run list minus
    # 3d_lowres; error when ensemble empty"): a run list of ONLY the
    # auxiliary lowres config yields an empty ensemble -> ValueError with
    # the reference's exact message. (The plan's task-1 case list sketch
    # suggested standalone lowres is ensemblable — that contradicts the
    # reference being replicated; the reference is controlling. See SUMMARY
    # deviations.)
    try:
        resolve_run_model_list(["3d_lowres"], plans, MODEL_ROOT)
    except ValueError as e:
        assert str(e) == (
            "At least one non-auxiliary model configuration is required "
            "for ensemble inference."
        ), f"got {e!r}"
    else:
        raise AssertionError("expected ValueError for lowres-only run list")
    print("PASS test_standalone_lowres_raises_reference_error")


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


if __name__ == "__main__":
    test_default_model_list()
    test_cascade_only_auto_inserts_previous_stage()
    test_lowres_cascade_explicit()
    test_standalone_lowres_raises_reference_error()
    test_standalone_fullres()
    test_reference_reorder_with_reversed_pair()
    test_preprocess_params_cascade()
    test_preprocess_params_fullres_not_cascade()
    test_inference_params_cascade_two_input_channels()
    test_seg_resample_replica()
    test_one_hot_vs_reference()
    print("ALL PASS (plan 03: model-list semantics, cascade params, seg-resample replica, one-hot)")
