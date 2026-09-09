#!/usr/bin/env python
"""Phase 29 (HYPER-06 enabler): plan-driven ``use_mirroring`` resolution.

Tests ``config.resolve_use_mirroring`` precedence (per-config plans key >
top-level plans key > True default — cchmc's shipped behavior) plus the
real-bundle integration fact that the 15-checkpoint airway bundle (no
``use_mirroring`` key anywhere in plans.json) still resolves to TTA on, so
29-01 is behavior-neutral by construction.

Plain asserts, runnable headless (the unit cases are pure; the real-bundle
case only loads plans + checkpoint metadata on CPU, no model weights).

Run:  /tmp/monai-env/.venv/bin/python scripts/test_p29_mirroring.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "my_app"))

from config import InferenceParams, load_inference_params, resolve_use_mirroring  # noqa: E402

REAL_BUNDLE = "/raid/monai-deploy-app-sdk/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models"


def _plans(per_config=None, top=None):
    plans: dict = {}
    if top is not None:
        plans["use_mirroring"] = top
    if per_config is not None:
        plans["configurations"] = {"3d_fullres": {"use_mirroring": per_config}}
    elif plans:
        plans["configurations"] = {"3d_fullres": {}}
    return plans


def test_per_config_true():
    assert resolve_use_mirroring(_plans(per_config=True, top=False), "3d_fullres") is True
    print("PASS test_per_config_true")


def test_per_config_wins_over_top():
    # per-config False + top-level True -> False (per-config precedence)
    assert resolve_use_mirroring(_plans(per_config=False, top=True), "3d_fullres") is False
    print("PASS test_per_config_wins_over_top")


def test_top_level_false():
    assert resolve_use_mirroring(_plans(top=False), "3d_fullres") is False
    print("PASS test_top_level_false")


def test_top_level_true():
    assert resolve_use_mirroring(_plans(top=True), "3d_fullres") is True
    print("PASS test_top_level_true")


def test_absent_everywhere_defaults_true():
    # cchmc default (differs from ts-fast's False default — intentional):
    # the shipped 15-ckpt bundle has no key anywhere and must stay TTA-on.
    assert resolve_use_mirroring({}, "3d_fullres") is True
    assert resolve_use_mirroring({"configurations": {"3d_fullres": {}}}, "3d_fullres") is True
    # unknown config name falls through to the default
    assert resolve_use_mirroring({"configurations": {"other": {}}}, "3d_fullres") is True
    print("PASS test_absent_everywhere_defaults_true")


def test_malformed_plans_never_raise():
    assert resolve_use_mirroring(None, "3d_fullres") is True
    assert resolve_use_mirroring({"configurations": None}, "3d_fullres") is True
    # non-dict config entry
    assert resolve_use_mirroring({"configurations": {"3d_fullres": "oops"}}, "3d_fullres") is True
    # non-dict plans
    assert resolve_use_mirroring([1, 2, 3], "3d_fullres") is True
    print("PASS test_malformed_plans_never_raise")


def test_real_bundle_integration():
    # The real 15-ckpt airway bundle: no use_mirroring key anywhere ->
    # True (TTA on, today's behavior), and folds 0..4 (re-confirms the
    # 28-census fold facts).
    params: InferenceParams = load_inference_params(REAL_BUNDLE, "3d_fullres")
    assert params.use_mirroring is True, f"expected use_mirroring True, got {params.use_mirroring!r}"
    assert params.folds == (
        0,
        1,
        2,
        3,
        4,
    ), f"expected folds (0..4), got {params.folds!r}"
    print(f"PASS test_real_bundle_integration (use_mirroring=True, folds={params.folds})")


def main():
    test_per_config_true()
    test_per_config_wins_over_top()
    test_top_level_false()
    test_top_level_true()
    test_absent_everywhere_defaults_true()
    test_malformed_plans_never_raise()
    test_real_bundle_integration()
    print("ALL PASS (test_p29_mirroring)")


if __name__ == "__main__":
    main()
