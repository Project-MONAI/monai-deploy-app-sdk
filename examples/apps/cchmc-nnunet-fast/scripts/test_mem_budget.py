#!/usr/bin/env python
"""Unit tests for the memory-budget calculator (INFR-03, D-15).

Plain asserts, runnable headless: ``free_vram_bytes`` is always passed
explicitly, so no GPU (and no live VRAM probe) is needed. The
synthetic sizes are deliberately chosen to force each strategy branch,
including a large-volume set that forces ``defer_to_incremental`` even on a
40 GB budget (D-15: the defer branch must be unit-testable even though the
real OOM path is never exercised on the A100-40GB airway study).

Run:  /tmp/monai-env/.venv/bin/python scripts/test_mem_budget.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "my_app"))

from mem_budget import compute_memory_budget  # noqa: E402

GB = 10**9
FREE_40GB = 40 * GB


def _cfg(name, preprocessed, cropped, heads, channels=1):
    return {
        "config_name": name,
        "num_input_channels": channels,
        "num_segmentation_heads": heads,
        "preprocessed_shape": preprocessed,
        "cropped_shape": cropped,
    }


def test_full_volume_airway_like():
    # Three airway-like configs (fullres / lowres / cascade semantics),
    # 1-channel 240^3 preprocessed, 216^3 cropped, 2 heads each — tiny vs
    # 40 GB, so the plan must be full_volume.
    cfgs = [
        _cfg("3d_fullres", (1, 240, 240, 240), (1, 216, 216, 216), heads=2),
        _cfg("3d_lowres", (1, 240, 240, 240), (1, 216, 216, 216), heads=2),
        _cfg("3d_cascade_fullres", (1, 240, 240, 240), (1, 216, 216, 216), heads=2),
    ]
    plan = compute_memory_budget(cfgs, free_vram_bytes=FREE_40GB)
    assert plan.strategy == "full_volume", f"expected full_volume, got {plan.strategy!r}"
    assert plan.total_bytes < FREE_40GB
    print(f"PASS test_full_volume_airway_like (total {plan.total_bytes / GB:.3f} GB < 40 GB)")


def test_defer_forced_synthetic():
    # Synthetic large volumes that force the defer branch on a 40 GB budget.
    # Per config the preprocessed volume (4, 600, 512, 512) fp32 alone is
    # ~2.5 GB; adding the logits + probability maps, several such configs
    # exceed the 40 GB budget after the safety factor.
    #
    # Note: the plan text sketched 3 configs, but under the plan's own
    # formula 3 configs of these shapes total ~24.4 GB (full_volume at 40
    # GB) — so 5 configs of the same shapes are used here to actually force
    # the defer branch (the D-15 intent; the shape below is unchanged).
    cfgs = [
        _cfg(f"synthetic_cfg{i}", (4, 600, 512, 512), (4, 550, 480, 480), heads=4, channels=4)
        for i in range(5)
    ]
    per_cfg = next(iter(cfgs))
    per_cfg_preprocessed_gb = 4 * 600 * 512 * 512 * 4 / GB
    assert 2.4 < per_cfg_preprocessed_gb < 2.6, (
        f"test premise drifted: per-config preprocessed volume is "
        f"{per_cfg_preprocessed_gb:.2f} GB, expected ~2.5 GB"
    )
    plan = compute_memory_budget(cfgs, free_vram_bytes=FREE_40GB)
    assert plan.strategy == "defer_to_incremental", (
        f"expected defer_to_incremental (total {plan.total_bytes / GB:.2f} GB "
        f"vs {FREE_40GB / GB} GB free), got {plan.strategy!r}"
    )
    print(
        f"PASS test_defer_forced_synthetic (total {plan.total_bytes / GB:.2f} GB "
        f"> 40 GB -> defer_to_incremental)"
    )


def test_boundary():
    # Inclusive <= semantics: free_vram_bytes EXACTLY equal to total_bytes
    # still fits -> full_volume; one byte less -> defer.
    cfgs = [_cfg("boundary", (1, 240, 240, 240), (1, 216, 216, 216), heads=2)]
    plan = compute_memory_budget(cfgs, free_vram_bytes=FREE_40GB)
    at_boundary = compute_memory_budget(cfgs, free_vram_bytes=plan.total_bytes)
    below_boundary = compute_memory_budget(cfgs, free_vram_bytes=plan.total_bytes - 1)
    assert at_boundary.strategy == "full_volume", (
        f"expected full_volume at exactly total_bytes, got {at_boundary.strategy!r}"
    )
    assert below_boundary.strategy == "defer_to_incremental", (
        f"expected defer_to_incremental one byte below total, got {below_boundary.strategy!r}"
    )
    print("PASS test_boundary (total == free -> full_volume; total-1 -> defer)")


def test_per_config_mb_keys():
    cfgs = [
        _cfg("3d_fullres", (1, 240, 240, 240), (1, 216, 216, 216), heads=2),
        _cfg("3d_lowres", (1, 202, 202, 202), (1, 190, 190, 190), heads=2),
        _cfg("3d_cascade_fullres", (2, 240, 240, 240), (2, 216, 216, 216), heads=2, channels=2),
    ]
    plan = compute_memory_budget(cfgs, free_vram_bytes=FREE_40GB)
    assert set(plan.per_config_mb) == {"3d_fullres", "3d_lowres", "3d_cascade_fullres"}, (
        f"per_config_mb keys {sorted(plan.per_config_mb)} do not match the input configs"
    )
    for name, mb in plan.per_config_mb.items():
        assert mb > 0, f"per_config_mb[{name!r}] must be positive, got {mb}"
    print(f"PASS test_per_config_mb_keys ({sorted(plan.per_config_mb)})")


def main():
    failures = 0
    for test in (
        test_full_volume_airway_like,
        test_defer_forced_synthetic,
        test_boundary,
        test_per_config_mb_keys,
    ):
        try:
            test()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {test.__name__}: {e}")
    if failures:
        print(f"RESULT: FAIL ({failures} test(s) failed)")
        sys.exit(1)
    print("RESULT: PASS (all mem_budget tests)")


if __name__ == "__main__":
    main()
