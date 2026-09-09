#!/usr/bin/env python
"""Phase 29 (CFG04-05 + CFG04-07): per-consumer cascade prev-stage mapping.

Tests ``app._auxiliary_prev_stages`` (replaces the single-aux assumption
``_auxiliary_prev_stage`` — CFG04-05: N cascade consumers each map to their
own previous stage) and ``app._validate_cascade_producers`` (CFG04-07: a
cascade consumer whose plans ``previous_stage`` is absent from the run list
must fail fast with a named error instead of building a DAG with an unwired
``lowres_seg`` input port that would hang at runtime).

Synthetic plans dicts only — no bundle files, no model weights; importing
``app`` headless is proven by test_mem_budget.py's pattern (gpu_bootstrap +
monai/holoscan import cleanly in the venv).

Run:
      cd examples/apps/cchmc-nnunet-fast
      /tmp/monai-env/.venv/bin/python scripts/test_p29_cascade_wiring.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "my_app"))

from app import _auxiliary_prev_stages, _validate_cascade_producers  # noqa: E402


def _plans(entries):
    """Synthetic plans.json with one configurations entry per (cfg, prev) pair."""
    return {"configurations": {cfg: ({"previous_stage": prev} if prev is not None else {}) for cfg, prev in entries}}


def test_single_cascade_regression():
    # The real-bundle shape (single cascade): same answer as today's
    # _auxiliary_prev_stage single-prev path.
    plans = _plans(
        [
            ("3d_fullres", None),
            ("3d_lowres", None),
            ("3d_cascade_fullres", "3d_lowres"),
            ("2d", None),
        ]
    )
    run = ["3d_fullres", "3d_lowres", "3d_cascade_fullres"]
    mapping = _auxiliary_prev_stages(run, plans)
    assert mapping == {"3d_cascade_fullres": "3d_lowres"}, f"got {mapping!r}"
    print("PASS test_single_cascade_regression")


def test_two_cascade_consumers():
    # CFG04-05: two cascade configs with DISTINCT previous stages — today's
    # _auxiliary_prev_stage returns None (the single-aux assumption).
    plans = _plans([("c1", "p1"), ("c2", "p2"), ("p1", None), ("p2", None)])
    run = ["c1", "c2", "p1", "p2"]
    mapping = _auxiliary_prev_stages(run, plans)
    assert mapping == {"c1": "p1", "c2": "p2"}, f"got {mapping!r}"
    print("PASS test_two_cascade_consumers")


def test_shared_previous_stage():
    # Two consumers sharing ONE previous stage: the producer is the mapping
    # value twice; set(values()) dedup gives one emitter.
    plans = _plans([("c1", "p"), ("c2", "p"), ("p", None)])
    run = ["c1", "c2", "p"]
    mapping = _auxiliary_prev_stages(run, plans)
    assert mapping == {"c1": "p", "c2": "p"}, f"got {mapping!r}"
    assert set(mapping.values()) == {"p"}
    print("PASS test_shared_previous_stage")


def test_non_cascade_empty_mapping():
    plans = _plans([("3d_fullres", None), ("3d_lowres", None)])
    mapping = _auxiliary_prev_stages(["3d_fullres", "3d_lowres"], plans)
    assert mapping == {}, f"got {mapping!r}"
    print("PASS test_non_cascade_empty_mapping")


def test_missing_producer_raises_named_error():
    # CFG04-07: consumer present, its previous_stage declared in plans but
    # ABSENT from the run list -> fail fast; the message must name BOTH the
    # consumer and the missing stage.
    plans = _plans([("3d_cascade_fullres", "3d_lowres")])
    run = ["3d_cascade_fullres"]  # 3d_lowres deliberately not in the list
    try:
        _validate_cascade_producers(run, plans)
    except Exception as e:
        msg = str(e)
        assert "3d_cascade_fullres" in msg, f"consumer not named: {msg!r}"
        assert "3d_lowres" in msg, f"missing stage not named: {msg!r}"
        print(f"PASS test_missing_producer_raises_named_error ({type(e).__name__})")
        return
    raise AssertionError("no error raised for a producer-less cascade consumer")


def test_valid_run_list_passes():
    # No raise when every consumer's previous stage is in the run list.
    _validate_cascade_producers(
        ["c1", "c2", "p1", "p2"],
        _plans([("c1", "p1"), ("c2", "p2"), ("p1", None), ("p2", None)]),
    )
    _validate_cascade_producers(["a", "b"], _plans([("a", None), ("b", None)]))
    print("PASS test_valid_run_list_passes")


def main():
    test_single_cascade_regression()
    test_two_cascade_consumers()
    test_shared_previous_stage()
    test_non_cascade_empty_mapping()
    test_missing_producer_raises_named_error()
    test_valid_run_list_passes()
    print("ALL PASS (test_p29_cascade_wiring)")


if __name__ == "__main__":
    main()
