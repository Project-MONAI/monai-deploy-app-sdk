#!/usr/bin/env python
"""Phase 26 plan 26-01 TDD suite: env-gated cascade zero one-hot stub
(HOLOSCAN_SYNTHETIC_CASCADE) — RED->GREEN.

House pattern: standalone executable, NOT pytest; exits 0 iff every check
passes. Pure helpers + CPU torch tensors only — no CUDA, no bundle loading.
os.environ is monkeypatched per-check (restored in finally).

Checks:
  1. Stub OFF (env unset): _pad_conv0_for_synthetic_cascade and
     _expand_input_for_synthetic_cascade are strict no-ops (same object
     identity, values untouched) for C=1 and C>1; synthetic_cascade_enabled()
     is False.
  2. Stub ON (HOLOSCAN_SYNTHETIC_CASCADE=1):
     a. synthetic_cascade_enabled() is True.
     b. fake fold state dict with a first ConvNd weight [4,1,3,3,3] + C=25 ->
        NEW dict (input never mutated), every in-1 conv weight becomes
        [C_out,25,3,3,3] with channel 0 bitwise equal to the original and
        channels 1..24 all zero (26-04: ALL in-1 convs padded — PlainConvUNet
        checkpoints store conv0 under multiple keys, incl. the decoder's
        mirrored encoder conv0); bias untouched (same object); non-conv
        entries untouched (same object).
     c. input tensor [1,1,4,4,4] + C=25 -> [1,25,4,4,4], channel 0 equal,
        channels 1..24 zero, input tensor unmutated.
     d. 4D input [1,4,4,4] + C=25 -> [25,4,4,4] (compute() passes 4D).
     e. C=1 (non-cascade config) -> strict no-op even with env set.
     f. no in-1 conv weight in state dict -> strict no-op.
  3. env set to anything other than "1" ("0", "true") -> disabled.

Run:
  cd examples/apps/totalsegmentator-fast
  /tmp/monai-env/.venv/bin/python scripts/test_p26_cascade_stub.py
Exit: 0 iff every check passes; 1 otherwise.
"""

import contextlib
import os
import sys
from pathlib import Path

import torch

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from my_app.operators.slidewindow_operator import SYNTHETIC_CASCADE_ENV  # noqa: E402
from my_app.operators.slidewindow_operator import (
    _expand_input_for_synthetic_cascade,
    _pad_conv0_for_synthetic_cascade,
    synthetic_cascade_enabled,
)


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        print(f"FAIL: {name} {detail}")
        sys.exit(1)
    print(f"PASS: {name}")


@contextlib.contextmanager
def _env(value=None):
    old = os.environ.get(SYNTHETIC_CASCADE_ENV)
    if value is None:
        os.environ.pop(SYNTHETIC_CASCADE_ENV, None)
    else:
        os.environ[SYNTHETIC_CASCADE_ENV] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(SYNTHETIC_CASCADE_ENV, None)
        else:
            os.environ[SYNTHETIC_CASCADE_ENV] = old


def _fake_state_dict():
    """First ConvNd entry with in=1, then a bias, then a second in-1 weight."""
    w0 = torch.arange(4 * 1 * 3 * 3 * 3, dtype=torch.float32).reshape(4, 1, 3, 3, 3)
    bias = torch.arange(4, dtype=torch.float32)
    w1 = torch.ones(8, 1, 3, 3, 3)
    return {"conv0.conv.weight": w0, "conv0.conv.bias": bias, "conv1.conv.weight": w1}


# ---------------------------------------------------------------------------
# 1. stub OFF: strict no-ops
# ---------------------------------------------------------------------------
with _env(None):
    check("1a env unset -> disabled", synthetic_cascade_enabled() is False)
    sd = _fake_state_dict()
    out = _pad_conv0_for_synthetic_cascade(sd, 25)
    check("1b pad no-op identity when off (C=25)", out is sd)
    x = torch.randn(1, 1, 4, 4, 4)
    check(
        "1c expand no-op identity when off (C=25)",
        _expand_input_for_synthetic_cascade(x, 25) is x,
    )
    check(
        "1d expand no-op identity when off (C=1)",
        _expand_input_for_synthetic_cascade(x, 1) is x,
    )

# ---------------------------------------------------------------------------
# 2. stub ON
# ---------------------------------------------------------------------------
with _env("1"):
    check("2a env=1 -> enabled", synthetic_cascade_enabled() is True)

    sd = _fake_state_dict()
    orig_w0 = sd["conv0.conv.weight"].clone()
    out = _pad_conv0_for_synthetic_cascade(sd, 25)
    check("2b returns a NEW dict", out is not sd)
    nw = out["conv0.conv.weight"]
    check(
        "2c weight expanded to [4,25,3,3,3]",
        tuple(nw.shape) == (4, 25, 3, 3, 3),
        f"got {tuple(nw.shape)}",
    )
    check(
        "2d input channel 0 bitwise equal to original",
        torch.equal(nw[:, 0], orig_w0[:, 0]),
    )
    check("2e input channels 1..24 all zero", bool((nw[:, 1:] == 0).all()))
    check(
        "2f bias untouched (same object)",
        out["conv0.conv.bias"] is sd["conv0.conv.bias"],
    )
    check(
        "2g second in-1 weight ALSO padded (26-04: all in-1 convs, decoder mirrors included)",
        out["conv1.conv.weight"] is not sd["conv1.conv.weight"]
        and tuple(out["conv1.conv.weight"].shape) == (8, 25, 3, 3, 3)
        and torch.equal(out["conv1.conv.weight"][:, 0], sd["conv1.conv.weight"][:, 0])
        and bool((out["conv1.conv.weight"][:, 1:] == 0).all()),
    )
    check(
        "2h input dict unmutated",
        tuple(sd["conv0.conv.weight"].shape) == (4, 1, 3, 3, 3),
    )

    x = torch.randn(1, 1, 4, 4, 4)
    orig_x = x.clone()
    y = _expand_input_for_synthetic_cascade(x, 25)
    check(
        "2i input [1,1,4,4,4] -> [1,25,4,4,4]",
        tuple(y.shape) == (1, 25, 4, 4, 4),
        f"got {tuple(y.shape)}",
    )
    check(
        "2j input channel 0 equal, rest zero",
        torch.equal(y[:, 0], x[:, 0]) and bool((y[:, 1:].contiguous() == 0).all()),
    )
    check(
        "2k input tensor unmutated",
        torch.equal(x, orig_x) and tuple(x.shape) == (1, 1, 4, 4, 4),
    )

    x4 = torch.randn(1, 4, 4, 4)
    y4 = _expand_input_for_synthetic_cascade(x4, 25)
    check(
        "2l 4D input [1,4,4,4] -> [25,4,4,4]",
        tuple(y4.shape) == (25, 4, 4, 4),
        f"got {tuple(y4.shape)}",
    )
    check(
        "2m 4D channel 0 equal, rest zero",
        torch.equal(y4[0], x4[0]) and bool((y4[1:].contiguous() == 0).all()),
    )

    x1 = torch.randn(1, 1, 4, 4, 4)
    check(
        "2n C=1 -> input untouched even with env set",
        _expand_input_for_synthetic_cascade(x1, 1) is x1,
    )
    sd_nofit = {"conv0.conv.weight": torch.ones(8, 4, 3, 3, 3)}
    check(
        "2o no in-1 conv weight -> pad no-op identity",
        _pad_conv0_for_synthetic_cascade(sd_nofit, 25) is sd_nofit,
    )

# ---------------------------------------------------------------------------
# 3. env values other than "1" keep the stub off
# ---------------------------------------------------------------------------
for val in ("0", "true", "2"):
    with _env(val):
        sd = _fake_state_dict()
        check(
            f"3 env={val!r} -> disabled (no-op identity)",
            synthetic_cascade_enabled() is False and _pad_conv0_for_synthetic_cascade(sd, 25) is sd,
        )

print("ALL PASS: test_p26_cascade_stub")
sys.exit(0)
