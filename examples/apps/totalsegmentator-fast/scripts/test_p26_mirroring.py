#!/usr/bin/env python
"""Phase 26 plan 26-01 TDD suite: D-04 plan-driven use_mirroring (RED->GREEN).

House pattern (mirrors scripts/test_p26_config_fixes.py): standalone
executable, NOT pytest; exits 0 iff every check passes. Pure functions only
— no CUDA, no bundle loading, no operator construction.

Checks:
  1. resolve_use_mirroring precedence (per-config key > top-level key > False):
     a. configurations[<cfg>]["use_mirroring"]=True -> True
     b. per-config False + top-level True -> False (per-config wins)
     c. no per-config key + top-level True -> True
     d. key absent everywhere -> False
     e. plans without "configurations" -> False (never raises)
     f. config entry not a dict (e.g. a string) -> False (never raises)
  2. InferenceParams carries a `use_mirroring` bool field.
  3. resolve_mirror_flag precedence: explicit True/False beats the plan value;
     None falls through to the plan value.

Run:
  cd examples/apps/totalsegmentator-fast
  /tmp/monai-env/.venv/bin/python scripts/test_p26_mirroring.py
Exit: 0 iff every check passes; 1 otherwise.
"""

import dataclasses
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

import my_app.config as cfg  # noqa: E402


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        print(f"FAIL: {name} {detail}")
        sys.exit(1)
    print(f"PASS: {name}")


# ---------------------------------------------------------------------------
# 1. resolve_use_mirroring precedence
# ---------------------------------------------------------------------------
rm = cfg.resolve_use_mirroring  # RED: AttributeError pre-26-01

check(
    "1a per-config True wins",
    rm({"configurations": {"3d_fullres": {"use_mirroring": True}}}, "3d_fullres") is True,
)

check(
    "1b per-config False beats top-level True",
    rm(
        {
            "use_mirroring": True,
            "configurations": {"3d_fullres": {"use_mirroring": False}},
        },
        "3d_fullres",
    )
    is False,
)

check(
    "1c no per-config key -> top-level True",
    rm({"use_mirroring": True, "configurations": {"3d_fullres": {}}}, "3d_fullres") is True,
)

check(
    "1d absent everywhere -> False",
    rm({"configurations": {"3d_fullres": {"spacing": [1.5, 1.5, 1.5]}}}, "3d_fullres") is False,
)

check("1e no configurations key -> False (never raises)", rm({}, "3d_fullres") is False)

check(
    "1f config entry not a dict -> False (never raises)",
    rm({"configurations": {"3d_fullres": "3d_lowres"}}, "3d_fullres") is False,
)

check(
    "1g unknown config name falls to top-level",
    rm(
        {
            "use_mirroring": True,
            "configurations": {"3d_lowres": {"use_mirroring": False}},
        },
        "3d_fullres",
    )
    is True,
)

# ---------------------------------------------------------------------------
# 2. InferenceParams field
# ---------------------------------------------------------------------------
fields = {f.name: f for f in dataclasses.fields(cfg.InferenceParams)}
check(
    "2a InferenceParams has use_mirroring field",
    "use_mirroring" in fields,
    f"fields={sorted(fields)}",
)
# NOTE: `from __future__ import annotations` stores annotations as strings.
check(
    "2b use_mirroring annotated bool",
    str(fields["use_mirroring"].type) == "bool",
    f"type={fields['use_mirroring'].type!r}",
)
# field order: immediately after mirror_axes (plan placement)
names = [f.name for f in dataclasses.fields(cfg.InferenceParams)]
check(
    "2c use_mirroring placed after mirror_axes",
    names[names.index("mirror_axes") + 1] == "use_mirroring",
    f"order={names}",
)

# ---------------------------------------------------------------------------
# 3. resolve_mirror_flag precedence (pure helper)
# ---------------------------------------------------------------------------
rf = cfg.resolve_mirror_flag
check("3a explicit True beats plan False", rf(True, False) is True)
check("3b explicit False beats plan True", rf(False, True) is False)
check("3c None falls through to plan value True", rf(None, True) is True)
check("3d None falls through to plan value False", rf(None, False) is False)

print("ALL PASS: test_p26_mirroring")
sys.exit(0)
