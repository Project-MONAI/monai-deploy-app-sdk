"""CPU unit test for the HOLOSCAN_EMIT_NPY gate (container-aware
default included).

The gate constant is module-level and read at IMPORT time (exactly like
totalsegmentator-fast seg_emit_operator.py). Importing the operator module
would pull the holoscan/torch CUDA stack — too heavy for a CPU test — so
this test (a) asserts the verbatim gate expression is present in the
operator source, and (b) replicates the 2-line expression to prove the
documented on/off convention, including the container-aware default:

- dev (HOLOSCAN_APP_MANIFEST_PATH unset): default ON
- MAP (HOLOSCAN_APP_MANIFEST_PATH set, as baked by the holoscan-cli runner):
  default OFF
- HOLOSCAN_EMIT_NPY always overrides: only exact "0" disables, "1" enables.

Run (venv, CPU):
    python -m pytest examples/apps/cchmc-tcv-fast/scripts/test_emit_npy.py -v
"""

import os
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1]
OP = APP / "my_app" / "operators" / "postprocess_operator.py"

GATE_LINES = (
    '_MAP_RUN = bool(os.environ.get("HOLOSCAN_APP_MANIFEST_PATH"))',
    "_EMIT_NPY_ENABLED = os.environ.get(",
    ' "HOLOSCAN_EMIT_NPY", "0" if _MAP_RUN else "1") != "0"',
)


@pytest.fixture(autouse=True)
def _clean_env():
    yield
    os.environ.pop("HOLOSCAN_EMIT_NPY", None)
    os.environ.pop("HOLOSCAN_APP_MANIFEST_PATH", None)


def test_gate_convention_source():
    """postprocess_operator.py carries the verbatim gate expression."""
    src = OP.read_text()
    for line in GATE_LINES:
        assert line in src, f"missing gate line: {line!r}"
    # the gate is actually wired into compute
    assert "if _EMIT_NPY_ENABLED and self.output_folder is not None:" in src
    # single np.save call site (the dev emit)
    assert src.count("np.save(") == 1


def _gate(env_val, map_run):
    """Verbatim expression (container-aware default)."""
    if map_run:
        os.environ["HOLOSCAN_APP_MANIFEST_PATH"] = "/opt/holoscan/app.json"
    else:
        os.environ.pop("HOLOSCAN_APP_MANIFEST_PATH", None)
    if env_val is None:
        os.environ.pop("HOLOSCAN_EMIT_NPY", None)
    else:
        os.environ["HOLOSCAN_EMIT_NPY"] = env_val
    _map_run = bool(os.environ.get("HOLOSCAN_APP_MANIFEST_PATH"))
    return os.environ.get("HOLOSCAN_EMIT_NPY", "0" if _map_run else "1") != "0"


def test_gate_dev_values():
    # dev (no container marker): unset -> ON; only exact "0" disables
    assert _gate(None, map_run=False) is True
    assert _gate("0", map_run=False) is False  # only exact "0" disables
    assert _gate("1", map_run=False) is True
    assert _gate("", map_run=False) is True  # empty string is not "0" -> on
    assert _gate("false", map_run=False) is True  # any other value stays on


def test_gate_map_values():
    # MAP (container marker present): unset -> OFF; "1" overrides to ON
    assert _gate(None, map_run=True) is False
    assert _gate("0", map_run=True) is False
    assert _gate("1", map_run=True) is True
    assert _gate("", map_run=True) is True  # any non-"0" value -> on
