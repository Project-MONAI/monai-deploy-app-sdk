#!/usr/bin/env python
"""24 pre-closeout (user-directed, 2026-09-07) unit suite: HOLOSCAN_EMIT_NPY
gate for the debug/terminal numpy .npy + emit_meta.json writes.

House pattern (mirrors scripts/test_task_specs.py): standalone executable,
NOT pytest. Exits 0 iff every check passes. Headless: stdlib + the app's own
modules only (monai.deploy + torch import, as every house suite does); no
GPU work, no model/corpus files.

What this proves (behavior-neutrality of the gate):
  1. env UNSET   -> module constant _EMIT_NPY_ENABLED is True (default
     "1" = current behavior, byte-identical write path).
  2. env "1" -> True; env "0" -> False (only "0" disables; any other
     value keeps current behavior, same convention as HOLOSCAN_GPU_RESAMPLE).
  3. mode="total", env "0": compute() still RECEIVES all 3 inputs (flow
     semantics unchanged), writes NOTHING (no .npy, no emit_meta.json),
     and logs exactly "emit disabled (HOLOSCAN_EMIT_NPY=0)".
  4. mode="total", env "1": compute() writes seg_total_sar.npy +
     seg_total_dhw.npy + emit_meta.json; arrays round-trip byte-exact.
  5. mode="single", env "0": receives both seg inputs + meta, writes nothing.
  6. mode="single", env "1": writes seg_<part>_modelspace.npy +
     seg_<part>_dicom.npy + emit_meta.json; arrays round-trip.
  7. source guards: seg_emit has the gate on BOTH compute paths; the
     merge_5part per-part dump is ANDed with HOLOSCAN_EMIT_NPY; every
     np.save site in cascade_ops is inside an _EMIT_NPY_ENABLED guard.

Run:
  cd examples/apps/totalsegmentator-fast
  /tmp/monai-env/.venv/bin/python scripts/test_seg_emit_gate.py
Exit: 0 iff every check passes; 1 otherwise (details printed).
No files are written outside a temp dir; all data is in-memory synthetic.
"""

import importlib
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
MY_APP = APP_ROOT / "my_app"
sys.path.insert(0, str(MY_APP))  # flat import (my_app dir on sys.path, as the app runner provides)

import numpy as np  # noqa: E402
from config import MODEL_PARTS  # noqa: E402
from operators import seg_emit_operator as m  # noqa: E402

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  ok   {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL {name} {detail}")


def reload_with(env_val):
    """Re-import the module so the module-level env read re-runs."""
    if env_val is None:
        os.environ.pop("HOLOSCAN_EMIT_NPY", None)
    else:
        os.environ["HOLOSCAN_EMIT_NPY"] = env_val
    importlib.reload(m)
    return m


class FakeOpInput:
    """Records which named inputs compute() received, feeds synthetic arrays."""

    def __init__(self, inputs):
        self.inputs = inputs
        self.received = []

    def receive(self, name):
        self.received.append(name)
        return self.inputs.get(name)


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def make_op(mod, mode, out_dir, part=None, offset=None):
    """Build an operator instance WITHOUT Operator.__init__ (no Holoscan
    fragment needed — compute() only touches our attributes).

    Holoscan's Operator.fragment is a read-only property on the C++ base
    class, so we shadow it with a Python property on a test-only subclass
    (the source under test — compute() — is inherited unchanged)."""
    cls = type(
        "TestSegEmit",
        (mod.SegEmitOperator,),
        {
            "fragment": property(lambda self: self._test_frag),
        },
    )
    op = cls.__new__(cls)
    op._test_frag = object()  # any object: gpu_util keys records by id()
    cap = LogCapture()
    # The operator logs to f"{mod.__name__}.{type(self).__name__}" — a CHILD of
    # the module logger (instance is a TestSegEmit subclass) — so capture on
    # the module logger and set DEBUG so INFO records pass the level check.
    logger = logging.getLogger(mod.__name__)
    logger.setLevel(logging.DEBUG)
    for h in list(logger.handlers):  # fresh capture per op instance
        logger.removeHandler(h)
    logger.addHandler(cap)
    op._logger = logger
    op._mode = mode
    op._output_dir = out_dir
    op._capture = cap
    if mode == "total":
        op._base_name = "total"
    else:
        op._part = part
        op._offset = offset
    return op


def synthetic_seg():
    a = np.zeros((4, 5, 6), dtype=np.uint8)
    a[1, :, :] = 7
    a[2, 2, 2] = 117
    return a


def main():
    tmp = Path(tempfile.mkdtemp(prefix="tsfast_emit_gate_"))
    total_part = MODEL_PARTS[0]
    single_part = total_part["name"]
    single_offset = int(total_part["label_offset"])

    print("1-2. module-level env read (default '1'; only '0' disables)")
    mod = reload_with(None)
    check("unset -> enabled", mod._EMIT_NPY_ENABLED is True)
    mod = reload_with("1")
    check("'1' -> enabled", mod._EMIT_NPY_ENABLED is True)
    mod = reload_with("banana")
    check(
        "non-'0' truthy -> enabled (GPU_RESAMPLE convention)",
        mod._EMIT_NPY_ENABLED is True,
    )
    mod = reload_with("0")
    check("'0' -> disabled", mod._EMIT_NPY_ENABLED is False)

    print("3. mode=total, env '0': consumes inputs, writes nothing, logs the line")
    mod = reload_with("0")
    out = tmp / "total_off"
    seg = synthetic_seg()
    op = make_op(mod, "total", out)
    fi = FakeOpInput(
        {
            "seg_merged": seg,
            "seg_image": seg.copy(),
            "preprocessed_meta": {"_pre_resample_spatial_shape": [6, 5, 4]},
        }
    )
    op.compute(fi, None, None)
    check(
        "received seg_merged + seg_image + preprocessed_meta",
        fi.received == ["seg_merged", "seg_image", "preprocessed_meta"],
        f"got {fi.received}",
    )
    check(
        "output dir empty (no .npy / emit_meta.json)",
        not (out.exists() and any(out.iterdir())),
        f"found {list(out.iterdir()) if out.exists() else 'dir exists'}",
    )
    check(
        "logged exactly 'emit disabled (HOLOSCAN_EMIT_NPY=0)' once",
        [r for r in op._capture.records if "emit disabled" in r] == ["emit disabled (HOLOSCAN_EMIT_NPY=0)"],
        f"got {op._capture.records}",
    )

    print("4. mode=total, env '1': writes the 3 files, arrays round-trip")
    mod = reload_with("1")
    out = tmp / "total_on"
    op = make_op(mod, "total", out)
    fi = FakeOpInput(
        {
            "seg_merged": seg,
            "seg_image": seg.copy(),
            "preprocessed_meta": {"_pre_resample_spatial_shape": [6, 5, 4]},
        }
    )
    op.compute(fi, None, None)
    files = sorted(p.name for p in out.iterdir())
    check(
        "exactly seg_total_sar.npy + seg_total_dhw.npy + emit_meta.json",
        files == ["emit_meta.json", "seg_total_dhw.npy", "seg_total_sar.npy"],
        f"got {files}",
    )
    check(
        "sar array round-trips byte-exact",
        np.array_equal(np.load(out / "seg_total_sar.npy"), seg),
    )
    check(
        "dhw array round-trips byte-exact",
        np.array_equal(np.load(out / "seg_total_dhw.npy"), seg),
    )
    meta = json.loads((out / "emit_meta.json").read_text())
    check(
        "emit_meta.json: mode=total, sar+dhw shape + label histogram",
        meta["mode"] == "total"
        and meta["sar"]["shape"] == [4, 5, 6]
        and meta["sar"]["label_histogram"] == {"0": 89, "7": 30, "117": 1},
        f"got {meta.get('mode')} {meta['sar']['label_histogram']}",
    )
    check(
        "no 'emit disabled' log when enabled",
        not any("emit disabled" in r for r in op._capture.records),
    )

    print("5. mode=single, env '0': consumes inputs, writes nothing")
    mod = reload_with("0")
    out = tmp / "single_off"
    op = make_op(mod, "single", out, part=single_part, offset=single_offset)
    ms = synthetic_seg()
    di = synthetic_seg()
    fi = FakeOpInput(
        {
            "seg_merged": ms,
            "seg_merged_dicom": di,
            "preprocessed_meta": {"_pre_resample_spatial_shape": [6, 5, 4]},
        }
    )
    op.compute(fi, None, None)
    check(
        "received seg_merged + seg_merged_dicom + preprocessed_meta",
        fi.received == ["seg_merged", "seg_merged_dicom", "preprocessed_meta"],
        f"got {fi.received}",
    )
    check(
        "output dir empty",
        not (out.exists() and any(out.iterdir())),
        f"found {list(out.iterdir()) if out.exists() else 'dir exists'}",
    )
    check(
        "logged the disabled line",
        any("emit disabled (HOLOSCAN_EMIT_NPY=0)" in r for r in op._capture.records),
    )

    print("6. mode=single, env '1': writes the 3 files, arrays round-trip")
    mod = reload_with("1")
    out = tmp / "single_on"
    op = make_op(mod, "single", out, part=single_part, offset=single_offset)
    fi = FakeOpInput(
        {
            "seg_merged": ms,
            "seg_merged_dicom": di,
            "preprocessed_meta": {"_pre_resample_spatial_shape": [6, 5, 4]},
        }
    )
    op.compute(fi, None, None)
    files = sorted(p.name for p in out.iterdir())
    expect = sorted(
        [
            f"seg_{single_part}_modelspace.npy",
            f"seg_{single_part}_dicom.npy",
            "emit_meta.json",
        ]
    )
    check(
        "exactly seg_<part>_modelspace/dicom.npy + emit_meta.json",
        files == expect,
        f"got {files} want {expect}",
    )
    check(
        "modelspace round-trips",
        np.array_equal(np.load(out / f"seg_{single_part}_modelspace.npy"), ms),
    )
    check(
        "dicom round-trips",
        np.array_equal(np.load(out / f"seg_{single_part}_dicom.npy"), di),
    )
    meta = json.loads((out / "emit_meta.json").read_text())
    check(
        "emit_meta.json: part + label_offset recorded",
        meta["part"] == single_part and meta["label_offset"] == single_offset,
        f"got {meta.get('part')} {meta.get('label_offset')}",
    )

    print("7. source guards on the other two emit sites")
    cascade_src = (MY_APP / "operators" / "cascade_ops.py").read_text()
    merge_src = (MY_APP / "operators" / "merge_5part_operator.py").read_text()
    emit_src = (MY_APP / "operators" / "seg_emit_operator.py").read_text()
    n_save = len(re.findall(r"\bnp\.save\(", cascade_src))
    n_guard = len(re.findall(r"if _EMIT_NPY_ENABLED:", cascade_src))
    check(
        "cascade_ops: all 7 np.save sites inside 6 guards (2 saves share 1 block)",
        n_save == 7 and n_guard == 6,
        f"np.save={n_save} guards={n_guard}",
    )
    check(
        "cascade_ops: module-level env read present",
        re.search(
            r'_EMIT_NPY_ENABLED = os\.environ\.get\("HOLOSCAN_EMIT_NPY", "1"\) != "0"',
            cascade_src,
        )
        is not None,
    )
    check(
        "merge_5part: per-part dump ANDed with HOLOSCAN_EMIT_NPY",
        re.search(
            r'os\.environ\.get\("HOLOSCAN_EMIT_PART_SEGS"\) == "1"\s*\n'
            r'\s*and os\.environ\.get\("HOLOSCAN_EMIT_NPY", "1"\) != "0"',
            merge_src,
        )
        is not None,
    )
    check(
        "seg_emit: gate present on both compute paths (single + total)",
        emit_src.count("if not _EMIT_NPY_ENABLED:") == 2,
        f"count={emit_src.count('if not _EMIT_NPY_ENABLED:')}",
    )

    print()
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    print("ALL GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
