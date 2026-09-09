#!/usr/bin/env python
"""Phase 18 (plan 18-01, Task 2) unit suite for the `liver_segments` registry
entry, the spec-driven model-root resolver, and the crop_cascade topology
construction (LIV-04 + LIV-02 setup).

House pattern (mirrors scripts/test_task_specs.py / test_builder.py):
standalone executable, NOT pytest; exits 0 iff every check passes. Headless:
stdlib + the app's own modules only, no GPU, no torch, no nnunetv2.

Checks:
  1. the `liver_segments` entry: task_id 570, family crop_cascade, 2 parts in
     order (total_6mm, ct_liver_segments), crop labels ("liver",) + addon
     (20,20,20), resample (0.8046879768371582, 0.8046879768371582, 1.5),
     resample_order 1, output_mode seg_sr, trainer, tta False, target_spacing 6.0
  2. build_topology(liver_segments spec, all 2 parts) -> family crop_cascade,
     crop_stages == CROP_STAGES (the 8-stage tuple)
  3. fail-fast: get_task_spec("liver_segments_typo") raises ValueError naming
     the bogus value, and the valid-set message contains BOTH
     "liver_segments" and "total"
  4. registry purity: importing my_app.task_specs in a PLAIN python subprocess
     imports no holoscan/torch/nnunetv2; + in-source check that task_specs
     carries no forbidden imports/literals
  5. resolver: resolve_task_model_root on the real liver_segments root
     succeeds; on a stripped temp dir (one part's jsonpkls removed) it raises
     FileNotFoundError naming the missing file; on the `total` root it returns
     the SAME path resolve_total_model_root returns (byte-identical behavior
     guard); 4-layout candidates (root itself / models/ / models/<task>/ /
     <task>/) all resolve
  6. labels table: inline dict has exactly 9 keys (background +
     cnd.lv.seg.1..8 -> 0..8)

Run:
  cd examples/apps/totalsegmentator-fast
  /tmp/monai-env/.venv/bin/python scripts/test_liver_specs.py
Exit: 0 iff every check passes; 1 otherwise.
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

import my_app.config as cfg  # noqa: E402
from my_app.config import resolve_task_model_root, resolve_total_model_root  # noqa: E402
from my_app.task_specs import CROP_STAGES, build_topology, get_task_spec  # noqa: E402

LIVER_ROOT = Path("/raid/monai-deploy-app-sdk/ct-totalsegmentator-map/models/liver_segments")
TOTAL_ROOT = Path("/raid/monai-deploy-app-sdk/ct-totalsegmentator-map/models/total")


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        print(f"FAIL: {name} {detail}")
        sys.exit(1)
    print(f"PASS: {name}")


spec = get_task_spec("liver_segments")

# ---------------------------------------------------------------------------
# 1. entry values
# ---------------------------------------------------------------------------
check("1a task_id 570", spec.task_id == 570, f"got {spec.task_id!r}")
check("1b family crop_cascade", spec.family == "crop_cascade", f"got {spec.family!r}")
check(
    "1c parts in order (crop model first)",
    [p.name for p in spec.parts] == ["total_6mm", "ct_liver_segments"],
    f"got {[p.name for p in spec.parts]}",
)
check(
    "1d part details",
    spec.parts[0].max_local_label == 117
    and spec.parts[0].config_name == "3d_fullres"
    and spec.parts[1].max_local_label == 8
    and spec.parts[1].config_name == "3d_fullres"
    and spec.parts[0].label_offset == 0
    and spec.parts[1].label_offset == 0,
    f"got {spec.parts!r}",
)
check(
    "1e crop spec (labels + addon)",
    spec.crop is not None
    and spec.crop.source is None
    and tuple(spec.crop.labels) == ("liver",)
    and tuple(spec.crop.addon) == (20, 20, 20),
    f"got {spec.crop!r}",
)
check(
    "1f resample vector (TS config verbatim)",
    spec.resample == (0.8046879768371582, 0.8046879768371582, 1.5),
    f"got {spec.resample!r}",
)
check("1g resample_order 1", spec.resample_order == 1, f"got {spec.resample_order!r}")
check("1h target_spacing 6.0", spec.target_spacing == 6.0, f"got {spec.target_spacing!r}")
check("1i trainer", spec.trainer == "nnUNetTrainerNoMirroring", f"got {spec.trainer!r}")
check("1j tta False", spec.tta is False)
check(
    "1k snomed attr key",
    spec.snomed == "liver_segment_descriptions",
    f"got {spec.snomed!r}",
)
check(
    "1k2 snomed module",
    spec.snomed_module == "ai_liver_segment_descriptions",
    f"got {spec.snomed_module!r}",
)
check(
    "1k3 total snomed module",
    get_task_spec("total").snomed_module == "ai_segment_descriptions",
)
check("1l output_mode seg_sr", spec.output_mode == "seg_sr", f"got {spec.output_mode!r}")

# ---------------------------------------------------------------------------
# 2. build_topology -> crop_cascade with the 8-stage tuple
# ---------------------------------------------------------------------------
plan = build_topology(spec, [p.name for p in spec.parts])
check("2a family crop_cascade", plan.family == "crop_cascade", f"got {plan.family!r}")
check(
    "2b crop_stages == CROP_STAGES (8-stage tuple)",
    tuple(plan.crop_stages) == CROP_STAGES and len(plan.crop_stages) == 8,
    f"got {plan.crop_stages!r}",
)
check(
    "2c parts round-trip (both, in order)",
    list(plan.parts) == ["total_6mm", "ct_liver_segments"],
    f"got {list(plan.parts)!r}",
)
check("2d not serialized", plan.serialized is False)

# ---------------------------------------------------------------------------
# 3. fail-fast now names both tasks
# ---------------------------------------------------------------------------
try:
    get_task_spec("liver_segments_typo")
    check("3a fail-fast raises", False, "no exception raised")
except ValueError as e:
    msg = str(e)
    check(
        "3a fail-fast names typo + both valid tasks",
        "liver_segments_typo" in msg and "liver_segments" in msg and "total" in msg,
        f"msg={msg!s}",
    )

# ---------------------------------------------------------------------------
# 4. registry purity: plain-python import pulls in no GPU/NN deps
# ---------------------------------------------------------------------------
r = subprocess.run(
    [
        sys.executable,
        "-c",
        "import sys; sys.path.insert(0, %r); "
        "import my_app.task_specs; "
        "bad = [m for m in ('holoscan', 'torch', 'nnunetv2', 'numpy', 'monai') if m in sys.modules]; "
        "print('BAD:' + ','.join(bad) if bad else 'PURE')" % str(APP_ROOT),
    ],
    capture_output=True,
    text=True,
    timeout=60,
)
check(
    "4a plain-python import is pure (no holoscan/torch/nnunetv2/numpy/monai)",
    r.returncode == 0 and "PURE" in r.stdout,
    f"rc={r.returncode} out={r.stdout!r} err={r.stderr[-200:]!r}",
)
src = (APP_ROOT / "my_app" / "task_specs" / "__init__.py").read_text()
plans_src = (APP_ROOT / "my_app" / "task_specs" / "plans.py").read_text()
forbidden = re.compile(r"^\s*(import|from)\s+(holoscan|torch|nnunetv2|numpy|monai|my_app\.config)", re.M)
hits = [m.group(0).strip() for m in forbidden.finditer(src + "\n" + plans_src)]
check("4b no forbidden imports in task_specs source", not hits, f"hits={hits}")

# ---------------------------------------------------------------------------
# 5. spec-driven model-root resolver
# ---------------------------------------------------------------------------
part_names = [p.name for p in spec.parts]

# 5a. real liver_segments root resolves
got = resolve_task_model_root(LIVER_ROOT, "liver_segments", part_names)
check("5a real liver_segments root resolves", got == LIVER_ROOT, f"got {got!r}")

# 5b. 4-layout candidates all resolve to the same root
for layout in (
    LIVER_ROOT,
    LIVER_ROOT.parent,  # .../models (p/models candidate)
    LIVER_ROOT.parent.parent,  # repo root (p/models/<task> candidate)
):
    g2 = resolve_task_model_root(layout, "liver_segments", part_names)
    check(
        f"5b layout {layout.name or str(layout)} resolves to root",
        g2 == LIVER_ROOT,
        f"got {g2!r}",
    )

# 5c. stripped root (one part's jsonpkls removed) -> hard error naming the file
tmp = Path(tempfile.mkdtemp(prefix="p18_stripped_", dir="/raid/tmp"))
try:
    (tmp / "ct_liver_segments").mkdir(parents=True)
    (tmp / "ct_liver_segments" / "3d_fullres" / "fold_0").mkdir(parents=True)
    (tmp / "ct_liver_segments" / "3d_fullres" / "nnunet_checkpoint.pth").write_bytes(b"x")
    (tmp / "ct_liver_segments" / "3d_fullres" / "fold_0" / "final_model.pt").write_bytes(b"x")
    # total_6mm present with BOTH required files
    (tmp / "total_6mm" / "jsonpkls").mkdir(parents=True)
    (tmp / "total_6mm" / "jsonpkls" / "plans.json").write_text("{}")
    (tmp / "total_6mm" / "3d_fullres" / "fold_0").mkdir(parents=True)
    (tmp / "total_6mm" / "3d_fullres" / "fold_0" / "final_model.pt").write_bytes(b"x")
    (tmp / "total_6mm" / "3d_fullres" / "nnunet_checkpoint.pth").write_bytes(b"x")
    try:
        resolve_task_model_root(tmp, "liver_segments", part_names)
        check("5c stripped root raises", False, "no exception raised")
    except FileNotFoundError as e:
        msg = str(e)
        check(
            "5c stripped root raises FileNotFoundError naming missing file",
            "ct_liver_segments/jsonpkls/plans.json" in msg,
            f"msg={msg!s}",
        )

    # 5d. completely bogus path -> FileNotFoundError (no valid root)
    try:
        resolve_task_model_root(tmp / "does_not_exist", "liver_segments", part_names)
        check("5d bogus path raises", False, "no exception raised")
    except FileNotFoundError as e:
        check("5d bogus path raises FileNotFoundError", True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# 5e. total byte-identical behavior guard: same path as the old resolver
for layout in (TOTAL_ROOT, TOTAL_ROOT.parent, TOTAL_ROOT.parent.parent):
    old = resolve_total_model_root(layout)
    new = resolve_task_model_root(layout, "total", [p["name"] for p in cfg.MODEL_PARTS])
    check(
        f"5e total root identical for layout {layout.name}",
        old == new,
        f"old={old!r} new={new!r}",
    )

# ---------------------------------------------------------------------------
# 6. labels table: exactly 9 keys
# ---------------------------------------------------------------------------
labels = spec.labels
check("6a labels is an inline dict", isinstance(labels, dict), f"got {type(labels)!r}")
check("6b exactly 9 keys", len(labels) == 9, f"got {len(labels)}")
check(
    "6c keys/values == background 0 + cnd.lv.seg.1..8 -> 1..8",
    labels
    == {
        "background": 0,
        **{f"cnd.lv.seg.{i}": i for i in range(1, 9)},
    },
    f"got {json.dumps(labels)}",
)

print("ALL PASS: test_liver_specs")
sys.exit(0)
