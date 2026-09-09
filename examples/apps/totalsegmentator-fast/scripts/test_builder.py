#!/usr/bin/env python
"""Phase 17.3 unit suite for my_app/task_specs/plans.py (plan 17-03, Task 1).

House pattern (mirrors scripts/test_task_specs.py): standalone executable,
NOT pytest; exits 0 iff every check passes. Headless: stdlib + the app's own
pure-data modules only, no GPU, no model/corpus files.

Checks:
  1. multi-part (total): family/serialized + the implied N-1 serialization
     pair list is EXACTLY post_{i-1} -> swin_i for i in 1..N-1 (4 edges for
     the 5-part total; ports ("seg_argmax", "prev_part_seg") — the P9 OOM
     fix; the roadmap's "3 DAG gate edges" refers to this loop and ALL of
     it must survive)
  2. single-model: a one-part selection -> family single_model, not serialized
  3. crop-cascade (synthetic spec, Phase 18 preview): 8-stage tuple
  4. config_name plumbing (Pitfall 4): a synthetic 3d_fullres_high spec
     round-trips — the plan never bakes a config_name; the spec stays the
     source of truth
  5. subset guard: a 2-of-5 subset raises NotImplementedError ("not supported")
  6. no task/part name literals in plans.py source (grep-equivalent, in-process)
  7. subset validation: wrong-order selection raises ValueError
  8. non-first single-part selection (regression guard): the selected part
     resolves to ITS OWN PartSpec (offset 24 / 91), never spec.parts[0]
  9. Phase 19 (19-01): body spec builds single_model with 1 part; tile_step_size
     is a generic TaskSpec field (total default 0.8, body 0.5); zero quoted
     "body" literals in app.py + every operators/*.py (house grep gate)

Run:
  cd examples/apps/totalsegmentator-fast
  /tmp/monai-env/.venv/bin/python scripts/test_builder.py
Exit: 0 iff every check passes; 1 otherwise.
"""

import re
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from my_app.task_specs import TASK_REGISTRY, CropSpec, PartSpec, TaskSpec, get_task_spec  # noqa: E402
from my_app.task_specs.plans import CROP_STAGES, build_topology, resolve_part  # noqa: E402


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        print(f"FAIL: {name} {detail}")
        sys.exit(1)
    print(f"PASS: {name}")


TOTAL = get_task_spec("total")
ORACLE_ORDER = ["organs", "vertebrae", "cardiac", "muscles", "ribs"]

# ---------------------------------------------------------------------------
# 1. multi-part (total)
# ---------------------------------------------------------------------------
plan = build_topology(TOTAL, list(ORACLE_ORDER))
check("1a multi_part family", plan.family == "multi_part", f"got {plan.family!r}")
check("1b serialized True", plan.serialized is True, f"got {plan.serialized!r}")
check(
    "1c parts in oracle order",
    list(plan.parts) == ORACLE_ORDER,
    f"got {list(plan.parts)}",
)
# The implied serialization edges: for i in 1..N-1, post_{i-1} -> swin_i
# with port ("seg_argmax", "prev_part_seg"). For the 5-part total that is
# EXACTLY these 4 pairs (N-1 = 4).
pairs = [(plan.parts[i - 1], plan.parts[i]) for i in range(1, len(plan.parts)) if plan.serialized]
check(
    "1d serialization pairs exact (4 edges, P9 OOM fix)",
    pairs
    == [
        ("organs", "vertebrae"),
        ("vertebrae", "cardiac"),
        ("cardiac", "muscles"),
        ("muscles", "ribs"),
    ],
    f"got {pairs}",
)
check("1e crop_stages empty", plan.crop_stages == ())

# ---------------------------------------------------------------------------
# 2. single-model
# ---------------------------------------------------------------------------
plan1 = build_topology(TOTAL, ["organs"])
check(
    "2 single_model",
    plan1.family == "single_model"
    and plan1.serialized is False
    and plan1.parts == ("organs",)
    and plan1.crop_stages == (),
    f"got {plan1!r}",
)

# ---------------------------------------------------------------------------
# 3. crop-cascade (synthetic spec — constructed, NOT registered; Phase 18
#    preview for the liver_segments-style task)
# ---------------------------------------------------------------------------
crop_spec = TaskSpec(
    task_id=570,
    family="single_model",
    parts=(PartSpec(name="ct_liver_segments", label_offset=0),),
    crop=CropSpec(source=None, labels=("liver",), addon=(20, 20, 20)),
    resample=(0.8, 0.8, 1.5),
    target_spacing=(0.8, 0.8, 1.5),
)
cplan = build_topology(crop_spec, ["ct_liver_segments"])
check(
    "3a crop_cascade family",
    cplan.family == "crop_cascade",
    f"got {cplan.family!r}",
)
check(
    "3b 8 crop stages",
    tuple(cplan.crop_stages) == CROP_STAGES and len(cplan.crop_stages) == 8,
    f"got {cplan.crop_stages!r}",
)
check("3c not serialized", cplan.serialized is False, f"got {cplan.serialized!r}")
check(
    "3d parts round-trip",
    cplan.parts == ("ct_liver_segments",),
    f"got {cplan.parts!r}",
)

# ---------------------------------------------------------------------------
# 4. config_name plumbing (Pitfall 4)
# ---------------------------------------------------------------------------
high_spec = TaskSpec(
    task_id=957,
    family="single_model",
    parts=(PartSpec(name="total_highres", label_offset=0, config_name="3d_fullres_high"),),
    crop=None,
    resample=(0.75, 0.75, 1.0),
    target_spacing=(0.75, 0.75, 1.0),
)
hplan = build_topology(high_spec, ["total_highres"])
# The plan carries part NAMES only; config_name stays on the spec so the
# materializer can read part.config_name at build time (nothing bakes
# "3d_fullres" into the plan).
check(
    "4 config_name stays on the spec (Pitfall 4)",
    hplan.family == "single_model"
    and high_spec.parts[0].config_name == "3d_fullres_high"
    and "config_name" not in {f.name for f in __import__("dataclasses").fields(hplan)},
    f"got {hplan!r}",
)

# ---------------------------------------------------------------------------
# 5. subset guard (pre-refactor 2-4-part behavior preserved)
# ---------------------------------------------------------------------------
try:
    build_topology(TOTAL, ["organs", "vertebrae"])
    check("5 subset guard", False, "no exception raised")
except NotImplementedError as e:
    check(
        "5 subset guard NotImplementedError",
        "not supported" in str(e),
        f"msg={e!s}",
    )

# ---------------------------------------------------------------------------
# 6. no task/part name literals in plans.py (constraint 8)
# ---------------------------------------------------------------------------
plans_src = (APP_ROOT / "my_app" / "task_specs" / "plans.py").read_text()
pat = re.compile(r'["\x27](total|organs|vertebrae|cardiac|muscles|ribs|liver_segments|body)["\x27]')
hits = [l.strip() for l in plans_src.splitlines() if pat.search(l)]
check("6 no task/part literals in plans.py", not hits, f"hits={hits}")

# ---------------------------------------------------------------------------
# 7. subset validation: wrong order -> ValueError
# ---------------------------------------------------------------------------
try:
    build_topology(TOTAL, ["ribs", "organs"])
    check("7 wrong-order subset", False, "no exception raised")
except ValueError as e:
    check(
        "7 wrong-order subset ValueError",
        "ribs" in str(e) and "organs" in str(e),
        f"msg={e!s}",
    )
# unknown part name -> ValueError too
try:
    build_topology(TOTAL, ["organs", "spleen"])
    check("7b unknown part", False, "no exception raised")
except ValueError as e:
    check("7b unknown part ValueError", "spleen" in str(e), f"msg={e!s}")

# ---------------------------------------------------------------------------
# 8. non-first single-part selection (regression guard: the materializer
#    must wire the SELECTED part's model, never spec.parts[0])
# ---------------------------------------------------------------------------
vplan = build_topology(TOTAL, ["vertebrae"])
check(
    "8a non-first single-part plan",
    vplan.family == "single_model" and vplan.parts == ("vertebrae",),
    f"got {vplan!r}",
)
vp = resolve_part(TOTAL, "vertebrae")
check(
    "8b resolve vertebrae -> offset 24",
    vp.name == "vertebrae" and vp.label_offset == 24,
    f"got name={vp.name!r} offset={vp.label_offset!r}",
)
rp = resolve_part(TOTAL, "ribs")
check(
    "8c resolve ribs -> offset 91",
    rp.name == "ribs" and rp.label_offset == 91,
    f"got name={rp.name!r} offset={rp.label_offset!r}",
)
try:
    resolve_part(TOTAL, "spleen")
    check("8d resolve unknown", False, "no exception raised")
except KeyError as e:
    check("8d resolve unknown KeyError", "spleen" in str(e), f"msg={e!s}")

# ---------------------------------------------------------------------------
# 9. Phase 19 (19-01): body spec, generic tile_step_size field, no-literal gate
# ---------------------------------------------------------------------------
check(
    "9a total tile_step_size default 0.8 (behavior lock)",
    TASK_REGISTRY["total"].tile_step_size == 0.8,
    f"got {TASK_REGISTRY['total'].tile_step_size!r}",
)
check(
    "9b body tile_step_size 0.5",
    TASK_REGISTRY["body"].tile_step_size == 0.5,
    f"got {TASK_REGISTRY['body'].tile_step_size!r}",
)
bplan = build_topology(TASK_REGISTRY["body"], ["body"])
check(
    "9c body builds single_model with 1 part",
    bplan.family == "single_model" and tuple(bplan.parts) == ("body",) and bplan.serialized is False,
    f"got {bplan!r}",
)
# House grep gate (BODY-03): zero quoted "body" literals in builder/operator
# source — body must be pure registry data + data modules, not builder code.
body_pat = re.compile(r'"body"|\x27body\x27')
impure = []
for f in [APP_ROOT / "my_app" / "app.py"] + sorted((APP_ROOT / "my_app" / "operators").glob("*.py")):
    for line in f.read_text().splitlines():
        if body_pat.search(line):
            impure.append(f"{f.name}: {line.strip()}")
check("9d no quoted 'body' literal in app.py/operators/*.py", not impure, f"hits={impure}")

print("ALL PASS: test_builder")
sys.exit(0)
