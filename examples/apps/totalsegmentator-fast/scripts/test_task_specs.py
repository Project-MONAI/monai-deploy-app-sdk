#!/usr/bin/env python
"""Phase 17.2 unit suite for my_app/task_specs (plan 17-02, Task 1).

House pattern (mirrors scripts/test_merge_remap.py): standalone executable,
NOT pytest; exits 0 iff every check passes. Headless: stdlib + the app's own
modules only, no GPU, no model/corpus files.

Checks:
  1. the `total` entry equals the live my_app.config constants (names,
     offsets, max local labels, config names)
  2. get_task_spec("bogus") fails fast, naming the bogus value and the valid set
  3. specs are frozen (dataclass immutability)
  4. catalog-expressibility smoke: synthetic single-model + named-crop +
     scalar- and vector-resample specs construct
  5. registry shape: exactly 45 entries (Phase 21 — 3 shipped + 40 generated;
     Phase 26 plan 26-01 — +2 dev layout-test entries)
  6. the `total` labels reference resolves to the 117-label volume_labels table
     (118 entries: background 0 + labels 1..117)
  7. the `body` entry (Phase 19) field assertions + total tile_step_size default
  8. C+M headneck_muscles: parts, offsets, config_names, build_topology dispatch
  9. config_name variants (3d_lowres_high / 3d_fullres_high / 3d_fullres guards)
 10. resample variants (None / scalar / 3-vector)
 11. robust_crop trio (total_3mm crop parts) + remove_outside (heartchambers_highres)
 12. folds (0..4) vascular trio + licensed 14-task set
 13. new-entry invariants (resample_order 1, empty snomed/postprocess, seg_sr, no tta)
 14. total_v3 (5 parts, offsets, tile 0.8, 118-key labels, vertebrae_L6 == 26)
 15. test task (deprecated string-crop form, task 517)
 16. frozen + defaulted-fields round-trip (dataclasses.replace vs FrozenInstanceError)
 17. family dispatch sweep: build_topology for all 43 entries

Run:
  cd examples/apps/totalsegmentator-fast
  /tmp/monai-env/.venv/bin/python scripts/test_task_specs.py
Exit: 0 iff every check passes; 1 otherwise.
"""

import dataclasses
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

import my_app.config as cfg  # noqa: E402
from my_app.task_specs import build_topology  # noqa: E402
from my_app.task_specs import (
    CROP_STAGES,
    TASK_REGISTRY,
    CropSpec,
    PartSpec,
    TaskSpec,
    get_task_spec,
)


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        print(f"FAIL: {name} {detail}")
        sys.exit(1)
    print(f"PASS: {name}")


# ---------------------------------------------------------------------------
# 1. total entry == live config constants
# ---------------------------------------------------------------------------
spec = get_task_spec("total")
expected_names = ["organs", "vertebrae", "cardiac", "muscles", "ribs"]
check(
    "1a part names",
    [p.name for p in spec.parts] == expected_names,
    f"got {[p.name for p in spec.parts]}",
)
check(
    "1b part names == config.MODEL_PARTS",
    [p["name"] for p in cfg.MODEL_PARTS] == expected_names,
)
check(
    "1c offsets",
    [p.label_offset for p in spec.parts] == [0, 24, 50, 68, 91],
    f"got {[p.label_offset for p in spec.parts]}",
)
check(
    "1d offsets == config.MODEL_PARTS",
    [p["label_offset"] for p in cfg.MODEL_PARTS] == [p.label_offset for p in spec.parts],
)
expected_max = {"organs": 24, "vertebrae": 26, "cardiac": 18, "muscles": 23, "ribs": 26}
check(
    "1e max local labels",
    {p.name: p.max_local_label for p in spec.parts} == expected_max,
    f"got {{{', '.join(f'{p.name}:{p.max_local_label}' for p in spec.parts)}}}",
)
check(
    "1f max local labels == config.EXPECTED_MAX_LOCAL_LABEL",
    {p.name: p.max_local_label for p in spec.parts} == cfg.EXPECTED_MAX_LOCAL_LABEL,
)
check(
    "1g all config_name == 3d_fullres",
    all(p.config_name == "3d_fullres" for p in spec.parts),
)

# ---------------------------------------------------------------------------
# 2. unknown task fails fast, naming the bogus value and the valid set
# ---------------------------------------------------------------------------
try:
    get_task_spec("bogus")
    check("2 fail-fast", False, "no exception raised")
except ValueError as e:
    check("2 fail-fast", "bogus" in str(e) and "total" in str(e), f"msg={e!s}")
    check(
        "2b fail-fast names representative names across the 43-task set",
        all(
            t in str(e)
            for t in (
                "total",
                "liver_segments",
                "body",
                "total_v3",
                "headneck_muscles",
                "teeth",
                "test",
            )
        ),
        f"msg={e!s}",
    )

# ---------------------------------------------------------------------------
# 3. frozen
# ---------------------------------------------------------------------------
p = PartSpec(name="x", label_offset=0)
try:
    p.name = "y"  # type: ignore[misc]
    check("3 frozen", False, "no exception raised")
except dataclasses.FrozenInstanceError:
    check("3 frozen", True)

# ---------------------------------------------------------------------------
# 4. catalog-expressibility smoke (constructed, NOT registered)
# ---------------------------------------------------------------------------
TaskSpec(
    task_id=113,
    family="single_model",
    parts=(PartSpec(name="teeth", label_offset=0, config_name="3d_lowres_high"),),
    crop=CropSpec(source="craniofacial_structures", labels=("jaw",), addon=(10, 10, 10)),
    resample=0.5,
    target_spacing=0.5,
    output_mode="seg_sr",
)
TaskSpec(
    task_id=957,
    family="single_model",
    parts=(PartSpec(name="total_highres", label_offset=0, config_name="3d_fullres_high"),),
    crop=None,
    resample=(0.75, 0.75, 1.0),
    target_spacing=0.75,
    output_mode="seg_sr",
)
check("4 expressibility smoke", True)

# ---------------------------------------------------------------------------
# 5. registry shape (Phase 21: 3 shipped + 40 generated = 43)
# ---------------------------------------------------------------------------
# Phase 26 (26-01) dev layout-test tasks: single-part organs bundles in the
# lowres / cascade layouts so the CFG-01 config matrix cells are RUN-able.
DEV_26 = {"organs_lowres_test", "organs_cascade_test"}

EXPECTED_43 = {
    "total",
    "liver_segments",
    "body",
    "total_v3",
    "total_highres_test",
    "lung_vessels",
    "lung_vessels_LEGACY",
    "cerebral_bleed",
    "hip_implant",
    "pleural_pericard_effusion",
    "liver_vessels",
    "head_glands_cavities",
    "headneck_bones_vessels",
    "head_muscles",
    "headneck_muscles",
    "oculomotor_muscles",
    "lung_nodules",
    "kidney_cysts",
    "breasts",
    "ventricle_parts",
    "liver_lesions",
    "craniofacial_structures",
    "abdominal_muscles",
    "teeth",
    "trunk_cavities",
    "vertebrae_body",
    "vertebrae_pp",
    "vertebrae_pp_refined",
    "heartchambers_highres",
    "appendicular_bones",
    "tissue_types",
    "tissue_4_types",
    "face",
    "brain_structures",
    "thigh_shoulder_muscles",
    "coronary_arteries",
    "coronary_arteries_LEGACY",
    "aortic_sinuses",
    "renal_arteries",
    "aorta_annulus",
    "aortic_dissection",
    "pulmonary_artery_landmarks",
    "test",
}
check(
    "5a registry has 45 entries (43 + organs_lowres_test + organs_cascade_test dev entries)",
    len(TASK_REGISTRY) == 45,
    f"got {len(TASK_REGISTRY)}",
)
check(
    "5b registry shape == the 43 TS 2.18 CT task names + 2 Phase-26 dev entries",
    set(TASK_REGISTRY) == EXPECTED_43 | DEV_26,
    f"missing={sorted((EXPECTED_43 | DEV_26) - set(TASK_REGISTRY))} extra={sorted(set(TASK_REGISTRY) - (EXPECTED_43 | DEV_26))}",
)

# ---------------------------------------------------------------------------
# 6. label-table resolution (TASK-01)
# ---------------------------------------------------------------------------
import my_app.ai_segment_descriptions as a  # noqa: E402  (test only — never inside task_specs)

check("6a labels reference", spec.labels == "volume_labels", f"got {spec.labels!r}")
t = getattr(a, spec.labels)
ids = sorted(t.values())
check(
    "6b resolves to 117-label volume_labels (bg 0 + 1..117)",
    t is a.volume_labels and t.get("background") == 0 and [x for x in ids if x > 0] == list(range(1, 118)),
    f"len={len(t)}",
)

# ---------------------------------------------------------------------------
# 7. body entry (Phase 19, plan 19-01) — field assertions verbatim
# ---------------------------------------------------------------------------
b = get_task_spec("body")
check(
    "7a body task_id/family",
    b.task_id == 299 and b.family == "single_model",
    f"got {b.task_id!r} {b.family!r}",
)
check(
    "7b body parts",
    b.parts == (PartSpec(name="body", label_offset=0, config_name="3d_fullres", max_local_label=2),),
    f"got {b.parts!r}",
)
check("7c body crop None", b.crop is None, f"got {b.crop!r}")
check(
    "7d body resample fields",
    b.resample == 1.5 and b.target_spacing == 1.5 and b.resample_order == 1,
    f"got resample={b.resample!r} target_spacing={b.target_spacing!r} order={b.resample_order!r}",
)
check(
    "7e body tile_step_size 0.5 (TS nnunet.py:568-572: 0.8 only for total family)",
    b.tile_step_size == 0.5,
    f"got {b.tile_step_size!r}",
)
check(
    "7f body trainer/tta",
    b.trainer == "nnUNetTrainer" and b.tta is False,
    f"got {b.trainer!r} tta={b.tta!r}",
)
check(
    "7g body snomed table pointers",
    b.snomed == "body_descriptions" and b.snomed_module == "ai_body_descriptions",
    f"got {b.snomed!r} / {b.snomed_module!r}",
)
check(
    "7h body labels inline dict",
    b.labels == {"background": 0, "body_trunc": 1, "body_extremities": 2},
    f"got {b.labels!r}",
)
check("7i body output_mode seg_sr", b.output_mode == "seg_sr", f"got {b.output_mode!r}")
# 7j: total keeps the default tile_step_size (behavior lock)
check(
    "7j total tile_step_size default 0.8",
    get_task_spec("total").tile_step_size == 0.8,
    f"got {get_task_spec('total').tile_step_size!r}",
)
# 7k (19-03): resample_order is the spec field the pre-resample plumb-through
# reads — total stays 3 (byte-locked oracle default), liver_segments stays 1.
check(
    "7k total resample_order 3 (byte-locked oracle default)",
    get_task_spec("total").resample_order == 3,
    f"got {get_task_spec('total').resample_order!r}",
)
check(
    "7k liver_segments resample_order 1 (regression guard)",
    get_task_spec("liver_segments").resample_order == 1,
    f"got {get_task_spec('liver_segments').resample_order!r}",
)
# 7l (19-05): body task postprocessing = TS nnunet.py:686-695 verbatim
# (keep-largest body_trunc + remove_small_blobs body_extremities 50,000 mm3,
# model resolution, pre-back-resample); total/liver_segments keep the empty
# default (TS applies no task postprocessing to them).
check(
    "7l body postprocess rules (TS nnunet.py:686-695)",
    b.postprocess
    == (
        ("keep_largest_blob", "body_trunc"),
        ("remove_small_blobs", "body_extremities", 50000.0),
    ),
    f"got {b.postprocess!r}",
)
check(
    "7l total/liver_segments postprocess empty (default)",
    get_task_spec("total").postprocess == () and get_task_spec("liver_segments").postprocess == (),
    f"got {get_task_spec('total').postprocess!r} {get_task_spec('liver_segments').postprocess!r}",
)

# ---------------------------------------------------------------------------
# 8. C+M — headneck_muscles (Phase 21, 21-02): crop + 2 main parts
# ---------------------------------------------------------------------------
h = get_task_spec("headneck_muscles")
check(
    "8a headneck_muscles family crop_cascade",
    h.family == "crop_cascade",
    f"got {h.family!r}",
)
check(
    "8b headneck_muscles part names (Phase-22 bundle-placement contract)",
    tuple(p.name for p in h.parts) == ("total_6mm", "part1", "part2"),
    f"got {tuple(p.name for p in h.parts)!r}",
)
p1, p2 = h.parts[1], h.parts[2]
check(
    "8c headneck_muscles part1/part2 offsets + max local labels",
    p1.label_offset == 0 and p1.max_local_label == 11 and p2.label_offset == 11 and p2.max_local_label == 12,
    f"got p1=({p1.label_offset}, {p1.max_local_label}) p2=({p2.label_offset}, {p2.max_local_label})",
)
check(
    "8d headneck_muscles config_names: crop 3d_fullres, main parts 3d_fullres_high",
    [p.config_name for p in h.parts] == ["3d_fullres", "3d_fullres_high", "3d_fullres_high"],
    f"got {[p.config_name for p in h.parts]}",
)
check(
    "8e headneck_muscles resample vector + not licensed",
    h.resample == (0.75, 0.75, 1.0) and h.licensed is False,
    f"got {h.resample!r} licensed={h.licensed!r}",
)
hplan = build_topology(h, ["total_6mm", "part1", "part2"])
check(
    "8f build_topology(headneck_muscles) -> crop_cascade with CROP_STAGES",
    hplan.family == "crop_cascade" and hplan.crop_stages == CROP_STAGES,
    f"got family={hplan.family!r} crop_stages={hplan.crop_stages!r}",
)

# ---------------------------------------------------------------------------
# 9. config_name variants
# ---------------------------------------------------------------------------
te = get_task_spec("teeth")
check(
    "9a teeth main part 3d_lowres_high + named crop source",
    te.parts[1].config_name == "3d_lowres_high" and te.parts[0].name == "craniofacial_structures",
    f"got main={te.parts[1].config_name!r} crop_part={te.parts[0].name!r}",
)
check(
    "9b teeth crop source craniofacial_structures",
    te.crop is not None and te.crop.source == "craniofacial_structures",
    f"got {te.crop!r}",
)
check(
    "9c aorta_annulus 3d_fullres_high",
    get_task_spec("aorta_annulus").parts[0].config_name == "3d_fullres_high",
    f"got {get_task_spec('aorta_annulus').parts[0].config_name!r}",
)
check(
    "9d total/body parts still 3d_fullres (regression guard)",
    all(p.config_name == "3d_fullres" for p in get_task_spec("total").parts)
    and all(p.config_name == "3d_fullres" for p in get_task_spec("body").parts),
)

# ---------------------------------------------------------------------------
# 10. resample variants
# ---------------------------------------------------------------------------
check(
    "10a cerebral_bleed resample None",
    get_task_spec("cerebral_bleed").resample is None,
    f"got {get_task_spec('cerebral_bleed').resample!r}",
)
check(
    "10b vertebrae_body resample scalar 1.5 (not a tuple)",
    isinstance(get_task_spec("vertebrae_body").resample, float) and get_task_spec("vertebrae_body").resample == 1.5,
    f"got {get_task_spec('vertebrae_body').resample!r}",
)
check(
    "10c total_highres_test resample 3-vector",
    get_task_spec("total_highres_test").resample == (0.75, 0.75, 1.0),
    f"got {get_task_spec('total_highres_test').resample!r}",
)

# ---------------------------------------------------------------------------
# 11. robust_crop trio + remove_outside
# ---------------------------------------------------------------------------
ROBUST = {"lung_vessels", "liver_lesions", "heartchambers_highres"}
check(
    "11a robust_crop True for exactly the 3 oracle tasks",
    {k for k, v in TASK_REGISTRY.items() if v.robust_crop} == ROBUST,
    f"got {sorted(k for k, v in TASK_REGISTRY.items() if v.robust_crop)}",
)
check(
    "11b robust tasks carry the total_3mm crop part",
    all(get_task_spec(t).parts[0].name == "total_3mm" for t in ROBUST),
    f"got {[get_task_spec(t).parts[0].name for t in sorted(ROBUST)]}",
)
check(
    "11c heartchambers_highres remove_outside + dilation",
    get_task_spec("heartchambers_highres").remove_outside == ("heart", "aorta", "inferior_vena_cava")
    and get_task_spec("heartchambers_highres").remove_outside_dilation == 10.0,
    f"got {get_task_spec('heartchambers_highres').remove_outside!r} "
    f"{get_task_spec('heartchambers_highres').remove_outside_dilation!r}",
)
check(
    "11d every other entry remove_outside empty",
    all(v.remove_outside == () for k, v in TASK_REGISTRY.items() if k != "heartchambers_highres"),
)

# ---------------------------------------------------------------------------
# 12. folds + licensed
# ---------------------------------------------------------------------------
FIVEFOLD = {"aorta_annulus", "aortic_dissection", "pulmonary_artery_landmarks"}
check(
    "12a folds (0,1,2,3,4) for exactly the 3 vascular tasks",
    {k for k, v in TASK_REGISTRY.items() if v.folds == (0, 1, 2, 3, 4)} == FIVEFOLD,
    f"got {sorted(k for k, v in TASK_REGISTRY.items() if v.folds == (0, 1, 2, 3, 4))}",
)
check(
    "12b all other entries folds == (0,)",
    all(v.folds == (0,) for k, v in TASK_REGISTRY.items() if k not in FIVEFOLD),
)
LICENSED = {
    "heartchambers_highres",
    "appendicular_bones",
    "tissue_types",
    "tissue_4_types",
    "face",
    "brain_structures",
    "thigh_shoulder_muscles",
    "coronary_arteries",
    "coronary_arteries_LEGACY",
    "aortic_sinuses",
    "renal_arteries",
    "aorta_annulus",
    "aortic_dissection",
    "pulmonary_artery_landmarks",
}
check(
    "12c licensed True for exactly the 14 catalog tasks",
    {k for k, v in TASK_REGISTRY.items() if v.licensed} == LICENSED,
    f"got {sorted(k for k, v in TASK_REGISTRY.items() if v.licensed)}",
)

# ---------------------------------------------------------------------------
# 13. new-entry invariants (40 generated: D-21-3 resample_order 1; snomed gap = Phase 22)
# ---------------------------------------------------------------------------
NEW_40 = EXPECTED_43 - {"total", "liver_segments", "body"}

# 22-02: the 5 Phase-22 sampled tasks with their SNOMED modules (plan 22-02)
SNOMED_5 = {
    "total_v3": ("total_v3_descriptions", "ai_total_v3_descriptions", 117),
    "headneck_muscles": (
        "headneck_muscles_descriptions",
        "ai_headneck_muscles_descriptions",
        23,
    ),
    "vertebrae_pp": ("vertebrae_pp_descriptions", "ai_vertebrae_pp_descriptions", 24),
    "pleural_pericard_effusion": (
        "pleural_pericard_effusion_descriptions",
        "ai_pleural_pericard_effusion_descriptions",
        3,
    ),
    "trunk_cavities": (
        "trunk_cavities_descriptions",
        "ai_trunk_cavities_descriptions",
        4,
    ),
}

check(
    "13a all 40 new entries resample_order 1 (D-21-3)",
    all(TASK_REGISTRY[k].resample_order == 1 for k in NEW_40),
)
check(
    "13b 35 new entries snomed/snomed_module empty (codes never invented before validation)",
    all(TASK_REGISTRY[k].snomed == "" and TASK_REGISTRY[k].snomed_module == "" for k in NEW_40 if k not in SNOMED_5),
)
check(
    "13c all other new entries postprocess empty, seg_sr, no tta",
    all(
        TASK_REGISTRY[k].postprocess == ()
        and TASK_REGISTRY[k].output_mode == "seg_sr"
        and TASK_REGISTRY[k].tta is False
        for k in NEW_40
        if k not in ("vertebrae_pp",)
    ),
)
check(
    "13d vertebrae_pp postprocess restored (22-09: TS nnunet.py:693 postprocess_vertebrae_pp)",
    TASK_REGISTRY["vertebrae_pp"].postprocess == (("dilate_labels", 3.0, 100.0),),
    f"got {TASK_REGISTRY['vertebrae_pp'].postprocess!r}",
)

# ---------------------------------------------------------------------------
# 14. total_v3 (the total-family twin with the L6 rename)
# ---------------------------------------------------------------------------
tv3 = get_task_spec("total_v3")
check(
    "14a total_v3 5 main part names + offsets",
    [p.name for p in tv3.parts] == ["organs", "vertebrae", "cardiac", "muscles", "ribs"]
    and [p.label_offset for p in tv3.parts] == [0, 24, 50, 68, 91],
    f"got {[(p.name, p.label_offset) for p in tv3.parts]}",
)
check(
    "14b total_v3 tile_step_size 0.8 + resample 1.5",
    tv3.tile_step_size == 0.8 and tv3.resample == 1.5,
    f"got tile={tv3.tile_step_size!r} resample={tv3.resample!r}",
)
check(
    "14c total_v3 labels 118-key dict with vertebrae_L6 == 26 (vs total's vertebrae_S1)",
    isinstance(tv3.labels, dict)
    and len(tv3.labels) == 118
    and tv3.labels.get("background") == 0
    and tv3.labels.get("vertebrae_L6") == 26
    and "vertebrae_S1" not in tv3.labels,
    f"len={len(tv3.labels) if isinstance(tv3.labels, dict) else 'n/a'}",
)

# ---------------------------------------------------------------------------
# 15. test task (dev-only; deprecated TS string-crop form)
# ---------------------------------------------------------------------------
tst = get_task_spec("test")
check(
    "15a test task_id 517 (generator emits the one-tuple form)",
    tuple(tst.task_id) == (517,),
    f"got {tst.task_id!r}",
)
check(
    "15b test crop = CropSpec(source='body', labels=(), addon=(20,20,20))",
    tst.crop is not None and tst.crop.source == "body" and tst.crop.labels == () and tst.crop.addon == (20, 20, 20),
    f"got {tst.crop!r}",
)
check(
    "15c test resample None + 2-key carpal label table",
    tst.resample is None and tst.labels == {"background": 0, "carpal": 1},
    f"got resample={tst.resample!r} labels={tst.labels!r}",
)

# ---------------------------------------------------------------------------
# 16. frozen + defaulted-fields round-trip (Phase 21 schema fields)
# ---------------------------------------------------------------------------
mn = TaskSpec(
    task_id=999,
    family="single_model",
    parts=(PartSpec(name="x", label_offset=0),),
    crop=None,
    resample=None,
    target_spacing=1.0,
)
check(
    "16a omitted Phase-21 fields take defaults",
    mn.folds == (0,)
    and mn.robust_crop is False
    and mn.remove_outside == ()
    and mn.remove_outside_dilation == 0.0
    and mn.licensed is False,
    f"got folds={mn.folds!r} robust={mn.robust_crop!r} remove_outside={mn.remove_outside!r} "
    f"dilation={mn.remove_outside_dilation!r} licensed={mn.licensed!r}",
)
check(
    "16b dataclasses.replace works on the frozen TaskSpec",
    dataclasses.replace(h, resample_order=1).resample_order == 1,
)
try:
    h.resample_order = 2  # type: ignore[misc]
    check("16c direct assignment still raises", False, "no exception raised")
except dataclasses.FrozenInstanceError:
    check("16c direct assignment still raises FrozenInstanceError", True)

# ---------------------------------------------------------------------------
# 17. family dispatch sweep — every catalog variant at the plans level
# ---------------------------------------------------------------------------
sweep_bad = []
for name, s in TASK_REGISTRY.items():
    try:
        plan = build_topology(s, [p.name for p in s.parts])
    except Exception as e:  # noqa: BLE001
        sweep_bad.append(f"{name}: raised {e!r}")
        continue
    if plan.family != s.family:
        sweep_bad.append(f"{name}: plan.family={plan.family!r} != spec.family={s.family!r}")
    if plan.serialized is not (plan.family == "multi_part"):
        sweep_bad.append(f"{name}: serialized={plan.serialized!r}")
    expected_stages = CROP_STAGES if s.crop is not None else ()
    if plan.crop_stages != expected_stages:
        sweep_bad.append(f"{name}: crop_stages={plan.crop_stages!r} != {expected_stages!r}")
check(
    "17 build_topology dispatch sweep over all 45 entries",
    not sweep_bad,
    "; ".join(sweep_bad),
)

# ---------------------------------------------------------------------------
# 20. Phase 26 (26-01) dev layout-test entries
# ---------------------------------------------------------------------------
ol = get_task_spec("organs_lowres_test")  # must not raise
oc = get_task_spec("organs_cascade_test")  # must not raise
check(
    "20a organs_lowres_test: single_model, config 3d_lowres, max_local 24, seg_sr",
    ol.family == "single_model"
    and ol.parts == (PartSpec(name="organs", label_offset=0, config_name="3d_lowres", max_local_label=24),)
    and ol.output_mode == "seg_sr"
    and ol.crop is None
    and ol.tta is False
    and ol.licensed is False
    and ol.folds == (0,),
    f"got {ol}",
)
check(
    "20b organs_cascade_test: single_model, config 3d_cascade_fullres, max_local 24, seg_sr",
    oc.family == "single_model"
    and oc.parts
    == (
        PartSpec(
            name="organs",
            label_offset=0,
            config_name="3d_cascade_fullres",
            max_local_label=24,
        ),
    )
    and oc.output_mode == "seg_sr"
    and oc.crop is None
    and oc.tta is False
    and oc.licensed is False
    and oc.folds == (0,),
    f"got {oc}",
)
check(
    "20c both dev entries: body-template fields (tile 0.8, order 3, NoMirroring trainer, empty postprocess)",
    ol.tile_step_size == 0.8
    and ol.resample_order == 3
    and ol.trainer == "nnUNetTrainerNoMirroring"
    and ol.postprocess == ()
    and oc.tile_step_size == 0.8
    and oc.resample_order == 3
    and oc.trainer == "nnUNetTrainerNoMirroring"
    and oc.postprocess == (),
    f"got ol={(ol.tile_step_size, ol.resample_order, ol.trainer, ol.postprocess)} "
    f"oc={(oc.tile_step_size, oc.resample_order, oc.trainer, oc.postprocess)}",
)
check(
    "20d both dev entries: task_id (291,) (organs = Dataset291, one-tuple style)",
    tuple(ol.task_id) == (291,) and tuple(oc.task_id) == (291,),
    f"got ol={ol.task_id!r} oc={oc.task_id!r}",
)
check(
    "20e labels == organs 24-label subset of volume_labels (ids 1..24, bg 0)",
    ol.labels == {k: v for k, v in a.volume_labels.items() if v <= 24} and oc.labels == ol.labels,
    f"got {ol.labels!r}",
)
check(
    "20f dev entries snomed_module ai_segment_descriptions",
    ol.snomed_module == "ai_segment_descriptions" and oc.snomed_module == "ai_segment_descriptions",
    f"got ol={ol.snomed_module!r} oc={oc.snomed_module!r}",
)

# ---------------------------------------------------------------------------
# 18. 22-02: per-part config_name completeness check in resolve_task_model_root
# ---------------------------------------------------------------------------
import tempfile as _tempfile


def _mk_fake_bundle(root: Path, part_configs):
    """Build a minimal bundle tree: jsonpkls/plans.json +
    <config>/fold_0/final_model.pt per part (resolver only stats files)."""
    for name, cfg_name in part_configs:
        pdir = root / name
        (pdir / "jsonpkls").mkdir(parents=True)
        (pdir / "jsonpkls" / "plans.json").write_text("{}")
        (pdir / cfg_name / "fold_0").mkdir(parents=True)
        (pdir / cfg_name / "fold_0" / "final_model.pt").write_bytes(b"")


with _tempfile.TemporaryDirectory() as _td:
    root = Path(_td)
    _mk_fake_bundle(root, [("part1", "3d_fullres_high"), ("part2", "3d_fullres_high")])
    got = cfg.resolve_task_model_root(
        root,
        "fake_cascade",
        ["part1", "part2"],
        part_configs={"part1": "3d_fullres_high", "part2": "3d_fullres_high"},
    )
    check(
        "18a 3d_fullres_high pair resolves with per-part configs",
        got == root,
        f"got {got}",
    )

with _tempfile.TemporaryDirectory() as _td:
    root = Path(_td)
    _mk_fake_bundle(root, [("part1", "3d_fullres_high"), ("part2", "3d_fullres_high")])
    (root / "part2" / "3d_fullres_high" / "fold_0" / "final_model.pt").unlink()
    try:
        cfg.resolve_task_model_root(
            root,
            "fake_cascade",
            ["part1", "part2"],
            part_configs={"part1": "3d_fullres_high", "part2": "3d_fullres_high"},
        )
        check("18b fail-fast on missing highres checkpoint", False, "no exception")
    except FileNotFoundError as e:
        # 26-01 (CFG02-01): any-fold checkpoint requirement — the message names
        # the any-fold pattern instead of the old fold_0/final_model.pt literal.
        check(
            "18b fail-fast on missing highres checkpoint",
            "part2/3d_fullres_high/fold_*/" in str(e) and "final_model.pt" in str(e),
            f"msg={e!s}",
        )

with _tempfile.TemporaryDirectory() as _td:
    root = Path(_td)
    _mk_fake_bundle(root, [("total_6mm", "3d_fullres"), ("mainx", "3d_fullres")])
    got = cfg.resolve_task_model_root(root, "fake_task", ["total_6mm", "mainx"])
    check(
        "18c default (no part_configs) still requires 3d_fullres",
        got == root,
        f"got {got}",
    )

# ---------------------------------------------------------------------------
# 19. 22-02: 5 Phase-22 SNOMED description modules + registry snomed backfill
# ---------------------------------------------------------------------------
import importlib  # noqa: E402

# 19a: registry points at exactly the 5 sampled tasks AMONG THE GENERATED 40
# (the 3 shipped entries keep their pre-existing tables)
_backfilled = {
    k: (v.snomed, v.snomed_module) for k, v in TASK_REGISTRY.items() if k in NEW_40 and (v.snomed or v.snomed_module)
}
check(
    "19a exactly the 5 sampled tasks have non-empty snomed/snomed_module",
    set(_backfilled) == set(SNOMED_5) and all(_backfilled[k] == (SNOMED_5[k][0], SNOMED_5[k][1]) for k in SNOMED_5),
    f"got {_backfilled}",
)

for task, (attr, modname, n) in SNOMED_5.items():
    mod = importlib.import_module(f"my_app.{modname}")
    descs = getattr(mod, attr)
    vol = getattr(mod, f"{task}_volume_labels")
    check(
        f"19b {task}: table length {n} == registry label count, ids exactly 1..{n}",
        len(descs) == n and len(vol) == n and sorted(vol.values()) == list(range(1, n + 1)),
        f"desc={len(descs)} vol={len(vol)}",
    )
    # entry i (0-based i-1) must carry the registry name of label id i
    by_id = {vid: nm for nm, vid in vol.items()}
    order_ok = (
        all(sd._segment_label == by_id[i] for i, sd in enumerate(descs, 1)) if task != "total_v3" else True
    )  # total_v3 keeps total's display names (116) + vertebrae_L6 (26)
    check(f"19c {task}: entries in label-id order 1..{n}", order_ok)
    # every code is SCT; generic-fallback entries use Tissue/Tissue on BOTH fields
    fallback_ok = all(
        sd._segmented_property_category.scheme_designator == "SCT"
        and sd._segmented_property_type.scheme_designator == "SCT"
        and (sd._segmented_property_category.value != "85756007" or sd._segmented_property_type.value == "85756007")
        for sd in descs
    )
    check(f"19d {task}: all codes SCT, fallbacks are pure Tissue/Tissue", fallback_ok)

# 19e: total_v3 entry 26 is vertebrae_L6 with the generic fallback (not the CSV vertebrae_S1 row)
_tv3mod = importlib.import_module("my_app.ai_total_v3_descriptions")
e26 = _tv3mod.total_v3_descriptions[25]
check(
    "19e total_v3 entry 26 == vertebrae_L6 with Tissue/Tissue fallback",
    e26._segment_label == "vertebrae_L6"
    and e26._segmented_property_category.value == "85756007"
    and e26._segmented_property_type.value == "85756007",
    f"got label={e26._segment_label} type={e26._segmented_property_type.value}",
)

print("ALL PASS: test_task_specs (45-entry registry, incl. Phase-26 dev entries)")
sys.exit(0)
