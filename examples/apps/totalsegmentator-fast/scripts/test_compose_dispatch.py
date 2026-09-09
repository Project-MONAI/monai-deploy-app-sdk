#!/usr/bin/env python
"""Phase 18 (plan 18-02, Task 3) headless compose-dispatch smoke for the
`liver_segments` crop_cascade topology.

Why this exists (Phase 17 lesson): 17-03 never actually EXECUTED compose()
headlessly, so a missing package re-export only surfaced at 17-04. This
suite drives the REAL compose() for HOLOSCAN_TASK=liver_segments up to
(but before) pipeline launch and asserts the built DAG:

  (a) the crop_cascade branch executes — no NotImplementedError, no missing
      import on the dispatch path;
  (b) all 8 cascade nodes (+ PasteOp + emit) are present in the built
      Subgraph, named per the p18_run_study.sh span contract;
  (c) EVERY declared input port of EVERY built operator has a receiver
      (18-RESEARCH Pitfall 5 — an unwired declared input = silent hang);
      every declared output has >= 1 receiver EXCEPT the documented
      conditional `seg_cropped` output of the crop_mask op (empty-path
      only — its edge to paste always exists statically);
  (d) the gated main SlidingWindow op declares exactly its 2 inputs
      (preprocessed + prev_part_seg) — the 6mm-release-first serialization;
  (e) Phase 19 (19-01): body single_model compose — full seg_sr P10 chain
      (seg_metrics_op + dicom_seg_writer + dicom_sr_writer, 6 flow edges,
      Pitfall-5 clean, swin step 0.5); total + HOLOSCAN_MODEL_PARTS=organs
      compose — emit-only (writers absent, swin step 0.8).

The dry-run hook (plan-sanctioned): `Application.init_app_context` is
monkeypatched to a fake context — compose() is otherwise 100% real
(operator construction, model self-loads in setup on GPU 0, flows,
scheduler). No pipeline launch, no DICOM read.

NOTE (documented, pre-existing): the total-family SegEmitOperator in mode
"single" declares `seg_merged` and `seg_merged_dicom` inputs that the
single_model branch never wires (the 5-part chain is the live family; the
single path is P8 debug/triage). The Pitfall-5 audit below therefore
exempts ONLY those two known-dead ports when the family is not
crop_cascade. For the crop_cascade family under test here, the exemption
list is empty — every declared input must be wired.

Run:
  cd examples/apps/totalsegmentator-fast
  ulimit -s unlimited && CUDA_VISIBLE_DEVICES=0 \
    /tmp/monai-env/.venv/bin/python scripts/test_compose_dispatch.py
Exit: 0 iff all assertions pass.
"""

import dataclasses
import importlib
import logging
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[2]  # monai-deploy-app-sdk
STUDY_DIR = "/raid/map_outputs/ct_ts_validation/tcia-dcm/gpu/06-20-2009-NA-CT-44238"
MODEL_ROOT = str(REPO_ROOT / "ct-totalsegmentator-map/models/liver_segments")
OUT_DIR = "/tmp/t18_compose_dispatch_out"

os.environ["HOLOSCAN_TASK"] = "liver_segments"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, str(APP_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")

import my_app.app as appmod  # noqa: E402  (module-level _TASK read happens here)

assert appmod._TASK == "liver_segments", f"_TASK != liver_segments: {appmod._TASK!r}"

# ---------------------------------------------------------------------------
# Dry-run hook: replace app-context initialization (argv is read-only on
# the pybind Application). Everything else in compose() runs for real.
# ---------------------------------------------------------------------------


class _FakeContext:
    def __init__(self, study_dir=STUDY_DIR, model_root=MODEL_ROOT, out_dir=OUT_DIR):
        self.input_path = Path(study_dir)
        self.output_path = Path(out_dir)
        self.model_path = Path(model_root)
        self.args = {"input": study_dir, "output": out_dir, "model": model_root}


_CTX = {}  # per-compose context (Phase 19 additions compose more scenarios)


def _fake_init_app_context(argv, runtime_env=None):
    return _FakeContext(**_CTX)


appmod.Application.init_app_context = _fake_init_app_context

app = appmod.TotalSegFastApp()
app.compose()  # real construction: operator setup + model self-loads + flows

graph = app.graph
nodes = graph.get_nodes()
node_names = [n.name for n in nodes]
print(f"\nnodes ({len(node_names)}): {sorted(node_names)}")

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if (detail and not cond) else ''}")
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------------------------
# (b) the 8 cascade nodes (+ paste + emit) are present, span-contract names
# ---------------------------------------------------------------------------
expected_cascade = [
    "cascade_prep_6mm",  # downsample (crop_prep_6mm span)
    "swin_total_6mm",  # crop_infer (inference_3d_fullres_total_6mm)
    "postresample_total_6mm",  # (postresample_3d_fullres_total_6mm)
    "crop_mask",  # crop_mask (order-0 native mask build)
    "crop_to_mask",  # crop (native-res TS crop)
    "cascade_prep_taskres",  # back to task-res (crop_prep_taskres span)
    "swin_ct_liver_segments",  # main_infer (inference_3d_fullres_ct_liver_segments)
    "postresample_ct_liver_segments",  # (postresample_3d_fullres_ct_liver_segments)
    "paste",  # paste
    "emit_5part",  # emit (P10 node/span name — runner asserts)
]
for name in expected_cascade:
    check(f"node present: {name}", name in node_names)

# P10 output chain nodes
for name in ["seg_metrics_op", "dicom_seg_writer", "dicom_sr_writer"]:
    check(f"node present: {name}", name in node_names)

# ---------------------------------------------------------------------------
# (d) the gated main SwIn: exactly 2 declared inputs (preprocessed + gate)
# ---------------------------------------------------------------------------
by_name = {n.name: n for n in nodes}
swin_m = by_name.get("swin_ct_liver_segments")
if swin_m is not None:
    in_names = sorted(swin_m.spec.inputs)  # operator spec inputs are plain names
    check(
        "gated main swin inputs == [preprocessed, prev_part_seg]",
        in_names == ["preprocessed", "prev_part_seg"],
        f"got {in_names}",
    )

# ---------------------------------------------------------------------------
# (c) Pitfall-5 port audit: every declared input of every op has a receiver
# ---------------------------------------------------------------------------
# Connectivity maps: (input_map, output_map) where keys/vals are "node.port".
in_edges, out_edges = graph.get_port_connectivity_maps()
# in_edges: "dst.port" -> [src.port, ...] ; out_edges: "src.port" -> [dst.port, ...]

known_dead = set()
if app._task_spec.family != "crop_cascade":
    # Pre-existing P8 single-family gap (see module docstring): NOT under test.
    known_dead = {"emit_organs.seg_merged", "emit_organs.seg_merged_dicom"}
# ConditionType.NONE config ports ("optional input not requiring a sender"):
# never wired in ANY family incl. the shipped P10 total chain (the writers/
# loader get their folders from constructor args / app_context). A NONE
# condition cannot hang — it simply fires with no sender — so exempting them
# matches both GXF semantics and the proven total-family wiring.
none_condition_inputs = {
    "dicom_seg_writer.output_folder",
    "dicom_sr_writer.output_folder",
    "study_loader_op.input_folder",
}

audit_fail = 0
for n in nodes:
    for i in n.spec.inputs:  # operator spec ports are plain names
        key = f"{n.name}.{i}"
        if key in known_dead or key in none_condition_inputs:
            continue
        if not in_edges.get(key):
            print(f"  UNWIRED INPUT: {key}")
            audit_fail += 1
check("every declared input port has a receiver (Pitfall 5)", audit_fail == 0)

# Every declared output has >= 1 receiver, EXCEPT the documented conditional
# seg_cropped output of crop_mask (empty-path only message; the edge to
# paste exists statically, so it is wired — it just may carry 0 messages).
cond_outputs = {"crop_mask.seg_cropped"}
out_fail = 0
for n in nodes:
    for o in n.spec.outputs:  # operator spec ports are plain names
        key = f"{n.name}.{o}"
        if key in cond_outputs:
            continue
        if not out_edges.get(key):
            print(f"  UNWIRED OUTPUT: {key}")
            out_fail += 1
check("every declared output port has a receiver", out_fail == 0)

# The conditional edge must still exist statically (empty path fires paste).
senders = sorted(in_edges.get("paste.seg_cropped", []))
check(
    "crop_mask.seg_cropped -> paste.seg_cropped edge exists",
    "crop_mask.seg_cropped" in senders,
)
check(
    "paste.seg_cropped has BOTH senders (post_main + crop_mask)",
    senders == ["crop_mask.seg_cropped", "postresample_ct_liver_segments.seg_argmax_dicom"],
    f"got {senders}",
)

# ---------------------------------------------------------------------------
# (a) the crop branch executed: spec-driven parts, no task literals leaked
# ---------------------------------------------------------------------------
check("family == crop_cascade", app._task_spec.family == "crop_cascade")
check("spec carries crop (build_topology -> CROP_STAGES)", app._task_spec.crop is not None)

src = (APP_ROOT / "my_app/app.py").read_text()
check(
    "no quoted '3d_fullres' literal in app.py",
    '"3d_fullres"' not in src and "'3d_fullres'" not in src,
)
check(
    "no crop_cascade NotImplementedError left",
    "crop-cascade topology materialization" not in src,
)

# ---------------------------------------------------------------------------
# Phase 19 (19-01): single_model body — full seg_sr P10 chain wired + 6 flow
# edges + Pitfall-5 port audit; total + HOLOSCAN_MODEL_PARTS=organs — emit-only
# (writers ABSENT, 17-01 A/B contract); swin tile_step_size spec-driven
# (body 0.5 / total subset 0.8) via a constructor-capture wrapper.
# ---------------------------------------------------------------------------
BODY_MODEL_ROOT = str(REPO_ROOT / "ct-totalsegmentator-map/models/body")
TOTAL_MODEL_ROOT = str(REPO_ROOT / "ct-totalsegmentator-map/models/total")
HEADNECK_MODEL_ROOT = str(REPO_ROOT / "ct-totalsegmentator-map/models/headneck_muscles")
captured_steps = []
captured_pre_orders = []
captured_emits = []


def _compose(task, model_root, parts_env=None, capture=False):
    os.environ.pop("HOLOSCAN_MODEL_PARTS", None)
    if parts_env is not None:
        os.environ["HOLOSCAN_MODEL_PARTS"] = parts_env
    _CTX.clear()
    _CTX.update({"study_dir": STUDY_DIR, "model_root": model_root, "out_dir": OUT_DIR})
    orig_swin = appmod.SlideWindowOperator
    orig_pre = appmod.PreprocessOperator
    orig_emit = appmod.SegEmitOperator
    if capture:

        class _CapSwin(orig_swin):
            def __init__(self, *a, **kw):
                captured_steps.append((kw.get("name"), kw.get("tile_step_size")))
                super().__init__(*a, **kw)

        class _CapPre(orig_pre):
            def __init__(self, *a, **kw):
                captured_pre_orders.append((kw.get("name"), kw.get("input_resample_order")))
                super().__init__(*a, **kw)

        class _CapEmit(orig_emit):
            def __init__(self, *a, **kw):
                captured_emits.append((kw.get("name"), kw.get("mode"), kw.get("base_name")))
                super().__init__(*a, **kw)

        appmod.SlideWindowOperator = _CapSwin
        appmod.PreprocessOperator = _CapPre
        appmod.SegEmitOperator = _CapEmit
    try:
        appmod._TASK = task  # compose() reads the module-level knob
        app = appmod.TotalSegFastApp()
        app.compose()
        return app
    finally:
        appmod.SlideWindowOperator = orig_swin
        appmod.PreprocessOperator = orig_pre
        appmod.SegEmitOperator = orig_emit
        os.environ.pop("HOLOSCAN_MODEL_PARTS", None)
        appmod._TASK = "liver_segments"


bapp = _compose("body", BODY_MODEL_ROOT, capture=True)
check(
    "19i body preprocess input_resample_order 1 (spec-driven, TS CLI parity)",
    ("preprocess_body", 1) in captured_pre_orders,
    f"got {captured_pre_orders}",
)
b_nodes = [n.name for n in bapp.graph.get_nodes()]
print(f"\nbody nodes ({len(b_nodes)}): {sorted(b_nodes)}")
check("19a body family single_model", bapp._task_spec.family == "single_model")
for name in [
    "merge_body",
    "emit_body",
    "taskpp_body",
    "backresample_body",
    "seg_metrics_op",
    "dicom_seg_writer",
    "dicom_sr_writer",
]:
    check(f"19b body node present: {name}", name in b_nodes)

b_in, _ = bapp.graph.get_port_connectivity_maps()
# 19-02 Rule-1 fix: the single-model seg goes through backresample_body
# (original-spacing back-resample + P10 writer flip) before the metrics op
# and the SEG writer — model-resolution seg_argmax_dicom cannot feed the
# SEG writer directly (P10 writer contract) and its voxel counts would be
# wrong for metrics.
p10_edges = {
    # 19-05: seg now flows through the spec-driven task postprocess op
    # (TaskSpec.postprocess, TS nnunet.py:686-695 body rules) before the
    # original-spacing back-resample.
    "taskpp_body.seg": "postresample_body.seg_argmax_dicom",
    "backresample_body.seg": "taskpp_body.seg",
    "backresample_body.preprocessed_meta": "preprocess_body.preprocessed_meta",
    "seg_metrics_op.input_scan": "series_to_vol_op.image",
    "seg_metrics_op.segmentation_mask": "backresample_body.seg_orig",
    "dicom_sr_writer.dict": "seg_metrics_op.metrics_dict",
    "dicom_sr_writer.study_selected_series_list": "series_selector_op.study_selected_series_list",
    "dicom_seg_writer.study_selected_series_list": "series_selector_op.study_selected_series_list",
    "dicom_seg_writer.seg_image": "backresample_body.seg_image",
}
for dst, src in p10_edges.items():
    check(
        f"19c edge {src} -> {dst}",
        dst in b_in and src in b_in[dst],
        f"got {b_in.get(dst)}",
    )

body_none = {
    "dicom_seg_writer.output_folder",
    "dicom_sr_writer.output_folder",
    "study_loader_op.input_folder",
}
b_unwired = []
for n in bapp.graph.get_nodes():
    for i in n.spec.inputs:
        key = f"{n.name}.{i}"
        if key in body_none:
            continue
        if not b_in.get(key):
            b_unwired.append(key)
check(
    "19d body: every declared input wired (Pitfall 5)",
    not b_unwired,
    f"unwired={b_unwired}",
)

check(
    "19e body swin tile_step_size 0.5 (spec-driven)",
    ("swin_body", 0.5) in captured_steps,
    f"got {captured_steps}",
)

captured_steps.clear()
captured_pre_orders.clear()
tapp = _compose("total", TOTAL_MODEL_ROOT, parts_env="organs", capture=True)
t_nodes = [n.name for n in tapp.graph.get_nodes()]
print(f"\ntotal/organs nodes ({len(t_nodes)}): {sorted(t_nodes)}")
from my_app.task_specs import build_topology  # noqa: E402  (pure data, headless-safe)

_tplan = build_topology(tapp._task_spec, ["organs"])
check(
    "19f total organs-subset TOPOLOGY family single_model",
    _tplan.family == "single_model" and tuple(_tplan.parts) == ("organs",),
    f"got {_tplan!r}",
)
for name in ["dicom_seg_writer", "dicom_sr_writer", "seg_metrics_op"]:
    check(
        f"19g total organs subset: {name} ABSENT (emit-only)",
        name not in t_nodes,
    )
check(
    "19h total subset swin tile_step_size 0.8 (default field value)",
    ("swin_organs", 0.8) in captured_steps,
    f"got {captured_steps}",
)
check(
    "19j total preprocess input_resample_order 3 (spec default, byte-locked)",
    ("preprocess_organs", 3) in captured_pre_orders,
    f"got {captured_pre_orders}",
)

# ---------------------------------------------------------------------------
# Phase 22 (22-02): headneck_muscles C+M (crop + 2 main parts + spec-driven
# merge) and total_v3 multi_part (spec-driven table selection + emit contract)
# ---------------------------------------------------------------------------
from my_app.config import EXPECTED_MAX_LOCAL_LABEL as _EMLL  # noqa: E402
from my_app.config import MODEL_PARTS as _MODEL_PARTS  # noqa: E402
from my_app.task_specs import TASK_REGISTRY as _REG  # noqa: E402

# --- 22-02a: headneck_muscles (crop + 2 main parts) graph shape ------------
_hn_orig = _REG["headneck_muscles"]
_hn_snomed_live = bool(_hn_orig.snomed_module)
if _hn_snomed_live:
    try:
        importlib.import_module(f"my_app.{_hn_orig.snomed_module}")
    except ImportError:
        _hn_snomed_live = False
if not _hn_snomed_live:
    # 22-02 Task 1 lands before the 23-entry SNOMED module (Task 2): point the
    # spec at the existing liver table for the DAG-shape assertions only — the
    # description table CONTENT is irrelevant to graph topology and is covered
    # separately (test_task_specs + the Task-2 module import check).
    _REG["headneck_muscles"] = dataclasses.replace(
        _hn_orig,
        snomed="liver_segment_descriptions",
        snomed_module="ai_liver_segment_descriptions",
    )

captured_steps.clear()
captured_pre_orders.clear()
captured_emits.clear()
happ = _compose("headneck_muscles", HEADNECK_MODEL_ROOT, capture=True)
_REG["headneck_muscles"] = _hn_orig

h_nodes = [n.name for n in happ.graph.get_nodes()]
print(f"\nheadneck_muscles nodes ({len(h_nodes)}): {sorted(h_nodes)}")
for name in [
    "cascade_prep_6mm",
    "swin_total_6mm",
    "postresample_total_6mm",
    "crop_mask",
    "crop_to_mask",
    "cascade_prep_taskres",
    "swin_part1",
    "postresample_part1",
    "swin_part2",
    "postresample_part2",
    "merge_remap",
    "paste",
    "emit_5part",
    "crop_seg_sink",
    "seg_metrics_op",
    "dicom_seg_writer",
    "dicom_sr_writer",
]:
    check(f"22a headneck node present: {name}", name in h_nodes)

h_in, _ = happ.graph.get_port_connectivity_maps()
check(
    "22b headneck: both main swins at spec step 0.5 (non-total family)",
    ("swin_part1", 0.5) in captured_steps and ("swin_part2", 0.5) in captured_steps,
    f"got {captured_steps}",
)
check(
    "22c headneck: part1 gated by the 6mm postresample (P9 pattern)",
    "postresample_total_6mm.seg_argmax" in (h_in.get("swin_part1.prev_part_seg") or []),
    f"got {h_in.get('swin_part1.prev_part_seg')}",
)
check(
    "22d headneck: part2 gated by part1's postresample (serialization)",
    "postresample_part1.seg_argmax" in (h_in.get("swin_part2.prev_part_seg") or []),
    f"got {h_in.get('swin_part2.prev_part_seg')}",
)
check(
    "22e headneck: merge_remap receives BOTH main parts' seg_argmax_dicom",
    "postresample_part1.seg_argmax_dicom" in (h_in.get("merge_remap.seg_part1") or [])
    and "postresample_part2.seg_argmax_dicom" in (h_in.get("merge_remap.seg_part2") or []),
    f"got seg_part1={h_in.get('merge_remap.seg_part1')} seg_part2={h_in.get('merge_remap.seg_part2')}",
)
h_senders = sorted(h_in.get("paste.seg_cropped", []))
check(
    "22f headneck: paste.seg_cropped == [crop_mask (empty path), merge_remap.merged]",
    h_senders == ["crop_mask.seg_cropped", "merge_remap.merged"],
    f"got {h_senders}",
)
_mp_offsets = {p["name"]: p["label_offset"] for p in _MODEL_PARTS}
check(
    "22g headneck: MODEL_PARTS part1/part2 offsets 0/11",
    _mp_offsets.get("part1") == 0 and _mp_offsets.get("part2") == 11,
    "got {k: v for k, v in _mp_offsets.items() if k in ('part1', 'part2')}",
)
check(
    "22h headneck: EXPECTED_MAX_LOCAL_LABEL part1=11 part2=12",
    _EMLL.get("part1") == 11 and _EMLL.get("part2") == 12,
    f"got part1={_EMLL.get('part1')} part2={_EMLL.get('part2')}",
)
h_unwired = []
for n in happ.graph.get_nodes():
    for i in n.spec.inputs:
        key = f"{n.name}.{i}"
        if key in body_none:  # ConditionType.NONE exemptions (reuse liver set)
            continue
        if not h_in.get(key):
            h_unwired.append(key)
check(
    "22i headneck: every declared input wired (Pitfall 5)",
    not h_unwired,
    f"unwired={h_unwired}",
)
check(
    "22j headneck: emit mode=total base_name='headneck_muscles' (seg_headneck_muscles_{sar,dhw}.npy)",
    ("emit_5part", "total", "headneck_muscles") in captured_emits,
    f"got {captured_emits}",
)

# --- 22-02b: total_v3 multi_part (same part names/configs as total -> the
# total model root resolves; graph = total's 5-part chain; emit base name
# stays "total" = the seg_total_dhw.npy gate contract) -----------------------
captured_steps.clear()
captured_pre_orders.clear()
captured_emits.clear()
tvapp = _compose("total_v3", TOTAL_MODEL_ROOT, capture=True)
tv_nodes = [n.name for n in tvapp.graph.get_nodes()]
print(f"\ntotal_v3 nodes ({len(tv_nodes)}): {sorted(tv_nodes)}")
expected_tv3_nodes = {
    "preprocess",
    "swin_organs",
    "postresample_organs",
    "swin_vertebrae",
    "postresample_vertebrae",
    "swin_cardiac",
    "postresample_cardiac",
    "swin_muscles",
    "postresample_muscles",
    "swin_ribs",
    "postresample_ribs",
    "merge_5part",
    "emit_5part",
    "seg_metrics_op",
    "dicom_seg_writer",
    "dicom_sr_writer",
}

check(
    "22k total_v3: full 5-part multi_part chain present",
    all(
        n in tv_nodes
        for n in [
            "preprocess",
            "swin_organs",
            "postresample_organs",
            "swin_vertebrae",
            "postresample_vertebrae",
            "swin_cardiac",
            "postresample_cardiac",
            "swin_muscles",
            "postresample_muscles",
            "swin_ribs",
            "postresample_ribs",
            "merge_5part",
            "emit_5part",
            "seg_metrics_op",
            "dicom_seg_writer",
            "dicom_sr_writer",
        ]
    ),
    f"missing={sorted(expected_tv3_nodes - set(tv_nodes))}",
)
check(
    "22l total_v3: emit mode=total with base name 'total' (default = seg_total_dhw.npy)",
    any(n == "emit_5part" and m == "total" and bn in (None, "total") for n, m, bn in captured_emits),
    f"got {captured_emits}",
)
# Spec-driven table selection: whatever the registry points at, the SEG writer
# and metrics op consume it (Task 1: empty snomed_module -> ai_segment_descriptions
# / volume_labels fallback; Task 2: ai_total_v3_descriptions + the spec's 118-key
# label dict). Data-driven so this section is valid in BOTH states.
_tv3 = tvapp._task_spec
if _tv3.snomed_module:
    _tv3mod = importlib.import_module(f"my_app.{_tv3.snomed_module}")
    _tv3descs = getattr(_tv3mod, _tv3.snomed)
    _tv3labels = getattr(_tv3mod, _tv3.labels) if isinstance(_tv3.labels, str) else _tv3.labels
else:
    import my_app.ai_segment_descriptions as _a117

    _tv3descs = _a117.ai_segment_descriptions
    _tv3labels = _a117.volume_labels
_tv3by = {n.name: n for n in tvapp.graph.get_nodes()}
_tv3got = [sd.segment_label for sd in _tv3by["dicom_seg_writer"]._seg_descs]
_tv3exp = [d._segment_label for d in _tv3descs]
check(
    "22m total_v3: SEG writer descriptions == registry-pointed table (117 entries)",
    _tv3got == _tv3exp and len(_tv3got) == 117,
    f"got {len(_tv3got)} exp {len(_tv3exp)}; first diff "
    f"{next(((i, a, b) for i, (a, b) in enumerate(zip(_tv3got, _tv3exp)) if a != b), None)}",
)
_tv3mlabels = {k: v for k, v in _tv3labels.items() if k != "background"}
check(
    "22n total_v3: metrics labels == registry label table minus background",
    dict(_tv3by["seg_metrics_op"].labels_dict) == _tv3mlabels,
    f"got {len(_tv3by['seg_metrics_op'].labels_dict)} keys, exp {len(_tv3mlabels)}",
)

print()
if FAIL:
    print(f"COMPOSE-DISPATCH: FAIL — {len(FAIL)} assertion(s) failed: {FAIL}")
    sys.exit(1)
print(
    "COMPOSE-DISPATCH: PASS — crop_cascade branch executed; 10/10 cascade/P10 nodes present; "
    "every declared port wired; gated main swin verified; body single_model full seg_sr chain "
    "(6 edges, Pitfall-5 clean, step 0.5); total organs-subset emit-only (step 0.8); "
    "headneck_muscles C+M (2 main swin/post pairs + merge_remap + serialization + offsets 0/11); "
    "total_v3 5-part chain with spec-driven tables + emit base_name 'total'"
)
sys.exit(0)
