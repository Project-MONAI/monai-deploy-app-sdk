# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""totalsegmentator-fast application: GPU-resident nnU-Net TotalSegmentator pipeline.

P9 scope (live): the full 5-part chain — 1 PreprocessOperator + 5x
(SlideWindowOperator -> PostResampleOperator) + MergeRemapOperator +
the terminal 5-part emit op (SegEmitOperator, op-mode total) — over ONE
shared preprocessed
volume in the single existing Subgraph (plan 09-02, 09-RESEARCH §4 port
table). Phase 10: the 5-part chain also terminates in real DICOM SEG+SR
writers (oracle-exact, SEG/SR subfolders) alongside the numpy emit.
The single-part organs debug/triage path (HOLOSCAN_MODEL_PARTS=
organs, P8 E2E) is preserved byte-for-byte (decision D-9-2).

Bundle ownership (decision D-9-5): the app NO LONGER loads model weights
in setup — each SlideWindowOperator self-loads its part in setup (INF-008
house pattern, v1.1); all 5 weights stay resident for the process
lifetime (PIP-03 "load once"); per-part logits buffers are cleared after
each part's argmax (release_buffers) so 5 logit stacks never coexist.

The series-selection rule is the oracle CT rule ("Standard Axial CT
Series"), BYTE-IDENTICAL to ct-totalsegmentator-map/app_total/app.py.
"""

# INFR-01/D-14: RMM must be the FIRST import (importing rmm after holoscan
# raises ImportError: undefined symbol __cxa_call_terminate — live-reproduced
# 2026-08-19, see gpu_bootstrap.py docstring).
try:
    from my_app import gpu_bootstrap
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    import gpu_bootstrap

gpu_bootstrap.install_torch_allocator()

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

# torch before holoscan is fine — only rmm-after-holoscan trips the hazard.
import torch

# D-21 port (Phase 14-01): concurrent independent-fragment scheduler knob
# (v1.1-proven, cchmc-nnunet-fast D-21). DEFAULT OFF ("0") for the A/B;
# the default flip to ON is Plan 14-03. The scheduler is set on the app via
# self.scheduler(...) as the final statement of compose() — holoscan's
# app.run() takes no scheduler kwarg.
from holoscan.schedulers import EventBasedScheduler

from monai.deploy.conditions import CountCondition
from monai.deploy.core import Application, Operator, OperatorSpec
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.dicom_utils import EquipmentInfo, ModelInfo

try:  # package-style import (my_app.*)
    import importlib as _importlib

    from my_app.ai_segment_descriptions import (
        _algorithm_name,
        _algorithm_version,
        ai_segment_descriptions,
        volume_labels,
    )
    from my_app.config import (
        EXPECTED_MAX_LOCAL_LABEL,
        MODEL_PARTS,
        load_preprocess_params,
        resolve_active_parts,
        resolve_task_model_root,
    )
    from my_app.dicom_seg_writer_operator import DICOMSegmentationWriterOperator
    from my_app.dicom_text_sr_writer_operator import DICOMTextSRWriterOperator
    from my_app.operators import (
        DICOMSeriesSelectorOperator,
        MergeOperator,
        MergeRemapOperator,
        PostResampleOperator,
        PreprocessOperator,
        SegEmitOperator,
        SegVolumeMetricsOperator,
        SlideWindowOperator,
        VolumeSinkOperator,
    )
    from my_app.operators.cascade_ops import CascadePrepOp, CropMaskOp, CropOp, PasteOp
    from my_app.operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_on_gpu,
        get_study_id,
        nvtx_range,
    )
    from my_app.operators.merge_remap import (
        back_resample_merged,
        flip_sar_to_dhw,
        flip_sar_to_writer,
    )
    from my_app.operators.preprocess_operator import to_holoscan_gpu_tensor
    from my_app.operators.task_postprocess_operator import TaskPostprocessOperator
    from my_app.segmentation_metrics_operator import SegmentationMetricsOperator
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    import importlib as _importlib

    from ai_segment_descriptions import (
        _algorithm_name,
        _algorithm_version,
        ai_segment_descriptions,
        volume_labels,
    )
    from config import (
        EXPECTED_MAX_LOCAL_LABEL,
        MODEL_PARTS,
        load_preprocess_params,
        resolve_active_parts,
        resolve_task_model_root,
    )
    from dicom_seg_writer_operator import DICOMSegmentationWriterOperator
    from dicom_text_sr_writer_operator import DICOMTextSRWriterOperator
    from operators import (
        DICOMSeriesSelectorOperator,
        MergeOperator,
        MergeRemapOperator,
        PostResampleOperator,
        PreprocessOperator,
        SegEmitOperator,
        SegVolumeMetricsOperator,
        SlideWindowOperator,
        VolumeSinkOperator,
    )
    from operators.cascade_ops import CascadePrepOp, CropMaskOp, CropOp, PasteOp
    from operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_on_gpu,
        get_study_id,
        nvtx_range,
    )
    from operators.merge_remap import (
        back_resample_merged,
        flip_sar_to_dhw,
        flip_sar_to_writer,
    )
    from operators.preprocess_operator import to_holoscan_gpu_tensor
    from operators.task_postprocess_operator import TaskPostprocessOperator
    from segmentation_metrics_operator import SegmentationMetricsOperator

# P13 (13-02 Task 5, post-gate): the volume-metrics selection knob, read ONCE
# at module level. Default ON ("1") -> GPU single-pass SegVolumeMetricsOperator
# (NVTX `volume_metrics`, ~150 ms vs the legacy 2.75/62.6/96.3 s per-label
# loop; 3-pin gate PASS 3/3 at locked bars, p13_gate_report.json). Set
# HOLOSCAN_GPU_METRICS=0 to restore the legacy SegmentationMetricsOperator
# (knob fully intact; revert = one `git revert` of this line + comment).
_GPU_METRICS_ENABLED = os.environ.get("HOLOSCAN_GPU_METRICS", "1") == "1"

# P14 (14-01): the EventBasedScheduler worker count, read ONCE at module level.
# Default 5 (matches v1.1 D-21). Only used when HOLOSCAN_CONCURRENT_FRAGMENTS
# != "0"; the OFF path never constructs a scheduler.
_SCHED_WORKERS = int(os.environ.get("HOLOSCAN_SCHEDULER_WORKERS", "5"))

# Phase 17 (17-02): the task selection knob, read ONCE at module level
# (house pattern). Default = the 5-part TotalSegmentator chain.
# Unknown names fail fast in compose() BEFORE any GPU work (no silent
# fallback); the error names the valid task set (my_app.task_specs).
_TASK = os.environ.get("HOLOSCAN_TASK", "total")


# Total-aware RMM warm-pool target (plan 07-02, re-sized from v1.1's ~0.97 GB):
# largest single-part fp32 probability stack (26 heads x ~36 M voxels @1.5 mm
# x 4 B ~= 3.74 GB) + 5x ~120 MB weights (~0.63 GB) + shared preprocessed
# volume (~0.15 GB) ~= 4.5 GB. Do NOT warm to the all-5-held 16.85 GB figure
# (unreachable under the sequential single-part strategy).
# P10 (10-01): timed writer wrappers — ported VERBATIM from
# cchmc-nnunet-fast/my_app/app.py:149-185 (the v1.1 SC writer is out of
# Phase 10 scope). Base classes are the LOCAL my_app writer modules
# (byte-identical oracle copies), not the SDK paths.


def timed_writer_compute(operator, base_class, name, op_input, op_output, context):
    """Shared compute wrapper for the timed writer subclasses: NVTX range +
    a structured timing record (INFR-005/INFR-006) around the unmodified
    writer compute (subclassed, never edited)."""
    with nvtx_range(name):
        timing = GpuTiming(name)
        timing.start()
        try:
            return base_class.compute(operator, op_input, op_output, context)
        finally:
            record = timing.stop()
            # key the registry/collector by the shared fragment (the app),
            # not the individual operator instance
            fragment = getattr(operator, "fragment", operator)
            record["study"] = get_study_id(fragment)
            StudyTimingCollector.record(fragment, record)
            # the SEG writer does not define _logger; fall back to module logger
            logger = getattr(operator, "_logger", None) or logging.getLogger(f"timed_{type(operator).__name__}")
            logger.info("timing: %s", json.dumps(record))


class TimedDICOMSegmentationWriterOperator(DICOMSegmentationWriterOperator):
    """SEG writer (local my_app oracle copy) with an NVTX range + structured timing record."""

    def compute(self, op_input, op_output, context):
        return timed_writer_compute(self, DICOMSegmentationWriterOperator, "write_seg", op_input, op_output, context)


class TimedDICOMTextSRWriterOperator(DICOMTextSRWriterOperator):
    """SR writer (local my_app oracle copy) with an NVTX range + structured timing record."""

    def compute(self, op_input, op_output, context):
        return timed_writer_compute(self, DICOMTextSRWriterOperator, "write_sr", op_input, op_output, context)


class SegBackResampleFlipOperator(Operator):
    """Phase 19 (19-02): back-resample + writer flip for single-model seg_sr.

    The single_model chain has no MergeRemapOperator (that op exists only in
    the multi_part/crop_cascade branches), so the post-inference seg —
    PostResample's `seg_argmax_dicom` port: model-resolution, slidewindow-
    native SAR, CUDA uint8 — must be brought to ORIGINAL study spacing and
    the P10 writer flip BEFORE it can feed the SEG writer (pixel_array must
    match the source series geometry) or the metrics op (voxel volumes come
    from the native DICOM Image — model-space counts would be wrong).

    Reuses the MergeRemap pure helpers EXACTLY (back_resample_merged:
    order-0 nearest, skip-branch on pre_resample_skipped; flip_sar_to_writer:
    in-plane flip only) — the same two steps the multi_part branch applies
    in MergeRemapOperator.compute, here on the single-model seg.

    Fully generic: no task/part-name literals; the branch gate is spec-driven
    (output_mode == "seg_sr" and full-task-ness) in compose().

    Named Inputs:
        seg: uint8 3D CUDA tensor, model-resolution SAR (PostResample's
            `seg_argmax_dicom` port).
        preprocessed_meta: PreprocessOperator metadata dict (must carry
            `_pre_resample_spatial_shape` + `pre_resample_skipped`).
    Named Outputs:
        seg_image: uint8 3D CPU numpy at ORIGINAL spacing, flip(1,2) SAR ->
            writer order — the oracle-exact SEG-writer input (P10 contract).
        seg_orig: same map, SAR unflipped — the metrics-op input (native-
            spacing counts x native voxel volume).
        seg_dhw: same map, flip(0,1,2) — the dcm-decoded (z,y,x) layout;
            the emit npy that is the dcm<->npy parity pair (P10/p18 contract:
            what the SEG dcm encodes, byte-exact).

    The D2H of this port is a second deliberate one besides the merge's
    emit path (both are terminal-output hops; the merge's own D2H is
    unchanged — 17-01 A/B contract).
    """

    INPUT_SEG = "seg"
    INPUT_META = "preprocessed_meta"
    OUTPUT_IMAGE = "seg_image"
    OUTPUT_ORIG = "seg_orig"
    OUTPUT_DHW = "seg_dhw"

    def __init__(self, fragment, *args, **kwargs):
        # NOTE: holoscan 4.2's Operator.__init__ invokes self.setup(spec)
        # before this constructor body finishes — initialize all state first.
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input(self.INPUT_SEG)
        spec.input(self.INPUT_META)
        spec.output(self.OUTPUT_IMAGE)
        spec.output(self.OUTPUT_ORIG)
        spec.output(self.OUTPUT_DHW)

    def compute(self, op_input, op_output, context) -> None:
        with nvtx_range("back_resample"):
            t0 = time.time()
            seg_gpu = op_input.receive(self.INPUT_SEG)
            meta = op_input.receive(self.INPUT_META)
            if seg_gpu is None:
                raise ValueError("SegBackResampleFlipOperator received no 'seg' input.")
            if not meta or "_pre_resample_spatial_shape" not in meta:
                raise ValueError(
                    "SegBackResampleFlipOperator requires 'preprocessed_meta' with "
                    "'_pre_resample_spatial_shape' (pre-resample metadata)"
                )
            tensor = torch.utils.dlpack.from_dlpack(seg_gpu)
            assert_on_gpu(tensor)
            seg_model = tensor.detach().cpu().numpy().astype(np.uint8, copy=False)
            orig_ras = meta["_pre_resample_spatial_shape"]
            skipped = bool(meta.get("pre_resample_skipped", False))
            seg_orig, info = back_resample_merged(seg_model, orig_ras, skipped)
            seg_writer = flip_sar_to_writer(seg_orig)
            seg_dhw = flip_sar_to_dhw(seg_orig)
            op_output.emit(np.ascontiguousarray(seg_orig), self.OUTPUT_ORIG)
            op_output.emit(np.ascontiguousarray(seg_writer), self.OUTPUT_IMAGE)
            op_output.emit(np.ascontiguousarray(seg_dhw), self.OUTPUT_DHW)
            record = {
                "op": "back_resample",
                "study": get_study_id(self.fragment),
                "model_shape": list(seg_model.shape),
                "native_shape": list(seg_orig.shape),
                "skipped": info.get("skipped", skipped),
                "wall_time_s": round(time.time() - t0, 4),
            }
            self._logger.info("back_resample record: %s", json.dumps(record))


class MultiMainMergeOperator(Operator):
    """Phase 22 (22-02): spec-driven multi-main-part merge for crop_cascade
    C+M tasks (headneck_muscles = 6mm crop + part1 + part2 -> 23 unified
    labels).

    Implements the EXACT later-part-wins contract of MergeRemapOperator
    (merge_remap.py oracle anchor: only non-background voxels are offset by
    the part's label_offset; later parts overwrite earlier ones on overlap;
    background is NEVER shifted). The 5-part MergeRemapOperator cannot be
    reused here: its input ports are hard-coded to the total parts and the
    operators/ purity set is locked by the Phase 22 config-change-only
    audit — so the merge lives here as an app-level class, mirroring the
    SegBackResampleFlipOperator precedent (zero operator-file edits).

    All main parts infer on the SAME cropped task-resolution volume, so the
    merge is elementwise on the PostResampleOperator seg_argmax_dicom output
    (taskres SAR). The merged map is wired to PasteOp's seg_cropped port;
    PasteOp then does the order-0 back-resample to cropped native + canvas
    paste + P10 flips (unchanged).

    Named Inputs: seg_<part> per main part (uint8 3D CUDA, taskres SAR).
    Named Outputs:
        merged: uint8 3D CUDA, taskres SAR, unified labels (1..N).

    CountCondition(n_main_parts) — the Pitfall-5 fan-in guard: the merge
    never fires on a partial part set.
    """

    OUTPUT_MERGED = "merged"

    def __init__(
        self,
        fragment: Any,
        *args: Any,
        main_parts: Optional[Sequence[Tuple[str, int, Optional[int]]]] = None,
        **kwargs: Any,
    ):
        # holoscan 4.2: Operator.__init__ invokes setup() before this body
        # finishes — initialize all state first.
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._main_parts = list(main_parts or [])
        if not self._main_parts:
            raise ValueError(
                "MultiMainMergeOperator requires non-empty main_parts: "
                "[(name, label_offset, max_local_label), ...] in spec "
                "(later-part-wins) order."
            )
        super().__init__(fragment, CountCondition(fragment, len(self._main_parts)), *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        for name, _offset, _cap in self._main_parts:
            spec.input(f"seg_{name}")
        spec.output(self.OUTPUT_MERGED)

    def compute(self, op_input, op_output, context) -> None:
        with nvtx_range("merge_remap"):
            timing = GpuTiming("merge_remap")
            timing.start()

            combined = None
            per_part: Dict[str, Any] = {}
            for name, offset, cap in self._main_parts:
                port = f"seg_{name}"
                holo = op_input.receive(port)
                if holo is None:
                    raise ValueError(f"MultiMainMergeOperator received no {port!r} input.")
                tensor = torch.utils.dlpack.from_dlpack(holo)
                assert_on_gpu(tensor)
                seg = tensor.detach().cpu().numpy().astype(np.uint8, copy=False)
                if combined is None:
                    combined = np.zeros(seg.shape, dtype=np.uint8)
                if cap is not None and seg.size:
                    mx = int(seg.max())
                    if mx > cap:
                        raise ValueError(
                            f"MultiMainMergeOperator: part {name!r} local label "
                            f"{mx} exceeds its max_local_label {cap} "
                            f"(bundle/spec mismatch)."
                        )
                # Oracle later-part-wins merge (merge_remap.py contract):
                # only non-zero voxels are offset; background never shifts.
                mask = seg > 0
                combined[mask] = seg[mask] + offset
                per_part[name] = {
                    "offset": int(offset),
                    "max_local": int(seg.max()) if seg.size else 0,
                }

            # One deliberate D2H hop (terminal-output class, same as the P8
            # merge/emit hops); the emit is a CUDA tensor because PasteOp's
            # seg_cropped reception asserts on-device input.
            op_output.emit(
                to_holoscan_gpu_tensor(torch.as_tensor(np.ascontiguousarray(combined)).to(torch.device("cuda"))),
                self.OUTPUT_MERGED,
            )

            record = timing.stop()
            record["study"] = get_study_id(self.fragment)
            record["parts"] = per_part
            record["max_label"] = int(combined.max()) if combined.size else 0
            record["shape"] = list(combined.shape)
            StudyTimingCollector.record(self.fragment, record)
            self._logger.info("merge record: %s", json.dumps(record))


WARM_POOL_BYTES = 4 * 1024**3 + 512 * 1024**2  # 4.5 GiB = 4831838208 bytes


class TotalSegFastApp(Application):
    """GPU-resident nnU-Net TotalSegmentator 5-part task (117 classes).

    P9: the 5-part inference chain is live (one Subgraph over one shared
    preprocessed volume). P10 (10-01): the 5-part chain now terminates in
    real DICOM SEG+SR writers (oracle-exact) alongside the numpy emit.
    Model bundle ownership: each SlideWindowOperator self-loads its part in
    setup (INF-008); the app keeps NO bundle references (D-9-5).
    """

    def __init__(self, *args, **kwargs):
        """Creates an application instance."""
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__(*args, **kwargs)

    def _setup_models(self, model_path: Path) -> None:
        """Resolve the total model root (fail-fast) and log the active parts.

        P9 (plan 09-02, decision D-9-5) — NO weight loads here anymore.
        Bundle ownership: each SlideWindowOperator self-loads its part in
        setup (INF-008 house pattern, v1.1); weights stay resident for the
        process lifetime (PIP-03 "load once" — 5 x ~120 MB fits trivially);
        per-part logits buffers are cleared after each part's argmax
        (release_buffers) so 5 stacks never coexist. The duplicate
        app-level organs-bundle load is gone (carried item 4).
        """
        # Phase 18 (18-01): spec-driven model-root resolution (LIV-04) — the
        # root is resolved from the RESOLVED TaskSpec (task name + its part
        # names), not the hard-coded total layout. For `total` this returns
        # the identical path the old resolve_total_model_root returned for
        # every layout (regression-locked in test_liver_specs.py). Same 4
        # candidate layouts, same fail-fast naming every missing file.
        # 22-02: per-part config_name flows into the completeness check so
        # non-3d_fullres bundles (e.g. headneck_muscles' 3d_fullres_high parts)
        # resolve; `total`-family callers see the exact pre-22-02 check.
        model_root = resolve_task_model_root(
            model_path,
            _TASK,
            [p.name for p in self._task_spec.parts],
            part_configs={p.name: p.config_name for p in self._task_spec.parts},
        )
        self._logger.info(f"Resolved model root (task={_TASK}): {model_root}")
        self._model_root = model_root

        active_parts = resolve_active_parts()
        self._logger.info(f"Active parts: {[p['name'] for p in active_parts]}")
        self._logger.info(
            "Bundle ownership: each SlideWindowOperator self-loads its part "
            "in setup (INF-008); weights stay resident (PIP-03); per-part "
            "logit buffers are cleared after each part's argmax (release_buffers)."
        )

    def compose(self):
        """Creates the app-specific operators and chains them into the
        processing DAG."""
        logging.info(f"Begin {self.compose.__name__}")

        # Phase 17 (17-02): resolve the task spec FIRST — before the RMM
        # assert, _setup_models, or ANY GPU work. Unknown HOLOSCAN_TASK ->
        # ValueError (non-zero exit) naming the valid task set.
        try:  # package-style import (my_app.*)
            from my_app.task_specs import build_topology, get_task_spec, resolve_part
        except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
            from task_specs import build_topology, get_task_spec, resolve_part
        self._task_spec = get_task_spec(_TASK)
        self._logger.info(
            "task: %s (family=%s, parts=%s)", _TASK, self._task_spec.family, [p.name for p in self._task_spec.parts]
        )

        # Use command-line options over environment variables to init context
        app_context = Application.init_app_context(self.argv)
        app_input_path = Path(app_context.input_path)
        app_output_path = Path(app_context.output_path)
        model_path = Path(app_context.model_path)

        # --- RMM allocator check (INFR-01/D-14) ---
        # The gpu_bootstrap import at the top of this module installed RMM as
        # torch's CUDA allocator; if this fails, rmm was imported after
        # holoscan (the undefined-symbol hazard) or the bootstrap import was
        # lost.
        backend = torch.cuda.memory.get_allocator_backend()
        self._logger.info("memory_allocator_backend: %s", backend)
        assert backend == "pluggable", (
            f"RMM torch allocator not active (backend={backend!r}); "
            "gpu_bootstrap must be imported before holoscan (INFR-01)"
        )

        # --- Model loading (plan 07-02): 5x (or subset) load, once, in setup ---
        # resolve_task_model_root() covers all layouts (the task root itself,
        # MAP bakes with models/ or models/<task>/ subfolders, or a task-model
        # dir) and fails fast on any incomplete root (18-01).
        self._setup_models(model_path)

        # --- Total-aware RMM warm pool (plan 07-02): ~4.5 GiB, re-sized from
        # v1.1's per-bundle ~0.97 GB. The 4 GiB initial_pool_size pin in
        # gpu_bootstrap.py stays unchanged (it is a floor, not the warm
        # target). RMM warm-up is an app-level concern (it is NOT a weight
        # load); the 5 weights load per-operator in SlideWindowOperator.setup.
        gpu_bootstrap.warm_pool(WARM_POOL_BYTES)
        self._logger.info(f"RMM warm pool: {WARM_POOL_BYTES / 1024**3:.2f} GiB")

        # --- DICOM I/O (SDK, unchanged) ---
        study_loader_op = DICOMDataLoaderOperator(
            self, CountCondition(self, 1), input_folder=app_input_path, name="study_loader_op"
        )

        # custom DICOM Series Selector (copied from the v1.1 skeleton);
        # oracle CT rule, all_matched + SOP sorting: downstream runs on the
        # 1st selected series.
        series_selector_op = DICOMSeriesSelectorOperator(
            self,
            rules=Sample_Rules_Text,
            all_matched=True,
            sort_by_sop_instance_count=True,
            name="series_selector_op",
        )

        series_to_vol_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_op")

        # P7 terminal: gives the to-volume output a receiver (GXF rule)
        # until Phases 8-9 wire the TotalSegSubgraph.
        volume_sink_op = VolumeSinkOperator(self, name="volume_sink")

        # --- Inference chain (Phase 17: spec-driven builder; plan 09-02,
        # 09-RESEARCH §4 port-table wiring preserved verbatim) ---
        # Topology dispatch is driven by the task spec (plan.family), never
        # by task/part names:
        #   * single_model  — one-part debug/triage wiring (P8 verbatim:
        #     plain span names, no part= kwarg, byte-identical behavior);
        #   * multi_part    — the serialized all-parts chain (order from the
        #     spec, FIXED oracle order) with per-part NVTX/timing tags, bundle
        #     self-load (D-9-5), per-part release_buffers, and the
        #     MergeRemapOperator + total-mode terminal emit;
        #   * crop_cascade  — the TS 2.18-faithful two-stage liver cascade
        #     (Phase 18, 18-02): cascade prep x2 + crop-mask + crop +
        #     gated main SlidingWindow + PasteOp + P10 output chain.
        # The 1->N fan-out of preprocessed/preprocessed_meta is free GXF
        # behavior (one emitted message reaches all subscribers).
        if _TASK == "total":
            # The HOLOSCAN_MODEL_PARTS debug knob is a total-task-only
            # mechanism; total keeps its pre-Phase-18 behavior byte-for-byte.
            active_parts = resolve_active_parts()
            active_names = [p["name"] for p in active_parts]
        else:
            # crop_cascade (liver_segments): the spec's FULL part list is the
            # active set (parts[0] = crop model, parts[1] = main model — the
            # registry ordering convention, 18-01). No env knob.
            # (Grep gates: config names only via spec/config_name; no quoted
            # config-name literals introduced by this branch.)
            active_names = [p.name for p in self._task_spec.parts]
        plan = build_topology(self._task_spec, active_names)
        self._logger.info(
            "topology: family=%s parts=%s serialized=%s",
            plan.family,
            list(plan.parts),
            plan.serialized,
        )
        if plan.family == "single_model":
            # Single-part debug/triage path — P8 wiring verbatim, with the
            # part resolved from the spec by its selected name (incl.
            # seg_argmax_dicom to emit; P8 plain span names, NO part= kwarg
            # on pre/swin/post). The selected part is looked up by name —
            # never the spec's first part — so a non-first single-part
            # selection wires its OWN model.
            # Phase 19 (19-01): register spec parts into the SAME module-level
            # lists operators validate against (cascade precedent, 18-02).
            # No-op for parts already present (total parts / debug subset);
            # required for single-model tasks whose part name is new to
            # MODEL_PARTS / EXPECTED_MAX_LOCAL_LABEL.
            _known_parts = {p["name"] for p in MODEL_PARTS}
            for _p in self._task_spec.parts:
                if _p.name not in _known_parts:
                    MODEL_PARTS.append({"name": _p.name, "label_offset": _p.label_offset})
                if _p.max_local_label is not None:
                    EXPECTED_MAX_LOCAL_LABEL[_p.name] = _p.max_local_label
            part = resolve_part(self._task_spec, plan.parts[0])
            part_path = self._model_root / part.name
            pre_op = PreprocessOperator(
                self,
                model_path=part_path,
                config_name=part.config_name,
                # spec-driven input pre-resample order (19-03): body=1 (TS CLI),
                # total default 3 (byte-locked)
                input_resample_order=self._task_spec.resample_order,
                name=f"preprocess_{part.name}",
            )
            swin_op = SlideWindowOperator(
                self,
                model_path=part_path,  # self-loads in setup (INF-008 pattern)
                config_name=part.config_name,
                use_mirroring=None,
                # D-04 (26): plan-driven — configurations.<cfg>.use_mirroring
                # (default False; no shipped plans set it, so behavior-neutral)
                use_gaussian=True,
                # spec-driven (Phase 19, 19-01): TS nnunet.py:568-572 uses 0.8
                # only for the total family; default field value keeps total
                # byte/behavior-identical
                tile_step_size=self._task_spec.tile_step_size,
                name=f"swin_{part.name}",
            )
            # P9 (09-03): per-part postprocess gate wired at the oracle's
            # exact placement (per-part, model-space, PRE-merge). The
            # bundle ships 0 postprocessing.pkl, so the live path is a
            # true identity no-op (zero-copy) with one log line;
            # model_path enables the setup-time pkl check + record line.
            post_op = PostResampleOperator(
                self,
                config_name=part.config_name,
                emit_argmax_seg=True,
                emit_probabilities=False,
                model_path=part_path,  # 09-03: oracle-parity per-part postprocess gate
                # 26-04: plans-semantic tasks (resample=None, e.g. 3d_lowres / cascade)
                # revert through the plans probability resample so seg_argmax_dicom
                # is crop-native, matching the 5-part path (22-11 audit S4).
                resample_seg_to_original=(self._task_spec.resample is None),
                name=f"postresample_{part.name}",
            )
            merge_op = MergeOperator(self, part=part.name, name=f"merge_{part.name}")
            emit_op = SegEmitOperator(self, part=part.name, output_dir=app_output_path, name=f"emit_{part.name}")

            # P7 fan-out (volume_sink) stays wired; the inference chain branches off to-volume.
            self.add_flow(series_to_vol_op, pre_op, {("image", "image")})
            self.add_flow(pre_op, swin_op, {("preprocessed", "preprocessed")})
            self.add_flow(swin_op, post_op, {("logits", "logits")})
            self.add_flow(pre_op, post_op, {("preprocessed_meta", "preprocessed_meta")})
            self.add_flow(pre_op, emit_op, {("preprocessed_meta", "preprocessed_meta")})
            self.add_flow(post_op, merge_op, {("seg_argmax", f"seg_{part.name}")})
            # P8 base emit flows — used by the debug / emit-only path (non-
            # seg_sr or a part SUBSET). The seg_sr full-task chain below
            # re-wires emit's seg ports instead (19-02 Rule-1): the emit
            # numpy pair must sit at ORIGINAL spacing — the dcm<->npy parity
            # pair (p10/p18 contract) — which the model-resolution
            # seg_argmax_dicom cannot provide. In that case merge's
            # seg_merged port legitimately has no receiver: the merge op
            # stays in the DAG for the merge_<part> span contract, its
            # downstream array is superseded by backresample's outputs.
            _seg_sr_full = self._task_spec.output_mode == "seg_sr" and set(active_names) == {
                p.name for p in self._task_spec.parts
            }
            if not _seg_sr_full:
                self.add_flow(post_op, emit_op, {("seg_argmax_dicom", "seg_merged_dicom")})
                self.add_flow(merge_op, emit_op, {("seg_merged", "seg_merged")})

            # Phase 19 (19-01): generic P10 output chain (DICOM SEG + SR) for
            # full single-model tasks with output_mode "seg_sr" — mirroring the
            # multi_part/crop_cascade chains, spec-driven only (no task or
            # part-name literals). _full_task requires the active part set to
            # equal the spec's full part set, so the total debug subset (the
            # HOLOSCAN_MODEL_PARTS knob selecting 1 of 5 parts) stays
            # emit-only — the 17-01 A/B contract. 19-02 Rule-1 fix: the
            # single-model seg (post_op's seg_argmax_dicom — model-resolution
            # SAR, CUDA) is back-resampled to ORIGINAL spacing (order-0,
            # MergeRemap's exact helpers) by SegBackResampleFlipOperator,
            # which feeds the metrics op (SAR — voxel volume comes from the
            # native Image) and the SEG writer (flip(1,2) only, the P10
            # oracle-exact writer contract). Pitfall-5: every new port has a
            # receiver; the existing single-mode emit flows are unchanged.
            _full_task = set(active_names) == {p.name for p in self._task_spec.parts}
            if _seg_sr_full:
                # 19-02 Rule-1 fix: back-resample the single-model seg to
                # ORIGINAL study spacing + P10 writer flip. Model-resolution
                # seg_argmax_dicom cannot feed the SEG writer (pixel_array
                # must match source series geometry) and its voxel counts
                # would be wrong for metrics.
                backresample_op = SegBackResampleFlipOperator(self, name=f"backresample_{part.name}")
                # 19-05: TS-CLI task postprocessing (model resolution,
                # pre-back-resample) — data-driven via TaskSpec.postprocess;
                # empty for total/liver_segments so their wiring stays
                # byte-identical. TS nnunet.py:686-695 (body: keep-largest
                # body_trunc + remove_small_blobs body_extremities 50,000 mm3).
                if self._task_spec.postprocess:
                    taskpp_op = TaskPostprocessOperator(
                        self,
                        rules=self._task_spec.postprocess,
                        label_map=dict(self._task_spec.labels),
                        model_spacing=self._task_spec.target_spacing,
                        name=f"taskpp_{part.name}",
                    )
                    self.add_flow(post_op, taskpp_op, {("seg_argmax_dicom", "seg")})
                    self.add_flow(taskpp_op, backresample_op, {("seg", "seg")})
                else:
                    self.add_flow(post_op, backresample_op, {("seg_argmax_dicom", "seg")})
                self.add_flow(pre_op, backresample_op, {("preprocessed_meta", "preprocessed_meta")})
                # Re-wire emit's seg ports (supersedes the P8 base flows above):
                # seg_merged <- SAR original-spacing map; seg_merged_dicom <-
                # the exact array the SEG dcm encodes (flip(0,1,2) = the p9
                # dcm-decode layout), so seg_<part>_dicom.npy is the parity pair.
                self.add_flow(backresample_op, emit_op, {("seg_orig", "seg_merged")})
                self.add_flow(backresample_op, emit_op, {("seg_dhw", "seg_merged_dicom")})
                try:
                    seg_desc_mod = _importlib.import_module(f"my_app.{self._task_spec.snomed_module}")
                except ImportError:  # flat import (my_app dir on sys.path)
                    seg_desc_mod = _importlib.import_module(self._task_spec.snomed_module)
                seg_descriptions = getattr(seg_desc_mod, self._task_spec.snomed)
                _algorithm = seg_descriptions[0]._algorithm_identification
                my_model_info = ModelInfo(
                    creator="TotalSegmentator",
                    name=_algorithm.name,
                    version=_algorithm.version,
                    uid="0.1.0",
                )
                my_equipment_info = EquipmentInfo(
                    manufacturer="The MONAI Consortium",
                    manufacturer_model="MONAI Deploy App SDK",
                    software_version_number="3.5.0",
                )
                custom_tags_seg = {
                    "SeriesDescription": "AI Generated DICOM SEG; Not for Clinical Use.",
                    "AlgorithmName": f"{my_model_info.name}:{my_model_info.version}:{my_model_info.uid}",
                }
                custom_tags_sr = {
                    "SeriesDescription": "AI Generated DICOM SR; Not for Clinical Use.",
                    "AlgorithmName": f"{my_model_info.name}:{my_model_info.version}:{my_model_info.uid}",
                }
                _metrics_labels = {k: v for k, v in self._task_spec.labels.items() if k != "background"}
                if _GPU_METRICS_ENABLED:
                    seg_metrics_op = SegVolumeMetricsOperator(
                        self,
                        name="seg_metrics_op",
                        labels_dict=_metrics_labels,
                    )
                else:
                    seg_metrics_op = SegmentationMetricsOperator(
                        self,
                        name="seg_metrics_op",
                        use_gpu=True,
                        labels_dict=_metrics_labels,
                    )
                dicom_seg_writer = TimedDICOMSegmentationWriterOperator(
                    self,
                    segment_descriptions=seg_descriptions,
                    model_info=my_model_info,
                    custom_tags=custom_tags_seg,
                    output_folder=app_output_path / "SEG",
                    omit_empty_frames=False,
                    name="dicom_seg_writer",
                )
                dicom_sr_writer = TimedDICOMTextSRWriterOperator(
                    self,
                    copy_tags=True,
                    model_info=my_model_info,
                    equipment_info=my_equipment_info,
                    custom_tags=custom_tags_sr,
                    included_fields=["vol"],
                    report_code_value="25045-6",
                    report_coding_scheme_designator="LN",
                    report_code_meaning="CT Report",
                    output_folder=app_output_path / "SR",
                    name="dicom_sr_writer",
                )

                self.add_flow(series_to_vol_op, seg_metrics_op, {("image", "input_scan")})
                # Metrics op gets the back-resampled SAR map (original spacing —
                # voxel volume comes from the native Image, flip-invariant).
                self.add_flow(backresample_op, seg_metrics_op, {("seg_orig", "segmentation_mask")})
                self.add_flow(seg_metrics_op, dicom_sr_writer, {("metrics_dict", "dict")})
                self.add_flow(
                    series_selector_op, dicom_sr_writer, {("study_selected_series_list", "study_selected_series_list")}
                )
                self.add_flow(
                    series_selector_op, dicom_seg_writer, {("study_selected_series_list", "study_selected_series_list")}
                )
                # Writer input = flip(1,2) only (P10 oracle-exact writer contract).
                self.add_flow(backresample_op, dicom_seg_writer, {("seg_image", "seg_image")})
        elif plan.family == "multi_part":
            # Serialized all-parts chain (09-RESEARCH §4), driven by the
            # spec's part list. Order is FIXED (the spec's oracle order,
            # later-part-wins). Preprocess runs ONCE (model_path = first
            # part's bundle); the preprocessed volume fans out to all
            # SlideWindowOperators.
            #
            # P9 OOM FIX: the chain is serialized via DAG edges, not just
            # naming. Parts 2..N are constructed gated=True and each gets an
            # ordering edge post_{prev} -> swin_{part} carrying the previous
            # part's seg_argmax as a sync gate (consumed, values unused).
            # The pre-fan-out alone no longer determines start order: each
            # part begins only after the previous part's postresample
            # (including release_buffers()) has run, so per-part logit
            # stacks (~4.4-4.7 GiB) are reused sequentially instead of all
            # accumulating concurrently (31322 RMM pool OOM). These N-1
            # edges are now emitted for EVERY multi-part spec, not just the
            # 5-part task. The release_fn wiring below is unchanged.
            #
            # P9 (09-04) per-part normalization fix: the shared preprocess
            # emits the resampled+cropped UNNORMALIZED volume
            # (normalize=False); each SlideWindowOperator normalizes with
            # ITS OWN part's CT properties (normalize=True) — oracle parity,
            # where the shared ResampleToModelSpacingd never normalizes and
            # each part's run_case_npy applies its own plans' CT properties.
            # (The old shared first-part normalization — first part's clip
            # [-1024,276] — corrupted the other parts' inputs, e.g. all
            # bone clipped to a single constant for the vertebrae model.)
            parts = self._task_spec.parts
            pre_op = PreprocessOperator(
                self,
                model_path=self._model_root / parts[0].name,
                config_name=parts[0].config_name,
                name="preprocess",
                normalize=False,
                # spec-driven input pre-resample order (19-03): total=3
                # (byte-locked oracle default — identical to the previous
                # hard-coded 3; no behavior change for total)
                input_resample_order=self._task_spec.resample_order,
            )
            swin_ops, post_ops = {}, {}
            for i, part in enumerate(parts):
                name = part.name
                swin_ops[name] = SlideWindowOperator(
                    self,
                    model_path=self._model_root / name,  # self-loads in setup (INF-008, D-9-5)
                    config_name=part.config_name,
                    use_mirroring=None,
                    # D-04 (26): plan-driven — configurations.<cfg>.use_mirroring
                    # (default False; no shipped plans set it, so behavior-neutral)
                    use_gaussian=True,
                    tile_step_size=0.8,  # oracle CT tile_step_size
                    part=name,  # per-part NVTX/timing tag
                    normalize=True,  # 09-04: per-part CT normalization (own plans' properties)
                    gated=(i > 0),  # P9 OOM fix: parts 2..N wait for prev part's postresample
                    name=f"swin_{name}",
                )
                post_ops[name] = PostResampleOperator(
                    self,
                    config_name=part.config_name,
                    emit_argmax_seg=True,
                    emit_probabilities=False,
                    emit_argmax_dicom=False,  # port discipline: only seg_argmax is wired
                    part=name,  # per-part NVTX/timing tag
                    release_fn=swin_ops[name].release_buffers,  # per-part cache clear
                    model_path=self._model_root / name,  # 09-03: per-part postprocess gate
                    name=f"postresample_{name}",
                )
            merge_op = MergeRemapOperator(self, output_dir=app_output_path, name="merge_5part")
            emit_op = SegEmitOperator(self, mode="total", output_dir=app_output_path, name="emit_5part")

            self.add_flow(series_to_vol_op, pre_op, {("image", "image")})
            for part in parts:
                name = part.name
                self.add_flow(pre_op, swin_ops[name], {("preprocessed", "preprocessed")})
                self.add_flow(swin_ops[name], post_ops[name], {("logits", "logits")})
                self.add_flow(pre_op, post_ops[name], {("preprocessed_meta", "preprocessed_meta")})
                self.add_flow(post_ops[name], merge_op, {("seg_argmax", f"seg_{name}")})
            # P9 OOM fix: serialized-chain ordering edges — N-1 of them;
            # part i starts only after part i-1's postresample has emitted
            # its seg_argmax (post-emit work incl. release_buffers() follows
            # immediately in the same compute). seg_argmax is fan-out to
            # merge + the gate; all other flows above are unchanged.
            for i in range(1, len(parts)):
                self.add_flow(
                    post_ops[parts[i - 1].name],
                    swin_ops[parts[i].name],
                    {("seg_argmax", "prev_part_seg")},
                )
            self.add_flow(pre_op, merge_op, {("preprocessed_meta", "preprocessed_meta")})
            self.add_flow(pre_op, emit_op, {("preprocessed_meta", "preprocessed_meta")})
            self.add_flow(merge_op, emit_op, {("seg_merged", "seg_merged")})
            self.add_flow(
                merge_op, emit_op, {("seg_dhw", "seg_image")}
            )  # emit's port is named seg_image; carries spatial DHW (flip 0,1,2)

            # --- P10 output chain (SEG + SR) — 10-RESEARCH §Port table +
            # §Pinned Oracle Semantics (oracle-exact args, no substitutions).
            # Flip contract (10-RESEARCH §5, P10 root-cause fix): metrics gets
            # the pre-flip SAR seg_merged (vol = pixel_count x voxel_volume is
            # flip-invariant). The SEG writer gets seg_image = flip(1,2) ONLY
            # (oracle-exact nnunet_seg_operator:633 contract); highdicom's
            # _prepare_spatial_metadata sorts planes superior-first, D-flipping
            # the input, so the on-disk SEG decodes to the spatial DHW
            # (flip 0,1,2) = the emit's seg_total_dhw.npy / seg_dhw port.
            # Pitfall-5 audit: every required port of the 3 new operators is
            # wired below; the SR writer's study_selected_series_list port is
            # declared ConditionType.NONE but effectively mandatory under
            # copy_tags=True (raises at runtime without it) — hence wired.
            # 22-02 (Q4.3): spec-driven table selection. Non-empty
            # spec.snomed_module -> importlib the module (same pattern as the
            # single_model branch) and read its description table + labels;
            # empty -> the legacy globals. For `total` the spec points at the
            # same module + table (snomed_module="ai_segment_descriptions",
            # labels="volume_labels"), so this resolves to the byte-identical
            # values (p10 re-lock proves it — 22-03 gate).
            if self._task_spec.snomed_module:
                try:
                    seg_desc_mod = _importlib.import_module(f"my_app.{self._task_spec.snomed_module}")
                except ImportError:  # flat import (my_app dir on sys.path)
                    seg_desc_mod = _importlib.import_module(self._task_spec.snomed_module)
                seg_descriptions = getattr(seg_desc_mod, self._task_spec.snomed)
                _desc_alg = seg_descriptions[0]._algorithm_identification
                my_model_info = ModelInfo(
                    creator="TotalSegmentator",
                    name=_desc_alg.name,
                    version=_desc_alg.version,
                    uid="0.1.0",
                )
                _labels_table = self._task_spec.labels
                if isinstance(_labels_table, str):
                    # str form = attribute name on the descriptions module
                    # (the `total` convention — "volume_labels").
                    _labels_table = getattr(seg_desc_mod, _labels_table)
                _metrics_labels = {k: v for k, v in _labels_table.items() if k != "background"}
            else:
                seg_descriptions = ai_segment_descriptions
                my_model_info = ModelInfo(
                    creator="TotalSegmentator",
                    name=_algorithm_name,
                    version=_algorithm_version,
                    uid="0.1.0",
                )
                _metrics_labels = {k: v for k, v in volume_labels.items() if k != "background"}
            my_equipment_info = EquipmentInfo(
                manufacturer="The MONAI Consortium",
                manufacturer_model="MONAI Deploy App SDK",
                software_version_number="3.5.0",
            )
            custom_tags_seg = {
                "SeriesDescription": "AI Generated DICOM SEG; Not for Clinical Use.",
                "AlgorithmName": f"{my_model_info.name}:{my_model_info.version}:{my_model_info.uid}",
            }
            custom_tags_sr = {
                "SeriesDescription": "AI Generated DICOM SR; Not for Clinical Use.",
                "AlgorithmName": f"{my_model_info.name}:{my_model_info.version}:{my_model_info.uid}",
            }

            # 22-02: _metrics_labels is the spec-driven value from the branch above
            # (snomed_module set -> module/registry table; empty -> global
            # volume_labels). Removed the unconditional global overwrite that
            # clobbered spec-driven tables for tasks like total_v3.
            if _GPU_METRICS_ENABLED:
                # 13-02: GPU single-pass volume-metrics op (NVTX `volume_metrics`,
                # same input/output contract as the legacy op -> DAG shape, edge
                # names, flow names, and the `metrics_dict` output key are all
                # unchanged; only the op class swaps).
                seg_metrics_op = SegVolumeMetricsOperator(
                    self,
                    name="seg_metrics_op",
                    labels_dict=_metrics_labels,
                )
            else:
                seg_metrics_op = SegmentationMetricsOperator(
                    self,
                    name="seg_metrics_op",
                    use_gpu=True,
                    labels_dict=_metrics_labels,
                )
            dicom_seg_writer = TimedDICOMSegmentationWriterOperator(
                self,
                segment_descriptions=seg_descriptions,  # spec-driven (22-02); total resolves to the full 117-entry list
                model_info=my_model_info,
                custom_tags=custom_tags_seg,
                output_folder=app_output_path / "SEG",
                omit_empty_frames=False,  # keep 117 x D frames (baseline: 4914 on 44238)
                name="dicom_seg_writer",
            )
            dicom_sr_writer = TimedDICOMTextSRWriterOperator(
                self,
                copy_tags=True,  # same Study UID as source; REQUIRES study_selected_series_list
                model_info=my_model_info,
                equipment_info=my_equipment_info,
                custom_tags=custom_tags_sr,
                included_fields=["vol"],
                report_code_value="25045-6",
                report_coding_scheme_designator="LN",
                report_code_meaning="CT Report",
                output_folder=app_output_path / "SR",
                name="dicom_sr_writer",
            )

            self.add_flow(series_to_vol_op, seg_metrics_op, {("image", "input_scan")})
            self.add_flow(merge_op, seg_metrics_op, {("seg_merged", "segmentation_mask")})
            self.add_flow(seg_metrics_op, dicom_sr_writer, {("metrics_dict", "dict")})
            self.add_flow(
                series_selector_op, dicom_sr_writer, {("study_selected_series_list", "study_selected_series_list")}
            )
            self.add_flow(
                series_selector_op, dicom_seg_writer, {("study_selected_series_list", "study_selected_series_list")}
            )
            self.add_flow(
                merge_op, dicom_seg_writer, {("seg_image", "seg_image")}
            )  # writer input = flip(1,2) only (P10)
        else:
            # crop_cascade (liver_segments) — Phase 18 (18-02): the 8-stage
            # TS 2.18-faithful cascade (plan.crop_stages), materialized with
            # the Phase-17 operators (SlideWindow/PostResample reuse) + the
            # four cascade_ops (prep x2 / crop-mask / crop / paste). Everything
            # below is driven by self._task_spec — zero task-name literals
            # (grep gates: config names only via PartSpec.config_name; part
            # names only via resolve_part/spec.parts). 18-RESEARCH §Critical Correction:
            # NATIVE-resolution crop — 6mm labelmap back-resampled to native
            # (order 0) before the mask is built; addon mm->voxels TRUNCATED
            # at native zooms; task-res order-1 resample runs AFTER the crop.
            #
            # EMPTY-MASK PATH (LIV-03, TS nnunet.py:493 — flag-gated skip,
            # REQUIRED contract): crop_mask detects the empty mask and fans
            # an all-zero native canvas into paste's seg_cropped port; the
            # main chain (cascade_prep_taskres / swin / postresample of
            # ct_liver_segments) NEVER fires because its required inputs
            # never arrive (cascade_prep_taskres emits nothing on empty; the
            # gated swin never gets `preprocessed`). Spans crop_prep_taskres,
            # inference_3d_fullres_ct_liver_segments and
            # postresample_3d_fullres_ct_liver_segments are absent — exactly
            # the p18_run_study.sh --empty 10-present/3-absent contract.
            #
            # Span contract (p18_run_study.sh, 18-01): cascade_prep_6mm,
            # inference_3d_fullres_<part>, postresample_3d_fullres_<part>,
            # crop_mask, crop_to_mask, crop_prep_taskres, paste, emit_5part.
            parts = self._task_spec.parts
            crop_part = parts[0]  # registry convention: parts[0] = crop model
            main_parts = [resolve_part(self._task_spec, n) for n in plan.parts[1:]]
            main_part = main_parts[0]  # single-main (liver_segments) keeps this exact path
            multi_main = len(main_parts) > 1  # 22-02: C+M (crop + N main parts)
            crop_path = self._model_root / crop_part.name
            main_paths = {p.name: self._model_root / p.name for p in main_parts}
            main_path = main_paths[main_part.name]
            crop_addon = list(self._task_spec.crop.addon) if self._task_spec.crop else [20, 20, 20]
            # spec.resample is the TS config tuple in (x, y, z) order — the
            # taskres 3D volume (CropOp output) is in the SAME (x, y, z)
            # world order (and the native in-spacing is (x,y,z)), so pass
            # it straight through. (18-03 fix: 18-02 permuted it to (z,x,y)
            # for a 4D post-transpose resample the 3D-only resampler never
            # received — mixed axis orders produced a wrong main-model input
            # volume, e.g. 64199 (130,196,201) vs TS (242,196,108).)
            res = self._task_spec.resample
            # 22-11 (audit S2 root cause): `resample is None` tasks now run
            # the PLANS-SEMANTIC path — TS 2.18 passes the native crop
            # through unresampled and the nnUNet preprocessor does
            # f32 -> crop_to_nonzero -> CTNormalization (BEFORE resample)
            # -> resampling_fn_data (plans order 3 / order_z 0 / sep-z).
            # The pre-fix 22-04 fallback (world order-1 int32 resample +
            # in-swin normalize-after) is superseded by
            # CascadePrepOp(plans_semantic=True) + swin normalize=False +
            # postresample resample_seg_to_original=True below. The
            # plans-spacing target below is still computed and passed
            # (harmless — the plans-semantic branch derives its own).
            if res is None:
                # 22-04 decision A (superseded by 22-11 for the input side;
                # plans.json spacing is original (D,H,W) array order while
                # target_dhw is (x,y,z) world order, so reverse).
                _main_pp = load_preprocess_params(main_path, main_part.config_name)
                _plans_sp = _main_pp.spacing
                res = (_plans_sp[2], _plans_sp[1], _plans_sp[0])
            elif isinstance(res, (int, float)):
                res = (res, res, res)  # scalar TS resample = isotropic
            task_res_dhw = tuple(float(s) for s in res)
            plans_semantic = self._task_spec.resample is None
            crop_target_dhw = (float(self._task_spec.target_spacing),) * 3

            # Part-name validation in SlideWindowOperator/PostResampleOperator
            # runs against config.MODEL_PARTS (total's 5 parts). Register the
            # cascade part names into that SAME list object at runtime so the
            # per-part NVTX/timing tags (inference_3d_fullres_<part> etc.) work
            # unmodified — total-family behavior is untouched (its parts are
            # already present; this block never runs for total).
            _known_parts = {p["name"] for p in MODEL_PARTS}
            for _p in parts:
                if _p.name not in _known_parts:
                    MODEL_PARTS.append({"name": _p.name, "label_offset": _p.label_offset})
            # 22-02: C+M main parts also register max local labels (mirrors the
            # single_model registration). Gated on multi_main so the
            # single-main (liver_segments) module state stays byte-identical.
            if multi_main:
                for _p in main_parts:
                    if _p.max_local_label is not None:
                        EXPECTED_MAX_LOCAL_LABEL[_p.name] = _p.max_local_label

            # --- Stage 1+2: downsample + crop_infer (6mm crop model) ---
            pre6_op = CascadePrepOp(
                self,
                model_path=crop_path,
                config_name=crop_part.config_name,
                target_spacing_dhw=crop_target_dhw,
                name="cascade_prep_6mm",
            )
            swin6_op = SlideWindowOperator(
                self,
                model_path=crop_path,  # self-loads in setup (INF-008, D-9-5)
                config_name=crop_part.config_name,
                use_mirroring=None,
                # D-04 (26): plan-driven — configurations.<cfg>.use_mirroring
                # (default False; no shipped plans set it, so behavior-neutral)
                use_gaussian=True,
                tile_step_size=0.8,  # oracle CT tile_step_size
                part=crop_part.name,  # NVTX/timing: inference_3d_fullres_total_6mm
                normalize=True,  # 09-04 pattern: per-part CT normalization
                name=f"swin_{crop_part.name}",
            )
            post6_op = PostResampleOperator(
                self,
                config_name=crop_part.config_name,
                emit_argmax_seg=True,
                emit_probabilities=False,
                emit_argmax_dicom=True,  # CropMaskOp consumes the native-order 6mm labelmap
                part=crop_part.name,  # NVTX/timing: postresample_3d_fullres_total_6mm
                release_fn=swin6_op.release_buffers,  # 6mm logits freed before main inference
                model_path=crop_path,
                name=f"postresample_{crop_part.name}",
            )
            # --- Stages 3+4: crop_mask + crop (native resolution, TS cropping.py) ---
            crop_mask_op = CropMaskOp(
                self,
                dataset_json=crop_path / "jsonpkls" / "dataset.json",  # crop label ids resolved at setup
                output_dir=app_output_path,
                crop_labels=list(self._task_spec.crop.labels),  # spec-driven (e.g. liver vs lung lobes)
                name="crop_mask",
            )
            crop_op = CropOp(
                self,
                addon_mm=crop_addon,
                output_dir=app_output_path,
                name="crop_to_mask",
            )
            # --- Stages 5+6: task-res prep + main_infer (gated) + back_resample ---
            pre_task_op = CascadePrepOp(
                self,
                model_path=main_path,
                config_name=main_part.config_name,
                target_spacing_dhw=task_res_dhw,
                task_res=True,
                plans_semantic=plans_semantic,
                output_dir=app_output_path,
                name="cascade_prep_taskres",
            )
            if multi_main:
                # 22-02: C+M (crop + N main parts, e.g. headneck_muscles
                # part1/part2). One (swin+post) pair per main part, serialized
                # in spec order with the P9 OOM pattern: part 1 gates on the
                # 6mm postresample; each later part gates on the previous
                # part's postresample — per-part logit stacks never coexist.
                # Fully spec-driven: per-part config_name, tile_step_size
                # (TS nnunet.py:577 — 0.5 for non-total main models), and the
                # (name, offset, max_local) tuples feeding the merge op
                # (later-part-wins, the MergeRemapOperator contract).
                main_swin_ops, main_post_ops = {}, {}
                for _mp in main_parts:
                    main_swin_ops[_mp.name] = SlideWindowOperator(
                        self,
                        model_path=main_paths[_mp.name],  # self-loads in setup (INF-008)
                        config_name=_mp.config_name,
                        use_mirroring=None,
                        # D-04 (26): plan-driven — configurations.<cfg>.use_mirroring
                        # (default False; no shipped plans set it, so behavior-neutral)
                        use_gaussian=True,
                        tile_step_size=self._task_spec.tile_step_size,  # 0.5 for non-total main models
                        part=_mp.name,  # NVTX/timing: inference_<config>_<part>
                        normalize=not plans_semantic,  # 09-04: per-part CT normalization (22-11: pre-normalized input skips it)
                        gated=True,
                        name=f"swin_{_mp.name}",
                    )
                    main_post_ops[_mp.name] = PostResampleOperator(
                        self,
                        config_name=_mp.config_name,
                        emit_argmax_seg=True,
                        emit_probabilities=False,
                        emit_argmax_dicom=True,  # merge consumes (same crop grid/orientation)
                        resample_seg_to_original=plans_semantic,  # 22-11: crop-native seg emit
                        part=_mp.name,
                        release_fn=main_swin_ops[_mp.name].release_buffers,
                        model_path=main_paths[_mp.name],
                        name=f"postresample_{_mp.name}",
                    )
                merge_remap_op = MultiMainMergeOperator(
                    self,
                    main_parts=[(p.name, p.label_offset, p.max_local_label) for p in main_parts],
                    name="merge_remap",
                )
            else:
                swin_main_op = SlideWindowOperator(
                    self,
                    model_path=main_path,
                    config_name=main_part.config_name,
                    use_mirroring=None,
                    # D-04 (26): plan-driven — configurations.<cfg>.use_mirroring
                    # (default False; no shipped plans set it, so behavior-neutral)
                    use_gaussian=True,
                    # 18-03 fix (Rule 1): the 6mm CROP model is task_name="total"
                    # in TS, which uses tile_step_size 0.8 (nnunet.py:575 —
                    # "fewer overlapping tiles for every resolution variant of
                    # the total model"). The MAIN model is task_name=
                    # "liver_segments" -> step_size 0.5 (nnunet.py:577). The
                    # 18-02 copy used 0.8 for both; the 0.04%/0.65% final-stage
                    # deltas vs TS were seam-blending wobble from the wrong
                    # main-model overlap (study-size dependent: bigger crop ->
                    # more seams -> 31322 worst).
                    tile_step_size=0.5,
                    part=main_part.name,  # NVTX/timing: inference_3d_fullres_ct_liver_segments
                    normalize=not plans_semantic,  # 09-04: per-part CT normalization
                    # (22-11: plans-semantic input is pre-normalized f32)
                    gated=True,  # serialization gate: starts only after the 6mm
                    # postresample (incl. its release_buffers) has emitted
                    # prev_part_seg — 6mm VRAM released before main load
                    # (v2.1 part-serialization pattern)
                    name=f"swin_{main_part.name}",
                )
                post_main_op = PostResampleOperator(
                    self,
                    config_name=main_part.config_name,
                    emit_argmax_seg=True,
                    emit_probabilities=False,
                    emit_argmax_dicom=True,  # PasteOp consumes it
                    # (identity-branch paste for plans-semantic; order-0 back-resample otherwise)
                    resample_seg_to_original=plans_semantic,  # 22-11 (audit S4):
                    # crop-native seg emit for resample-None tasks
                    part=main_part.name,  # NVTX/timing: postresample_3d_fullres_ct_liver_segments
                    release_fn=swin_main_op.release_buffers,
                    model_path=main_path,
                    name=f"postresample_{main_part.name}",
                )
            # --- Stage 7: paste (undo_crop into the full-native canvas) ---
            paste_op = PasteOp(self, name="paste")
            # Stage 8: emit (P10 node/span names preserved for the runner).
            # base_name=_TASK -> seg_liver_segments_{sar,dhw}.npy per the
            # p18_run_study.sh output contract (spec-driven, no literal).
            emit_op = SegEmitOperator(
                self, mode="total", base_name=_TASK, output_dir=app_output_path, name="emit_5part"
            )
            # The gated swin's gate input (prev_part_seg) receives post6's
            # seg_argmax; post_main's seg_argmax has no other consumer — give
            # it the P7 sink so every declared port has a receiver (GXF rule).
            crop_seg_sink = VolumeSinkOperator(self, name="crop_seg_sink")

            self.add_flow(series_to_vol_op, pre6_op, {("image", "image")})
            self.add_flow(pre6_op, swin6_op, {("preprocessed", "preprocessed")})
            self.add_flow(swin6_op, post6_op, {("logits", "logits")})
            self.add_flow(pre6_op, post6_op, {("preprocessed_meta", "preprocessed_meta")})
            self.add_flow(post6_op, crop_mask_op, {("seg_argmax_dicom", "seg_labelmap")})
            self.add_flow(pre6_op, crop_mask_op, {("preprocessed_meta", "preprocessed_meta")})
            self.add_flow(crop_mask_op, crop_op, {("crop_mask", "crop_mask")})
            self.add_flow(crop_mask_op, crop_op, {("preprocessed_meta", "preprocessed_meta")})
            self.add_flow(series_to_vol_op, crop_op, {("image", "image")})
            self.add_flow(crop_op, pre_task_op, {("cropped_volume", "image")})
            self.add_flow(crop_op, pre_task_op, {("crop_meta", "crop_meta")})
            self.add_flow(crop_mask_op, pre_task_op, {("mask_nonempty", "mask_nonempty")})
            if multi_main:
                # Per-main-part chain + N-1 serialization gates (P9 pattern):
                # part i starts only after the previous stage's postresample
                # (including release_buffers) has emitted prev_part_seg.
                # Part 1's previous stage is the 6mm postresample.
                _prev_post = post6_op
                for _mp in main_parts:
                    self.add_flow(pre_task_op, main_swin_ops[_mp.name], {("preprocessed", "preprocessed")})
                    self.add_flow(pre_task_op, main_post_ops[_mp.name], {("preprocessed_meta", "preprocessed_meta")})
                    self.add_flow(main_swin_ops[_mp.name], main_post_ops[_mp.name], {("logits", "logits")})
                    self.add_flow(_prev_post, main_swin_ops[_mp.name], {("seg_argmax", "prev_part_seg")})
                    self.add_flow(main_post_ops[_mp.name], merge_remap_op, {("seg_argmax_dicom", f"seg_{_mp.name}")})
                    _prev_post = main_post_ops[_mp.name]
                # Merged unified-label map feeds paste (back-resample + canvas
                # paste + P10 flips) — identical to the single-main path.
                self.add_flow(merge_remap_op, paste_op, {("merged", "seg_cropped")})
                # The last part's seg_argmax has no other consumer (its only
                # role was gating) — P7 sink, same as the single-main path.
                self.add_flow(_prev_post, crop_seg_sink, {("seg_argmax", "image")})
            else:
                # Gate edge: 6mm released before the main model's inference starts.
                self.add_flow(post6_op, swin_main_op, {("seg_argmax", "prev_part_seg")})
                self.add_flow(pre_task_op, swin_main_op, {("preprocessed", "preprocessed")})
                self.add_flow(pre_task_op, post_main_op, {("preprocessed_meta", "preprocessed_meta")})
                self.add_flow(swin_main_op, post_main_op, {("logits", "logits")})
                self.add_flow(post_main_op, paste_op, {("seg_argmax_dicom", "seg_cropped")})
            # Empty-path fan-in: crop_mask emits the all-zero native canvas on
            # the SAME port only when the mask is empty (Pitfall 5: the edge
            # exists statically; exactly one of the two senders fires per run).
            self.add_flow(crop_mask_op, paste_op, {("seg_cropped", "seg_cropped")})
            self.add_flow(crop_op, paste_op, {("crop_bbox", "crop_bbox")})
            self.add_flow(crop_op, paste_op, {("crop_meta", "crop_meta")})
            self.add_flow(crop_mask_op, paste_op, {("mask_nonempty", "mask_nonempty")})
            self.add_flow(crop_op, paste_op, {("preprocessed_meta", "preprocessed_meta")})
            if not multi_main:
                self.add_flow(post_main_op, crop_seg_sink, {("seg_argmax", "image")})
            self.add_flow(pre6_op, emit_op, {("preprocessed_meta", "preprocessed_meta")})
            self.add_flow(paste_op, emit_op, {("seg_merged", "seg_merged")})
            self.add_flow(
                paste_op, emit_op, {("seg_dhw", "seg_image")}
            )  # emit's port name is seg_image; carries spatial DHW

            # --- P10 output chain (SEG + SR) — tables from the spec, never
            # hard-coded: segment descriptions come from the module named by
            # spec.snomed (attribute of the same name), metrics labels from
            # spec.labels (background excluded), ModelInfo from the table's
            # algorithm fields. Flip contract identical to multi_part
            # (10-RESEARCH §5): metrics gets pre-flip SAR seg_merged; the SEG
            # writer gets seg_image = flip(1,2) of SAR only; highdicom
            # superior-first sorting makes the on-disk SEG decode to spatial
            # DHW = the emit's seg_dhw port. Pitfall-5 audit: every required
            # port of the 3 output operators is wired below; the SR writer's
            # study_selected_series_list port is effectively mandatory under
            # copy_tags=True — hence wired.
            try:
                seg_desc_mod = _importlib.import_module(f"my_app.{self._task_spec.snomed_module}")
            except ImportError:  # flat import (my_app dir on sys.path)
                seg_desc_mod = _importlib.import_module(self._task_spec.snomed_module)
            seg_descriptions = getattr(seg_desc_mod, self._task_spec.snomed)
            _algorithm = seg_descriptions[0]._algorithm_identification
            my_model_info = ModelInfo(
                creator="TotalSegmentator",
                name=_algorithm.name,
                version=_algorithm.version,
                uid="0.1.0",
            )
            my_equipment_info = EquipmentInfo(
                manufacturer="The MONAI Consortium",
                manufacturer_model="MONAI Deploy App SDK",
                software_version_number="3.5.0",
            )
            custom_tags_seg = {
                "SeriesDescription": "AI Generated DICOM SEG; Not for Clinical Use.",
                "AlgorithmName": f"{my_model_info.name}:{my_model_info.version}:{my_model_info.uid}",
            }
            custom_tags_sr = {
                "SeriesDescription": "AI Generated DICOM SR; Not for Clinical Use.",
                "AlgorithmName": f"{my_model_info.name}:{my_model_info.version}:{my_model_info.uid}",
            }

            _metrics_labels = {k: v for k, v in self._task_spec.labels.items() if k != "background"}
            if _GPU_METRICS_ENABLED:
                seg_metrics_op = SegVolumeMetricsOperator(
                    self,
                    name="seg_metrics_op",
                    labels_dict=_metrics_labels,
                )
            else:
                seg_metrics_op = SegmentationMetricsOperator(
                    self,
                    name="seg_metrics_op",
                    use_gpu=True,
                    labels_dict=_metrics_labels,
                )
            dicom_seg_writer = TimedDICOMSegmentationWriterOperator(
                self,
                segment_descriptions=seg_descriptions,  # 8-entry 99COUINAUD table
                model_info=my_model_info,
                custom_tags=custom_tags_seg,
                output_folder=app_output_path / "SEG",
                omit_empty_frames=False,
                name="dicom_seg_writer",
            )
            dicom_sr_writer = TimedDICOMTextSRWriterOperator(
                self,
                copy_tags=True,
                model_info=my_model_info,
                equipment_info=my_equipment_info,
                custom_tags=custom_tags_sr,
                included_fields=["vol"],
                report_code_value="25045-6",
                report_coding_scheme_designator="LN",
                report_code_meaning="CT Report",
                output_folder=app_output_path / "SR",
                name="dicom_sr_writer",
            )

            self.add_flow(series_to_vol_op, seg_metrics_op, {("image", "input_scan")})
            self.add_flow(paste_op, seg_metrics_op, {("seg_merged", "segmentation_mask")})
            self.add_flow(seg_metrics_op, dicom_sr_writer, {("metrics_dict", "dict")})
            self.add_flow(
                series_selector_op, dicom_sr_writer, {("study_selected_series_list", "study_selected_series_list")}
            )
            self.add_flow(
                series_selector_op, dicom_seg_writer, {("study_selected_series_list", "study_selected_series_list")}
            )
            self.add_flow(
                paste_op, dicom_seg_writer, {("seg_image", "seg_image")}
            )  # writer input = flip(1,2) only (P10)

        logging.info(f"End {self.compose.__name__}")
        self.add_flow(study_loader_op, series_selector_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(
            series_selector_op,
            series_to_vol_op,
            {("study_selected_series_list", "study_selected_series_list")},
        )
        self.add_flow(series_to_vol_op, volume_sink_op, {("image", "image")})

        # --- Scheduler (D-21 port, Phase 14-01) ---
        # The v2.0 app has NO scheduler -> holoscan falls back to the default
        # GreedyScheduler, which serializes the SEG/SR writers behind
        # seg_metrics (45 s idle-writer gap at 31322, 12-RESEARCH §WS-3a).
        # The v1.1 app (cchmc-nnunet-fast D-21) proves this knob is
        # pixel-exact and safe: the scheduler changes WHEN codelets fire, not
        # the DAG. DEFAULT ON ("1") as of Plan 14-03 (was OFF for the 14-01
        # A/B, which was 6/6 PASS — p14_scheduler_ab.json); setting
        # HOLOSCAN_CONCURRENT_FRAGMENTS=0 forces serial (rollback path).
        # HOLOSCAN_SCHEDULER_WORKERS tunes the pool (default 5).
        if os.environ.get("HOLOSCAN_CONCURRENT_FRAGMENTS", "1") != "0":
            self.scheduler(EventBasedScheduler(self, worker_thread_number=_SCHED_WORKERS, name="concurrent"))
            self._logger.info(
                "scheduler: EventBasedScheduler worker_thread_number=%d (D-21 port)",
                _SCHED_WORKERS,
            )
        else:
            self._logger.info("scheduler: default GreedyScheduler (serial)")

        logging.info(f"End {self.compose.__name__}")


# Series selection rules (JSON) — BROADENED for full-corpus TCIA validation
# (user-approved 2026-08-28). Original CLINICAL rule (restorable verbatim):
#   "Standard Axial CT Series": StudyDescription "(.*?)", Modality "(?i)CT",
#   ImageOrientationPatient "Axial", ImageType ["PRIMARY"], SliceThickness [2, 5],
#   SeriesDescription "(?i)^(?!.*(cor|sag|lung)).*$"
# The clinical rule rejects 68/359 corpus studies (ST=0.625 thin-slice x62,
# non-PRIMARY ImageType x6). Broadened to "any CT series": every corpus study
# carries exactly one series, so selection is deterministic; for the 3 pinned
# studies the selected series is identical to the clinical rule's (pin
# baselines in /raid/tmp/ts_baseline remain valid — sha256 re-verified by gate).
# This Sample_Rules_Text block is BYTE-IDENTICAL in both apps (this file and
# ct-totalsegmentator-map/app_total/app.py) — parity requirement.
Sample_Rules_Text = """
{
    "selections": [
        {
            "name": "Any CT Series (full-corpus validation)",
            "conditions": {
                "Modality": "(?i)CT"
            }
        }
    ]
}
"""

if __name__ == "__main__":
    logging.info(f"Begin {__name__}")
    TotalSegFastApp().run()
    logging.info(f"End {__name__}")
