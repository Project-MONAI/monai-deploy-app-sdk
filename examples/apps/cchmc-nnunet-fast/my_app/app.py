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

"""cchmc-nnunet-fast application: GPU-resident nnU-Net pipeline.

DAG (Phase 2, multi-fragment — one ``Fragment`` per resolved nnUNet config,
PIPE-03/D-02; DICOM I/O, ensemble, postprocess and writers stay app-level):

    DICOMDataLoaderOperator -> DICOMSeriesSelectorOperator
      -> DICOMSeriesToVolumeOperator
        ├─ (image) -> [Fragment nnunet_<cfg>] for each cfg in run_list:
        │    Preprocess -> SlideWindow -> PostResample; the auxiliary
        │    fragment also emits lowres_seg (argmax uint8), crossed to the
        │    cascade fragment with zero disk I/O (PIPE-04)
        ├─ (probabilities) -> EnsembleAverageOperator (prob_<cfg> ports,
        │    ensemble_model_list order) -> PostprocessOperator
        │       -> DICOMSegmentationWriterOperator (SEG)
        │       -> DICOMTextSRWriterOperator       (SR)
        │       -> DICOMSCWriterOperator           (SC)

The reference's monolithic ``NNUnetSegOperator`` is replaced by
config-generic fragment instantiation over ``resolve_run_model_list``
(``HOLOSCAN_MODEL_LIST`` selects the list; default = reference default).
The SDK DICOM I/O operators are unchanged (no SDK core edits — the writers
are only subclassed here for timing/NVTX observability).
"""

# INFR-01/D-14: RMM must be the FIRST import (importing rmm after holoscan
# raises ImportError: undefined symbol __cxa_call_terminate — live-reproduced
# 2026-08-19, pinned by scripts/test_gpu_bootstrap.py).
try:
    from my_app import gpu_bootstrap
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    import gpu_bootstrap

gpu_bootstrap.install_torch_allocator()

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional, Tuple

# torch before holoscan is fine — only rmm-after-holoscan trips the hazard.
import torch

# INFR-004: per-fragment CUDA stream pools (best-effort overlap, D-16).
from holoscan.resources import CudaStreamPool
# D-21: concurrent independent-fragment execution (research-verified on
# holoscan-cu13 4.2 — worker-thread pool, GIL released at the C++ boundary).
from holoscan.schedulers import EventBasedScheduler

# pydicom SR coded dictionary — direct import (not part of the App SDK package)
from pydicom.sr.codedict import codes

from monai.deploy.conditions import CountCondition
from monai.deploy.core import Application, Subgraph
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator, SegmentDescription
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.dicom_text_sr_writer_operator import (
    DICOMTextSRWriterOperator,
    EquipmentInfo,
    ModelInfo,
)

try:  # package-style import (my_app.*)
    from my_app.config import find_jsonpkls_dir, load_inference_params, resolve_run_model_list
    from my_app.mem_budget import compute_memory_budget
    from my_app.operators import (
        DICOMSCWriterOperator,
        DICOMSeriesSelectorOperator,
        EnsembleAverageOperator,
        GpuTiming,
        PostprocessOperator,
        PostResampleOperator,
        PreprocessOperator,
        SlideWindowOperator,
        StudyTimingCollector,
        get_study_id,
        nvtx_range,
    )
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from config import find_jsonpkls_dir, load_inference_params, resolve_run_model_list
    from mem_budget import compute_memory_budget
    from operators import (
        DICOMSCWriterOperator,
        DICOMSeriesSelectorOperator,
        EnsembleAverageOperator,
        GpuTiming,
        PostprocessOperator,
        PostResampleOperator,
        PreprocessOperator,
        SlideWindowOperator,
        StudyTimingCollector,
        get_study_id,
        nvtx_range,
    )

# output_labels: which segmentations are desired in the DICOM SEG / SR / SC
# outputs — 1 = airway
OUTPUT_LABELS = [1]

# general algorithm information (DICOM VR LO type, 64-char limit)
_ALGORITHM_NAME = "CCHMC_nnunet_airway_fast"
_ALGORITHM_FAMILY = codes.DCM.ArtificialIntelligence
_ALGORITHM_VERSION = "1.0.0"
_MAP_UID = "1.0.0"

_DEFAULT_LABEL_NAMES = {0: "background", 1: "airway"}


def _previous_stage_of(cfg: str, plans: Any) -> Optional[str]:
    """The RAW plans.json ``previous_stage`` of ``cfg`` (plans-driven, D-02).

    ``None`` for non-cascade configs. Reading the raw entry (not the
    PlansManager-resolved one) is fine here: ``previous_stage`` is the one
    field a cascading entry carries directly (inherited fields are resolved
    from ``inherits_from``)."""
    cfgs = plans.get("configurations", {}) or {}
    entry = cfgs.get(cfg, {}) or {}
    return entry.get("previous_stage")


def _auxiliary_prev_stage(run_list: List[str], plans: Any) -> Optional[str]:
    """The (unique) config in ``run_list`` that is the previous stage of some
    other config in ``run_list`` — i.e. ``3d_lowres`` when the cascade is in
    the list, ``None`` otherwise. Plans-driven (D-02): no config-name
    literals. This is the ONLY fragment that emits ``lowres_seg`` (D-07:
    the auxiliary stage never feeds the ensemble, so its ``probabilities``
    output is simply not declared — conditional port table, RESEARCH
    Pitfall 7)."""
    consumers = {
        _previous_stage_of(c, plans) for c in run_list
    }
    prevs = [c for c in run_list if c in consumers]
    return prevs[0] if len(prevs) == 1 else None


def timed_writer_compute(operator, base_class, name, op_input, op_output, context):
    """Shared compute wrapper for the timed writer subclasses: NVTX range +
    a structured timing record (INFR-005/INFR-006) around the unmodified SDK
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
            # the SDK SEG writer does not define _logger; fall back to module logger
            logger = getattr(operator, "_logger", None) or logging.getLogger(f"timed_{type(operator).__name__}")
            logger.info("timing: %s", json.dumps(record))


class TimedDICOMSegmentationWriterOperator(DICOMSegmentationWriterOperator):
    """SDK SEG writer with an NVTX range + structured timing record."""

    def compute(self, op_input, op_output, context):
        return timed_writer_compute(
            self, DICOMSegmentationWriterOperator, "write_seg", op_input, op_output, context
        )


class TimedDICOMTextSRWriterOperator(DICOMTextSRWriterOperator):
    """SDK SR writer with an NVTX range + structured timing record."""

    def compute(self, op_input, op_output, context):
        return timed_writer_compute(
            self, DICOMTextSRWriterOperator, "write_sr", op_input, op_output, context
        )


class TimedDICOMSCWriterOperator(DICOMSCWriterOperator):
    """Custom SC writer with an NVTX range + structured timing record."""

    def compute(self, op_input, op_output, context):
        return timed_writer_compute(
            self, DICOMSCWriterOperator, "write_sc", op_input, op_output, context
        )


class NnUnetConfigSubgraph(Subgraph):
    """One nnUNet config's model chain as a subgraph (PIPE-03, D-02).

    Contains ``preprocess_<cfg> -> slidewindow_<cfg> -> postresample_<cfg>``
    and exposes interface ports so the app wires it config-generically:

    * input  ``image``          — always (the DICOM volume);
    * input  ``lowres_seg``     — cascade configs only (the previous stage's
      post-softmax argmax; zero disk I/O — PIPE-04/D-09/D-10);
    * output ``probabilities``  — ensemble members only (D-07: the auxiliary
      lowres never feeds the ensemble — its port is simply absent);
    * output ``lowres_seg``     — the auxiliary previous stage only.

    Subgraph is the holoscan-cu13 4.2 supported multi-fragment mechanism
    (interface ports + ``add_flow`` between subgraphs/operators): 4.2's
    app_driver rejects an app graph mixing C++ ``Fragment``s and
    app-level operators ("Both fragments and operators are added to the
    application graph"), and its fragment-to-fragment flow API has no
    operator-port addressing — Subgraph interface ports do.
    """

    def __init__(
        self,
        parent: Any,
        name: str,
        model_path: Any,
        config_name: str,
        n_entry_inputs: int,
        emit_probabilities: bool,
        emit_lowres_seg: bool,
    ):
        # Subgraph.__init__ runs compose() during construction — all state
        # touched by compose must exist first (holoscan 4.2 quirk, same
        # pattern as the operators' flags-before-super()).
        self._model_path = model_path
        self._config_name = config_name
        self._n_entry_inputs = n_entry_inputs
        self._emit_probabilities = bool(emit_probabilities)
        self._emit_lowres_seg = bool(emit_lowres_seg)
        super().__init__(parent, name)

    def compose(self) -> None:
        cfg = self._config_name
        pre = PreprocessOperator(
            self,
            # Entry condition (RESEARCH Pattern 1): the cascade preprocess
            # must fire once, after BOTH inputs — never on the image alone.
            CountCondition(self, self._n_entry_inputs),
            model_path=self._model_path,
            config_name=cfg,
            name=f"preprocess_{cfg}",
        )
        sw = SlideWindowOperator(
            self, model_path=self._model_path, config_name=cfg, name=f"slidewindow_{cfg}"
        )
        post = PostResampleOperator(
            self,
            config_name=cfg,
            emit_probabilities=self._emit_probabilities,
            emit_lowres_seg=self._emit_lowres_seg,
            name=f"postresample_{cfg}",
        )
        self.add_flow(pre, sw, {("preprocessed", "preprocessed")})
        self.add_flow(pre, post, {("preprocessed_meta", "preprocessed_meta")})
        self.add_flow(sw, post, {("logits", "logits")})

        # Interface ports — the ONLY ports crossing the subgraph boundary.
        # Conditional declarations implement the plan's port table (D-07,
        # RESEARCH Pitfall 7: no declared port left without a flow/receiver).
        self.add_input_interface_port("image", pre, "image")
        if self._n_entry_inputs > 1:
            self.add_input_interface_port("lowres_seg", pre, "lowres_seg")
        if self._emit_probabilities:
            self.add_output_interface_port("probabilities", post, "probabilities")
        if self._emit_lowres_seg:
            self.add_output_interface_port("lowres_seg", post, "lowres_seg")


class CCHMCNNUnetFastApp(Application):
    """Fast-track nnU-Net segmentation app for CCHMC models.

    Loads DICOM input, performs GPU-resident nnU-Net inference through the
    five-operator chain (preprocess, sliding-window inference, post-resample,
    ensemble average, postprocess), and writes segmentation outputs (DICOM
    SEG, SR, and Secondary Capture). Every operator emits structured timing
    records; an aggregated per-study latency summary is logged after the run.
    """

    def __init__(self, *args, **kwargs):
        """Creates an application instance."""
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        """Entry point — delegates to the base Application runner, then logs
        the aggregated per-study timing summary (preprocess / inference /
        postprocess / write)."""
        self._logger.info(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._log_study_timing_summaries()
        self._logger.info(f"End {self.run.__name__}")

    def _log_study_timing_summaries(self):
        """Emit one aggregate latency record per study (INFR-006)."""
        for study, records in StudyTimingCollector.studies(self).items():
            per_operator = {}
            for r in records:
                per_operator[r["operator"]] = r["duration_ms"]
            aggregate = {
                "study": study,
                "operators": per_operator,
                "total_ms": round(sum(r["duration_ms"] for r in records), 3),
                "n_records": len(records),
            }
            self._logger.info("study_timing_summary: %s", json.dumps(aggregate))

    def _compute_budget_plan(self, model_path, run_list):
        """Setup-time VRAM budget for the FULL resolved run list (INFR-03).

        Phase 2 Plan 04: extended from the single-config estimate to ALL
        configs in the run list — the fragments coexist in one DAG, so their
        full-volume footprints add. The preprocessed volume shape is
        study-dependent and unknown at compose time, so the per-dataset
        ``median_image_size_in_voxels`` (read from the PlansManager-RESOLVED
        configuration — inherited configs like the cascade carry the field
        via ``inherits_from``) is used as the estimate; the crop shape is
        bounded above by the same volume (crop ⊆ image). The plan is a
        safety net, not an exact prediction.
        """
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

        jsonpkls = find_jsonpkls_dir(model_path)
        plans = json.loads((jsonpkls / "plans.json").read_text())
        manager = PlansManager(plans)

        cfgs = []
        for cfg in run_list:
            params = load_inference_params(model_path, cfg)
            resolved = manager.get_configuration(cfg).configuration

            median = resolved.get("median_image_size_in_voxels")
            if median is None:
                orig_shape = plans["original_median_shape_after_transp"]
                orig_spacing = plans["original_median_spacing_after_transp"]
                spacing = resolved["spacing"]
                median = [
                    int(round(float(s) * float(o) / float(t)))
                    for s, o, t in zip(orig_shape, orig_spacing, spacing)
                ]
            volume_shape = (params.num_input_channels, *[int(round(float(s))) for s in median])

            cfgs.append(
                {
                    "config_name": cfg,
                    "num_input_channels": params.num_input_channels,
                    "num_segmentation_heads": params.num_segmentation_heads,
                    "preprocessed_shape": volume_shape,
                    # upper bound: the crop is always ⊆ the (resampled) image
                    "cropped_shape": volume_shape,
                }
            )
        return compute_memory_budget(cfgs)

    def compose(self):
        """Creates the app-specific operators and chains them into the
        processing DAG."""
        logging.info(f"Begin {self.compose.__name__}")

        # Use command-line options over environment variables to init context
        app_context = Application.init_app_context(self.argv)
        app_input_path = Path(app_context.input_path)
        app_output_path = Path(app_context.output_path)
        model_path = Path(app_context.model_path)

        # Temporary bug fix for MAP execution where model path copy is messed
        # up — check for a 'models' subfolder and use it if present (same as
        # the reference app).
        models_subfolder = model_path / "models"
        if models_subfolder.exists() and models_subfolder.is_dir():
            self._logger.info(f"Found 'models' subfolder in {model_path}. Setting model_path to {models_subfolder}")
            model_path = models_subfolder

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

        # --- Model list (PIPE-03, D-02) ---
        # HOLOSCAN_MODEL_LIST selects the run list (comma-separated configs);
        # unset = the reference default (plans.json order filtered to existing
        # model dirs). Reference semantics + data-driven previous-stage
        # auto-insertion live in config.resolve_run_model_list.
        model_list_arg = os.environ.get("HOLOSCAN_MODEL_LIST")
        model_list_arg = (
            [s.strip() for s in model_list_arg.split(",") if s.strip()]
            if model_list_arg
            else None
        )
        plans = json.loads((find_jsonpkls_dir(model_path) / "plans.json").read_text())
        run_list, ensemble_list = resolve_run_model_list(model_list_arg, plans, model_path)
        self._logger.info(
            "run_model_list=%s ensemble_model_list=%s", list(run_list), list(ensemble_list)
        )

        # --- Memory budget (INFR-03, D-15) ---
        # Setup-time estimate of the per-config full-volume VRAM footprint of
        # ALL configs in the run list vs free VRAM. Drives the ensemble defer
        # flag; the real OOM path is UNEXERCISED on the A100-40GB airway
        # study (documented, not faked).
        plan = self._compute_budget_plan(model_path, list(run_list))
        self._logger.info("memory_budget: %s", json.dumps(asdict(plan)))

        # --- DICOM I/O (SDK, unchanged) ---
        study_loader_op = DICOMDataLoaderOperator(
            self, CountCondition(self, 1), input_folder=app_input_path, name="study_loader_op"
        )

        # custom DICOM Series Selector op (copied from the reference app);
        # all_matched + SOP sorting: downstream runs on the 1st selected series
        series_selector_op = DICOMSeriesSelectorOperator(
            self, rules=Sample_Rules_Text, all_matched=True, sort_by_sop_instance_count=True,
            name="series_selector_op",
        )

        series_to_vol_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_op")

        # App-level postprocess (CC on GPU, SR/SC data) — sits OUTSIDE the
        # per-config fragments (Phase 2: only the model chain is fragmented).
        postprocess_op = PostprocessOperator(
            self,
            model_path=model_path,
            applied_labels=tuple(OUTPUT_LABELS),
            label_names=_DEFAULT_LABEL_NAMES,
            output_labels=tuple(OUTPUT_LABELS),
            output_folder=app_output_path,
            name="postprocess_op",
        )

        # --- One subgraph per resolved config (PIPE-03, D-02) ---
        # The loop iterates the resolved run list — NO config-name literals
        # in this block (D-02: a future 2d model drops in with zero code
        # changes). Port discipline (RESEARCH Pitfall 7): every declared
        # port has a flow / every declared output has a receiver in each
        # configuration (the conditional interface ports in
        # NnUnetConfigSubgraph implement the plan's port table).
        aux_prev = _auxiliary_prev_stage(list(run_list), plans)
        subgraphs: dict = {}
        for cfg in run_list:
            # Entry inputs: the image always; lowres_seg for cascade configs
            # (resolve_run_model_list auto-inserts the previous stage, so the
            # producer subgraph is guaranteed present).
            n_entry_inputs = 2 if _previous_stage_of(cfg, plans) is not None else 1
            sg = NnUnetConfigSubgraph(
                self,
                name=f"nnunet_{cfg}",
                model_path=model_path,
                config_name=cfg,
                n_entry_inputs=n_entry_inputs,
                # D-07 + conditional port table: probabilities are declared
                # ONLY for ensemble members (the auxiliary lowres never feeds
                # the ensemble — its port is simply absent); lowres_seg is
                # declared ONLY by the auxiliary previous stage (consumed by
                # the cascade subgraph below).
                emit_probabilities=(cfg in ensemble_list),
                emit_lowres_seg=(cfg == aux_prev),
            )
            # INFR-004: one CudaStreamPool per fragment (NonBlocking stream,
            # reserved_size=1, named for the nsys trace). Best-effort overlap
            # (D-16): a plus, not a gate.
            CudaStreamPool(
                sg,
                dev_id=0,
                stream_flags=1,
                reserved_size=1,
                nvtx_identifier=f"streams_{cfg}",
                name=f"cuda_stream_pool_{cfg}",
            )
            # App-level entry edge: the DICOM volume feeds every config.
            self.add_flow(series_to_vol_op, sg, {("image", "image")})
            subgraphs[cfg] = sg

        # --- Cascade edge (PIPE-04, D-09/D-10) ---
        # The auxiliary stage's post-softmax argmax (uint8, original DICOM
        # orientation) crosses the fragment boundary with ZERO disk I/O and
        # ZERO copy — it is consumed as the second channel of the cascade
        # preprocess input. Only present when the auxiliary stage is in the
        # run list (port table: no declared port left unwired).
        if aux_prev is not None:
            for cfg in run_list:
                if _previous_stage_of(cfg, plans) == aux_prev:
                    self.add_flow(subgraphs[aux_prev], subgraphs[cfg], {("lowres_seg", "lowres_seg")})

        # --- App-level ensemble (INF-009, ensemble_model_list order) ---
        # One named probability stream per ensemble config; CountCondition
        # inside the operator guarantees it runs on FULL arrival only. The
        # same code path covers the single-config lists (one port, count 1).
        ensemble_op = EnsembleAverageOperator(
            self,
            config_names=list(ensemble_list),
            emit_averaged_probabilities=False,  # no receiver — port discipline
            defer_strategy=(plan.strategy == "defer_to_incremental"),
            name="ensemble_average_op",
        )
        for cfg in ensemble_list:
            self.add_flow(subgraphs[cfg], ensemble_op, {("probabilities", f"prob_{cfg}")})
        self.add_flow(ensemble_op, postprocess_op, {("seg", "seg")})

        # --- Writers (subclassed for timing; SDK behavior unmodified) ---
        # SEG writer: segment description for the airway segment
        segment_descriptions = [
            SegmentDescription(
                segment_label="Airway",
                segmented_property_category=codes.SCT.BodyStructure,
                segmented_property_type=codes.SCT.TracheaAndBronchus,
                algorithm_name=_ALGORITHM_NAME,
                algorithm_family=_ALGORITHM_FAMILY,
                algorithm_version=_ALGORITHM_VERSION,
            ),
        ]

        my_model_info = ModelInfo(
            creator="CCHMC",
            name=_ALGORITHM_NAME,
            version=_ALGORITHM_VERSION,
            uid=_MAP_UID,
        )
        my_equipment = EquipmentInfo(
            manufacturer="The MONAI Consortium",
            manufacturer_model="MONAI Deploy App SDK",
            software_version_number="3.0.0",
        )

        custom_tags_seg = {
            "SeriesDescription": "AI Generated DICOM SEG; Not for Clinical Use.",
            "AlgorithmName": f"{my_model_info.name}:{my_model_info.version}:{my_model_info.uid}",
        }
        custom_tags_sr = {
            "SeriesDescription": "AI Generated DICOM SR; Not for Clinical Use.",
            "AlgorithmName": f"{my_model_info.name}:{my_model_info.version}:{my_model_info.uid}",
        }
        custom_tags_sc = {
            "SeriesDescription": "AI Generated DICOM Secondary Capture; Not for Clinical Use.",
            "AlgorithmName": f"{my_model_info.name}:{my_model_info.version}:{my_model_info.uid}",
        }

        dicom_seg_writer = TimedDICOMSegmentationWriterOperator(
            self,
            segment_descriptions=segment_descriptions,
            model_info=my_model_info,
            custom_tags=custom_tags_seg,
            output_folder=app_output_path / "SEG",
            # keep input/output series #s aligned (reference-app setting)
            omit_empty_frames=False,
            name="dicom_seg_writer",
        )

        dicom_sr_writer = TimedDICOMTextSRWriterOperator(
            self,
            # copy DICOM attributes so the SR carries the same Study UID
            copy_tags=True,
            model_info=my_model_info,
            equipment_info=my_equipment,
            custom_tags=custom_tags_sr,
            output_folder=app_output_path / "SR",
            name="dicom_sr_writer",
        )

        dicom_sc_writer = TimedDICOMSCWriterOperator(
            self,
            model_info=my_model_info,
            equipment_info=my_equipment,
            custom_tags=custom_tags_sc,
            output_folder=app_output_path / "SC",
            name="dicom_sc_writer",
        )

        # --- DAG wiring ---
        self.add_flow(study_loader_op, series_selector_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(
            series_selector_op,
            series_to_vol_op,
            {("study_selected_series_list", "study_selected_series_list")},
        )

        # volume -> per-config fragments (entry edges wired in the factory
        # above) -> ensemble -> postprocess (wired above); the original image
        # is also needed by postprocess (SR voxel volume + SC overlay)
        self.add_flow(series_to_vol_op, postprocess_op, {("image", "image")})

        # DICOM SEG: selected series + final segmentation
        self.add_flow(
            series_selector_op,
            dicom_seg_writer,
            {("study_selected_series_list", "study_selected_series_list")},
        )
        self.add_flow(postprocess_op, dicom_seg_writer, {("seg", "seg_image")})

        # DICOM SR: selected series + result text
        self.add_flow(
            series_selector_op,
            dicom_sr_writer,
            {("study_selected_series_list", "study_selected_series_list")},
        )
        self.add_flow(postprocess_op, dicom_sr_writer, {("result_text", "text")})

        # DICOM SC: selected series + temp SC dir (overlay .dcm)
        self.add_flow(
            series_selector_op,
            dicom_sc_writer,
            {("study_selected_series_list", "study_selected_series_list")},
        )
        self.add_flow(postprocess_op, dicom_sc_writer, {("dicom_sc_dir", "dicom_sc_dir")})

        # --- Pool pre-allocation (INFR-01/D-14) ---
        # Warm the RMM pool to the budget total at setup time so study 1's
        # per-tile allocations draw from the pool instead of cudaMalloc.
        gpu_bootstrap.warm_pool(plan.total_bytes)

        # --- Scheduler (D-21) ---
        # D-21: concurrent independent-fragment execution, ON by default
        # (flipped after the full gate suite passed with concurrency enabled —
        # .planning/phases/03-optimization/gates/03-GATE-concurrent.json,
        # all 4 pixel gates + SR + residency green; overlap evidence in
        # .planning/profiles/phase3/overlap.md). Explicit
        # HOLOSCAN_CONCURRENT_FRAGMENTS=0 restores the Phase 2 GreedyScheduler
        # serial behavior (byte-for-byte; verified by 03-GATE-serial.json).
        if os.environ.get("HOLOSCAN_CONCURRENT_FRAGMENTS", "1") != "0":
            self.scheduler(EventBasedScheduler(self, worker_thread_number=5, name="concurrent"))
            self._logger.info("scheduler: EventBasedScheduler worker_thread_number=5 (D-21)")
        else:
            self._logger.info("scheduler: default GreedyScheduler (serial, Phase 2 behavior)")

        logging.info(f"End {self.compose.__name__}")


# Series selection rules (JSON). Empty = no attribute conditions: with
# all_matched=True every series in the study is selected and downstream
# operators run on the 1st selected series (reference-app behavior).
Sample_Rules_Text = """
"""

if __name__ == "__main__":
    logging.info(f"Begin {__name__}")
    CCHMCNNUnetFastApp().run()
    logging.info(f"End {__name__}")
