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

DAG (Phase 1, single config ``3d_fullres``):

    DICOMDataLoaderOperator
      -> DICOMSeriesSelectorOperator
        -> DICOMSeriesToVolumeOperator
          -> PreprocessOperator -> SlideWindowOperator
             -> PostResampleOperator -> EnsembleAverageOperator
                -> PostprocessOperator
                  -> DICOMSegmentationWriterOperator (SEG)
                  -> DICOMTextSRWriterOperator       (SR)
                  -> DICOMSCWriterOperator           (SC)

The monolithic ``NNUnetSegOperator`` of the reference app is replaced by the
five-operator chain ``Preprocess -> SlideWindow -> PostResample ->
EnsembleAverage -> Postprocess``; the SDK DICOM I/O operators are unchanged
(no SDK core edits — the writers are only subclassed here for structured
timing/NVTX observability).
"""

# INFR-01/D-14: RMM must be imported before holoscan (undefined-symbol hazard:
# `import rmm` after `import holoscan` raises ImportError: undefined symbol
# __cxa_call_terminate — live-reproduced 2026-08-19, pinned by
# scripts/test_gpu_bootstrap.py). This MUST stay the FIRST import.
try:
    from my_app import gpu_bootstrap
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    import gpu_bootstrap

gpu_bootstrap.install_torch_allocator()

import json
import logging
from dataclasses import asdict
from pathlib import Path

# torch before holoscan is fine — only rmm-after-holoscan trips the hazard.
import torch

# pydicom SR coded dictionary — direct import (not part of the App SDK package)
from pydicom.sr.codedict import codes

from monai.deploy.conditions import CountCondition
from monai.deploy.core import Application
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator, SegmentDescription
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.dicom_text_sr_writer_operator import (
    DICOMTextSRWriterOperator,
    EquipmentInfo,
    ModelInfo,
)

try:  # package-style import (my_app.*)
    from my_app.config import find_jsonpkls_dir, load_inference_params
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
    from config import find_jsonpkls_dir, load_inference_params
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

# nnU-Net plans.json configuration (Phase 1: single config)
CONFIG_NAME = "3d_fullres"

# output_labels: which segmentations are desired in the DICOM SEG / SR / SC
# outputs — 1 = airway
OUTPUT_LABELS = [1]

# general algorithm information (DICOM VR LO type, 64-char limit)
_ALGORITHM_NAME = "CCHMC_nnunet_airway_fast"
_ALGORITHM_FAMILY = codes.DCM.ArtificialIntelligence
_ALGORITHM_VERSION = "1.0.0"
_MAP_UID = "1.0.0"

_DEFAULT_LABEL_NAMES = {0: "background", 1: "airway"}


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

    def _compute_budget_plan(self, model_path):
        """Setup-time VRAM budget for the current config (INFR-03).

        The preprocessed volume shape is study-dependent and unknown at
        compose time, so the per-dataset ``median_image_size_in_voxels`` from
        plans.json is used as the estimate (inherited configs that lack the
        key fall back to the original median shape scaled by the spacing
        ratio, matching the resample); the crop shape is bounded above by
        the same volume (crop ⊆ image). The plan is a safety net, not an
        exact prediction.
        """
        params = load_inference_params(model_path, CONFIG_NAME)
        jsonpkls = find_jsonpkls_dir(model_path)
        plans = json.loads((jsonpkls / "plans.json").read_text())
        cfg = plans["configurations"][CONFIG_NAME]

        median = cfg.get("median_image_size_in_voxels")
        if median is None:
            orig_shape = plans["original_median_shape_after_transp"]
            orig_spacing = plans["original_median_spacing_after_transp"]
            spacing = cfg["spacing"]
            median = [
                int(round(float(s) * float(o) / float(t)))
                for s, o, t in zip(orig_shape, orig_spacing, spacing)
            ]
        volume_shape = (params.num_input_channels, *[int(round(float(s))) for s in median])

        cfgs = [
            {
                "config_name": CONFIG_NAME,
                "num_input_channels": params.num_input_channels,
                "num_segmentation_heads": params.num_segmentation_heads,
                "preprocessed_shape": volume_shape,
                # upper bound: the crop is always ⊆ the (resampled) image
                "cropped_shape": volume_shape,
            }
        ]
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

        # --- Memory budget (INFR-03, D-15) ---
        # Setup-time estimate of the per-config full-volume VRAM footprint
        # vs free VRAM. Drives the ensemble defer flag; the real OOM path is
        # UNEXERCISED on the A100-40GB airway study (documented, not faked).
        plan = self._compute_budget_plan(model_path)
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

        # --- New GPU operator chain (replaces NNUnetSegOperator) ---
        preprocess_op = PreprocessOperator(
            self, model_path=model_path, config_name=CONFIG_NAME, name="preprocess_op"
        )
        slidewindow_op = SlideWindowOperator(
            self, model_path=model_path, config_name=CONFIG_NAME, name="slidewindow_op"
        )
        postresample_op = PostResampleOperator(self, name="postresample_op")
        # averaged_probabilities has no consumer in the DAG (postprocess
        # consumes the uint8 seg) — disable that output or the GXF scheduler
        # rejects the entity (declared output with no downstream receiver).
        ensemble_op = EnsembleAverageOperator(
            self,
            emit_averaged_probabilities=False,
            defer_strategy=(plan.strategy == "defer_to_incremental"),
            name="ensemble_average_op",
        )
        postprocess_op = PostprocessOperator(
            self,
            model_path=model_path,
            applied_labels=tuple(OUTPUT_LABELS),
            label_names=_DEFAULT_LABEL_NAMES,
            output_labels=tuple(OUTPUT_LABELS),
            output_folder=app_output_path,
            name="postprocess_op",
        )

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

        # volume -> preprocess -> slidewindow -> postresample -> ensemble -> postprocess
        self.add_flow(series_to_vol_op, preprocess_op, {("image", "image")})
        self.add_flow(preprocess_op, slidewindow_op, {("preprocessed", "preprocessed")})
        self.add_flow(preprocess_op, postresample_op, {("preprocessed_meta", "preprocessed_meta")})
        self.add_flow(slidewindow_op, postresample_op, {("logits", "logits")})
        self.add_flow(postresample_op, ensemble_op, {("probabilities", "probabilities")})
        self.add_flow(ensemble_op, postprocess_op, {("seg", "seg")})
        # the original image is also needed by postprocess (SR voxel volume +
        # SC overlay)
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
