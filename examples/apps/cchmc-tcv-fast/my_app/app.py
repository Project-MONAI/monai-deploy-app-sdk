# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""cchmc-tcv-fast application: GPU-resident nnU-Net pipeline (TCV cardiac+lung).

DAG (multi-fragment — one ``Fragment`` per resolved nnUNet config;
DICOM I/O, ensemble, postprocess and writers stay app-level):

    DICOMDataLoaderOperator -> DICOMSeriesSelectorOperator
      -> DICOMSeriesToVolumeOperator
        ├─ (image) -> [Fragment nnunet_<cfg>] for each cfg in run_list:
        │ Preprocess -> SlideWindow -> PostResample; the auxiliary
        │ fragment also emits lowres_seg (argmax uint8), crossed to the
        │ cascade fragment with zero disk I/O
        ├─ (probabilities) -> EnsembleAverageOperator (prob_<cfg> ports,
        │ ensemble_model_list order) -> PostprocessOperator
        │ -> DICOMSegmentationWriterOperator (SEG)
        │ -> DICOMTextSRWriterOperator (SR)
        │ -> DICOMSCWriterOperator (SC)

The reference's monolithic ``NNUnetSegOperator`` is replaced by
config-generic fragment instantiation over ``resolve_run_model_list``
(``HOLOSCAN_MODEL_LIST`` selects the list; default = reference default).
The SDK DICOM I/O operators are unchanged (no SDK core edits — the writers
are only subclassed here for timing/NVTX observability).
"""

# /: RMM must be the FIRST import (importing rmm after holoscan
# raises ImportError: undefined symbol __cxa_call_terminate — live-reproduced
# 2026-08-19; a dev-only GPU test pins the import order).
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
from typing import Any, Dict, List, Optional

# torch before holoscan is fine — only rmm-after-holoscan trips the hazard.
import torch

# per-fragment CUDA stream pools (best-effort overlap).
from holoscan.resources import CudaStreamPool

# concurrent independent-fragment execution (research-verified on
# holoscan-cu13 4.2 — worker-thread pool, GIL released at the C++ boundary).
from holoscan.schedulers import EventBasedScheduler

# pydicom SR coded dictionary — direct import (not part of the App SDK package)
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code

# The bundled pydicom SCT dictionary has no keyword entries for LeftLung /
# RightLung (probed 2026-10-10), so extend the shared dict once with the
# standard SNOMED CT body-structure codes the TCV segment descriptors use
# (26317004 left lung, 8214007 right lung). Idempotent; no existing keyword
# is altered (deviation from plan's highdicom probe: the app imports
# codes from pydicom.sr.codedict, and highdicom exposes no `codes` module in
# this venv).
if not hasattr(codes.SCT, "LeftLung"):
    codes.SCT.LeftLung = Code("26317004", "SCT", "Left lung")
if not hasattr(codes.SCT, "RightLung"):
    codes.SCT.RightLung = Code("8214007", "SCT", "Right lung")

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
# outputs — 1 = heart, 2 = l_lung, 3 = r_lung (TCV fine-tune bundle,
# 3-label 3d_fullres bundle)
OUTPUT_LABELS = [1, 2, 3]

# / : serve-mode bound on each operator's shape-keyed buffer cache.
# Rationale: per-instance key set is small — config-fixed shapes plus a few
# per-study crop shapes — so 8 keeps the 3 most-recent studies' shapes with
# headroom while bounding worst-case growth on an unbounded shape stream.
# Batch never passes it (default None = unbounded, behavior unchanged).
SERVE_SHAPE_CACHE_MAX = 8

# general algorithm information (DICOM VR LO type, 64-char limit)
_ALGORITHM_NAME = "cchmc_tcv_fast"
_ALGORITHM_FAMILY = codes.DCM.ArtificialIntelligence
_ALGORITHM_VERSION = "1.0.0"
_MAP_UID = "1.0.0"

_DEFAULT_LABEL_NAMES = {0: "background", 1: "heart", 2: "left lung", 3: "right lung"}


def _previous_stage_of(cfg: str, plans: Any) -> Optional[str]:
    """The RAW plans.json ``previous_stage`` of ``cfg`` (plans-driven).

    ``None`` for non-cascade configs. Reading the raw entry (not the
    PlansManager-resolved one) is fine here: ``previous_stage`` is the one
    field a cascading entry carries directly (inherited fields are resolved
    from ``inherits_from``)."""
    cfgs = plans.get("configurations", {}) or {}
    entry = cfgs.get(cfg, {}) or {}
    return entry.get("previous_stage")


def _auxiliary_prev_stages(run_list: List[str], plans: Any) -> Dict[str, str]:
    """Map EVERY cascade consumer in ``run_list`` to its previous stage.

    Plans-driven: no config-name literals. Replaces the old single-aux
    assumption: N cascade consumers each get their own producer mapping,
    so ``mapping.values`` is the set of fragments that emit
    ``lowres_seg`` (an auxiliary stage never feeds the ensemble, so its
    ``probabilities`` output is simply not declared — conditional port
    table: no declared port is left without a flow/receiver).
    Non-cascade run lists return ``{}``.

    NOTE: this does NOT check that a previous stage is present in the run
    list — that is ``_validate_cascade_producers``.
    """
    mapping: Dict[str, str] = {}
    for cfg in run_list:
        prev = _previous_stage_of(cfg, plans)
        if prev is not None:
            mapping[cfg] = prev
    return mapping


def _validate_cascade_producers(run_list: List[str], plans: Any) -> None:
    """Fail fast when a cascade consumer's previous stage is missing .

    Without this check the app built a DAG with an unwired ``lowres_seg``
    input port, which hangs at runtime. The error names BOTH the consumer
    config and the missing stage (fail-fast message style mirrors
    ``find_jsonpkls_dir``).
    """
    for cfg, prev in _auxiliary_prev_stages(run_list, plans).items():
        if prev not in run_list:
            raise RuntimeError(
                f"cascade config {cfg!r} requires previous stage {prev!r} "
                f"in the run list (model dir missing?); refusing to build a "
                f"DAG with an unwired lowres_seg input"
            )


def timed_writer_compute(operator, base_class, name, op_input, op_output, context):
    """Shared compute wrapper for the timed writer subclasses: NVTX range +
    a structured timing record around the unmodified SDK
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
        return timed_writer_compute(self, DICOMSegmentationWriterOperator, "write_seg", op_input, op_output, context)


class TimedDICOMTextSRWriterOperator(DICOMTextSRWriterOperator):
    """SDK SR writer with an NVTX range + structured timing record."""

    def compute(self, op_input, op_output, context):
        return timed_writer_compute(self, DICOMTextSRWriterOperator, "write_sr", op_input, op_output, context)


class TimedDICOMSCWriterOperator(DICOMSCWriterOperator):
    """Custom SC writer with an NVTX range + structured timing record."""

    def compute(self, op_input, op_output, context):
        return timed_writer_compute(self, DICOMSCWriterOperator, "write_sc", op_input, op_output, context)


class NnUnetConfigSubgraph(Subgraph):
    """One nnUNet config's model chain as a subgraph ().

    Contains ``preprocess_<cfg> -> slidewindow_<cfg> -> postresample_<cfg>``
    and exposes interface ports so the app wires it config-generically:

    * input ``image`` — always (the DICOM volume);
    * input ``lowres_seg`` — cascade configs only (the previous stage's
      post-softmax argmax; zero disk I/O);
    * output ``probabilities`` — ensemble members only (the auxiliary
      lowres never feeds the ensemble — its port is simply absent);
    * output ``lowres_seg`` — the previous stage(s) of any cascade
      consumer (auxiliary producers; they never feed the ensemble).

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
        serve_mode: bool = False,
        cache_max_entries: Optional[int] = None,
    ):
        # Subgraph.__init__ runs compose during construction — all state
        # touched by compose must exist first (holoscan 4.2 quirk, same
        # pattern as the operators' flags-before-super).
        self._model_path = model_path
        self._config_name = config_name
        self._n_entry_inputs = n_entry_inputs
        self._emit_probabilities = bool(emit_probabilities)
        self._emit_lowres_seg = bool(emit_lowres_seg)
        # Serve-mode buffer-cache cap (None = unbounded).
        self._cache_max_entries = cache_max_entries
        # CountCondition is ONE-SHOT in holoscan 4.2: a direct experiment
        # proved the count never resets (neither via the .count setter nor
        # via re-fire), so a latch consumed by study 1 permanently stalls a
        # long-lived serve graph (silent backpressure into the selector).
        # Serve mode uses the default per-message condition instead:
        # CPU-probed to fire when ALL declared input ports hold a message
        # and to RE-ARM every cycle (two-input probe), and production-proven
        # by PostprocessOperator (2 inputs) surviving study 1 under the
        # EventBasedScheduler. Batch keeps CountCondition (behavior
        # unchanged).
        self._serve_mode = bool(serve_mode)
        # Capture the top-level app BEFORE super.__init__: holoscan 4.2's
        # Subgraph.__init__ runs compose during construction, and it
        # exposes the parent as `self.fragment` (there is no `application`
        # attribute) — an explicit capture is the only reliable handle to
        # the app for the optional model_load_hook.
        self._parent_app = parent
        super().__init__(parent, name)

    def compose(self) -> None:
        cfg = self._config_name
        if self._serve_mode:
            # default per-message condition (see __init__ note) —
            # fires once per study on full arrival and re-arms for the next.
            pre = PreprocessOperator(
                self,
                model_path=self._model_path,
                config_name=cfg,
                name=f"preprocess_{cfg}",
                cache_max_entries=self._cache_max_entries,
            )
        else:
            pre = PreprocessOperator(
                self,
                # Entry condition: the cascade preprocess
                # must fire once, after BOTH inputs — never on the image alone.
                CountCondition(self, self._n_entry_inputs),
                model_path=self._model_path,
                config_name=cfg,
                name=f"preprocess_{cfg}",
                cache_max_entries=self._cache_max_entries,
            )
        sw = SlideWindowOperator(
            self,
            model_path=self._model_path,
            config_name=cfg,
            name=f"slidewindow_{cfg}",
            # The serve branch installs model_load_hook on the top-level app
            # before subgraph construction; batch never sets it → getattr →
            # None → no-op. Passed at construction because holoscan 4.2 runs
            # setup (and the model load that reads the hook) INSIDE the
            # operator constructor — a post-construction assignment would
            # race the load, leaving the registry empty and CheckHealth
            # reporting models: [].
            model_load_hook=getattr(self._parent_app, "_model_load_hook", None),
            cache_max_entries=self._cache_max_entries,
        )
        post = PostResampleOperator(
            self,
            config_name=cfg,
            emit_probabilities=self._emit_probabilities,
            emit_lowres_seg=self._emit_lowres_seg,
            # Free the aux fragment's 3d_lowres weights (~0.8 GB) after the
            # terminal lowres_seg emit — the cascade consumes only the
            # emitted tensor, so nothing downstream touches the released
            # bundle. Non-aux configs get release_fn=None (no behavior
            # change). HOLOSCAN_KEEP_LOWRES_WEIGHTS=1 bypasses the hook
            # (measurement opt-out for the ON-vs-OFF peak-VRAM comparison).
            # Note: the bound method sw.release keeps sw alive via post.
            # NEVER release in serve mode — the daemon is long-lived and
            # SlideWindowOperator has no reload path, so a released
            # 3d_lowres makes the second study's compute raise "compute
            # after release" and kill the whole graph (observed: daemon
            # died on a queue_full second study). Serve keeps all models
            # resident (the persistent-daemon design); this stays a
            # batch-only optimization. Batch behavior unchanged.
            release_fn=(
                sw.release
                if (
                    self._emit_lowres_seg
                    and not self._serve_mode
                    and os.environ.get("HOLOSCAN_KEEP_LOWRES_WEIGHTS") != "1"
                )
                else None
            ),
            name=f"postresample_{cfg}",
        )
        self.add_flow(pre, sw, {("preprocessed", "preprocessed")})
        self.add_flow(pre, post, {("preprocessed_meta", "preprocessed_meta")})
        self.add_flow(sw, post, {("logits", "logits")})

        # Interface ports — the ONLY ports crossing the subgraph boundary.
        # Conditional declarations implement the port table: no declared
        # port is left without a flow/receiver.
        self.add_input_interface_port("image", pre, "image")
        if self._n_entry_inputs > 1:
            self.add_input_interface_port("lowres_seg", pre, "lowres_seg")
        if self._emit_probabilities:
            self.add_output_interface_port("probabilities", post, "probabilities")
        if self._emit_lowres_seg:
            self.add_output_interface_port("lowres_seg", post, "lowres_seg")


class TCVNNUnetFastApp(Application):
    """Fast-track nnU-Net segmentation app for the CCHMC TCV bundle.

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
        """Emit one aggregate latency record per study ."""
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

    def _budget_cfgs(self, model_path, run_list):
        """Build the per-config budget mappings for the resolved run list
        . Reused by BOTH the compose-time plan and the per-RPC
        admission gate so the two always estimate from the
        same shapes.

        The preprocessed volume shape is study-dependent and unknown at
        admission/compose time, so the per-dataset
        ``median_image_size_in_voxels`` (PlansManager-RESOLVED; inherited
        configs carry it via ``inherits_from``) is the SHAPE term — an upper
        bound, since the crop is always ⊆ the resampled image. The per-RPC
        part of the admission budget gate is the LIVE free-VRAM term (pynvml).
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
                    int(round(float(s) * float(o) / float(t))) for s, o, t in zip(orig_shape, orig_spacing, spacing)
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
        return cfgs

    def _compute_budget_plan(self, model_path, run_list):
        """Setup-time VRAM budget for the FULL resolved run list .

        Extended from the single-config estimate to ALL
        configs in the run list — the fragments coexist in one DAG, so their
        full-volume footprints add. The estimate (median-size upper bound,
        crop ⊆ image) is built by :func:`_budget_cfgs`; the plan is a
        safety net, not an exact prediction.
        """
        return compute_memory_budget(self._budget_cfgs(model_path, run_list))

    def compose(self):
        """Creates the app-specific operators and chains them into the
        processing DAG."""
        logging.info(f"Begin {self.compose.__name__}")

        # Serve mode : a SEPARATE compose branch — `python -m my_app
        # serve` sets app._serve_mode = True (serve_daemon.run_serve). The
        # batch path below is untouched: when the flag is unset (batch),
        # this is a no-op check and every line that follows is the original
        # batch compose, byte-for-byte ( design rule).
        if getattr(self, "_serve_mode", False):
            return self._compose_serve()

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

        # --- RMM allocator check ---
        # The gpu_bootstrap import at the top of this module installed RMM as
        # torch's CUDA allocator; if this fails, rmm was imported after
        # holoscan (the undefined-symbol hazard) or the bootstrap import was
        # lost.
        backend = torch.cuda.memory.get_allocator_backend()
        self._logger.info("memory_allocator_backend: %s", backend)
        assert backend == "pluggable", (
            f"RMM torch allocator not active (backend={backend!r}); " "gpu_bootstrap must be imported before holoscan "
        )

        # --- Model list () ---
        # HOLOSCAN_MODEL_LIST selects the run list (comma-separated configs);
        # unset = the reference default (plans.json order filtered to existing
        # model dirs). Reference semantics + data-driven previous-stage
        # auto-insertion live in config.resolve_run_model_list.
        model_list_arg = os.environ.get("HOLOSCAN_MODEL_LIST")
        model_list_arg = [s.strip() for s in model_list_arg.split(",") if s.strip()] if model_list_arg else None
        plans = json.loads((find_jsonpkls_dir(model_path) / "plans.json").read_text())
        run_list, ensemble_list = resolve_run_model_list(model_list_arg, plans, model_path)
        self._logger.info("run_model_list=%s ensemble_model_list=%s", list(run_list), list(ensemble_list))
        # ergonomics: on the default path (no HOLOSCAN_MODEL_LIST),
        # log the plans configurations filtered out for lacking a model dir
        # (BEFORE auto-insertion — i.e. the filtered-out set; [2d] on the
        # real bundle). Explicit lists are used as given, so nothing is
        # "skipped" there.
        if model_list_arg is None:
            skipped = [c for c in (plans.get("configurations") or {}) if not (model_path / c).is_dir()]
            if skipped:
                self._logger.info("skipped configs without model dirs: %s", list(skipped))

        # --- Memory budget ---
        # Setup-time estimate of the per-config full-volume VRAM footprint of
        # ALL configs in the run list vs free VRAM. Drives the ensemble defer
        # flag; the real OOM path is UNEXERCISED on the reference 40 GB A100
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
            self,
            rules=Sample_Rules_Text,
            all_matched=True,
            sort_by_sop_instance_count=True,
            name="series_selector_op",
        )

        series_to_vol_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_op")

        # App-level postprocess (CC on GPU, SR/SC data) — sits OUTSIDE the
        # per-config fragments (only the model chain is fragmented).
        # TCV (TCV fine-tune bundle): official postprocessing.pkl-driven
        # cleanup — the bundle's pkl defines remove-largest-CC for labels 1
        # and 2 ONLY (heart, left lung); label 3 (right lung) carries no pkl
        # rule and is intentionally not cleaned (flows to SEG/SR as-is).
        # keep_top_k=None so the pkl rules apply; a top-k map would REPLACE
        # them and diverge from the oracle.
        postprocess_op = PostprocessOperator(
            self,
            model_path=model_path,
            applied_labels=(1, 2),
            keep_top_k=None,
            label_names=_DEFAULT_LABEL_NAMES,
            output_labels=tuple(OUTPUT_LABELS),
            output_folder=app_output_path,
            name="postprocess_op",
        )

        # --- One subgraph per resolved config ---
        # The loop iterates the resolved run list — NO config-name literals
        # in this block (a future 2d model drops in with zero code
        # changes). Port discipline: every declared port has a flow / every
        # declared output has a receiver in each configuration (the
        # conditional interface ports in NnUnetConfigSubgraph implement the
        # port table).
        # fail fast before building any subgraph with an unwired
        # lowres_seg input (a cascade consumer whose previous stage is not
        # in the run list would hang at runtime).
        _validate_cascade_producers(list(run_list), plans)
        # per-consumer mapping (replaces the single-aux assumption):
        # keys = cascade consumers, values = their previous-stage producers.
        aux_prev_map = _auxiliary_prev_stages(list(run_list), plans)
        subgraphs: dict = {}
        for cfg in run_list:
            # Entry inputs: the image always; lowres_seg for cascade configs
            # (resolve_run_model_list auto-inserts the previous stage on the
            # default path; the check above guarantees the producer
            # is present for explicit lists too).
            n_entry_inputs = 2 if cfg in aux_prev_map else 1
            sg = NnUnetConfigSubgraph(
                self,
                name=f"nnunet_{cfg}",
                model_path=model_path,
                config_name=cfg,
                n_entry_inputs=n_entry_inputs,
                # + conditional port table: probabilities are declared
                # ONLY for ensemble members (an auxiliary lowres never feeds
                # the ensemble — its port is simply absent); lowres_seg is
                # declared ONLY by previous-stage producers (consumed by the
                # cascade subgraph(s) below).
                emit_probabilities=(cfg in ensemble_list),
                emit_lowres_seg=(cfg in aux_prev_map.values()),
            )
            # One CudaStreamPool per fragment (NonBlocking stream,
            # reserved_size=1, named for the nsys trace). Best-effort overlap:
            # a plus, not a gate.
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

        # --- Cascade edge (, /) ---
        # Each auxiliary stage's post-softmax argmax (uint8, original DICOM
        # orientation) crosses the fragment boundary with ZERO disk I/O and
        # ZERO copy — it is consumed as the second channel of its cascade
        # consumer's preprocess input. One edge per (consumer, producer)
        # pair in the mapping (port table: no declared port left unwired; the
        # check above guarantees every producer subgraph exists).
        for consumer, prev in aux_prev_map.items():
            self.add_flow(subgraphs[prev], subgraphs[consumer], {("lowres_seg", "lowres_seg")})

        # --- App-level ensemble (, ensemble_model_list order) ---
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
        # SEG writer: segment descriptions for the TCV segments
        # (TCV fine-tune bundle: 3-label heart/l_lung/r_lung)
        segment_descriptions = [
            SegmentDescription(
                segment_label="Heart",
                segmented_property_category=codes.SCT.BodyStructure,
                segmented_property_type=codes.SCT.Heart,
                algorithm_name=_ALGORITHM_NAME,
                algorithm_family=_ALGORITHM_FAMILY,
                algorithm_version=_ALGORITHM_VERSION,
            ),
            SegmentDescription(
                segment_label="Left Lung",
                segmented_property_category=codes.SCT.BodyStructure,
                segmented_property_type=codes.SCT.LeftLung,
                algorithm_name=_ALGORITHM_NAME,
                algorithm_family=_ALGORITHM_FAMILY,
                algorithm_version=_ALGORITHM_VERSION,
            ),
            SegmentDescription(
                segment_label="Right Lung",
                segmented_property_category=codes.SCT.BodyStructure,
                segmented_property_type=codes.SCT.RightLung,
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
            # CCHMC v3.6 geometry fix: emit (W,H,D); declare explicitly so the
            # writer's ambiguous shape heuristic (square in-plane) cannot
            # transpose the in-plane axes.
            input_array_layout="whd",
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

        # --- Pool pre-allocation ---
        # Warm the RMM pool to the budget total at setup time so study 1's
        # per-tile allocations draw from the pool instead of cudaMalloc.
        gpu_bootstrap.warm_pool(plan.total_bytes)

        # --- Scheduler ---
        # Concurrent independent-fragment execution, ON by default
        # (flipped after the full verification suite passed with concurrency
        # enabled — all pixel-parity checks + SR + residency green).
        # Explicit HOLOSCAN_CONCURRENT_FRAGMENTS=0 restores the GreedyScheduler
        # serial behavior (byte-for-byte; verified by a serial parity run).
        if os.environ.get("HOLOSCAN_CONCURRENT_FRAGMENTS", "1") != "0":
            self.scheduler(EventBasedScheduler(self, worker_thread_number=5, name="concurrent"))
            self._logger.info("scheduler: EventBasedScheduler worker_thread_number=5 (concurrent)")
        else:
            self._logger.info("scheduler: default GreedyScheduler (serial)")

        logging.info(f"End {self.compose.__name__}")

    def _compose_serve(self):
        """Serve-mode compose : build ONCE at startup, parameterize at
        the I/O edges. The gRPC ingress operator replaces the batch DICOM
        loader position; the Serve* writers resolve their output folders per
        study from the StudyContext slot; the Egress operator collects the
        three writer done-ports and resolves the pending RPC. Everything in
        the middle (series selection, per-config subgraphs, ensemble,
        postprocess math, RMM warm pool, scheduler) is identical to the
        batch compose.

        NOTE: ``init_app_context`` resolves paths from argv/env — serve mode
        uses ONLY the model path (inputs/outputs are per-RPC), so the caller
        must provide a valid model path
        (HOLOSCAN_MODEL_PATH) and may leave HOLOSCAN_INPUT_PATH /
        HOLOSCAN_OUTPUT_PATH unset/empty (AppContext falls back to "";
        ModelFactory.create is the only path-dependent call at compose time).
        """
        # Serve-only imports stay LOCAL so the batch startup path never
        # imports gRPC/protobuf/serve modules. Dual package/flat idiom
        # (same as the module-level imports above).
        import json as _json
        from pathlib import Path as _Path

        try:  # package-style
            from my_app.operators.egress_operator import EgressOperator as _EgressOperator
            from my_app.operators.grpc_ingress_operator import GrpcIngressOperator as _GrpcIngressOperator
            from my_app.serve_state import STATE as _STATE
            from my_app.serve_writers import ServePostprocessOperator as _ServePostprocessOperator
            from my_app.serve_writers import ServeTimedDICOMSCWriterOperator as _ServeTimedDICOMSCWriterOperator
            from my_app.serve_writers import (
                ServeTimedDICOMSegmentationWriterOperator as _ServeTimedDICOMSegmentationWriterOperator,
            )
            from my_app.serve_writers import ServeTimedDICOMTextSRWriterOperator as _ServeTimedDICOMTextSRWriterOperator
        except ImportError:  # flat (my_app dir on sys.path)
            from operators.egress_operator import EgressOperator as _EgressOperator
            from operators.grpc_ingress_operator import GrpcIngressOperator as _GrpcIngressOperator
            from serve_state import STATE as _STATE
            from serve_writers import ServePostprocessOperator as _ServePostprocessOperator
            from serve_writers import ServeTimedDICOMSCWriterOperator as _ServeTimedDICOMSCWriterOperator
            from serve_writers import (
                ServeTimedDICOMSegmentationWriterOperator as _ServeTimedDICOMSegmentationWriterOperator,
            )
            from serve_writers import ServeTimedDICOMTextSRWriterOperator as _ServeTimedDICOMTextSRWriterOperator

        self._logger.info("Begin _compose_serve")

        # --- Preamble: identical to the batch compose ---
        # The "serve" router token is not an SDK argument (the batch arg parser has
        # no positional args), so strip it before init_app_context; in serve mode
        # input/output are per-RPC and paths come from env (HOLOSCAN_MODEL_PATH).
        serve_argv = [a for a in (self.argv or []) if a != "serve"]
        app_context = Application.init_app_context(serve_argv)
        model_path = _Path(app_context.model_path)

        models_subfolder = model_path / "models"
        if models_subfolder.exists() and models_subfolder.is_dir():
            self._logger.info(f"Found 'models' subfolder in {model_path}. Setting model_path to {models_subfolder}")
            model_path = models_subfolder

        backend = torch.cuda.memory.get_allocator_backend()
        self._logger.info("memory_allocator_backend: %s", backend)
        assert backend == "pluggable", (
            f"RMM torch allocator not active (backend={backend!r}); " "gpu_bootstrap must be imported before holoscan "
        )

        model_list_arg = os.environ.get("HOLOSCAN_MODEL_LIST")
        model_list_arg = [s.strip() for s in model_list_arg.split(",") if s.strip()] if model_list_arg else None
        plans = _json.loads((find_jsonpkls_dir(model_path) / "plans.json").read_text())
        run_list, ensemble_list = resolve_run_model_list(model_list_arg, plans, model_path)
        self._logger.info("run_model_list=%s ensemble_model_list=%s", list(run_list), list(ensemble_list))
        if model_list_arg is None:
            skipped = [c for c in (plans.get("configurations") or {}) if not (model_path / c).is_dir()]
            if skipped:
                self._logger.info("skipped configs without model dirs: %s", list(skipped))

        plan = self._compute_budget_plan(model_path, list(run_list))
        self._logger.info("memory_budget: %s", _json.dumps(asdict(plan)))

        # --- I/O edges (the ONLY serve differences vs batch) ---
        # Ingress replaces the batch DICOMDataLoaderOperator position:
        # hosts the gRPC server in-graph, admits exactly one study at a
        # time, and emits the same dicom_study_list message downstream.
        # / : the per-RPC admission gate receives the SAME
        # per-config budget mappings the compose-time plan used, so it can
        # recompute the study budget against live free VRAM at every
        # ProcessStudy admission (shape term = median upper bound; live
        # term = pynvml free bytes).
        ingress = _GrpcIngressOperator(
            self,
            name="grpc_ingress",
            budget_cfgs=self._budget_cfgs(model_path, list(run_list)),
            budget_model_path=model_path,
            budget_run_list=list(run_list),
        )

        series_selector_op = DICOMSeriesSelectorOperator(
            self,
            rules=Sample_Rules_Text,
            all_matched=True,
            sort_by_sop_instance_count=True,
            name="series_selector_op",
        )

        series_to_vol_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_op")

        # Placeholder output folders — the Serve* subclasses rewire
        # output_folder per study from the StudyContext slot at compute.
        _placeholder = _Path("/nonexistent/serve_placeholder")
        # TCV: same pkl-driven cleanup as the batch path (no top-k map).
        postprocess_op = _ServePostprocessOperator(
            self,
            model_path=model_path,
            applied_labels=(1, 2),
            keep_top_k=None,
            label_names=_DEFAULT_LABEL_NAMES,
            output_labels=tuple(OUTPUT_LABELS),
            output_folder=_placeholder,
            name="postprocess_op",
        )

        # --- Middle: verbatim from the batch compose ---
        # Serve mode only — publish model_load records to the
        # CheckHealth registry. Batch path never reaches here.
        try:
            from my_app.model_registry import models as _registry_models
            from my_app.model_registry import register as _register_models
        except ImportError:
            from model_registry import models as _registry_models
            from model_registry import register as _register_models
        self._model_load_hook = _register_models
        _validate_cascade_producers(list(run_list), plans)
        aux_prev_map = _auxiliary_prev_stages(list(run_list), plans)
        subgraphs: dict = {}
        for cfg in run_list:
            n_entry_inputs = 2 if cfg in aux_prev_map else 1
            sg = NnUnetConfigSubgraph(
                self,
                name=f"nnunet_{cfg}",
                model_path=model_path,
                config_name=cfg,
                n_entry_inputs=n_entry_inputs,
                emit_probabilities=(cfg in ensemble_list),
                emit_lowres_seg=(cfg in aux_prev_map.values()),
                serve_mode=True,
                cache_max_entries=SERVE_SHAPE_CACHE_MAX,
            )
            CudaStreamPool(
                sg,
                dev_id=0,
                stream_flags=1,
                reserved_size=1,
                nvtx_identifier=f"streams_{cfg}",
                name=f"cuda_stream_pool_{cfg}",
            )
            self.add_flow(series_to_vol_op, sg, {("image", "image")})
            subgraphs[cfg] = sg

        for consumer, prev in aux_prev_map.items():
            self.add_flow(subgraphs[prev], subgraphs[consumer], {("lowres_seg", "lowres_seg")})

        ensemble_op = EnsembleAverageOperator(
            self,
            config_names=list(ensemble_list),
            emit_averaged_probabilities=False,
            defer_strategy=(plan.strategy == "defer_to_incremental"),
            serve_mode=True,
            name="ensemble_average_op",
        )
        for cfg in ensemble_list:
            self.add_flow(subgraphs[cfg], ensemble_op, {("probabilities", f"prob_{cfg}")})
        self.add_flow(ensemble_op, postprocess_op, {("seg", "seg")})

        # --- Writers (Serve* subclasses; same constructor args as batch) ---
        # Segment descriptions for the TCV segments (3-label contract,
        # identical to the batch compose path).
        segment_descriptions = [
            SegmentDescription(
                segment_label="Heart",
                segmented_property_category=codes.SCT.BodyStructure,
                segmented_property_type=codes.SCT.Heart,
                algorithm_name=_ALGORITHM_NAME,
                algorithm_family=_ALGORITHM_FAMILY,
                algorithm_version=_ALGORITHM_VERSION,
            ),
            SegmentDescription(
                segment_label="Left Lung",
                segmented_property_category=codes.SCT.BodyStructure,
                segmented_property_type=codes.SCT.LeftLung,
                algorithm_name=_ALGORITHM_NAME,
                algorithm_family=_ALGORITHM_FAMILY,
                algorithm_version=_ALGORITHM_VERSION,
            ),
            SegmentDescription(
                segment_label="Right Lung",
                segmented_property_category=codes.SCT.BodyStructure,
                segmented_property_type=codes.SCT.RightLung,
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

        dicom_seg_writer = _ServeTimedDICOMSegmentationWriterOperator(
            self,
            segment_descriptions=segment_descriptions,
            model_info=my_model_info,
            custom_tags=custom_tags_seg,
            output_folder=_placeholder / "SEG",
            omit_empty_frames=False,
            # CCHMC v3.6 geometry fix (same as batch path): emit (W,H,D),
            # declare explicitly to bypass the ambiguous square-in-plane heuristic.
            input_array_layout="whd",
            name="dicom_seg_writer",
        )

        dicom_sr_writer = _ServeTimedDICOMTextSRWriterOperator(
            self,
            copy_tags=True,
            model_info=my_model_info,
            equipment_info=my_equipment,
            custom_tags=custom_tags_sr,
            output_folder=_placeholder / "SR",
            name="dicom_sr_writer",
        )

        dicom_sc_writer = _ServeTimedDICOMSCWriterOperator(
            self,
            model_info=my_model_info,
            equipment_info=my_equipment,
            custom_tags=custom_tags_sc,
            output_folder=_placeholder / "SC",
            name="dicom_sc_writer",
        )

        egress = _EgressOperator(self, ingress, name="egress")

        # --- DAG wiring (identical to batch) ---
        self.add_flow(ingress, series_selector_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(
            series_selector_op,
            series_to_vol_op,
            {("study_selected_series_list", "study_selected_series_list")},
        )
        self.add_flow(series_to_vol_op, postprocess_op, {("image", "image")})

        self.add_flow(
            series_selector_op,
            dicom_seg_writer,
            {("study_selected_series_list", "study_selected_series_list")},
        )
        self.add_flow(postprocess_op, dicom_seg_writer, {("seg", "seg_image")})

        self.add_flow(
            series_selector_op,
            dicom_sr_writer,
            {("study_selected_series_list", "study_selected_series_list")},
        )
        self.add_flow(postprocess_op, dicom_sr_writer, {("result_text", "text")})

        self.add_flow(
            series_selector_op,
            dicom_sc_writer,
            {("study_selected_series_list", "study_selected_series_list")},
        )
        self.add_flow(postprocess_op, dicom_sc_writer, {("dicom_sc_dir", "dicom_sc_dir")})

        # --- Serve-only edges: writer done-ports -> egress ---
        self.add_flow(dicom_seg_writer, egress, {("done", "seg_done")})
        self.add_flow(dicom_sr_writer, egress, {("done", "sr_done")})
        self.add_flow(dicom_sc_writer, egress, {("done", "sc_done")})

        # --- Pool pre-allocation + scheduler: identical to batch ---
        gpu_bootstrap.warm_pool(plan.total_bytes)

        # / (option C): measure the device pool headroom ONCE,
        # now that models are loaded and the RMM pool is warm. headroom =
        # device total - startup used - slack (MiB). A study's absolute peak
        # is startup_floor + working_set regardless of the current pool
        # floor (the grow-only pool reuses freed blocks for smaller studies
        # and extends up to this ceiling for larger ones), so this constant
        # is the correct admission constraint — NOT current free VRAM
        # (which sags after a large study yet admits no false positives on
        # reuse). NVML index follows CUDA_VISIBLE_DEVICES (F2 fix).
        # On pynvml failure the headroom stays None and the per-RPC gate
        # falls back to the plans-median flat estimate.
        try:
            from my_app.study_vram import nvml_device_index
        except ImportError:  # flat
            from study_vram import nvml_device_index

        try:
            import pynvml

            pynvml.nvmlInit()
            _mem = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(nvml_device_index()))
            _slack_mb = int(os.environ.get("BUDGET_CEILING_SLACK_MB", "512"))
            ingress.budget_headroom_mib = (_mem.total - _mem.used) / 2**20 - _slack_mb
            self._logger.info(
                "budget_ceiling: total=%.0f MiB startup_used=%.0f MiB slack=%d MiB -> headroom=%.0f MiB (index %d)",
                _mem.total / 2**20,
                _mem.used / 2**20,
                _slack_mb,
                ingress.budget_headroom_mib,
                nvml_device_index(),
            )
        except Exception as exc:  # noqa: BLE001 - gate degrades to flat estimate
            self._logger.warning(
                "budget_ceiling: pynvml unavailable (%s); per-RPC gate uses plans-median fallback", exc
            )

        if os.environ.get("HOLOSCAN_CONCURRENT_FRAGMENTS", "1") != "0":
            self.scheduler(EventBasedScheduler(self, worker_thread_number=5, name="concurrent"))
            self._logger.info("scheduler: EventBasedScheduler worker_thread_number=5 (concurrent)")
        else:
            self._logger.info("scheduler: default GreedyScheduler (serial)")

        # LAST: models loaded, graph built -> the daemon answers IDLE.
        _STATE.mark_ready()
        self._logger.info("grpc_serve: compose complete; status=IDLE (port bound at ingress init)")
        # structured lifecycle line — ready-at-IDLE.
        self._logger.info(
            "LIFECYCLE stage=ready mode=serve port=%d status=IDLE models=%d",
            int(os.environ.get("MAP_GRPC_PORT", "8765")),
            len(_registry_models()),
        )
        logging.info("End _compose_serve")


# Series selection rules (JSON). CT modality only, any slice thickness:
# downstream inference is thickness-agnostic (resampled per plans.json), and
# the largest matching series wins (sort_by_sop_instance_count).
Sample_Rules_Text = """
{
    "selections": [
        {
            "name": "TCV CT Series",
            "conditions": {
                "StudyDescription": "(.*?)",
                "Modality": "(?i)CT"
            }
        }
    ]
}
"""

if __name__ == "__main__":
    logging.info(f"Begin {__name__}")
    TCVNNUnetFastApp().run()
    logging.info(f"End {__name__}")
