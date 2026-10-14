# Copyright 2021-2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve-only Postprocess/writer subclasses (, /).

These wrap the BATCH classes in app.py (NEVER modified) and add three
serve-mode behaviors per writer:

1. Per-study STAGING folders resolved from the StudyContext slot at compute
   time: ``<output_path>.tmp-<study_uid>/<SUBDIR>`` (a construction-time
   path must never be reused across studies — the construction-time folder
   is only a placeholder). Egress publishes the staging subdirs to the
   final output path atomically on success (serve_publish.publish_study)
   — the final path is never written mid-pipeline.
2. A ``done`` output port so the Egress operator can match all three
   writers' completion (success OR failure) and finalize the study.
3. Fault injection: ``MAP_GRPC_FAULT_INJECT=write_seg`` (or write_sr /
   write_sc) records a typed failure at that stage and still emits
   ``done`` — a swallowed failure must not strand the Egress (deadlock
   class).

Constraint: ``ServePostprocessOperator`` has NO done port and NO fault
stage — in holoscan 4.2 an operator exception kills the graph, so
postprocess failures surface as a graph crash (fault injection covers
writer stages only).
"""

import os

try:  # package-style (my_app.*)
    from my_app.app import (
        TimedDICOMSCWriterOperator,
        TimedDICOMSegmentationWriterOperator,
        TimedDICOMTextSRWriterOperator,
    )
    from my_app.operators import PostprocessOperator
    from my_app.serve_publish import stage_dir_for
    from my_app.study_context import get_current
except ImportError:  # flat import (my_app dir on sys.path)
    from app import (
        TimedDICOMSCWriterOperator,
        TimedDICOMSegmentationWriterOperator,
        TimedDICOMTextSRWriterOperator,
    )
    from operators import PostprocessOperator
    from serve_publish import stage_dir_for
    from study_context import get_current

__all__ = [
    "ServePostprocessOperator",
    "ServeTimedDICOMSegmentationWriterOperator",
    "ServeTimedDICOMTextSRWriterOperator",
    "ServeTimedDICOMSCWriterOperator",
]


class _ServeWriterMixin:
    """Shared serve-mode writer behavior (mixes in FIRST so its
    setup/compute win; super chains land on the Timed* batch classes,
    preserving the existing NVTX + timing record flow)."""

    _serve_stage: str = ""
    _serve_output_subdir: str = ""

    def setup(self, spec):
        super().setup(spec)
        spec.output("done")

    def compute(self, op_input, op_output, context):
        ctx = get_current()
        if ctx is None:
            raise RuntimeError(f"{type(self).__name__}: compute with no study in flight (slot empty)")
        # Per-study output folder — resolved on EVERY compute, never the
        # construction-time placeholder.
        # Write into the per-study STAGING dir; Egress
        # publish_study atomically renames each subdir to the final output path on success —
        # the final path is never written mid-pipeline (no partial tree on kill).
        self.output_folder = stage_dir_for(ctx.output_path, ctx.study_uid) / self._serve_output_subdir
        if os.environ.get("MAP_GRPC_FAULT_INJECT") == self._serve_stage:
            ctx.record_failure(self._serve_stage, f"MAP_GRPC_FAULT_INJECT: injected {self._serve_stage} failure")
            # Consume this study's writer input before emitting done: an
            # unconsumed queued message would let the message condition
            # re-fire compute (duplicate completions, and a later call
            # running against a different study's slot).
            try:
                op_input.receive()
            except Exception:  # noqa: BLE001 - defensive; emit done either way
                self._logger.warning("%s: fault-inject: no writer input to consume", type(self).__name__)
            op_output.emit({"study_uid": ctx.study_uid, "stage": self._serve_stage}, "done")
            return None
        try:
            return super().compute(op_input, op_output, context)
        except Exception as exc:
            # Record the typed failure and let Egress resolve a typed INTERNAL
            # (the study's partial outputs are purged there); swallowing a
            # writer failure keeps the persistent graph alive for the next
            # study instead of crashing it.
            ctx.record_failure(self._serve_stage, f"{type(exc).__name__}: {exc}")
        finally:
            # ALWAYS emit done — Egress finalizes (success or typed failure
            # + scratch purge) either way; a swallowed failure must not
            # strand the Egress (deadlock class).
            # NOTE: holoscan 4.2 OutputContext.emit signature is (data, name).
            op_output.emit({"study_uid": ctx.study_uid, "stage": self._serve_stage}, "done")


class ServePostprocessOperator(PostprocessOperator):
    """Serve-mode postprocess: output_folder resolved per study from the
    StudyContext slot (covers the SC temp dir at output_path/"temp").

    No done port, no fault stage — see module docstring (postprocess
    failures kill the graph in 4.2).
    """

    def compute(self, op_input, op_output, context):
        ctx = get_current()
        if ctx is None:
            raise RuntimeError("ServePostprocessOperator: compute with no study in flight (slot empty)")
        # route the SC overlay temp dir (output_folder/"temp") into the
        # per-study scratch, so the overlay .dcm intermediate lands in the
        # study's scratch dir and is purged with the study. Final
        # SEG/SR/SC outputs are unaffected: the Serve* writer mixins resolve
        # their own output folders from ctx on every compute.
        self.output_folder = ctx.scratch_path if ctx.scratch_path is not None else ctx.output_path
        return super().compute(op_input, op_output, context)


class ServeTimedDICOMSegmentationWriterOperator(_ServeWriterMixin, TimedDICOMSegmentationWriterOperator):
    _serve_stage = "write_seg"
    _serve_output_subdir = "SEG"


class ServeTimedDICOMTextSRWriterOperator(_ServeWriterMixin, TimedDICOMTextSRWriterOperator):
    _serve_stage = "write_sr"
    _serve_output_subdir = "SR"


class ServeTimedDICOMSCWriterOperator(_ServeWriterMixin, TimedDICOMSCWriterOperator):
    _serve_stage = "write_sc"
    _serve_output_subdir = "SC"
