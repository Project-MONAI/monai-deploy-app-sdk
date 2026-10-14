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

"""Egress operator: study completion -> RPC resolution ().

Completion = the three writer ``done`` ports (SEG + SR + SC). This is the
only place that can see writer success/failure — postprocess cannot (it
runs before the writers; see the architecture doc (b)).

Because operator exceptions kill the graph in holoscan 4.2, finalize is
defensive: try/finally, no raises after the study slot is handed over.

Timeout semantics (pairs with grpc_ingress_operator.ProcessStudy): on an
RPC timeout the handler aborts DEADLINE_EXCEEDED but does NOT release the
study — this operator is the SOLE post-timeout release site (it purges,
clears the slot, ends the study, and unblocks the admitter when the
pipeline truly finishes).
"""

import logging
import os as _os
import shutil
import sys as _sys
import threading
import time
from typing import Any

import grpc

# --- committed pb2 stubs: flat-import convention --------------------
_PROTO_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "proto")
if _PROTO_DIR not in _sys.path:
    _sys.path.insert(0, _PROTO_DIR)
import map_service_pb2 as map_service_pb2

# --- my_app helpers: dual package/flat import (app.py idiom) -----------------
try:  # package-style (my_app.*)
    from my_app.operators.gpu_util import StudyTimingCollector
    from my_app.serve_publish import discard_stage, publish_study, stage_dir_for
    from my_app.serve_scratch import purge_study_scratch
    from my_app.serve_state import STATE
    from my_app.study_context import StudyContext
    from my_app.study_context import clear as study_context_clear
    from my_app.study_context import get_current
except ImportError:  # flat import (my_app dir on sys.path)
    from operators.gpu_util import StudyTimingCollector
    from serve_state import STATE
    from study_context import StudyContext, clear as study_context_clear, get_current
    from serve_scratch import purge_study_scratch
    from serve_publish import stage_dir_for, publish_study, discard_stage

from monai.deploy.core import Operator, OperatorSpec

__all__ = ["EgressOperator"]

_DONE_PORTS_REQUIRED = 3  # seg_done + sr_done + sc_done
_DONE_PORT_STAGE = {"seg_done": "write_seg", "sr_done": "write_sr", "sc_done": "write_sc"}
_ALL_STAGES = frozenset(_DONE_PORT_STAGE.values())


class EgressOperator(Operator):
    """Matches finished pipeline states to pending RPC futures.

    Inputs: ``seg_done`` / ``sr_done`` / ``sc_done`` (one per writer).
    No output ports.

    Completion wiring: a CountCondition(1) latches after its first fire for
    the graph's lifetime and a latched sink backpressures upstream, so a
    per-study CountCondition gate is NOT usable in a long-lived serve graph.
    This operator therefore fires per received done message (default
    condition) and keeps a per-study flag counter; it finalizes only when
    all three writers have reported.
    """

    def __init__(self, fragment: Any, ingress: Any, name: str = "egress"):
        # NOTE: holoscan 4.2's Operator.__init__ invokes self.setup(spec)
        # before this constructor body finishes — initialize all state first.
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._ingress = ingress
        # Design (pattern B): no gate condition — per-message fires +
        # per-study stage set with an atomic claim at finalize. The stage
        # set lives on the per-RPC StudyContext (ctx.serve_seen_stages), NOT
        # keyed by uid string: the admitter legitimately reuses the same
        # StudyID across studies, and a uid-keyed guard permanently blocked
        # the second study with the same uid (observed admission deadlock).
        self._counts_lock = threading.Lock()
        super().__init__(fragment, name=name)
        if False:
            # Unchosen alternative (pattern A): a CountCondition(3) gate
            # would fire compute once per study when all three done ports
            # arrived; valid only if CountCondition re-armed per cycle — a
            # direct experiment disproved that.
            self._done_counts = CountCondition(self, 3)  # noqa: F821 — unreachable

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("seg_done")
        spec.input("sr_done")
        spec.input("sc_done")

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        # Fix (root cause of a ~560 fires/s re-fire hot loop and phantom
        # ~1 ms "OK" finalizations): holoscan 4.2's event scheduler re-fires
        # an operator whenever its condition re-evaluates TRUE, and a done
        # message that compute never receives STAYS in the receiver queue.
        # A CPU-only probe confirmed the unconsumed message re-ticks the
        # operator (unconsumed_retick=True) and that consuming it fixes the
        # loop (consume_fixes=True).
        # Every fire therefore DRAINS all three done ports (receive consumes;
        # it returns None when a port is empty), and finalization is keyed on
        # DISTINCT (uid, stage) pairs so any residual re-delivery is a no-op.
        ctx = get_current()
        stages: set = set()
        for port, stage in _DONE_PORT_STAGE.items():
            while True:
                msg = op_input.receive(port)
                if msg is None:
                    break
                if not isinstance(msg, dict):
                    stages.add(stage)
                    continue
                # Consume-and-ignore stale dones from a previous study; only
                # the in-flight study's own done messages count.
                muid = msg.get("study_uid")
                if ctx is not None and muid is not None and muid != ctx.study_uid:
                    continue
                stages.add(msg.get("stage") or stage)
        if not stages:
            return  # spurious fire with nothing queued
        if ctx is None:
            # A done message with no in-flight study means a stale message
            # (post-timeout drain / re-delivery after finalize) — drop, never
            # crash the graph on bookkeeping.
            self._logger.error("egress: done message with no study in flight; dropping %s", sorted(stages))
            return
        with self._counts_lock:
            ctx.serve_seen_stages |= stages
            complete = _ALL_STAGES <= ctx.serve_seen_stages
        if complete:
            self._finalize(ctx)

    def _finalize(self, ctx: StudyContext) -> None:
        """Resolve the study's pending RPC (runs on a scheduler thread —
        NO blocking calls).

        The ENTRY-POP below is the exactly-once claim: only one concurrent
        finalize can obtain a non-None entry, so all release bookkeeping
        (response/error, purge, slot clear, end_study, done_event) runs once.
        An RPC timeout can NOT leave ``entry is None`` here, because the
        handler's finally does not pop timed-out entries. ``STATE`` stays
        BUSY until this ``end_study`` — i.e., until the pipeline truly
        finishes.
        """
        # POP + UID-RELEASE under the ingress lock — this is the study's
        # sole finalize bookkeeping site for the graph path. The handler's
        # finally pops AND discards the uid only when NOT timed_out, so a
        # timed-out entry is still here when its pipeline completes, its
        # done event is still settable, and its uid was NOT released by the
        # handler (a duplicate ProcessStudy(same uid) during the drain window
        # was still ALREADY_EXISTS). The discard mirrors the pop: the uid
        # becomes re-requestable ONLY once the study has truly drained; for
        # non-timed-out studies it is an idempotent no-op (the handler
        # already released it).
        with self._ingress._lock:
            # Entry identity = the entry the admitter ACTUALLY dequeued
            # (2026-10-12 TOCTOU fix): never
            # re-look-up _entries[uid] as the identity — that dict is only
            # the exactly-once claim (and, pre-fix, could be overwritten by
            # a racing same-uid enqueue, crossing responses).
            claimed = self._ingress._entries.pop(ctx.study_uid, None)
            entry = self._ingress._in_flight or claimed
            self._ingress._in_flight = None
            self._ingress._pending_uids.discard(ctx.study_uid)
            STATE.set_queue_depth(self._ingress.queue.qsize())
        # Guard on ENTRY identity, not on the dict
        # claim. claimed can be None while a valid entry is still reachable
        # via the _in_flight identity handoff (e.g. a lost dict-claim); the
        # only true "already resolved" state is entry IS None (both the
        # dict pop and _in_flight returned nothing).
        if entry is None:
            # A concurrent egress fire (or the Ingress-side _finalize_entry
            # on a load failure) already resolved this study and performed
            # its release bookkeeping — nothing left to do. (The uid-keyed
            # stale-done guard that lived here before the admission-deadlock
            # fix is gone: the entry pop is the claim, and it is correct
            # across uid reuse.)
            self._logger.warning(
                "egress: finalize for %s with no entry — already resolved; skipping",
                ctx.study_uid,
            )
            return
        latency_ms = (time.monotonic() - ctx.t_start) * 1000.0
        records = StudyTimingCollector.take(self.fragment)  # atomic pop
        if not entry.timed_out:
            if ctx.failure is not None:
                entry.error = (
                    grpc.StatusCode.INTERNAL,
                    f"stage_at_failure={ctx.failure[0]}; {ctx.failure[1]}",
                )
            else:
                resp = map_service_pb2.StudyResponse(status="OK", latency_ms=latency_ms)
                for r in records:
                    resp.stages.add(
                        stage=r.get("operator", "unknown"),
                        duration_ms=float(r.get("duration_ms", 0.0)),
                    )
                entry.response = resp
        # Atomic output publish. Writers wrote into the per-study staging
        # dir <output>.tmp-<uid> (serve_writers); the final output path is
        # touched ONLY here. A kill before this line therefore leaves no
        # partial tree at the output path (crash-safety).
        # NOTE: keyed on ctx.failure, NOT entry.timed_out — a timed-out RPC
        # whose pipeline completed still publishes (work is done; the next
        # re-submit overwrites atomically).
        stage = stage_dir_for(ctx.output_path, ctx.study_uid)
        try:
            if ctx.failure is not None:
                # (preserved): a failed run leaves NO outputs — stage discarded
                # and the final path cleared (callers who pre-created it stay clean).
                discard_stage(stage)
                shutil.rmtree(ctx.output_path, ignore_errors=True)
            else:
                publish_study(stage, ctx.output_path)
        except OSError as exc:
            # A publish/cleanup OSError (permission denied, read-only mount,
            # stage missing) must NOT escape _finalize: the entry claim was
            # popped above, so ONLY this site performs the release bookkeeping
            # below. Escaping would skip it — the admitter would then wait on
            # done_event forever (daemon bricked). Convert to a typed
            # INTERNAL and discard the pre-built success response.
            self._logger.error(
                "egress: output publish failed for %s: %s: %s",
                ctx.study_uid,
                type(exc).__name__,
                exc,
            )
            if not entry.timed_out:
                entry.response = None
                entry.error = (
                    grpc.StatusCode.INTERNAL,
                    f"publish_failed: {type(exc).__name__}: {exc}",
                )
        # scratch vanishes on every finalize path. _finalize is the
        # sole post-pipeline site (success AND failure AND timeout drain), so
        # the purge is UNCONDITIONAL — unlike the output rmtree above.
        purge_study_scratch(ctx.scratch_path)
        study_context_clear()
        STATE.end_study()
        # LAST, unconditionally (incl. timed_out): the SOLE site that
        # unblocks the admitter after an RPC timeout — the gRPC handler
        # never does; slot + STATE are already released above, so the
        # admitter can admit the next study without seeing an occupied slot.
        entry.done_event.set()
        self._logger.info(
            "egress: study %s finalized%s",
            ctx.study_uid,
            "; RPC already timed out" if entry.timed_out else "",
        )
