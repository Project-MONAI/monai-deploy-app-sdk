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

"""In-graph gRPC ingress operator for the serve daemon ().

Hosts a stock-grpcio sync server on a thread pool INSIDE the Holoscan
graph, so CheckHealth/ProcessStudy answer as soon as the operator is
constructed (models are loaded during operator construction; the port
binds at graph-build time and the gRPC handler thread serves
CheckHealth with status=INITIALIZING until ``_compose_serve`` finishes).

Admission contract (, exactly ONE study in flight):

* gRPC handler threads do ZERO graph work — they validate (pure CPU:
  pydicom + os), reject fail-fast BEFORE enqueue (INVALID_ARGUMENT),
  guard duplicates (ALREADY_EXISTS) and queue-full (RESOURCE_EXHAUSTED),
  then enqueue and BLOCK on the study's done event (per-study timeout ->
  DEADLINE_EXCEEDED without releasing the graph).
* The single **admitter** daemon thread is the ONLY path from the queue
  into the graph: it sets the StudyContext slot, flips DaemonState to
  BUSY, and wakes the operator's AsynchronousCondition. It then blocks
  on the done event until Egress finalizes the study — serialization at
  the gRPC layer, never in the pipeline.
* Wake pattern: direct cross-thread event_state + notify (via the
  ReleaseGuard transport), verified 10k/10k, 0 lost.
"""

import logging

# --- committed pb2 stubs: flat-import convention --------------------
import os
import os as _os
import queue as pyqueue
import shutil
import sys
import sys as _sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import grpc
import pydicom

_PROTO_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "proto")
if _PROTO_DIR not in _sys.path:
    _sys.path.insert(0, _PROTO_DIR)
import map_service_pb2 as map_service_pb2
import map_service_pb2_grpc as map_service_pb2_grpc

# --- my_app helpers: dual package/flat import (app.py idiom) -----------------
try:  # package-style (my_app.*)
    from my_app import model_registry, serve_release
    from my_app.mem_budget import check_study_budget
    from my_app.serve_publish import discard_stage, purge_stale_stage, stage_dir_for
    from my_app.serve_scratch import ScratchError, create_study_scratch, purge_study_scratch, scratch_dir_for
    from my_app.serve_state import STATE
    from my_app.study_context import StudyContext, clear, get_current, set_current
    from my_app.study_vram import StudyShapeError, estimate_study_peak_mib, nvml_device_index
except ImportError:  # flat import (my_app dir on sys.path)
    from serve_state import STATE
    from study_context import StudyContext, set_current, get_current, clear
    from serve_scratch import create_study_scratch, ScratchError, purge_study_scratch, scratch_dir_for
    from serve_publish import purge_stale_stage, discard_stage, stage_dir_for
    from mem_budget import check_study_budget
    from study_vram import estimate_study_peak_mib, nvml_device_index, StudyShapeError
    import model_registry
    import serve_release

from monai.deploy.conditions import AsynchronousCondition, AsynchronousEventState
from monai.deploy.core import Operator, OperatorSpec
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator

__all__ = ["GrpcIngressOperator", "StudyEntry", "_load_dicom_studies"]


@dataclass
class StudyEntry:
    """One admitted study: context + RPC resolution handle.

    ``done_event`` is set by EXACTLY one site in the whole daemon:
    Egress ``_finalize`` (or this operator's load-failure path for
    studies that never enter the pipeline). ``error`` carries
    ``(grpc.StatusCode value, detail)``; ``response`` carries the
    successful ``StudyResponse``.
    """

    ctx: StudyContext
    done_event: threading.Event
    response: Optional[Any] = None  # map_service_pb2.StudyResponse
    error: Optional[Tuple[int, str]] = None  # (grpc.StatusCode, detail)
    timed_out: bool = False  # RPC gave up; pipeline keeps running to Egress


def _load_dicom_studies(input_path: Path) -> List[Any]:
    """Load DICOM studies via the SDK loader WITHOUT instantiating its
    Operator C++ base (an ordinary method on the load path, callable from
    our operator).

    ``DICOMDataLoaderOperator.__new__`` skips the C++ ``Operator.__init__``
    (verified in the build venv); ``load_data_to_studies`` only touches the
    plain Python attrs set below plus module-level pydicom helpers.
    """
    loader = DICOMDataLoaderOperator.__new__(DICOMDataLoaderOperator)
    loader._logger = logging.getLogger("grpc_ingress.dicom_loader")
    loader.input_path = input_path
    loader._must_load = True
    return DICOMDataLoaderOperator.load_data_to_studies(loader, input_path)


def _reject_budget(ctx, logger, detail: str):
    """/: a budget rejection fires AFTER 's
    ``create_study_scratch`` succeeded — purge the per-study scratch before
    the RPC aborts with RESOURCE_EXHAUSTED so no ``study_*`` dir survives the
    reject path.

    (A bare ``_abort(RESOURCE_EXHAUSTED)`` here would leak the study's
    scratch dir: the RPC dies before Egress is reached and the
    ``_finalize_entry`` site only fires on admitter load failures. Purge is
    idempotent/rmtree-ignore_errors, so a double purge with Egress is
    harmless. We deliberately do NOT purge in a ProcessStudy ``finally``:
    the timeout abort (DEADLINE_EXCEEDED) happens while the study is STILL
    running in the pipeline (its uid stays in _pending_uids through the
    drain window), so an unconditional finally-purge would race the live
    study; the pre-enqueue reject branch is the only safe site because the
    study was never admitted.)
    """
    purge_study_scratch(ctx.scratch_path)
    logger.info("budget_check REJECT uid=%s: %s", ctx.study_uid, detail)
    return grpc.StatusCode.RESOURCE_EXHAUSTED


class _MapServiceServicer(map_service_pb2_grpc.MapServiceServicer):
    """gRPC handler threads: validate, guard, enqueue, block — no CUDA/torch."""

    def __init__(self, ingress: "GrpcIngressOperator"):
        self._ingress = ingress
        self._logger = logging.getLogger(f"{__name__}._MapServiceServicer")

    def CheckHealth(self, request, context):  # noqa: N802 - gRPC servicer interface name
        # pure Python state reads (no ML/GPU-framework calls in this
        # thread — the architecture doc rule 1): DaemonState snapshot + the
        # model_load records published through the optional hook into
        # model_registry. Answers fast even while a study is BUSY.
        snap = STATE.snapshot()
        resp = map_service_pb2.HealthResponse(
            status=snap["status"],
            queue_depth=snap["queue_depth"],
            queue_capacity=snap["queue_capacity"],
            in_flight_study_uid=snap["in_flight_study_uid"] or "",
        )
        for m in model_registry.models():
            resp.models.add(
                config=str(m.get("config", "")),
                trainer=str(m.get("trainer", "")),
                folds=int(m.get("folds", 0)),
                patch_size=[int(x) for x in m.get("patch_size", [])],
            )
        return resp

    def ProcessStudy(self, request, context):  # noqa: N802 - gRPC servicer interface name
        # 2026-10-12 TOCTOU fix: the duplicate + queue-full guards (step 3)
        # and the mutating enqueue (step 5) used to be TWO separate lock
        # sections with
        # non-locked multi-second work in between (scratch create, stale-
        # stage purge, NVML budget check). Concurrent handler threads could
        # all pass the guards and then race: queue.Full leaked out of
        # put_nowait as UNKNOWN "internal handler error: Full: ", and a
        # second same-uid call overwrote _entries[uid], crossing responses
        # (GATE 1 4/6: queue_full + duplicate_uid FAIL). Now the re-checks
        # immediately before the mutation and the mutation itself are ONE
        # critical section, put_nowait is wrapped in try/except queue.Full
        # (typed RESOURCE_EXHAUSTED mentioning the queue, full rollback),
        # and Egress _finalize resolves the in-flight entry identity set by
        # the admitter instead of re-looking-up _entries[uid].
        ctx: Optional[StudyContext] = None
        entry: Optional[StudyEntry] = None
        # grpcio 1.83.1's context.abort sets the typed
        # code/details on the RPC state and then raises a BARE Exception
        # (grpc/_server.py `raise Exception`); the server's own handler
        # (_unary_response_in_pool) swallows it and honors state.aborted.
        # Our broad except below would otherwise catch that bare Exception
        # and re-abort UNKNOWN ("internal handler error: Exception: "),
        # masking every typed rejection. Track aborts and re-raise.
        aborted = False
        # True ONLY after a successful put_nowait in the
        # atomic section. The finally-release must be gated on this: an entry
        # object allocated but never enqueued (duplicate / queue-full typed
        # rejection) must NOT trigger the by-uid release below — that pop
        # would delete the IN-FLIGHT same-uid study's _entries claim and
        # _pending_uids membership (rerun2: first call DEADLINE_EXCEEDED at
        # 600 s, daemon stuck BUSY -> invalid_args health cascade).
        admitted = False

        def _abort(code: int, msg: str) -> None:
            nonlocal aborted
            aborted = True
            context.abort(code, msg)  # raises; typed code already stored

        try:
            # 1. Fail fast while the graph is still being built (
            # flips it to IDLE only at the end of _compose_serve).
            if STATE.status == "INITIALIZING":
                _abort(grpc.StatusCode.UNAVAILABLE, "daemon still initializing (models loading)")

            # 2. Validation BEFORE enqueue — pure CPU (pydicom + os only).
            ctx, err = self._ingress._validate(request)
            if err is not None:
                _abort(err[0], err[1])

            # 3+4. Duplicate and queue-full guards (same lock scope so the
            # pair is atomic against concurrent handler threads).
            with self._ingress._lock:
                if ctx.study_uid in self._ingress._pending_uids:
                    _abort(
                        grpc.StatusCode.ALREADY_EXISTS,
                        f"study {ctx.study_uid!r} already in flight or queued",
                    )
                if self._ingress.queue.full():
                    _abort(
                        grpc.StatusCode.RESOURCE_EXHAUSTED,
                        f"study queue full (depth={STATE.queue_capacity}); retry later",
                    )

            # 4b (): per-study scratch — create BEFORE enqueue
            # so no pipeline work ever touches a missing dir; typed reject
            # on free-space shortfall (never a raw OOM/IO error in the
            # reply). Rejections above (INVALID_ARGUMENT / ALREADY_EXISTS /
            # RESOURCE_EXHAUSTED queue) happen before this line and never
            # create a dir.
            # If a same-uid call is ALREADY
            # admitted (in flight or queued), that call owns the scratch
            # dir — skip creation (create_study_scratch purges existing
            # content, which would destroy the LIVE study's scratch mid-run)
            # and point at the existing dir; this call will be rejected
            # ALREADY_EXISTS in the atomic section below.
            with self._ingress._lock:
                _uid_pending = ctx.study_uid in self._ingress._pending_uids
            if _uid_pending:
                ctx.scratch_path = scratch_dir_for(ctx.study_uid)
            else:
                try:
                    scratch_path, _ = create_study_scratch(ctx.study_uid)
                except ScratchError as exc:
                    _abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
                ctx.scratch_path = scratch_path

            # 4c (): purge a stale staging dir left by a killed
            # run of this uid BEFORE the pipeline writes (idempotent re-submit).
            # The final output path is NOT touched here — publish_study handles
            # a pre-existing final tree atomically at publish time.
            purge_stale_stage(ctx.output_path, ctx.study_uid)

            # 4.7. Per-RPC memory budget (/ + shape-aware
            # upgrade): recompute at admission so a study that will not fit
            # is rejected cleanly (typed RESOURCE_EXHAUSTED) instead of
            # OOM'ing the long-lived daemon mid-pipeline (the OOM-survival
            # probe showed that failure bricks the gRPC server).
            # pynvml = NVML, not a CUDA context, so the gRPC thread stays
            # ML-free ( rule); the device index follows
            # CUDA_VISIBLE_DEVICES (F2 fix: NVML indices are physical and
            # ignore the env, so index 0 read the wrong GPU on the test box).
            # The decision is CEILING-based (shape-aware estimate vs the
            # compose-time pool headroom), not free-VRAM-based: the RMM pool
            # is grow-only, so after a large study free VRAM is low yet
            # smaller studies still fit from pool reuse. Free VRAM is logged
            # informationally. Header-read failure falls back to the
            # plans-median flat estimate.
            if self._ingress.budget_cfgs:
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    free_vram = pynvml.nvmlDeviceGetMemoryInfo(
                        pynvml.nvmlDeviceGetHandleByIndex(nvml_device_index())
                    ).free
                except Exception as exc:
                    self._logger.warning("budget_check skipped (pynvml unavailable: %s)", exc)
                    free_vram = None
                if free_vram is not None:
                    study_peak_mib = None
                    try:
                        est, est_detail = estimate_study_peak_mib(
                            ctx.input_path, self._ingress.budget_model_path, self._ingress.budget_run_list
                        )
                        study_peak_mib = est
                        per_cfg = est_detail.get("per_config") or {}
                        shapes = ", ".join(f"{c}={'x'.join(map(str, v['new_shape']))}" for c, v in per_cfg.items())
                        self._logger.info(
                            "budget_check uid=%s: study estimate ~%.0f MiB (vsum=%.1f M-vox; %s)",
                            ctx.study_uid,
                            est,
                            est_detail.get("vsum_mvox", 0.0),
                            shapes,
                        )
                    except StudyShapeError as exc:
                        self._logger.warning(
                            "budget_check uid=%s: shape estimate unavailable (%s); using plans-median fallback",
                            ctx.study_uid,
                            exc,
                        )
                    ok, detail = check_study_budget(
                        self._ingress.budget_cfgs,
                        free_vram_bytes=int(free_vram),
                        study_peak_mib=study_peak_mib,
                        headroom_mib=self._ingress.budget_headroom_mib,
                    )
                    if ok:
                        self._logger.info("budget_check uid=%s: %s", ctx.study_uid, detail)
                    else:
                        # reject-after-create must purge — see _reject_budget docstring.
                        _abort(_reject_budget(ctx, self._logger, detail), detail)

            # 5. Enqueue — ATOMIC guard+mutation (TOCTOU fix, 2026-10-12).
            # The step-3 checks run BEFORE the
            # un-locked scratch/purge/budget work below, so they are
            # re-checked under the lock here, in the SAME critical section
            # as the mutation. (Moving scratch-create/budget to AFTER
            # enqueue was rejected: the reject branches must run before the
            # admitter can dequeue the entry — _reject_budget's invariant.)
            entry = StudyEntry(ctx=ctx, done_event=threading.Event())
            with self._ingress._lock:
                if ctx.study_uid in self._ingress._pending_uids:
                    # DO NOT purge here. The uid is
                    # pending, so any scratch dir at this path belongs to the
                    # IN-FLIGHT/queued same-uid study, not to this rejected
                    # call — purging here destroyed the live study's scratch
                    # mid-run ("emit_npy skipped (output folder missing)").
                    # If THIS call created the dir at 4b it is an empty
                    # recreation at the same path; the owning study reuses
                    # it and purges it at its own finalize.
                    self._logger.info(
                        "admission REJECT uid=%s: duplicate (atomic section)",
                        ctx.study_uid,
                    )
                    _abort(
                        grpc.StatusCode.ALREADY_EXISTS,
                        f"study {ctx.study_uid!r} already in flight or queued",
                    )
                if self._ingress.queue.full():
                    # Re-check under the lock immediately before the
                    # mutation — the check that matters is this one.
                    # Rejection AFTER scratch creation must purge (same
                    # invariant as _reject_budget: reject-before-admit).
                    purge_study_scratch(ctx.scratch_path)
                    self._logger.info(
                        "admission REJECT uid=%s: queue full (depth=%s)",
                        ctx.study_uid,
                        STATE.queue_capacity,
                    )
                    _abort(
                        grpc.StatusCode.RESOURCE_EXHAUSTED,
                        f"study queue full (depth={STATE.queue_capacity}); retry later",
                    )
                try:
                    self._ingress._pending_uids.add(ctx.study_uid)
                    self._ingress._entries[ctx.study_uid] = entry
                    self._ingress.queue.put_nowait(entry)
                    admitted = True  # the ONLY flag that authorizes
                    # the by-uid release in finally below
                except pyqueue.Full:
                    # Defense in depth: if Full
                    # leaks through despite full under the same lock, roll
                    # back every mutation (no phantom _pending_uids/_entries
                    # state), purge the scratch, and reject TYPED — a
                    # leaked queue.Full must never reach the catch-all
                    # below as UNKNOWN.
                    self._ingress._pending_uids.discard(ctx.study_uid)
                    self._ingress._entries.pop(ctx.study_uid, None)
                    purge_study_scratch(ctx.scratch_path)
                    self._logger.info(
                        "admission REJECT uid=%s: queue.Full on put (depth=%s)",
                        ctx.study_uid,
                        STATE.queue_capacity,
                    )
                    _abort(
                        grpc.StatusCode.RESOURCE_EXHAUSTED,
                        f"study queue full (depth={STATE.queue_capacity}); retry later",
                    )
                STATE.set_queue_depth(self._ingress.queue.qsize())

            # 6. Long-RPC wait with per-study timeout. On timeout: abort
            # DEADLINE_EXCEEDED, but NEVER release the study's done event
            # here — the graph is not cancellable, so the study keeps
            # running; Egress _finalize is the sole post-timeout release
            # site (it purges, clears the slot, and unblocks the admitter).
            timeout = float(os.environ.get("MAP_GRPC_STUDY_TIMEOUT", "900"))
            finished = entry.done_event.wait(timeout=timeout)
            if not finished:
                entry.timed_out = True
                entry.error = (grpc.StatusCode.DEADLINE_EXCEEDED, f"study exceeded {timeout:.0f}s")

            # 7. Resolve: typed abort on error, else the Egress-built response.
            if entry.error is not None:
                _abort(entry.error[0], entry.error[1])
                return None
            return entry.response
        except Exception as exc:
            if aborted:
                # Typed rejection already applied (state.code/details set
                # before abort raised) — propagate so the server finalizes
                # with OUR code, not a masked UNKNOWN.
                raise
            # Never let a handler exception propagate into the gRPC thread
            # pool — convert to a typed UNKNOWN abort.
            context.abort(grpc.StatusCode.UNKNOWN, f"internal handler error: {type(exc).__name__}: {exc}")
        finally:
            # UID/entry release — ONLY for studies this handler actually
            # ADMISSION-ENQUEUED (admitted) that have NOT timed out.
            # (a) A typed-rejection path allocates
            # the entry but never enqueues it; without the admitted gate
            # the by-uid pop/discard below deleted the IN-FLIGHT same-uid
            # study's bookkeeping (orphaned done_event -> 600 s deadline;
            # end_study never reached -> daemon stuck BUSY).
            # (b) On timeout the study is still mid-pipeline until Egress
            # drains it: its uid must STAY in _pending_uids through the
            # drain window (a duplicate ProcessStudy(same uid) before
            # Egress finalizes must still be rejected ALREADY_EXISTS),
            # and the entry must STAY in _entries so Egress can reach
            # its done event. Egress _finalize is the uid-release site
            # for admitted studies (this block is a no-op afterwards:
            # Egress popped the entry and discarded the uid first).
            # The pop is identity-checked as belt-and-braces.
            if admitted and entry is not None and not entry.timed_out:
                with self._ingress._lock:
                    if self._ingress._entries.get(ctx.study_uid) is entry:
                        self._ingress._entries.pop(ctx.study_uid, None)
                    self._ingress._pending_uids.discard(ctx.study_uid)
                    STATE.set_queue_depth(self._ingress.queue.qsize())


class GrpcIngressOperator(Operator):
    """In-graph gRPC server + serial study admitter (serve mode only).

    Replaces the batch ``DICOMDataLoaderOperator`` position: the same
    ``dicom_study_list`` output port feeds the unchanged
    series_selector downstream of it.
    """

    def __init__(
        self,
        fragment: Any,
        name: str = "grpc_ingress",
        port: Optional[int] = None,
        payloads_root: Optional[Any] = None,
        queue_depth: Optional[int] = None,
        budget_cfgs: Optional[List[dict]] = None,
        budget_model_path: Optional[Any] = None,
        budget_run_list: Optional[List[str]] = None,
    ):
        # NOTE: holoscan 4.2's Operator.__init__ invokes self.setup(spec)
        # before this constructor body finishes — initialize ALL state first
        # (same pattern as the in-repo operators).
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.port = int(port if port is not None else os.environ.get("MAP_GRPC_PORT", "8765"))
        self.payloads_root = Path(payloads_root or os.environ.get("MAP_PAYLOADS_ROOT", "/payloads"))
        depth = int(queue_depth if queue_depth is not None else os.environ.get("MAP_GRPC_QUEUE_DEPTH", "4"))
        self.queue_capacity = depth
        STATE.set_queue_capacity(depth)
        # / : per-RPC admission budget (same cfg mappings as the
        # compose-time plan); None disables the per-RPC gate (batch/tests).
        self.budget_cfgs = budget_cfgs
        # / (option C): shape-aware estimate inputs. model_path
        # + run_list drive my_app.study_vram's header-only resampled-shape
        # math; budget_headroom_mib is set by app._compose_serve after the
        # RMM warm pool (device total - startup used - slack, in MiB). None
        # -> check_study_budget falls back to the flat estimate.
        self.budget_model_path = budget_model_path
        self.budget_run_list = budget_run_list
        self.budget_headroom_mib: Optional[float] = None

        # Woken from the admitter thread; starts parked (a fresh
        # AsynchronousCondition actually begins READY and fires once
        # spuriously at graph start — compute re-parks on the empty slot,
        # so the phantom fire is absorbed).
        self.condition = AsynchronousCondition(fragment, name="ingress_cond")
        self.condition.event_state = AsynchronousEventState.EVENT_WAITING

        self._lock = threading.Lock()
        self._pending_uids: Set[str] = set()  # in flight OR queued (ALREADY_EXISTS guard)
        self._entries: Dict[str, StudyEntry] = {}
        # 2026-10-12 TOCTOU fix: the entry the admitter actually dequeued —
        # the identity Egress _finalize / _finalize_entry must resolve
        # against (never a re-look-up of _entries[uid]).
        self._in_flight: Optional[StudyEntry] = None
        self.queue: "pyqueue.Queue[StudyEntry]" = pyqueue.Queue(maxsize=depth)

        self._grpc_server: Optional[grpc.Server] = None
        super().__init__(fragment, self.condition, name=name)

        # Stock-grpcio sync server on a 4-worker pool. Bind + start happen
        # at graph-build time so CheckHealth answers INITIALIZING while
        # models load ( flips the status to IDLE).
        self._grpc_server = grpc.server(ThreadPoolExecutor(max_workers=4))
        self._servicer = _MapServiceServicer(self)
        map_service_pb2_grpc.add_MapServiceServicer_to_server(self._servicer, self._grpc_server)
        bound = self._grpc_server.add_insecure_port(f"0.0.0.0:{self.port}")
        if bound == 0:
            raise RuntimeError(f"grpc_ingress: failed to bind 0.0.0.0:{self.port}")
        self._grpc_server.start()
        self._logger.info(
            "grpc_ingress: serving map_service on 0.0.0.0:%d (queue_depth=%d, payloads_root=%s)",
            self.port,
            depth,
            self.payloads_root,
        )

        # the single guarded release site — the admitter never touches
        # event_state directly; every wake goes through the set-once guard
        # with this operator's fact-B transport as the wake mechanics.
        serve_release.GUARD = serve_release.ReleaseGuard(self._wake)

        # The admitter is the ONLY study-admission path; its wake is the
        # guard's single release site.
        self._admitter = threading.Thread(target=self._admitter_loop, name="grpc_admitter", daemon=True)
        self._admitter.start()

    # -- graph surface ---------------------------------------------------------

    def setup(self, spec: OperatorSpec) -> None:
        # Same message type + port name the batch loader emits — everything
        # downstream of the series selector is unmodified.
        spec.output("dicom_study_list")

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        ctx = get_current()
        if ctx is None:
            # Spurious startup fire (caveat 1) — the admitter only wakes
            # after set_current, so an empty slot means no admission.
            self._logger.warning("grpc_ingress: compute with no study in flight; re-parking")
            self._repark()
            return

        # Dev-gate fault injection : reject at load, never touch the
        # pipeline.
        if os.environ.get("MAP_GRPC_FAULT_INJECT") == "ingress_load":
            ctx.record_failure("ingress_load", "MAP_GRPC_FAULT_INJECT: injected load failure")
            self._finalize_entry(ctx, ok=False)
            self._repark()
            return

        try:
            studies = _load_dicom_studies(ctx.input_path)
        except Exception as exc:
            # The RPC must never die silently: typed failure -> Egress-less
            # resolve path (load happens here, before the pipeline runs).
            ctx.record_failure("ingress_load", f"{type(exc).__name__}: {exc}")
            self._logger.exception("grpc_ingress: load failed for study %s", ctx.study_uid)
            self._finalize_entry(ctx, ok=False)
            self._repark()
            return

        op_output.emit(studies, "dicom_study_list")
        self._repark()  # parked until the NEXT admission; slot cleared by Egress

    # -- internals --------------------------------------------------------------

    def _repark(self) -> None:
        self.condition.event_state = AsynchronousEventState.EVENT_WAITING

    def _wake(self, condition: Optional[Any] = None) -> None:
        """Wake the given operator condition from this
        (non-scheduler) thread. Called ONLY via ReleaseGuard.release — the
        guard owns the set-once registry; this owns the wake mechanics."""
        cond = condition if condition is not None else self.condition
        # Direct cross-thread event_state + notify, verified 10,000/10,000
        # wakes with 0 lost / 0 double-fires (p50 0.219 ms).
        cond.event_state = AsynchronousEventState.EVENT_DONE
        cond.notify_scheduler()  # transport
        if False:
            # Unchosen alternative: marshal every wake through ONE dedicated
            # dispatcher thread (single queue of (condition, state) pairs
            # drained sequentially).
            self._wakes.put((cond, AsynchronousEventState.EVENT_DONE))

    def _finalize_entry(self, ctx: StudyContext, ok: bool) -> None:
        """Ingress-side resolve path (load failure / fault injection).

        Releases the slot and DaemonState BEFORE unblocking the admitter:
        the admitter proceeds straight to the next admission and must not
        see an occupied slot. (The full Egress-side finalize lives in
        egress_operator.py and is the other, mutually exclusive, release
        site.)
        """
        with self._lock:
            # Entry identity = the in-flight entry the admitter dequeued
            # (2026-10-12 TOCTOU fix) — the dict pop stays as the claim.
            claimed = self._entries.pop(ctx.study_uid, None)
            entry = self._in_flight or claimed
            self._in_flight = None
            self._pending_uids.discard(ctx.study_uid)
        # on failure ALWAYS purge the partial output dir (never
        # leave a half SEG tree). : writers never ran on this
        # path — discard the (empty) stage; final-path clear preserved for
        # pre-created dirs.
        discard_stage(stage_dir_for(ctx.output_path, ctx.study_uid))
        shutil.rmtree(ctx.output_path, ignore_errors=True)
        # the ingress-side exit path (load failure / fault injection)
        # never reaches Egress, so it purges the scratch dir itself.
        purge_study_scratch(ctx.scratch_path)
        if entry is not None and not entry.timed_out:
            stage, detail = ctx.failure if ctx.failure is not None else ("ingress_load", "unknown failure")
            entry.error = (grpc.StatusCode.INTERNAL, f"stage_at_failure={stage}; {detail}")
        clear()
        STATE.end_study()
        STATE.set_queue_depth(self.queue.qsize())
        if entry is not None:
            # LAST: unblock the gRPC handler (non-timeout) / admitter
            # (timeout drain) only after slot + STATE are released.
            entry.done_event.set()

    def _admitter_loop(self) -> None:
        """Serial admitter: admit exactly one study, block until Egress
        finalizes it, repeat. The block on the done event IS the
        one-study-at-a-time guarantee."""
        while True:
            entry = self.queue.get()
            with self._lock:
                self._in_flight = entry  # identity handoff for Egress _finalize
            set_current(entry.ctx)  # RuntimeError on a race = a bug: crash
            STATE.begin_study(entry.ctx.study_uid)  # BUSY the instant of admission, before the wake
            serve_release.GUARD.release(entry, self.condition)  # single release site, set-once
            entry.done_event.wait()  # released by Egress _finalize (sole site
            # for the graph path) or _finalize_entry (ingress-side failure)
            serve_release.GUARD.forget(entry)  # frees the set-once slot for the next entry object

    def _validate(
        self, req: "map_service_pb2.StudyRequest"
    ) -> Tuple[Optional[StudyContext], Optional[Tuple[int, str]]]:
        """Pure-CPU fail-fast validation (gRPC handler thread, BEFORE
        enqueue). Returns (ctx, None) on success, (None, (code, detail))
        on rejection. Paths are canonicalized with resolve and checked
        under the payloads root with is_relative_to (path-traversal guard:
        rejects '..' and symlinks escaping the root)."""
        for fname in ("study_uid", "input_path", "output_path"):
            if not getattr(req, fname):
                return None, (grpc.StatusCode.INVALID_ARGUMENT, f"missing field: {fname}")

        root = self.payloads_root.resolve()
        try:
            p_in = Path(req.input_path).resolve()
            p_out = Path(req.output_path).resolve()
        except (OSError, RuntimeError) as exc:
            return None, (grpc.StatusCode.INVALID_ARGUMENT, f"unresolvable path: {exc}")
        if not p_in.is_relative_to(root):
            return None, (grpc.StatusCode.INVALID_ARGUMENT, f"{req.input_path} is not under payloads root {root}")
        if not p_out.is_relative_to(root):
            return None, (grpc.StatusCode.INVALID_ARGUMENT, f"{req.output_path} is not under payloads root {root}")
        # The failure path in egress removes output_path wholesale, so it
        # must be a DEDICATED directory: never the payloads root itself and
        # never overlapping input_path (which would delete the source DICOM
        # tree or let a nested input survive inside a deleted output).
        if p_out == root:
            return None, (
                grpc.StatusCode.INVALID_ARGUMENT,
                "output_path must be a dedicated subdirectory, not the payloads root",
            )
        if p_out == p_in or p_out.is_relative_to(p_in) or p_in.is_relative_to(p_out):
            return None, (
                grpc.StatusCode.INVALID_ARGUMENT,
                "output_path must not overlap input_path (failure cleanup removes output_path)",
            )
        if not p_in.is_dir():
            return None, (grpc.StatusCode.INVALID_ARGUMENT, "input_path does not exist")
        dcm_files = sorted(f for f in p_in.rglob("*") if f.suffix.lower() in (".dcm", ".dicom"))
        if not dcm_files:
            return None, (grpc.StatusCode.INVALID_ARGUMENT, "no DICOM instances under input_path")
        try:
            ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
            on_disk = str(ds.StudyID).strip()
        except Exception as exc:
            return None, (
                grpc.StatusCode.INVALID_ARGUMENT,
                f"cannot read StudyID from {dcm_files[0].name}: {type(exc).__name__}: {exc}",
            )
        if on_disk != req.study_uid.strip():
            return None, (
                grpc.StatusCode.INVALID_ARGUMENT,
                f"study_uid mismatch: request={req.study_uid!r} on-disk={on_disk!r}",
            )
        if p_out.exists() and not p_out.is_dir():
            return None, (grpc.StatusCode.INVALID_ARGUMENT, "output_path exists and is a file, not a directory")
        return (
            StudyContext(
                study_uid=req.study_uid,
                input_path=p_in,
                output_path=p_out,
                t_start=time.monotonic(),
            ),
            None,
        )
