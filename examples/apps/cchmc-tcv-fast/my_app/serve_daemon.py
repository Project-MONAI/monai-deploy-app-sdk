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

"""Serve-mode runner: ``python -m my_app serve``.

Sets ``_serve_mode`` on the app so ``compose`` takes the serve branch
(in-graph gRPC daemon), then blocks in ``app.run`` until the graph
terminates.

Crash-safe-by-construction: startup scratch sweep here
(serve_publish.sweep_stale_scratch); atomic output publish in Egress;
structured LIFECYCLE log lines: startup/ready/shutdown.

Note on hard kills: under the stock no-trap PID-1 tools.sh a hard kill
(SIGKILL) never reaches Python, so a SIGKILL'd daemon has startup + ready
LIFECYCLE lines but NO shutdown line — that is the expected, contracted
behavior, not a gap.
"""

import logging
import os
import sys

try:  # package-style (my_app.*)
    from my_app.app import TCVNNUnetFastApp
    from my_app.serve_publish import sweep_stale_scratch
    from my_app.serve_state import STATE as _DAEMON_STATE
except ImportError:  # flat import (my_app dir on sys.path)
    from app import TCVNNUnetFastApp
    from serve_publish import sweep_stale_scratch
    from serve_state import STATE as _DAEMON_STATE


def run_serve() -> int:
    # Purge-at-start: a fresh daemon owns no study state; any leftover
    # <MAP_SCRATCH_ROOT>/study_* is crash residue from a killed prior
    # process (it got no chance to clean up). serve_publish's
    # module logger carries its own stderr handler so this line is visible
    # before holoscan configures root logging inside app.run.
    sweep_stale_scratch()
    port = int(os.environ.get("MAP_GRPC_PORT", "8765"))
    # Bounded in-process supervisor. In holoscan
    # 4.2 an unhandled operator exception tears down the whole graph (and
    # with it the in-graph gRPC server). With a recovery budget, re-compose a
    # FRESH app instance (models re-load, gRPC server re-binds) instead of
    # exiting the process. Default 0 = historical behavior: exit code 1 and
    # let the container's RestartPolicy (on-failure, set by the TM's
    # MapEndpointManager) recover. Caveat: if the dead server still holds the
    # port, the re-compose fails at grpc_ingress bind (RuntimeError) and
    # burns budget, then falls back to process exit + container restart — so
    # this can never loop worse than the historical path.
    max_recoveries = int(os.environ.get("MAP_SERVE_MAX_RECOVERIES", "0"))
    # structured lifecycle line — startup (port/mode).
    logging.info(
        "LIFECYCLE stage=startup mode=serve port=%d pid=%d cuda=%s max_recoveries=%d",
        port,
        os.getpid(),
        os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        max_recoveries,
    )
    code, reason, attempt = 0, "graph_terminated", 0
    while True:
        attempt += 1
        # FRESH app instance per attempt: _compose_serve re-loads the models
        # and re-binds the in-graph gRPC server. No module re-import (the
        # 'my_app.app' import happened once above), so the RMM torch
        # allocator is NOT re-initialized (see __main__.py lazy-import note).
        app = TCVNNUnetFastApp()
        app._serve_mode = True  # read by app.compose via getattr default — batch never sets it
        logging.info(
            "serve_daemon: starting persistent gRPC daemon (map_service, port from MAP_GRPC_PORT) attempt=%d",
            attempt,
        )
        try:
            app.run()  # compose(serve branch) + blocks until the graph terminates
            code, reason = 0, "graph_terminated"
            break  # clean graph termination -> real shutdown
        except KeyboardInterrupt:
            code, reason = 130, "interrupt"
            break
        except BaseException:  # noqa: B036 - supervisor must catch all to bound recovery budget
            code, reason = 1, "graph_error"
            logging.exception(
                "serve_daemon: graph error on attempt %d (port=%d)",
                attempt,
                port,
            )
            if attempt > max_recoveries:
                # Recovery budget exhausted -> exit non-zero; the container
                # RestartPolicy (on-failure) / orchestrator takes over.
                break
            # Reset the module-singleton state machine so the re-compose's
            # mark_ready can transition INITIALIZING -> IDLE again (the
            # singleton persists across graph re-composes within this process).
            _DAEMON_STATE.reset()
            logging.info(
                "serve_daemon: in-process recovery %d/%d — re-composing graph (models reload)",
                attempt,
                max_recoveries,
            )
    # Structured lifecycle line — shutdown. Under the stock no-trap tools.sh
    # a hard kill (SIGKILL) never reaches this line — that is contracted,
    # not a gap (module docstring).
    logging.info(
        "LIFECYCLE stage=shutdown mode=serve port=%d reason=%s code=%d attempts=%d",
        port,
        reason,
        code,
        attempt,
    )
    # Force-exit. A graph error tears down the graph but the in-graph gRPC
    # server thread (non-daemon) keeps the process alive, so a plain return
    # / sys.exit hangs in threading._shutdown and the container's
    # on-failure restart never fires (observed: zombie PID 1, 341 threads,
    # port 8765 closed, daemon bricked after one bad study).
    # os._exit skips thread joins and exits immediately with the right code.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
    return code  # unreachable (os._exit never returns); satisfies linters
