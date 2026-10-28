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

# __main__.py allows `python my_app` to resolve to this package.

"""Dual-mode entrypoint.

`python -m my_app serve` -> persistent gRPC daemon (serve branch)
`python -m my_app -i <in> ...` -> legacy batch app (batch branch)
`python -m my_app` (no args) -> batch (the SDK batch parser decides)

Routing is EXACT token equality on argv[0] == "serve" (case-sensitive, never a
substring scan): a study path that merely CONTAINS the word "serve" must not
misroute. Unknown bare positionals fail fast (exit 2, clear
stderr naming the argument). Dashed arguments are delegated to the SDK batch
parser — original behavior, byte-identical batch contract.

Both heavy imports stay LAZY in their branch ON PURPOSE:
importing app.py as module 'app' and later as 'my_app.app' would execute it
a second time and re-run gpu_bootstrap.install_torch_allocator -> "Can't
swap an already initialized allocator".
"""

import json
import logging
import sys

_USAGE = "usage: python -m my_app serve" " | python -m my_app -i <input> -o <output> [-m <models>] [SDK batch flags]"


class RouterError(ValueError):
    """Unroutable argv for the cchmc-tcv-fast dual-mode router."""


def route(argv: list[str]) -> str:
    """Map argv (program name excluded) to a mode: 'serve' or 'batch'.

    Raises RouterError on unroutable argv (see module docstring table).
    """
    if not argv:
        return "batch"
    first = argv[0]
    if first == "serve":
        if len(argv) > 1:
            raise RouterError("'serve' takes no arguments; inbound paths are per-RPC (ProcessStudy)")
        return "serve"
    if first.startswith("-"):
        return "batch"
    raise RouterError(f"unrecognized argument {first!r} (expected 'serve' or batch flags such as -i/-o)")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    # make the router-era Begin/End lines and the LIFECYCLE
    # lines visible before the SDK's set_up_logging runs. force=False keeps
    # any pre-existing configuration untouched.
    logging.basicConfig(level=logging.INFO, force=False)
    try:
        mode = route(argv)
    except RouterError as exc:
        print(f"cchmc-tcv-fast: {exc}\n{_USAGE}", file=sys.stderr)
        return 2
    logging.info(f"Begin {__name__} mode={mode}")
    if mode == "serve":
        # Lazy import, serve branch ONLY (see module docstring).
        try:
            from my_app.serve_daemon import run_serve
        except ImportError:  # flat import (my_app dir on sys.path)
            from serve_daemon import run_serve
        return run_serve()
    # Flat 'app' import lives in the batch branch ON PURPOSE ( fix):
    # importing it at module top would load app.py as module 'app' and then
    # serve_daemon's 'my_app.app' import would execute it a second time,
    # re-running gpu_bootstrap.install_torch_allocator -> "Can't swap an
    # already initialized allocator".
    # structured lifecycle line — batch startup (argv is the
    # full batch invocation, json-dumped for grep stability).
    logging.info("LIFECYCLE stage=startup mode=batch argv=%s", json.dumps(argv))
    from app import TCVNNUnetFastApp

    try:
        TCVNNUnetFastApp().run()
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        logging.info("LIFECYCLE stage=shutdown mode=batch code=%d", code)
        raise
    except BaseException:
        logging.info("LIFECYCLE stage=shutdown mode=batch code=1")
        raise
    logging.info("LIFECYCLE stage=shutdown mode=batch code=0")
    logging.info(f"End {__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
