#!/usr/bin/env python3
"""CPU smoke test for the serve_daemon import path (no GPU, no daemon).

Checks (all CPU-only, venv python + stdlib):
  1. AST: my_app/serve_daemon.py imports exactly {TCVNNUnetFastApp} from
     my_app.app / app; the string CCHMCNNUnetFastApp appears nowhere.
  2. Import smoke: `import my_app.serve_daemon` succeeds from the app dir.
  3. Router: my_app.__main__.route dispatch:
       ["serve"] -> "serve" ; ["serve","x"] -> RouterError
       ["-i","a","-o","b"] -> "batch" ; ["bogus"] -> RouterError
       [] -> "batch"

Prints `SERVE-IMPORT-SMOKE: PASS` and exits 0 on success; otherwise prints
the failing check and exits 1.
"""

import ast
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
DAEMON = APP_DIR / "my_app" / "serve_daemon.py"
VENV_PY = "python"


def fail(msg: str) -> None:
    print(f"SERVE-IMPORT-SMOKE: FAIL — {msg}")
    sys.exit(1)


def check_ast(source: str) -> None:
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("my_app.app", "app"):
            imported |= {a.name for a in node.names}
    if imported != {"TCVNNUnetFastApp"}:
        fail(f"AST: imports from my_app.app/app are {sorted(imported)}, " f"expected ['TCVNNUnetFastApp']")
    if "CCHMCNNUnetFastApp" in source:
        fail("AST: 'CCHMCNNUnetFastApp' still present in serve_daemon.py")


def check_import() -> None:
    env = {"PYTHONPATH": str(APP_DIR)}
    r = subprocess.run(
        [VENV_PY, "-c", "import my_app.serve_daemon"],
        cwd=str(APP_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        fail(f"import smoke rc={r.returncode}\n{r.stderr[-2000:]}")


def check_router() -> None:
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from my_app.__main__ import route, RouterError\n"
        "assert route(['serve']) == 'serve'\n"
        "try:\n route(['serve','x']); sys.exit('no RouterError for serve+x')\n"
        "except RouterError: pass\n"
        "assert route(['-i','a','-o','b']) == 'batch'\n"
        "try:\n route(['bogus']); sys.exit('no RouterError for bogus')\n"
        "except RouterError: pass\n"
        "assert route([]) == 'batch'\n"
        "print('router-ok')\n" % str(APP_DIR)
    )
    r = subprocess.run([VENV_PY, "-c", code], cwd=str(APP_DIR), capture_output=True, text=True)
    if r.returncode != 0 or "router-ok" not in r.stdout:
        fail(f"router check rc={r.returncode}\n{r.stderr[-2000:]}")


def main() -> None:
    source = DAEMON.read_text()
    check_ast(source)
    check_import()
    check_router()
    print("SERVE-IMPORT-SMOKE: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
