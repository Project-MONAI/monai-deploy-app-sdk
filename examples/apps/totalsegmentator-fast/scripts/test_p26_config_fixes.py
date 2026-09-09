#!/usr/bin/env python
"""Phase 26 plan 26-01 TDD suite: CFG02-01 (any-fold completeness) + CFG02-02
(mandatory part_configs) — RED->GREEN for the config/__init__.py fixes.

House pattern (mirrors scripts/test_liver_specs.py): standalone executable,
NOT pytest; exits 0 iff every check passes. Headless: stdlib only, temp
bundle trees built by the test itself (empty checkpoint files — the resolver
only stats paths).

Checks:
  A (RED first): a bundle whose checkpoint lives ONLY in a non-zero fold
     (fold_2, no fold_0) resolves through resolve_task_model_root
     (pre-fix: rejected — the fold_0 literal assumption, CFG02-01).
  B (regression): fold_0 + fold_2 both present -> resolves.
  C (regression): jsonpkls/plans.json + config dir but NO fold dir / no
     checkpoint -> hard FileNotFoundError naming the searched paths
     (fail-fast preserved).
  D (RED first): _required_files_for_part("organs", {}) raises
     ValueError/KeyError naming part "organs" (pre-fix: silently returned the
     3d_fullres fold_0 paths — the silent fallback, CFG02-02).
  E (regression): legacy resolve_total_model_root on a synthetic complete
     total bundle (fold_0) returns the same root path as pre-change behavior
     (4-candidate layout lock).

Run:
  cd examples/apps/totalsegmentator-fast
  /tmp/monai-env/.venv/bin/python scripts/test_p26_config_fixes.py
Exit: 0 iff every check passes; 1 otherwise.
"""

import sys
import tempfile
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

import my_app.config as cfg  # noqa: E402

# Real minimal plans.json (the resolver only requires the file to exist, but
# copying the actual corpus file keeps the fixture faithful).
_PLANS_SRC = APP_ROOT.parents[2] / "ct-totalsegmentator-map" / "models" / "total" / "organs" / "jsonpkls" / "plans.json"


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        print(f"FAIL: {name} {detail}")
        sys.exit(1)
    print(f"PASS: {name}")


def _plans_text() -> str:
    if _PLANS_SRC.is_file():
        return _PLANS_SRC.read_text()
    return "{}"


def _mk_part(root: Path, part: str, folds_and_ckpts):
    """Create {root}/{part}/jsonpkls/plans.json + {config}/fold_N/ckpt files.

    folds_and_ckpts: list of (config_name, fold_or_None, checkpoint) —
    fold_or_None=None creates the config dir WITHOUT any fold dir (test C).
    """
    pdir = root / part
    (pdir / "jsonpkls").mkdir(parents=True)
    (pdir / "jsonpkls" / "plans.json").write_text(_plans_text())
    for config_name, fold, ckpt in folds_and_ckpts:
        if fold is None:
            (pdir / config_name).mkdir(parents=True, exist_ok=True)
            continue
        fdir = pdir / config_name / f"fold_{fold}"
        fdir.mkdir(parents=True)
        (fdir / ckpt).write_bytes(b"")


# ---------------------------------------------------------------------------
# A. any-fold completeness: fold_2-ONLY bundle resolves (CFG02-01, RED first)
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    _mk_part(root, "organs", [("3d_fullres", 2, "final_model.pt")])
    try:
        got = cfg.resolve_task_model_root(
            root,
            "organs_lowres_test",
            ["organs"],
            part_configs={"organs": "3d_fullres"},
        )
        check("A non-zero-fold-only bundle resolves (CFG02-01)", got == root, f"got {got}")
    except FileNotFoundError as e:
        check("A non-zero-fold-only bundle resolves (CFG02-01)", False, f"raised {e!s}")

# ---------------------------------------------------------------------------
# B. regression: fold_0 + fold_2 both present resolves
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    _mk_part(
        root,
        "organs",
        [
            ("3d_fullres", 0, "final_model.pt"),
            ("3d_fullres", 2, "final_model.pt"),
        ],
    )
    got = cfg.resolve_task_model_root(
        root,
        "organs_lowres_test",
        ["organs"],
        part_configs={"organs": "3d_fullres"},
    )
    check("B fold_0+fold_2 bundle resolves (regression)", got == root, f"got {got}")

# ---------------------------------------------------------------------------
# C. fail-fast preserved: config dir without ANY fold checkpoint raises,
#    naming the searched paths
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    _mk_part(root, "organs", [("3d_fullres", None, None)])
    try:
        cfg.resolve_task_model_root(
            root,
            "organs_lowres_test",
            ["organs"],
            part_configs={"organs": "3d_fullres"},
        )
        check(
            "C no-fold bundle raises FileNotFoundError (fail-fast)",
            False,
            "no exception",
        )
    except FileNotFoundError as e:
        msg = str(e)
        check(
            "C no-fold bundle raises FileNotFoundError (fail-fast)",
            "organs" in msg and "3d_fullres" in msg and "final_model.pt" in msg,
            f"msg={msg}",
        )

# ---------------------------------------------------------------------------
# D. CFG02-02: missing part_configs key is a named error (RED first)
# ---------------------------------------------------------------------------
try:
    result = cfg._required_files_for_part("organs", {})
    check(
        "D missing part_configs key raises (CFG02-02)",
        False,
        f"no exception, returned {result!r}",
    )
except (ValueError, KeyError) as e:
    check("D missing part_configs key raises (CFG02-02)", "organs" in str(e), f"msg={e!s}")

# present key still works and returns the per-part config paths
got = cfg._required_files_for_part("organs", {"organs": "3d_lowres"})
check(
    "D2 present key returns {config}/fold paths",
    got[0] == "jsonpkls/plans.json" and "3d_lowres" in got[1],
    f"got {got!r}",
)

# ---------------------------------------------------------------------------
# E. regression: legacy resolve_total_model_root on a complete fold_0 total
#    bundle returns the same root (4-candidate layout + total completeness)
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    for part in cfg.MODEL_PARTS:
        _mk_part(root, part["name"], [("3d_fullres", 0, "final_model.pt")])
    got = cfg.resolve_total_model_root(root)
    check(
        "E legacy total resolver: complete fold_0 bundle resolves (regression)",
        got == root,
        f"got {got}",
    )

# E2: same root via the nested models/total layout (candidate #3)
with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "models" / "total"
    for part in cfg.MODEL_PARTS:
        _mk_part(root, part["name"], [("3d_fullres", 0, "final_model.pt")])
    got = cfg.resolve_total_model_root(Path(td))
    check(
        "E2 legacy total resolver: models/total layout (regression)",
        got == root,
        f"got {got}",
    )

# E3: legacy total resolver fail-fast names missing files
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    for part in cfg.MODEL_PARTS:
        _mk_part(root, part["name"], [("3d_fullres", None, None)])
    try:
        cfg.resolve_total_model_root(root)
        check(
            "E3 legacy total resolver fail-fast on incomplete bundle",
            False,
            "no exception",
        )
    except FileNotFoundError as e:
        check(
            "E3 legacy total resolver fail-fast on incomplete bundle",
            "organs" in str(e) and "3d_fullres" in str(e),
            f"msg={e!s}",
        )

print("ALL PASS: test_p26_config_fixes")
sys.exit(0)
