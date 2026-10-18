#!/usr/bin/env python3
"""Strict load-verification of the TCV 3d_fullres bundle.

CPU-only. For each fold 0..4 in <models>/3d_fullres/:
  - torch.load (map_location=cpu, weights_only=True, numpy metadata allow-listed)
  - assert full nnUNet checkpoint dict: trainer_name, init_args.configuration,
    inference_allowed_mirroring_axes is None (no-op-TTA fact), current_epoch
  - sd = ckpt['network_weights'] (NEVER strict-load the outer dict — pitfall 3)
  - assert len(sd) == 1160
  - network built on CPU from models/jsonpkls/plans.json (vendored nnunetv2
    2.8.1 plans path) .load_state_dict(sd, strict=True) -> 0 missing / 0 unexpected

Writes a JSON report. Exit 0 iff all 5 folds pass.

Usage:
    python load_verify.py [--models DIR] [--config 3d_fullres] [--report out.json]
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import torch

APP_DIR = Path(__file__).resolve().parents[1]  # examples/apps/cchmc-tcv-fast
sys.path.insert(0, str(APP_DIR))  # my_app importable regardless of cwd

DEFAULT_MODELS = APP_DIR / "models"
DEFAULT_REPORT = APP_DIR / "scripts" / "load-verify-report.json"

EXPECTED_TRAINER = "nnUNetTrainer_250epochs_NoMirroring"
EXPECTED_TENSORS = 1160
EXPECTED_EPOCH = 250
FOLDS = (0, 1, 2, 3, 4)


def build_network(cfg: str, jp: Path):
    """Build the plans-defined network on CPU (vendored nnunetv2 2.8.1 path)."""
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.label_handling.label_handling import (
        determine_num_input_channels,
    )
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans = json.loads((jp / "plans.json").read_text())
    dataset_json = json.loads((jp / "dataset.json").read_text())
    pm = PlansManager(plans)
    cm = pm.get_configuration(cfg)
    num_input_channels = determine_num_input_channels(pm, cm, dataset_json)
    num_heads = pm.get_label_manager(dataset_json).num_segmentation_heads
    return get_network_from_plans(
        cm.network_arch_class_name,
        cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import,
        num_input_channels,
        num_heads,
        allow_init=True,
        deep_supervision=False,
    )  # CPU default


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=str(DEFAULT_MODELS))
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    args = ap.parse_args()

    from my_app.config import load_torch_checkpoint

    t0 = time.time()
    models = Path(args.models).expanduser().resolve()
    jp = models / "jsonpkls"
    if not (jp / "plans.json").is_file() or not (jp / "dataset.json").is_file():
        print(f"ERROR: jsonpkls not found under {models}", file=sys.stderr)
        sys.exit(1)

    cfg = args.config
    if not (models / cfg).is_dir():
        print(f"ERROR: model dir not found: {models / cfg}", file=sys.stderr)
        sys.exit(1)

    try:
        nnunetv2_version = importlib.metadata.version("nnunetv2")
    except importlib.metadata.PackageNotFoundError:
        nnunetv2_version = "NOT-INSTALLED"

    network = build_network(cfg, jp)
    folds_out: dict[str, dict] = {}
    all_ok = True

    for n in FOLDS:
        key = f"fold_{n}"
        fp = models / cfg / key / "final_model.pt"
        rec: dict = {
            "trainer_name": None,
            "configuration": None,
            "current_epoch": None,
            "mirroring_axes": None,
            "n_tensors": None,
            "missing": None,
            "unexpected": None,
            "ok": False,
            "error": None,
        }
        if not fp.is_file():
            rec["error"] = "checkpoint missing"
            folds_out[key] = rec
            all_ok = False
            continue

        try:
            ckpt = load_torch_checkpoint(fp, map_location="cpu")
            ia = ckpt.get("init_args") or {}
            rec["trainer_name"] = ckpt.get("trainer_name")
            rec["configuration"] = ia.get("configuration") if hasattr(ia, "get") else getattr(ia, "configuration", None)
            rec["current_epoch"] = ckpt.get("current_epoch")
            rec["mirroring_axes"] = ckpt.get("inference_allowed_mirroring_axes")

            sd = ckpt["network_weights"]  # pitfall 3: never strict the outer dict
            rec["n_tensors"] = len(sd)

            if rec["trainer_name"] != EXPECTED_TRAINER:
                rec["error"] = f"trainer_name {rec['trainer_name']!r} != {EXPECTED_TRAINER!r}"
            elif rec["configuration"] != cfg:
                rec["error"] = f"init_args.configuration {rec['configuration']!r} != {cfg!r}"
            elif ckpt.get("inference_allowed_mirroring_axes") is not None:
                rec["error"] = "mirroring axes not None (expected no-op-TTA contract)"
            elif rec["current_epoch"] != EXPECTED_EPOCH:
                rec["error"] = f"current_epoch {rec['current_epoch']!r} != {EXPECTED_EPOCH}"
            elif len(sd) != EXPECTED_TENSORS:
                rec["error"] = f"n_tensors {len(sd)} != {EXPECTED_TENSORS}"
            else:
                try:
                    network.load_state_dict(sd, strict=True)
                    rec["missing"] = 0
                    rec["unexpected"] = 0
                    rec["ok"] = True
                except Exception as e:  # noqa: BLE001
                    non_strict = network.load_state_dict(sd, strict=False)
                    rec["missing"] = len(non_strict.missing_keys)
                    rec["unexpected"] = len(non_strict.unexpected_keys)
                    rec["error"] = (
                        f"strict load failed: {type(e).__name__}: {e}; "
                        f"missing_sample={list(non_strict.missing_keys)[:5]!r}; "
                        f"unexpected_sample={list(non_strict.unexpected_keys)[:5]!r}"
                    )
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"

        folds_out[key] = rec
        if not rec["ok"]:
            all_ok = False

    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "venv_python": sys.executable,
        "nnunetv2_version": nnunetv2_version,
        "torch_version": torch.__version__,
        "config": cfg,
        "models": str(models),
        "folds": folds_out,
        "all_ok": bool(all_ok and len(folds_out) == 5),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    rp = Path(args.report).expanduser().resolve()
    if not Path(args.report).is_absolute():
        # Containment guard (Sonar S8707): a relative --report path must
        # resolve inside the working directory; an absolute path is explicit
        # operator intent and is allowed.
        try:
            rp.relative_to(Path.cwd().resolve())
        except ValueError:
            print(f"ERROR: --report escapes the working directory: {rp}", file=sys.stderr)
            sys.exit(2)
    if rp.is_dir():
        print(f"ERROR: --report path is a directory: {rp}", file=sys.stderr)
        sys.exit(2)
    rp.parent.mkdir(parents=True, exist_ok=True)
    # rp is validated above: relative paths must resolve inside the working
    # directory and directory targets are rejected, so the write cannot
    # escape it. Sonar's Python taint tracker does not recognize the
    # relative_to() containment check as a sanitizer (S8707 false positive).
    rp.write_text(json.dumps(report, indent=1))  # NOSONAR

    print(
        json.dumps(
            {
                k: {"ok": v["ok"], "missing": v["missing"], "unexpected": v["unexpected"], "n_tensors": v["n_tensors"]}
                for k, v in folds_out.items()
            },
            indent=1,
        )
    )
    print(
        f"all_ok={report['all_ok']}; nnunetv2={nnunetv2_version}; torch={torch.__version__}; "
        f"elapsed={report['elapsed_seconds']}s; report={rp}"
    )
    sys.exit(0 if report["all_ok"] else 1)


if __name__ == "__main__":
    main()
