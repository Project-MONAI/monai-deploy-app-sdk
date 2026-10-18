#!/usr/bin/env python3
"""Generate models/3d_fullres/nnunet_checkpoint.pth from fold_0/final_model.pt.

The nnUNet app contract (my_app/config/__init__.py::load_inference_params)
reads trainer_name / init_args / inference_allowed_mirroring_axes from a
small metadata file named ``nnunet_checkpoint.pth`` next to the fold
directories. The TCV bundle ships without it,
so no-TTA would only hold by accidental fallback (trainer_name="unknown").
This script records the metadata EXPLICITLY:

* trainer_name = nnUNetTrainer_250epochs_NoMirroring
* init_args (configuration must match "3d_fullres")
* inference_allowed_mirroring_axes = None (no TTA, recorded not accidental)

Only the three metadata keys are copied — NEVER the network weights
(the cchmc reference file is 5.9 KB for exactly this reason).

Idempotent: re-running when the file already matches prints UNCHANGED
with the same sha256. Usage:

    python scripts/sync_checkpoint_meta.py [--models <models_dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

META_KEYS = ("trainer_name", "init_args", "inference_allowed_mirroring_axes")
OUT_NAME = "nnunet_checkpoint.pth"

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from my_app.config import load_torch_checkpoint  # noqa: E402


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_models = Path(__file__).resolve().parents[1] / "models"
    parser.add_argument(
        "--models",
        type=Path,
        default=default_models,
        help=f"bundle root (default: {default_models})",
    )
    args = parser.parse_args(argv)

    import torch

    models_root = args.models.resolve()
    cfg_dir = models_root / "3d_fullres"
    src = cfg_dir / "fold_0" / "final_model.pt"
    if not src.is_file():
        print(f"ERROR: missing {src}", file=sys.stderr)
        return 1

    ckpt = load_torch_checkpoint(src, map_location="cpu")
    if not isinstance(ckpt, dict) or "trainer_name" not in ckpt:
        print(
            f"ERROR: {src} has no trainer_name metadata (keys={list(ckpt) if isinstance(ckpt, dict) else type(ckpt)})",
            file=sys.stderr,
        )
        return 1

    meta = {k: ckpt.get(k) for k in META_KEYS}

    out = cfg_dir / OUT_NAME
    if out.is_file():
        existing = load_torch_checkpoint(out, map_location="cpu")
        if all(existing.get(k) == meta[k] for k in META_KEYS):
            print(f"UNCHANGED {sha256_of(out)}")
            return 0

    # write atomically so a crash never leaves a half-written metadata file
    tmp = out.with_suffix(".pth.tmp")
    torch.save(meta, tmp)
    tmp.replace(out)
    print(
        f"WROTE {out} sha256={sha256_of(out)} "
        f"trainer={meta['trainer_name']} axes={meta['inference_allowed_mirroring_axes']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
