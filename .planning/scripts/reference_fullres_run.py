#!/usr/bin/env python
"""reference_fullres_run.py — per-config reference runs (Phase 1/2 gates).

Generates a freshly regenerated reference SC/SEG/SR for the pixel-exact
gates by running the reference MAP (``cchmc_nnunet_fifteen_ckpt_app``)
with ``model_list`` pinned to the ``--config`` list.

Why this exists (Phase 1, Plan 04 gate-calibration finding):
``testdata/current_output`` was produced by the reference app's DEFAULT
full-bundle run — a 2-config cascade ensemble (3d_fullres +
3d_cascade_fullres, 2447 post-CC voxels). Phase 1 scope for
``cchmc-nnunet-fast`` is the single config ``3d_fullres`` (3655 post-CC
voxels), so the pixel-exact gate must compare against a 3d_fullres-ONLY
reference run — exactly what this script produces with the default
``--config 3d_fullres``. The reference app itself is not modified
(example-app modification is out of scope): the app module is loaded by
file path and its ``NNUnetSegOperator`` constructor is monkey-patched in
the loaded module's namespace to pin ``model_list`` before the DAG is
composed.

Phase 2 per-config oracles (D-05) — ``--config`` accepts a COMMA-SEPARATED
list so the pin can carry more than one config:
  # 3d_lowres-only oracle:
  python .planning/scripts/reference_fullres_run.py \
      --config 3d_lowres --output testdata/ref_lowres_only
  # cascade-only oracle:
  python .planning/scripts/reference_fullres_run.py \
      --config 3d_lowres,3d_cascade_fullres --output testdata/ref_cascade_only
  # WHY lowres must be in the cascade-only list: with model_list pinned to
  # ['3d_cascade_fullres'] the reference CRASHES (live-probed 2026-08-19)
  # with "RuntimeError: ... tmp/3d_lowres/Img_in_context.nii.gz does not
  # exist" — the reference cascade reads the previous stage's exported
  # .nii.gz, which exists only if lowres ACTUALLY RAN; the reference list
  # logic reorders but does not auto-insert. With lowres in the list it
  # runs (auxiliary), cascade consumes its export, and the reference's own
  # ensemble_model_list excludes lowres — the SEG is cascade-only, exactly
  # matching the fast app's HOLOSCAN_MODEL_LIST=3d_cascade_fullres
  # semantics (D-07).

Usage (venv python; ``ulimit -s unlimited``; ~124 s per run on A100):
  python .planning/scripts/reference_fullres_run.py \
      [--input testdata/airway_input] \
      [--model examples/apps/cchmc_nnunet_fifteen_ckpt_app/models] \
      [--output testdata/ref_fullres_only] \
      [--config 3d_fullres]      # comma-separated list supported
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default=str(REPO_ROOT / "testdata" / "airway_input"))
    ap.add_argument("--model",
                    default=str(REPO_ROOT / "examples" / "apps" /
                                "cchmc_nnunet_fifteen_ckpt_app" / "models"))
    ap.add_argument("--output", default=str(REPO_ROOT / "testdata" / "ref_fullres_only"))
    ap.add_argument("--config", default="3d_fullres",
                    help="comma-separated nnUNet config list to pin as model_list")
    args = ap.parse_args(argv)
    model_list = [c.strip() for c in args.config.split(',') if c.strip()]
    if not model_list:
        print("reference_fullres_run: --config resolved to an empty list",
              file=sys.stderr)
        return 2

    ref_app_dir = (REPO_ROOT / "examples" / "apps" /
                   "cchmc_nnunet_fifteen_ckpt_app" / "my_app")
    app_py = ref_app_dir / "app.py"
    if not app_py.is_file():
        print(f"reference_fullres_run: missing {app_py}", file=sys.stderr)
        return 2

    # app.py performs top-level sibling imports (from nnunet_seg_operator
    # import ...), so the my_app dir must be first on sys.path.
    sys.path.insert(0, str(ref_app_dir))

    spec = importlib.util.spec_from_file_location("ref_airway_app", app_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ref_airway_app"] = mod
    spec.loader.exec_module(mod)

    # Pin the model list BEFORE compose() instantiates the operator.
    OrigOperator = mod.NNUnetSegOperator

    def pinned_operator(*a, **kwargs):
        kwargs["model_list"] = model_list
        return OrigOperator(*a, **kwargs)

    mod.NNUnetSegOperator = pinned_operator

    output_dir = Path(args.output)
    print(f"reference_fullres_run: model_list={model_list!r} "
          f"input={args.input} model={args.model} output={output_dir}")
    sys.argv = ["reference_fullres_run.py", "-i", args.input,
                "-m", args.model, "-o", str(output_dir)]
    mod.UTEAirwayNNUnetApp().run()

    segs = list((output_dir / "SEG").glob("*.dcm")) if (output_dir / "SEG").is_dir() else []
    print(f"reference_fullres_run: done — {len(segs)} SEG file(s) under {output_dir}")
    return 0 if segs else 1


if __name__ == "__main__":
    sys.exit(main())
