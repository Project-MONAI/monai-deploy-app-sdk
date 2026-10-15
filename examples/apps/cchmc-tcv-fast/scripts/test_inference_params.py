"""CPU-only parameter probe for the TCV bundle (no GPU).

Asserts that the app's bundle resolution yields the full TCV inference
parameter set — and, critically, that no-TTA is RECORDED (read from the
generated nnunet_checkpoint.pth metadata), not an accidental fallback:

* folds (0..4) and every fold's final_model.pt present
* mirror_axes is None (trainer nnUNetTrainer_250epochs_NoMirroring)
* trainer_name recorded (fallback would be "unknown" — proves the
  nnunet_checkpoint.pth metadata file is actually read)
* patch (256,256,256), 4 output channels (3 foreground + background,
  per nnUNet LabelManager), 1 input channel,
  checkpoint final_model.pt
* build_mirror_axis_combinations(None, 5) == [] (zero TTA passes)

Run (venv, CPU):
    python -m pytest examples/apps/cchmc-tcv-fast/scripts/test_inference_params.py -v
"""

import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "my_app"))  # flat import, mirrors my_app/__init__.py

import config as cfg  # my_app/config/__init__.py

# NOTE: the plan's snippet referenced cfg.build_mirror_axis_combinations, but
# the function actually lives in operators/slidewindow_operator.py (config
# does not re-export it) — using the real location per the plan's own
# "use the real name" escape hatch.
from operators.slidewindow_operator import build_mirror_axis_combinations

MODELS = APP / "models"


def test_inference_params_tcv():
    p = cfg.load_inference_params(MODELS)
    assert p.folds == (0, 1, 2, 3, 4)
    assert p.mirror_axes is None  # no-TTA is RECORDED, not fallback
    assert (
        p.trainer_name == "nnUNetTrainer_250epochs_NoMirroring"
    )  # proves nnunet_checkpoint.pth was read (fallback would be "unknown")
    assert p.patch_size == (256, 256, 256)
    assert p.num_segmentation_heads == 4
    # NOTE (deviation from plan): the plan's probe snippet asserted 3 (foreground
    # classes only), but the vendored nnUNetv2 LabelManager defines
    # num_segmentation_heads == len(all_labels) INCLUDING background
    # (nnUNet/nnunetv2/utilities/label_handling/label_handling.py:241-245):
    # labels {0 background, 1 heart, 2 l_lung, 3 r_lung} -> 4 output channels.
    # The network's output channel count IS 4; the 3 *foreground* classes are
    # heart/l_lung/r_lung. Value asserted from the real dataclass per the
    # plan's own "use the real one" escape hatch.
    assert p.checkpoint_name == "final_model.pt"
    assert p.num_input_channels == 1
    for f in p.folds:
        assert (APP / "models" / "3d_fullres" / f"fold_{f}" / "final_model.pt").is_file()


def test_mirror_combinations_empty():
    assert build_mirror_axis_combinations(None, 5) == []
