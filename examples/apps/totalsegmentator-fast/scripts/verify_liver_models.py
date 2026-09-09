#!/usr/bin/env python
"""Phase 18 (plan 18-01, Task 1, LIV-04) — re-runnable bundle-content verification
for the two on-disk `liver_segments` bundles.

Headless (CPU torch only, /tmp/monai-env/.venv convention). Reads:
  /raid/monai-deploy-app-sdk/ct-totalsegmentator-map/models/liver_segments/
    ct_liver_segments/  (main model, task 570)
    total_6mm/          (crop model, task 298)

Verifies (all values from 18-RESEARCH §"Models (LIV-04) — Verified State",
re-confirmed against live TS 2.18 config this plan):
  1. ct_liver_segments checkpoint trainer_name == nnUNetTrainerNoMirroring
  2. total_6mm checkpoint trainer_name == nnUNetTrainer_4000epochs_NoMirroring
  3. ct_liver_segments dataset.json labels == background + liver_segment_1..8 (ids 0-8)
  4. total_6mm dataset.json label id for "liver" == 5 (118-entry full table)
  5. ct_liver_segments plans.json 3d_fullres: spacing [1.5, 0.804688, 0.804688]
     (nnUNet z,y,x order => xy 0.804688, z 1.5 — matches the TS task-res vector),
     patch_size [64, 192, 192]
  6. total_6mm plans.json 3d_fullres: spacing 6.0 isotropic, patch_size [64, 64, 64]

Exit: 0 iff every check passes; 1 otherwise. Read-only on all inputs.
"""

import json
import sys
from pathlib import Path

ROOT = Path("/raid/monai-deploy-app-sdk/ct-totalsegmentator-map/models/liver_segments")

FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        print(f"FAIL: {name} {detail}")
        FAILURES.append(name)
    else:
        print(f"PASS: {name}")


def main() -> int:
    if not ROOT.is_dir():
        print(f"FAIL: model root missing: {ROOT}")
        return 1

    ct = ROOT / "ct_liver_segments"
    t6 = ROOT / "total_6mm"

    # ------------------------------------------------------------------
    # 1+2. checkpoint trainer names (CPU torch, read-only)
    # ------------------------------------------------------------------
    import torch

    for part, expected in (
        (ct, "nnUNetTrainerNoMirroring"),
        (t6, "nnUNetTrainer_4000epochs_NoMirroring"),
    ):
        ckpt_path = part / "3d_fullres" / "nnunet_checkpoint.pth"
        if not ckpt_path.is_file():
            check(f"{part.name} checkpoint present", False, f"missing {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        trainer = ckpt.get("trainer_name")
        check(
            f"{part.name} trainer_name",
            trainer == expected,
            f"got {trainer!r}, expected {expected!r}",
        )

    # ------------------------------------------------------------------
    # 3+4. dataset.json label tables
    # ------------------------------------------------------------------
    ct_labels = json.loads((ct / "jsonpkls" / "dataset.json").read_text())["labels"]
    check(
        "3 ct_liver_segments labels == background + liver_segment_1..8 (0-8)",
        ct_labels
        == {
            "background": 0,
            **{f"liver_segment_{i}": i for i in range(1, 9)},
        },
        f"got {ct_labels!r}",
    )

    t6_labels = json.loads((t6 / "jsonpkls" / "dataset.json").read_text())["labels"]
    check(
        "4a total_6mm 'liver' label id == 5",
        t6_labels.get("liver") == 5,
        f"got {t6_labels.get('liver')!r}",
    )
    check(
        "4b total_6mm full 118-entry label table (bg 0 + 1..117)",
        len(t6_labels) == 118
        and t6_labels.get("background") == 0
        and sorted(v for v in t6_labels.values()) == list(range(118)),
        f"n={len(t6_labels)}",
    )

    # ------------------------------------------------------------------
    # 5+6. plans.json 3d_fullres spacing + patch
    # (key is 'configurations' in these bundles' nnUNet 2 plans files)
    # ------------------------------------------------------------------
    ct_cfg = json.loads((ct / "jsonpkls" / "plans.json").read_text())["configurations"]["3d_fullres"]
    check(
        "5a ct_liver_segments 3d_fullres spacing (z,y,x) == [1.5, 0.804688, 0.804688]",
        abs(ct_cfg["spacing"][0] - 1.5) < 1e-9
        and abs(ct_cfg["spacing"][1] - 0.8046879768371582) < 1e-9
        and abs(ct_cfg["spacing"][2] - 0.8046879768371582) < 1e-9,
        f"got {ct_cfg['spacing']!r}",
    )
    check(
        "5b ct_liver_segments 3d_fullres patch_size == [64, 192, 192]",
        list(ct_cfg["patch_size"]) == [64, 192, 192],
        f"got {ct_cfg['patch_size']!r}",
    )

    t6_cfg = json.loads((t6 / "jsonpkls" / "plans.json").read_text())["configurations"]["3d_fullres"]
    check(
        "6a total_6mm 3d_fullres spacing == 6.0 isotropic",
        all(abs(s - 6.0) < 1e-9 for s in t6_cfg["spacing"]),
        f"got {t6_cfg['spacing']!r}",
    )
    check(
        "6b total_6mm 3d_fullres patch_size == [64, 64, 64]",
        list(t6_cfg["patch_size"]) == [64, 64, 64],
        f"got {t6_cfg['patch_size']!r}",
    )

    if FAILURES:
        print(f"\nFAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("\nALL PASS: verify_liver_models (8/8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
