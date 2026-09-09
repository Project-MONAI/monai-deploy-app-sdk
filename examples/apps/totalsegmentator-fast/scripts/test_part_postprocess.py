#!/usr/bin/env python
"""Phase 9 (09-03) synthetic suite for per-part postprocess — PIP-06 proof.

House pattern (mirrors scripts/test_pre_resample_zoom.py): standalone
executable, NOT pytest. Exits 0 iff every case passes.

What this proves (roadmap 9.3 / success criterion 4):
  1. IDENTITY (live path, 0 pkls): a 32^3 volume with 3 disjoint blobs
     (labels 1,2,3) passes through apply_part_postprocess UNMODIFIED —
     the SAME OBJECT is returned (zero copy, zero compute),
     info["no_op"] is True.
  2. DORMANT PATH BYTE TEST: with a REAL jsonpkls/postprocessing.pkl
     (pickled list of the actual vendored
     nnunetv2 remove_all_but_largest_component_from_segmentation +
     kwargs), apply_part_postprocess (GPU CC port) must be
     BYTE-IDENTICAL (np.array_equal) to the vendored CPU reference
     apply_postprocessing on a synthetic 26-label volume (labels 0..25)
     where labels 1 and 2 carry multiple disjoint components INCLUDING
     a SIZE-TIE pair at the maximum size (two 40-voxel blobs) — acvl
     semantics keep BOTH tied max components; the GPU port must too.
  3. NON-PORTED RULE: a pkl carrying any other rule family raises
     NotImplementedError naming the function (loud — never a silent
     no-op on real data).
  4. REGION-TUPLE GUARD: rule kwargs with a region tuple raise
     (ValueError from the GPU port's documented limitation — no silent
     pass).

Run:
  cd examples/apps/totalsegmentator-fast
  ulimit -s unlimited && CUDA_VISIBLE_DEVICES=0 \
    /tmp/monai-env/.venv/bin/python scripts/test_part_postprocess.py
Exit: 0 iff every case passes; 1 otherwise (details printed).
No files are written outside a temp dir; all data is in-memory synthetic.
"""

import os
import pickle
import sys
import tempfile
from pathlib import Path

# Raise the stack limit the same way the house run recipe does
# (ulimit -s unlimited) so CuPy kernels are safe even when launched
# directly (best effort — ignored if the hard limit forbids it).
try:
    import resource

    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
except (ImportError, OSError, ValueError):
    pass

APP_ROOT = Path(__file__).resolve().parents[1]
MY_APP = APP_ROOT / "my_app"
sys.path.insert(0, str(MY_APP))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

try:
    import cupy as cp
except Exception as e:  # noqa: BLE001
    print(f"SKIP-FAIL: CuPy import failed ({e}); this suite REQUIRES a CUDA GPU.")
    sys.exit(2)

import gpu_bootstrap  # noqa: E402, F401  (RMM bootstrap BEFORE anything else — load-bearing)
import numpy as np  # noqa: E402

if not cp.cuda.is_available():
    print(
        "SKIP-FAIL: CUDA device unavailable; this suite REQUIRES a CUDA GPU "
        "(run with CUDA_VISIBLE_DEVICES=0, GPUs 4-7 forbidden)."
    )
    sys.exit(2)

from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing  # noqa: E402
from nnunetv2.postprocessing.remove_connected_components import (
    remove_all_but_largest_component_from_segmentation,
)
from operators.part_postprocess import RULE_KEEP_LARGEST, apply_part_postprocess, find_postprocessing_pkl  # noqa: E402

FAILURES = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def write_pkl(model_dir: Path, pp_fns: list, pp_fn_kwargs: list) -> Path:
    """Write a real bundle-layout pkl: <model_dir>/jsonpkls/postprocessing.pkl."""
    jdir = model_dir / "jsonpkls"
    jdir.mkdir(parents=True, exist_ok=True)
    pkl = jdir / "postprocessing.pkl"
    with open(pkl, "wb") as f:
        pickle.dump((pp_fns, pp_fn_kwargs), f)
    return pkl


def build_26label_volume() -> np.ndarray:
    """48^3 uint8 volume carrying 26 distinct label values (0..25).

    Labels 1 and 2 carry the rule-relevant structure:
      * label 1: A = 40 voxels (5x4x2), B = 16 (4x2x2), C = 5 (5x1x1)
      * label 2: D = 40 voxels (4x5x2), E = 16 (4x2x2), F = 5 (5x1x1)
    The UNION mask {1,2} therefore has 6 disjoint components with sizes
    {40, 40, 16, 16, 5, 5} — A and D TIE for the maximum. acvl semantics
    (filter_fn keeps every component with size == max) keep BOTH A and D
    and remove B, C, E, F. Labels 3..25 are sparse 3x3x3 blocks (23 of
    them) so the volume genuinely carries 26 label values; no rule
    touches them, so they must pass through untouched.
    """
    seg = np.zeros((48, 48, 48), dtype=np.uint8)
    # Label 1 components.
    seg[2:7, 2:6, 2:4] = 1  # A: 5x4x2 = 40
    seg[2:6, 20:22, 2:4] = 1  # B: 4x2x2 = 16
    seg[2:7, 36:37, 2:3] = 1  # C: 5x1x1 = 5
    # Label 2 components.
    seg[2:6, 2:7, 20:22] = 2  # D: 4x5x2 = 40
    seg[20:24, 2:6, 20:22] = 2  # E: 4x2x2 = 16
    seg[36:37, 2:3, 20:21] = 2  # F: 5x1x1 = 5
    # Labels 3..25: 23 sparse 3x3x3 blocks on a 12-voxel grid (well
    # separated from each other and from the label 1/2 blobs — no
    # 26-neighbor contact).
    positions = [
        (3, 3, 0),
        (3, 3, 1),
        (3, 3, 2),
        (3, 3, 3),
        (0, 3, 0),
        (1, 3, 0),
        (2, 3, 0),
        (3, 0, 1),
        (3, 1, 1),
        (3, 2, 1),
        (0, 3, 1),
        (1, 3, 1),
        (2, 3, 1),
        (0, 3, 2),
        (1, 3, 2),
        (2, 3, 2),
        (0, 3, 3),
        (1, 3, 3),
        (2, 3, 3),
        (0, 0, 1),
        (1, 1, 2),
        (2, 2, 3),
        (3, 1, 3),
    ]
    for label, (r, c, s) in zip(range(3, 26), positions):
        seg[r * 12 + 2 : r * 12 + 5, c * 12 + 2 : c * 12 + 5, s * 12 + 2 : s * 12 + 5] = label
    return seg


def _unknown_rule(segmentation, **kwargs):  # noqa: ARG001 — pickled by reference
    """An arbitrary non-ported rule family (pickled into a pkl by test 3)."""
    return segmentation


def main() -> int:
    print("test_part_postprocess (P9 09-03, roadmap 9.3)")
    tmp = Path(tempfile.mkdtemp(prefix="p9_pp_test_"))

    # --- Test 1: identity, live path (no pkl) — same object, zero copy ---
    print("[1] identity (live path, 0 pkl)")
    (tmp / "live" / "jsonpkls").mkdir(parents=True)
    seg = np.zeros((32, 32, 32), dtype=np.uint8)
    seg[2:6, 2:6, 2:6] = 1
    seg[2:6, 20:24, 2:6] = 2
    seg[20:24, 2:6, 20:24] = 3
    out, info = apply_part_postprocess(seg, tmp / "live")
    ok("identity: same object returned (out is seg)", out is seg)
    ok("identity: array unchanged", np.array_equal(out, seg))
    ok("identity: info.no_op is True", info["no_op"] is True)
    ok("identity: info.pkl is None", info["pkl"] is None)
    ok(
        "identity: find_postprocessing_pkl -> None",
        find_postprocessing_pkl(tmp / "live") is None,
    )

    # --- Test 2: dormant path — GPU CC port byte-identical to vendored CPU ---
    print("[2] dormant path: GPU CC port vs vendored nnunetv2 apply_postprocessing")
    vol = build_26label_volume()
    distinct_labels = int(np.unique(vol).size)
    ok(
        "volume carries 26 distinct label values",
        distinct_labels == 26,
        f"got {distinct_labels}",
    )
    kwargs = {"labels_or_regions": [1, 2], "background_label": 0}

    # Vendored CPU reference (the oracle's exact function chain).
    ref = apply_postprocessing(vol.copy(), [remove_all_but_largest_component_from_segmentation], [kwargs])

    # Our path: real pkl on disk -> apply_part_postprocess (GPU CC port).
    model_dir = tmp / "bundle"
    pkl = write_pkl(model_dir, [remove_all_but_largest_component_from_segmentation], [kwargs])
    got, info = apply_part_postprocess(vol.copy(), model_dir)
    ok("dormant: info.no_op is False", info["no_op"] is False)
    ok("dormant: info.pkl points at the written pkl", info["pkl"] == str(pkl))
    ok(
        "dormant: one rule recorded",
        len(info["rules"]) == 1 and info["rules"][0] == RULE_KEEP_LARGEST,
        f"rules={info['rules']}",
    )
    ok(
        "dormant: BYTE-IDENTICAL to vendored reference",
        np.array_equal(ref, got),
        f"{int((ref != got).sum())} voxels differ",
    )

    # Tie-case assertions (acvl keeps BOTH 40-voxel blobs A and D).
    ok("tie: blob A (label 1, 40 vox) kept", got[4, 4, 3] == 1, f"got {got[4, 4, 3]}")
    ok("tie: blob D (label 2, 40 vox) kept", got[4, 4, 21] == 2, f"got {got[4, 4, 21]}")
    ok(
        "small blob B (label 1, 16 vox) removed",
        got[4, 21, 3] == 0,
        f"got {got[4, 21, 3]}",
    )
    ok(
        "small blob E (label 2, 16 vox) removed",
        got[22, 4, 21] == 0,
        f"got {got[22, 4, 21]}",
    )
    ok(
        "small blob C (label 1, 5 vox) removed",
        got[4, 36, 2] == 0,
        f"got {got[4, 36, 2]}",
    )
    ok(
        "small blob F (label 2, 5 vox) removed",
        got[36, 2, 20] == 0,
        f"got {got[36, 2, 20]}",
    )
    ok(
        "untouched labels 3..25 preserved",
        np.array_equal(vol[36:39, 38:41, 2:5], got[36:39, 38:41, 2:5])
        and np.array_equal(vol[2:5, 2:5, 14:17], got[2:5, 2:5, 14:17]),
    )

    # --- Test 3: non-ported rule raises loudly ---
    print("[3] non-ported rule family")
    model_dir3 = tmp / "bad_rule"
    write_pkl(model_dir3, [_unknown_rule], [{"foo": 1}])
    try:
        apply_part_postprocess(vol.copy(), model_dir3)
        ok("non-ported rule raises NotImplementedError", False, "no exception raised")
    except NotImplementedError as e:
        ok("non-ported rule raises NotImplementedError", True)
        ok("error names the function", "_unknown_rule" in str(e), str(e))
    except Exception as e:  # noqa: BLE001
        ok(
            "non-ported rule raises NotImplementedError",
            False,
            f"wrong exception: {type(e).__name__}: {e}",
        )

    # --- Test 4: region-tuple kwargs raise (no silent pass) ---
    print("[4] region-tuple guard")
    model_dir4 = tmp / "region"
    write_pkl(
        model_dir4,
        [remove_all_but_largest_component_from_segmentation],
        [{"labels_or_regions": [(1, 2)], "background_label": 0}],
    )
    try:
        apply_part_postprocess(vol.copy(), model_dir4)
        ok("region tuple raises", False, "no exception raised")
    except (ValueError, NotImplementedError) as e:
        ok("region tuple raises", True, f"{type(e).__name__}")
    except Exception as e:  # noqa: BLE001
        ok("region tuple raises", False, f"wrong exception: {type(e).__name__}: {e}")

    print()
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}):")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("ALL PASS (4/4 cases, incl. GPU-vs-CPU byte test with size tie)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
