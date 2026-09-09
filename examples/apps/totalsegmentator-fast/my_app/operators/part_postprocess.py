# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-part postprocess (P9 task 09-03 / roadmap 9.3, PIP-06).

Oracle anchor (ct-totalsegmentator-map, pinned SHA — oracle/nnunet_bundle.py,
``PostProcessNNUnet``):

    self._has_postprocessing = pp_pkl_file_path is not None and os.path.exists(pp_pkl_file_path)
    if self._has_postprocessing:
        self.pp_fns, self.pp_fn_kwargs = load_pickle(pp_pkl_file_path)
    else:
        logging.info(f"No postprocessing pkl found at '{...}'; PostProcessNNUnet will be a no-op.")

    # __call__:
    if not self._has_postprocessing:
        return data   # TRUE identity — data untouched, no copy

The oracle checks ``<part_path>/jsonpkls/postprocessing.pkl`` and runs the
rules per-part in MODEL space, PRE-merge (inside ``_post_process_for_part``,
before the 5-part merge). This module replicates those semantics exactly.

0-pkl finding (research 09-RESEARCH §3, re-verified at 09-03 execution):
all 5 shipped bundles contain ZERO ``postprocessing.pkl`` files
(``find models/total -name "*.pkl" | wc -l`` -> 0), so the LIVE path is a
true identity no-op — the input array is returned as the SAME OBJECT (zero
copy, zero compute), matching the oracle's ``return data`` line.

Correct-but-dormant decision (roadmap 9.3): if a pkl DOES appear, its rules
are applied per-part pre-merge. The only supported rule family is
``remove_all_but_largest_component_from_segmentation`` (the only family
nnUNet 2.8.1's ``determine_postprocessing`` writes), routed to the
v1.1-proven GPU CC port ``remove_all_but_largest_component_gpu`` (CuPy
two-pass, 26-neighbor full connectivity, max-size ties KEPT — acvl_utils
``generic_filter_components`` semantics; byte-tested vs the vendored
``nnunetv2.postprocessing.apply_postprocessing`` in
``scripts/test_part_postprocess.py``). ANY other rule family raises
``NotImplementedError`` loudly — a silent no-op on real data is the PIP-06
failure mode.

Setup-time-only pkl stat: a deployed bundle's pkl cannot appear at runtime,
so the existence check happens ONCE in the operator's ``setup()`` (cached
boolean); the live per-study path is a pure-Python branch with no
filesystem access and no D2H.

PIP-06 scope: postprocess runs PER-PART in model space, NEVER on the merged
117-label map. The v1.1 airway second-stage keep-largest transform (the
MONAI applied-labels keep-largest on label 1) is NOT a pkl rule family and
is deliberately NOT ported into this 5-part path — this module dispatches
ONLY on pkl rule fns (the airway-stage-name grep over this file is the
code proof; it must stay zero-hit).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

__all__ = [
    "RULE_KEEP_LARGEST",
    "find_postprocessing_pkl",
    "apply_part_postprocess",
]

# The only rule family nnUNet 2.8.1's determine_postprocessing writes into
# postprocessing.pkl — matched by "<module>.<qualname>". This is exactly
# what the oracle's PostProcessNNUnet would run; the GPU port
# (remove_all_but_largest_component_gpu) is v1.1-proven voxel-identical.
RULE_KEEP_LARGEST = (
    "nnunetv2.postprocessing.remove_connected_components." "remove_all_but_largest_component_from_segmentation"
)


def find_postprocessing_pkl(model_path: Union[str, Path]) -> Optional[Path]:
    """Return ``<model_path>/jsonpkls/postprocessing.pkl`` if it exists, else None.

    Oracle's checked path (nnunet_seg_operator.py:771 —
    ``jsonpkls/postprocessing.pkl`` per part bundle). Call this ONCE at
    setup; a deployed bundle's pkl cannot appear at runtime.
    """
    pkl = Path(model_path) / "jsonpkls" / "postprocessing.pkl"
    return pkl if pkl.exists() else None


def apply_part_postprocess(seg_np: np.ndarray, model_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the part's ``postprocessing.pkl`` rules to a per-part,
    model-space (SAR) argmax array — the oracle's exact placement
    (PRE-merge). PURE function (CPU numpy in/out) for testability; the
    GPU CC port is invoked internally when a pkl rule requires it.

    Args:
        seg_np: uint8 3D segmentation (per-part local label space).
        model_path: part bundle root (contains ``jsonpkls/``).

    Returns:
        ``(seg, info)``:
          * no pkl  -> the SAME OBJECT passed in (true identity, zero copy,
            zero compute — oracle parity) with
            ``info = {"no_op": True, "pkl": None}``.
          * pkl     -> the post-processed array with
            ``info = {"no_op": False, "pkl": str(pkl), "rules": [...]}``.

    Raises:
        NotImplementedError: any pkl rule that is not
            ``remove_all_but_largest_component_from_segmentation`` (loud —
            never a silent no-op on real data).
    """
    pkl = find_postprocessing_pkl(model_path)
    if pkl is None:
        return seg_np, {"no_op": True, "pkl": None}

    with open(pkl, "rb") as f:
        pp_fns, pp_fn_kwargs = pickle.load(f)

    applied: list = []
    for fn, kwargs in zip(pp_fns, pp_fn_kwargs):
        rule_name = f"{fn.__module__}.{fn.__qualname__}"
        if rule_name != RULE_KEEP_LARGEST:
            raise NotImplementedError(
                f"dormant-path postprocessing rule not ported: {rule_name} "
                "(only remove_all_but_largest_component_from_segmentation is "
                "supported — refusing to silently skip a rule on real data)"
            )
        # Function-local: CuPy + the GPU-port operator import only on the
        # dormant path, so this module stays importable headless without GPU.
        import cupy as cp

        try:  # package-style import (my_app.*)
            from my_app.operators.postprocess_operator import (
                remove_all_but_largest_component_gpu,
            )
        except ImportError:  # flat import (my_app dir on sys.path)
            from operators.postprocess_operator import (  # type: ignore[no-redef]
                remove_all_but_largest_component_gpu,
            )

        labels_or_regions = kwargs.get("labels_or_regions", [1])
        if not isinstance(labels_or_regions, (list, tuple)):
            labels_or_regions = [labels_or_regions]
        background_label = int(kwargs.get("background_label", 0))

        # CPU numpy -> CuPy -> GPU two-pass CC (in-place) -> CPU numpy.
        seg_gpu = cp.asarray(seg_np)
        remove_all_but_largest_component_gpu(seg_gpu, list(labels_or_regions), background_label)
        seg_np = np.ascontiguousarray(seg_gpu.get())
        applied.append(rule_name)

    return seg_np, {"no_op": False, "pkl": str(pkl), "rules": applied}
