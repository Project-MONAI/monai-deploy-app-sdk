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

"""MergeOperator: P8 degenerate single-part merge (plan 08-05).

Receives the per-part model-resolution segmentation (uint8 3D CUDA tensor)
and produces the "merged" segmentation by adding the part's label offset
from ``config.MODEL_PARTS`` to every non-background label — the oracle's
exact merge semantics (ct-totalsegmentator-map/app_total/
nnunet_seg_operator.py:703-723) in one pass. For organs (offset 0) this is
the identity; for all-background input (max == 0) the pass is a legal no-op.

P9 note: Phase 9 replaces this with the 5-port MergeRemapOperator
(CountCondition on 6 inputs, fixed-order loop over organs/vertebrae/
cardiac/muscles/ribs — roadmap 9.1). This class stays minimal and the
merge math is exposed headless-testable as ``merge_single_part()``.

The output is a CPU numpy uint8 array (merge is the single deliberate D2H
point on this path — the terminal SegEmitOperator writes .npy files, so
keeping the merged array on CPU here avoids an extra GPU round-trip).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from monai.deploy.core import Operator, OperatorSpec

try:  # package-style import (my_app.*)
    from my_app.config import EXPECTED_MAX_LOCAL_LABEL, MODEL_PARTS
    from my_app.operators.gpu_util import (
        assert_cuda_available,
        assert_on_gpu,
        get_study_id,
        nvtx_range,
    )
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from config import EXPECTED_MAX_LOCAL_LABEL, MODEL_PARTS
    from operators.gpu_util import (
        assert_cuda_available,
        assert_on_gpu,
        get_study_id,
        nvtx_range,
    )

__all__ = ["MergeOperator", "merge_single_part"]

# P9: the per-part head-cap dict moved to my_app.config (single source of
# truth, headless-clean). Alias kept so this module's references unchanged.
_EXPECTED_MAX_LOCAL_LABEL = EXPECTED_MAX_LOCAL_LABEL


def merge_single_part(seg: np.ndarray, part_name: str) -> np.ndarray:
    """Oracle-exact single-part merge: add the part's label offset to all
    non-background labels. Headless-testable (no fragment, no GPU).

    * offset comes from ``config.MODEL_PARTS`` (organs -> 0, identity);
    * max-label guard: ``seg.max() > expected heads`` raises ValueError
      (a part predicting a label outside its own head space is a bug, not
      data — the oracle would silently emit an out-of-range unified label);
    * all-background (max == 0) is legal and returns the input values
      unchanged (offset is only applied where ``seg > 0``).

    Returns a new uint8 array (input is not mutated).
    """
    if part_name not in _EXPECTED_MAX_LOCAL_LABEL:
        raise ValueError(f"Unknown model part {part_name!r}. Valid parts: {sorted(_EXPECTED_MAX_LOCAL_LABEL)}")
    part = next(p for p in MODEL_PARTS if p["name"] == part_name)
    offset = int(part["label_offset"])
    expected_max = _EXPECTED_MAX_LOCAL_LABEL[part_name]

    seg = np.asarray(seg)
    if seg.dtype != np.uint8:
        seg = seg.astype(np.uint8)
    if seg.ndim != 3:
        raise ValueError(f"merge_single_part expects a 3D array, got shape {seg.shape}")
    if seg.size == 0:
        raise ValueError("merge_single_part received an empty segmentation")
    if seg.max() > expected_max:
        raise ValueError(
            f"Part {part_name!r} produced max local label {int(seg.max())} but its model "
            f"only has {expected_max} foreground heads (labels 1-{expected_max})."
        )

    merged = np.where(seg > 0, seg + offset, seg).astype(np.uint8, copy=False)
    return merged


class MergeOperator(Operator):
    """Degenerate single-part merge (P8).

    Named Inputs:
        seg_<part> (e.g. "seg_organs"): uint8 3D CUDA tensor from
            PostResampleOperator's ``seg_argmax`` (model-resolution,
            slidewindow-native order).

    Named Outputs:
        seg_merged: uint8 3D CPU numpy array with the part's label offset
            applied (identity for organs) — consumed by SegEmitOperator.

    P9: replaced by the 5-port MergeRemapOperator (roadmap 9.1).
    """

    OUTPUT_MERGED = "seg_merged"

    def __init__(self, fragment: Any, *args: Any, part: str = "organs", **kwargs: Any):
        """Create the operator.

        Args:
            fragment: the owning application (passed to ``Operator``).
            part: model part name; must be a key of ``MODEL_PARTS``.
                The input port name is derived as ``seg_<part>``.
        """
        # NOTE: holoscan 4.2's Operator.__init__ invokes self.setup(spec)
        # before this constructor body finishes — initialize all state first.
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        if part not in {p["name"] for p in MODEL_PARTS}:
            raise ValueError(f"Unknown model part {part!r}. Valid: {[p['name'] for p in MODEL_PARTS]}")
        self._part = part
        self._input_seg = f"seg_{part}"
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input(self._input_seg)
        spec.output(self.OUTPUT_MERGED)

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        """Merge the single part (offset + max-label guard), emit to emit op."""
        # INFR-005: the NVTX range name carries the part.
        with nvtx_range(f"merge_{self._part}"):
            assert_cuda_available()

            holo_tensor = op_input.receive(self._input_seg)
            if holo_tensor is None:
                raise ValueError(f"MergeOperator received no {self._input_seg!r} input.")

            tensor = torch.utils.dlpack.from_dlpack(holo_tensor)
            assert_on_gpu(tensor)

            # Single deliberate D2H point: the terminal emit writes .npy.
            seg_np = tensor.detach().cpu().numpy().astype(np.uint8, copy=False)
            merged = merge_single_part(seg_np, self._part)

            # Oracle parity log (nnunet_seg_operator.py:731-733):
            labels = np.unique(merged)
            self._logger.info(
                "Part '%s' produced labels %s (max %d) -> merged max %d",
                self._part,
                labels.tolist(),
                int(labels.max()),
                int(merged.max()),
            )

            op_output.emit(merged, self.OUTPUT_MERGED)
            record = {
                "op": f"merge_{self._part}",
                "study": get_study_id(self.fragment),
                "shape": list(merged.shape),
                "max_label": int(merged.max()),
            }
            self._logger.info("merge record: %s", record)
