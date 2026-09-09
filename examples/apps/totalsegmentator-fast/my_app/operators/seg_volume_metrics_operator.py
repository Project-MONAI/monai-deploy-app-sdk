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

"""SegVolumeMetricsOperator: Phase 13 (WS-1) thin operator wrapping the pure
single-pass volume-metrics core (``operators/volume_metrics.py``).

Drop-in replacement for ``my_app/segmentation_metrics_operator.py`` (legacy
per-label CPU loop + scipy connected components, ~96 s at 31322 scale):

* Same named inputs: ``segmentation_mask`` (the ``seg_merged`` 3-D uint8 CPU
  torch tensor from MergeRemapOperator) + ``input_scan`` (DICOM Image, used
  ONLY for its spacing metadata — the scan array itself is never read, which
  is what makes this op fast: no per-label fancy-index of the full scan).
* Same named output: ``metrics_dict`` — ``{organ_name: {"vol": float (mL),
  "num.slices": int, "slice.range": (first, last) | None, "pixel.count": int}}``
  with exactly the 117 organ keys (background excluded), so the DICOM SR
  writer (``included_fields=["vol"]``) and the p10 SR-parity gate are
  untouched.
* Gated by ``HOLOSCAN_GPU_METRICS`` (default ``"1"`` → this op is in the DAG
  by default after the 13-02 post-gate flip; ``HOLOSCAN_GPU_METRICS=0``
  restores the legacy op via app.py selection). If this class is ever
  constructed while the flag is off it raises — a construction failure is a
  wiring bug, never a silent fallback.
* NVTX span name is EXACTLY ``volume_metrics`` and the house timing line is
  ``timing:volume_metrics ms=%f`` — the 13-02 benchmark-parser patch keys on
  both (this kills the legacy ``metrics_est`` ESTIMATED gap, 12-RESEARCH
  Pitfall 6).

Bit-identity guarantee: the vol math in ``volume_metrics.compute_volume_metrics``
mirrors the oracle ``_compute_volume_or_area`` exactly (same float64
expression + evaluation order, spacing read from the SAME Image metadata via
the identical ``_get_spacing`` extraction), so SR ``vol`` values are
bit-identical to the legacy op on the same input (proven headless on the
44238 pin by ``scripts/test_volume_metrics.py``).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np

from monai.deploy.core import ConditionType, Fragment, Operator, OperatorSpec
from monai.deploy.core.domain.image import Image

if TYPE_CHECKING:  # annotation only — no torch runtime dependency in this op
    import torch

try:  # package-style import (my_app.*)
    from my_app.operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        get_study_id,
        nvtx_range,
    )
    from my_app.operators.volume_metrics import compute_slice_stats
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        get_study_id,
        nvtx_range,
    )
    from operators.volume_metrics import compute_slice_stats

__all__ = ["SegVolumeMetricsOperator"]


def _get_spacing(image_obj: Union["torch.Tensor", np.ndarray, Image]) -> Optional[Tuple[float, ...]]:
    """Extract spacing from Image object metadata.

    VERBATIM copy of the legacy ``SegmentationMetricsOperator._get_spacing``
    (my_app/segmentation_metrics_operator.py) — the spacing MUST come from the
    same source with the same extraction order so the float inputs to the
    volume math are identical (bit-identity contract, 13-01).
    """
    if not isinstance(image_obj, Image):
        raise ValueError("Spacing required: input must be an Image with metadata containing spacing.")

    metadata = image_obj.metadata() or {}

    spacing = None
    if metadata:
        # Try common spacing keys in order of preference
        spacing = metadata.get("spacing") or metadata.get("pixdim") or metadata.get("pixel_spacing")

        # If not found, try DICOM-specific pixel spacing keys
        if spacing is None:
            row_spacing = metadata.get("row_pixel_spacing")
            col_spacing = metadata.get("col_pixel_spacing")
            depth_spacing = metadata.get("depth_pixel_spacing")

            if row_spacing is not None and col_spacing is not None and depth_spacing is not None:
                spacing = (float(row_spacing), float(col_spacing), float(depth_spacing))

        if spacing is not None and not isinstance(spacing, (list, tuple, np.ndarray)):
            raise ValueError(f"Spacing required: expected list/tuple/ndarray, got {type(spacing).__name__}.")

    if spacing is None:
        affine = getattr(image_obj, "affine", None)
        if affine is not None:
            affine_arr = np.asarray(affine)
            if affine_arr.shape[0] < 3 or affine_arr.shape[1] < 3:
                raise ValueError("Spacing required: affine matrix missing spatial axes.")
            spacing = (
                float(np.linalg.norm(affine_arr[:3, 0])),
                float(np.linalg.norm(affine_arr[:3, 1])),
                float(np.linalg.norm(affine_arr[:3, 2])),
            )
        else:
            raise ValueError(
                "Spacing required: metadata missing and affine attribute not available for spacing extraction."
            )

    return tuple(spacing)


class SegVolumeMetricsOperator(Operator):
    """Single-pass SR volume metrics (13-01; replaces the 96 s CPU per-label
    loop; see module docstring for the contract and gating).

    Named Inputs (2):
        segmentation_mask: 3-D (D,H,W) uint8 CPU torch tensor (``seg_merged``
            from MergeRemapOperator). A 4-D (1,D,H,W) tensor is accepted
            defensively (leading singleton dropped).
        input_scan: DICOM Image — used ONLY for spacing metadata (the scan
            array is never read; the intensity stats + connected components
            the legacy op computed from it are non-gated and dropped).

    Named Output (1):
        metrics_dict: {organ_name: {"vol", "num.slices", "slice.range",
            "pixel.count"}} — 117 keys, same shape the legacy op emitted.
    """

    INPUT_SEG = "segmentation_mask"
    INPUT_SCAN = "input_scan"
    OUTPUT_METRICS = "metrics_dict"

    def __init__(
        self,
        fragment: Fragment,
        *args: Any,
        labels_dict: Optional[Dict[str, int]] = None,
        **kwargs: Any,
    ):
        """Create the operator.

        Args:
            fragment: the owning application (passed to ``Operator``).
            labels_dict: organ-name -> label-value map (same dict app.py passes
                to the legacy op: the 117 organ entries of ``volume_labels``
                with ``"background"`` excluded).
        """
        # NOTE: holoscan 4.2's Operator.__init__ invokes self.setup(spec)
        # before this constructor body finishes — initialize all state first.
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.input_name_seg_mask = self.INPUT_SEG
        self.input_name_scan = self.INPUT_SCAN
        self.output_name_metrics = self.OUTPUT_METRICS
        self.labels_dict: Dict[str, int] = labels_dict if labels_dict is not None else {}

        # Gate: default ON (13-02 Task 5 post-gate flip; matches app.py's
        # single read site). If this class is constructed while the flag is
        # explicitly off, that is a wiring bug — raise instead of silently
        # running a non-default path.
        self._enabled = os.environ.get("HOLOSCAN_GPU_METRICS", "1") == "1"
        if not self._enabled:
            raise RuntimeError(
                "SegVolumeMetricsOperator was constructed while HOLOSCAN_GPU_METRICS is not "
                "'1' (default '1'; set '0' to select the legacy SegmentationMetricsOperator). "
                "Either set HOLOSCAN_GPU_METRICS=1 or wire the legacy op in app.py (13-02)."
            )

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input(self.input_name_seg_mask)
        spec.input(self.input_name_scan)
        spec.output(self.output_name_metrics).condition(ConditionType.NONE)

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        """Receive seg_merged + scan Image, run the single-pass metrics core,
        emit the legacy-shaped metrics_dict. NVTX span: exactly 'volume_metrics'."""
        segmentation_mask = op_input.receive(self.input_name_seg_mask)
        input_scan = op_input.receive(self.input_name_scan)

        if segmentation_mask is None:
            raise ValueError("SegVolumeMetricsOperator received no 'segmentation_mask' input.")
        if input_scan is None:
            raise ValueError("SegVolumeMetricsOperator received no 'input_scan' input.")

        # Spacing from the SAME Image metadata the legacy op used (verbatim
        # _get_spacing) — required for bit-identical vol math.
        spacing = _get_spacing(input_scan)

        with nvtx_range("volume_metrics"):  # 13-02 benchmark parser keys on this span
            timing = GpuTiming("volume_metrics")
            timing.start()

            # Single pass: bincount counts + (label, slice) presence. No scan
            # read, no scipy ndimage, no per-label Python loop.
            per_label = compute_slice_stats(segmentation_mask, spacing, num_labels=117)

            # Re-key by organ name: exactly the legacy dict shape
            # ({name: {vol, num.slices, slice.range, pixel.count}}, 117 keys).
            metrics: Dict[str, Dict[str, Any]] = {}
            for name, idx in self.labels_dict.items():
                entry = per_label.get(idx)
                if entry is None:  # labels_dict entry outside 1..117 (defensive)
                    entry = {"vol": 0.0, "num.slices": 0, "slice.range": None, "pixel.count": 0}
                metrics[name] = dict(entry)

            record = timing.stop()
            ms = record["duration_ms"]

        self._logger.info("timing:volume_metrics ms=%f", ms)  # house timing line (13-02 parser)
        record["study"] = get_study_id(self.fragment)
        record["n_labels"] = len(metrics)
        record["n_nonzero"] = sum(1 for e in metrics.values() if e["pixel.count"] > 0)
        StudyTimingCollector.record(self.fragment, record)
        self._logger.info("timing: %s", __import__("json").dumps(record))

        op_output.emit(metrics, self.output_name_metrics)
