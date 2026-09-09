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

"""SegEmitOperator: terminal numpy emit for segmentations.

mode="single" (P8, byte-identical):

* ``seg_<part>_modelspace.npy`` — uint8, model-resolution (1.5 mm),
  slidewindow-native (nnUNet post-transpose) order. The 08-05 GATE TARGET:
  byte-compared against the oracle's pre-merge, pre-back-resample per-part dump.
* ``seg_<part>_dicom.npy`` — uint8, original-DICOM-orientation
  (crop/transpose-reverted at model resolution).
* ``emit_meta.json`` — part, offset, shapes, dtypes, label histograms,
  study id, and the pre-resample metadata carried by the pipeline.

mode="total" (P9, plan 09-02, decision D-9-3):

* ``seg_total_sar.npy`` — uint8, original spacing, SAR (pre-flip) — the
  09-04 GATE TARGET vs the oracle's post-merge, post-back-resample SAR dump.
* ``seg_total_dhw.npy`` — uint8, original spacing, DHW (DICOM orientation) —
  the exact array Phase 10's SEG writer consumes.
* ``emit_meta.json`` — mode, shapes, dtypes, 0..117 label histograms,
  study id, pre-resample metadata, wall time.

Terminal operator: declares NO outputs (satisfies the GXF "declared output
needs a receiver" rule). Zoom factors live in the merge's
``back_resample record:`` log line (09-01), not on any port.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from monai.deploy.core import Operator, OperatorSpec

try:  # package-style import (my_app.*)
    from my_app.config import MODEL_PARTS
    from my_app.operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        get_study_id,
        nvtx_range,
    )
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from config import MODEL_PARTS
    from operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        get_study_id,
        nvtx_range,
    )

__all__ = ["SegEmitOperator"]

# 24 pre-closeout (user-directed, 2026-09-07): gate for the debug/terminal
# numpy .npy + emit_meta.json writes. Default "1" (or any non-"0" value)
# keeps the current behavior; "0" disables the disk dumps. Tensor data flows
# via ports only — nothing in the flow reads these files — so gating them is
# behavior-neutral for the DICOM-SEG/SR outputs. The shipped MAP manifests
# bake HOLOSCAN_EMIT_NPY=0.
_EMIT_NPY_ENABLED = os.environ.get("HOLOSCAN_EMIT_NPY", "1") != "0"


def _to_cpu_uint8(x: Any) -> np.ndarray:
    """Normalize a received seg (holoscan PyTensor / torch tensor / ndarray)
    to a contiguous CPU numpy uint8 array.

    holoscan PyTensors cross the boundary via DLPack (same proven path as
    MergeOperator — bare np.asarray on a PyTensor raises TypeError).
    """
    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(np.asarray(x, dtype=np.uint8))
    import torch

    t = x if hasattr(x, "detach") else torch.utils.dlpack.from_dlpack(x)
    out = t.detach().to("cpu").numpy()
    return np.ascontiguousarray(np.asarray(out, dtype=np.uint8))


def _histogram(a: np.ndarray) -> dict:
    """Label histogram {label: count} for a uint8 array (sparse output)."""
    counts = np.bincount(a.ravel(), minlength=256)
    return {str(int(lab)): int(c) for lab, c in enumerate(counts) if c > 0}


class SegEmitOperator(Operator):
    """Terminal numpy emit: single-part (P8) or 5-part total (P9, 09-02).

    mode="single" (default, P8 byte-identical):
        Named Inputs:
            seg_merged: uint8 3D model-resolution seg with part offset applied.
            seg_merged_dicom: uint8 3D original-DICOM-orientation seg.
            preprocessed_meta: metadata dict.
        Named Outputs: none (terminal).

    mode="total" (P9 5-part, plan 09-02):
        Named Inputs:
            seg_merged: uint8 3D (SAR, original spacing) from MergeRemapOperator.
            seg_image: uint8 3D (DHW, original spacing) from MergeRemapOperator.
            preprocessed_meta: metadata dict.
        Named Outputs: none (terminal).
        Writes: seg_total_sar.npy (09-04 gate target) + seg_total_dhw.npy +
        emit_meta.json; NVTX/timing label ``emit_5part``.
    """

    INPUT_MERGED = "seg_merged"
    INPUT_DICOM = "seg_merged_dicom"
    INPUT_DHW = "seg_image"  # P9 total mode: DHW array from MergeRemapOperator
    INPUT_META = "preprocessed_meta"

    def __init__(
        self,
        fragment: Any,
        *args: Any,
        part: str = "organs",
        output_dir: Optional[Union[str, Path]] = None,
        mode: str = "single",
        base_name: str = "total",
        **kwargs: Any,
    ):
        """Create the operator.

        Args:
            fragment: the owning application (passed to ``Operator``).
            part: model part name (mode="single" only); must be a key of
                ``MODEL_PARTS``.
            output_dir: directory for the .npy + JSON writes (the app's
                ``-o`` path, plumbed from app.py).
            mode: P9 (09-02). ``"single"`` (default) = P8 single-part emit,
                byte-identical behavior. ``"total"`` = 5-part terminal emit:
                inputs are exactly ``seg_merged`` (SAR) + ``seg_image`` (DHW)
                + ``preprocessed_meta``; writes ``seg_total_sar.npy`` +
                ``seg_total_dhw.npy`` + ``emit_meta.json``; NVTX/timing
                label ``emit_5part``.
        """
        # NOTE: holoscan 4.2's Operator.__init__ invokes self.setup(spec)
        # before this constructor body finishes — initialize all state first.
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        if mode not in ("single", "total"):
            raise ValueError(f"SegEmitOperator mode must be 'single' or 'total', got {mode!r}")
        self._mode = mode
        # 18-02: total-mode emit file prefix. Default "total" keeps the P9/P10
        # filenames byte-identical (seg_total_sar/dhw.npy); the crop_cascade
        # task passes its task name so the p18 runner contract
        # (seg_liver_segments_sar/dhw.npy) is met without touching the total
        # path. The NVTX/timing label stays `emit_5part` for both.
        self._base_name = base_name
        if mode == "single":
            if part not in {p["name"] for p in MODEL_PARTS}:
                raise ValueError(f"Unknown model part {part!r}. Valid: {[p['name'] for p in MODEL_PARTS]}")
            self._part = part
            self._offset = int(next(p for p in MODEL_PARTS if p["name"] == part)["label_offset"])
        self._output_dir = Path(output_dir) if output_dir is not None else None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        if self._mode == "total":
            # P9 5-part terminal: exactly 3 declared inputs, all wired in
            # app.py compose(). No outputs (terminal sink).
            spec.input(self.INPUT_MERGED)  # SAR, original spacing
            spec.input(self.INPUT_DHW)  # DHW, original spacing
            spec.input(self.INPUT_META)
            return
        spec.input(self.INPUT_MERGED)
        spec.input(self.INPUT_DICOM)
        spec.input(self.INPUT_META)
        # Terminal: no outputs declared.

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        """Write segmentation .npy files + emit_meta.json to the output dir."""
        if self._mode == "total":
            self._compute_total(op_input, op_output, context)
            return
        # --- P8 single-part mode (unchanged) ---
        with nvtx_range(f"emit_{self._part}"):
            t0 = time.time()
            merged = op_input.receive(self.INPUT_MERGED)
            dicom = op_input.receive(self.INPUT_DICOM)
            meta = op_input.receive(self.INPUT_META) or {}
            if merged is None:
                raise ValueError("SegEmitOperator received no 'seg_merged' input.")
            if dicom is None:
                raise ValueError("SegEmitOperator received no 'seg_merged_dicom' input.")

            merged = _to_cpu_uint8(merged)
            dicom = _to_cpu_uint8(dicom)
            if merged.ndim != 3 or dicom.ndim != 3:
                raise ValueError(f"expected 3D segmentations, got shapes {merged.shape} / {dicom.shape}")
            if merged.shape != dicom.shape:
                # Different ORIENTATION does not change the shape tuple for
                # axis flips; a shape mismatch signals a wiring bug.
                raise ValueError(f"modelspace/dicom shape mismatch: {merged.shape} vs {dicom.shape}")

            if not _EMIT_NPY_ENABLED:
                self._logger.info("emit disabled (HOLOSCAN_EMIT_NPY=0)")
                return

            out_dir = self._output_dir
            if out_dir is None:
                raise RuntimeError("SegEmitOperator requires output_dir (plumb the app -o path).")
            out_dir.mkdir(parents=True, exist_ok=True)

            modelspace_path = out_dir / f"seg_{self._part}_modelspace.npy"
            dicom_path = out_dir / f"seg_{self._part}_dicom.npy"
            np.save(modelspace_path, merged)
            self._logger.info(
                "wrote %s shape=%s dtype=%s max=%d", modelspace_path, merged.shape, merged.dtype, int(merged.max())
            )
            np.save(dicom_path, dicom)
            self._logger.info(
                "wrote %s shape=%s dtype=%s max=%d", dicom_path, dicom.shape, dicom.dtype, int(dicom.max())
            )

            emit_meta = {
                "part": self._part,
                "label_offset": self._offset,
                "modelspace": {
                    "path": str(modelspace_path),
                    "shape": list(merged.shape),
                    "dtype": "uint8",
                    "label_histogram": _histogram(merged),
                },
                "dicom": {
                    "path": str(dicom_path),
                    "shape": list(dicom.shape),
                    "dtype": "uint8",
                    "label_histogram": _histogram(dicom),
                },
                "note": (
                    "modelspace = model-resolution (1.5 mm) slidewindow-native "
                    "order — the 08-05 gate target; dicom = crop/transpose-reverted "
                    "original-DICOM-orientation at model resolution (Phase 9 "
                    "back-resamples it to original spacing)."
                ),
                "study_id": get_study_id(self.fragment),
                "_pre_resample_spatial_shape": meta.get("_pre_resample_spatial_shape"),
                "pre_resample_skipped": meta.get("pre_resample_skipped"),
                "pre_resample_spacing": meta.get("pre_resample_spacing"),
                "wall_time_s": round(time.time() - t0, 4),
            }
            meta_path = out_dir / "emit_meta.json"
            meta_path.write_text(json.dumps(emit_meta, indent=2))
            self._logger.info("wrote %s", meta_path)
            # Span-name log line (plan 08-05 NVTX check greps this): the
            # nvtx_range above wraps this whole compute() as `emit_<part>`.
            self._logger.info(
                "emit record: op=emit_%s wall_time_s=%s max_label=%d",
                self._part,
                emit_meta["wall_time_s"],
                int(merged.max()),
            )

    # ------------------------------------------------------------------
    # P9 5-part total mode (plan 09-02, decision D-9-3)
    # ------------------------------------------------------------------

    def _compute_total(self, op_input: Any, op_output: Any, context: Any) -> None:
        """P9 5-part terminal emit: write seg_total_sar.npy + seg_total_dhw.npy
        + emit_meta.json; NVTX/timing label ``emit_5part``.

        Inputs (exactly 3, all wired in app.py):
            seg_merged  — SAR, original spacing (09-04 GATE TARGET).
            seg_image   — DHW, original spacing (SEG-writer orientation).
            preprocessed_meta — metadata dict.
        """
        with nvtx_range("emit_5part"):
            t0 = time.time()
            timing = GpuTiming("emit_5part")
            timing.start()

            sar = op_input.receive(self.INPUT_MERGED)
            dhw = op_input.receive(self.INPUT_DHW)
            meta = op_input.receive(self.INPUT_META) or {}
            if sar is None or dhw is None:
                raise ValueError("SegEmitOperator (total) received no 'seg_merged' or 'seg_image' input.")

            sar = _to_cpu_uint8(sar)
            dhw = _to_cpu_uint8(dhw)
            # Guard: ndim + non-empty only (plan 09-02). The SAR->DHW flip
            # preserves the shape tuple, so a mismatch is suspicious but we
            # record both shapes rather than raise (anisotropic in-plane
            # could theoretically differ in other corpora).
            for tag, a in (("sar", sar), ("dhw", dhw)):
                if a.ndim != 3 or a.size == 0:
                    raise ValueError(f"total emit: {tag} array must be non-empty 3D, got shape {a.shape}")
            if sar.shape != dhw.shape:
                self._logger.warning(
                    "total emit: sar/dhw shape mismatch sar=%s dhw=%s (recording both)",
                    sar.shape,
                    dhw.shape,
                )

            if not _EMIT_NPY_ENABLED:
                self._logger.info("emit disabled (HOLOSCAN_EMIT_NPY=0)")
                return

            out_dir = self._output_dir
            if out_dir is None:
                raise RuntimeError("SegEmitOperator requires output_dir (plumb the app -o path).")
            out_dir.mkdir(parents=True, exist_ok=True)

            sar_path = out_dir / f"seg_{self._base_name}_sar.npy"
            dhw_path = out_dir / f"seg_{self._base_name}_dhw.npy"
            np.save(sar_path, sar)
            self._logger.info("wrote %s shape=%s dtype=%s max=%d", sar_path, sar.shape, sar.dtype, int(sar.max()))
            np.save(dhw_path, dhw)
            self._logger.info("wrote %s shape=%s dtype=%s max=%d", dhw_path, dhw.shape, dhw.dtype, int(dhw.max()))

            wall_time_s = round(time.time() - t0, 4)
            emit_meta = {
                "mode": "total",
                "base_name": self._base_name,
                "sar": {
                    "path": str(sar_path),
                    "shape": list(sar.shape),
                    "dtype": "uint8",
                    "label_histogram": _histogram(sar),
                },
                "dhw": {
                    "path": str(dhw_path),
                    "shape": list(dhw.shape),
                    "dtype": "uint8",
                    "label_histogram": _histogram(dhw),
                },
                "study_id": get_study_id(self.fragment),
                "_pre_resample_spatial_shape": meta.get("_pre_resample_spatial_shape"),
                "pre_resample_skipped": meta.get("pre_resample_skipped"),
                "pre_resample_spacing": meta.get("pre_resample_spacing"),
                "wall_time_s": wall_time_s,
            }
            meta_path = out_dir / "emit_meta.json"
            meta_path.write_text(json.dumps(emit_meta, indent=2))
            self._logger.info("wrote %s", meta_path)

            # House-standard GpuTiming record (09-04 VRAM report selects
            # emit_5part among >=6 boundaries).
            record = timing.stop()
            record["study"] = get_study_id(self.fragment)
            record["max_label"] = int(sar.max())
            record["shape"] = list(sar.shape)
            StudyTimingCollector.record(self.fragment, record)
            self._logger.info("timing: %s", json.dumps(record))

            # Custom emit record line (plan 09-02 Task 3 greps this).
            self._logger.info(
                "emit record: op=emit_5part wall_time_s=%s max_label=%d",
                wall_time_s,
                int(sar.max()),
            )
