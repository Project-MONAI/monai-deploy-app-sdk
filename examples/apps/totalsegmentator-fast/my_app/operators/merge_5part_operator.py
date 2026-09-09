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

"""MergeRemapOperator: P9 keystone — 5-part merge + back-resample + flip (roadmap 9.1/9.4, plan 09-01).

Receives the five per-part model-resolution segmentations (uint8 3D CUDA
tensors from PostResampleOperator ``seg_argmax``, model-resolution 1.5 mm,
SAR order) plus the ``preprocessed_meta`` fan-out, and produces:

* ``seg_merged`` — the unified 117-label map (SAR, ORIGINAL spacing,
  uint8 CPU torch tensor). This is the 9.4 GATE TARGET: byte-compared
  against the oracle's post-merge, post-back-resample SAR dump. Emitted
  as a torch tensor (not bare numpy) to match the oracle's metrics-input
  contract: the oracle's SegmentationMetricsOperator probes
  ``mask.device.type``, which crashes on numpy 2.2 where
  ``ndarray.device`` is the str ``'cpu'`` (P10 bug: smoke 44238
  AttributeError in seg_metrics_op).
* ``seg_image`` — the SAME map flipped SAR->DHW once (oracle
  compute_impl :785-792), original spacing, uint8 — the orientation the
  SEG writer consumes.

Oracle anchors (pinned SHA 7676d95, ct-totalsegmentator-map/app_total/
nnunet_seg_operator.py):
* merge: fixed-order later-part-wins loop, non-background-only writes
  (:672-748); head caps 24/26/18/23/26 (config.EXPECTED_MAX_LOCAL_LABEL);
* back-resample: ONE scipy ndimage.zoom(order=0, mode="nearest") on the
  MERGED seg, factors = reversed(_pre_resample_spatial_shape) / current
  SAR shape, fp64 widen, uint8 back-cast (:154-222);
* flip: single np.flip(axis=[1,2]) after back-resample (:785-792).

The math lives in the pure, headless-testable ``merge_remap`` module
(scripts/test_merge_remap.py); this operator only moves data and logs.

PIP-06 (09-03): per-part postprocess runs PRE-merge, on each part's own
local-label map — NEVER on this 117-label map.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from monai.deploy.conditions import CountCondition
from monai.deploy.core import Operator, OperatorSpec

try:  # package-style import (my_app.*)
    from my_app.config import MODEL_PARTS
    from my_app.operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        get_study_id,
        nvtx_range,
    )
    from my_app.operators.merge_remap import (
        back_resample_merged,
        flip_sar_to_dhw,
        flip_sar_to_writer,
        merge_remap,
    )
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from config import MODEL_PARTS
    from operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        get_study_id,
        nvtx_range,
    )
    from operators.merge_remap import (
        back_resample_merged,
        flip_sar_to_dhw,
        flip_sar_to_writer,
        merge_remap,
    )

__all__ = ["MergeRemapOperator"]


class MergeRemapOperator(Operator):
    """5-part merge + single post-merge back-resample + SAR->DHW flip.

    Named Inputs (6 — explicit CountCondition(self, 6) is the Pitfall-5
    silent-hang guard; the operator NEVER runs on a partial fan-in):
        seg_organs, seg_vertebrae, seg_cardiac, seg_muscles, seg_ribs:
            uint8 3D CUDA tensors (PostResampleOperator seg_argmax, model
            resolution, SAR).
        preprocessed_meta: the preprocess fan-out dict carrying
            ``_pre_resample_spatial_shape`` (RAS) + ``pre_resample_skipped``
            (08-02 keys, present on BOTH skip and resample branches).

    Named Outputs (3):
        seg_merged: uint8 3D CPU torch tensor (SAR, original spacing) — the
            metrics path and the 9.4 gate target.
        seg_image: uint8 3D CPU numpy (source-series slice order, in-plane
            flipped [1,2]) — the oracle-exact SEG-writer input (P10).
        seg_dhw: uint8 3D CPU numpy (spatial DHW = decoded SEG frame order,
            flip [0,1,2]) — the terminal emit's seg_total_dhw.npy target.
    """

    INPUT_SEGS = ["seg_organs", "seg_vertebrae", "seg_cardiac", "seg_muscles", "seg_ribs"]
    INPUT_META = "preprocessed_meta"
    OUTPUT_MERGED = "seg_merged"
    OUTPUT_IMAGE = "seg_image"  # SEG-writer input: flip(1,2) only (oracle-exact, P10)
    OUTPUT_DHW = "seg_dhw"  # spatial DHW (decoded SEG frame order): flip(0,1,2)

    def __init__(self, fragment: Any, *args: Any, output_dir: Optional[Path] = None, **kwargs: Any):
        """Create the operator.

        Args:
            fragment: the owning application (passed to ``Operator``).
            output_dir: optional dir for the ``HOLOSCAN_EMIT_PART_SEGS=1``
                triage dump of raw per-part model-space arrays (D-9-3).
        """
        # NOTE: holoscan 4.2's Operator.__init__ invokes self.setup(spec)
        # before this constructor body finishes — initialize all state first.
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._output_dir = Path(output_dir) if output_dir else None
        # P9 (09-02) fix: this holoscan version has no spec.add_condition —
        # conditions are constructor args (see DICOMDataLoaderOperator in
        # app.py). The explicit 6-input condition (Pitfall-5 hang guard) is
        # therefore injected here, keeping the caller signature unchanged.
        super().__init__(fragment, CountCondition(fragment, 6), *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("seg_organs")
        spec.input("seg_vertebrae")
        spec.input("seg_cardiac")
        spec.input("seg_muscles")
        spec.input("seg_ribs")
        spec.input("preprocessed_meta")
        spec.output(self.OUTPUT_MERGED)
        spec.output(self.OUTPUT_IMAGE)
        spec.output(self.OUTPUT_DHW)
        # Pitfall 5: explicit 6-input condition (all 5 parts + meta must
        # arrive before compute() fires — no silent hang on partial fan-in).
        # Applied in __init__ as a constructor arg (this holoscan build has
        # no OperatorSpec.add_condition).

    def _receive_part(self, op_input: Any, name: str) -> np.ndarray:
        """Receive one part tensor; single deliberate D2H point (P8 merge
        precedent: the terminal emit writes .npy)."""
        holo_tensor = op_input.receive(name)
        if holo_tensor is None:
            raise ValueError(f"MergeRemapOperator received no {name!r} input.")
        tensor = torch.utils.dlpack.from_dlpack(holo_tensor)
        assert_cuda_available()
        return tensor.detach().cpu().numpy().astype(np.uint8, copy=False)

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        """Merge (later-part-wins) -> back-resample (order-0) -> flip to DHW."""
        # INFR-005: NVTX range carries the stage name; the GpuTiming record
        # (parseable `timing:` line, ISO-8601 end) is a 09-04 VRAM-report
        # boundary (p9_vram_report selects merge_5part among >=6 boundaries).
        with nvtx_range("merge_5part"):
            timing = GpuTiming("merge_5part")
            timing.start()

            segs = {name: self._receive_part(op_input, name) for name in self.INPUT_SEGS}

            meta = op_input.receive(self.INPUT_META) or {}
            if "_pre_resample_spatial_shape" not in meta:
                raise ValueError(
                    "MergeRemapOperator: preprocessed_meta is missing "
                    "'_pre_resample_spatial_shape' — the back-resample MUST use "
                    "pre-resample metadata (08-02 carried item 3)."
                )
            orig_shape_ras = meta["_pre_resample_spatial_shape"]
            skipped = bool(meta.get("pre_resample_skipped", False))

            # (c) fixed MODEL_PARTS order — merge_remap re-checks names.
            # P9 (09-02) fix: segs is keyed by PORT name (seg_<part>); the
            # part-name lookup below must use the prefixed key.
            parts = [(p["name"], segs[f"seg_{p['name']}"]) for p in MODEL_PARTS]
            combined, minfo = merge_remap(parts)

            # (d) the ONE back-resample, in model space (SAR).
            with nvtx_range("back_resample_merged"):
                merged_orig, binfo = back_resample_merged(combined, orig_shape_ras, skipped)

            # (e) two DISTINCT flips (P10 root cause — see merge_remap
            # docstring): the SEG writer gets the oracle-exact in-plane-only
            # flip (it assigns frame IOPs from the source series' slice
            # order and position-sorts internally); the emit gets the
            # spatially-correct DHW (decoded SEG frame order).
            seg_writer = flip_sar_to_writer(merged_orig)
            seg_dhw = flip_sar_to_dhw(merged_orig)

            # (f) contract log lines (09-04 greps `op=` values).
            self._logger.info(
                "merge record: %s",
                {
                    "op": "merge_5part",
                    "study": get_study_id(self.fragment),
                    "parts": {
                        n: {
                            "max_local": minfo["parts"][n]["max_local"],
                            "bg_in_range": minfo["invariants"]["per_part_bg_voxels_in_range"][n],
                        }
                        for n in (p["name"] for p in MODEL_PARTS)
                    },
                    "max_label": minfo["max_label"],
                    "n_labels": len(minfo["labels"]),
                    "shape": list(merged_orig.shape),
                },
            )
            self._logger.info(
                "back_resample record: %s",
                {
                    "op": "back_resample_merged",
                    "zoom_factors": binfo["zoom_factors"],
                    "skipped": binfo["skipped"],
                    "target_sar": binfo["target_sar"],
                    "shape": list(merged_orig.shape),
                },
            )

            # (g) triage emit (D-9-3): raw per-part model-space dumps.
            # 24 pre-closeout (user-directed, 2026-09-07): also gated by
            # HOLOSCAN_EMIT_NPY (default "1"; "0" disables — shipped MAPs).
            if (
                os.environ.get("HOLOSCAN_EMIT_PART_SEGS") == "1"
                and os.environ.get("HOLOSCAN_EMIT_NPY", "1") != "0"
                and self._output_dir is not None
            ):
                self._output_dir.mkdir(parents=True, exist_ok=True)
                for p in MODEL_PARTS:
                    np.save(self._output_dir / f"seg_{p['name']}_modelspace.npy", segs[f"seg_{p['name']}"])

            # (h) triple emit: SAR metrics path + writer input + spatial DHW.
            # seg_merged is emitted as a CPU torch tensor (oracle contract —
            # nnunet_seg_operator emits a detached torch tensor to the metrics
            # op). A bare numpy array crashes the metrics op's device probe
            # under numpy 2.2 (ndarray.device == 'cpu' str, no .type).
            op_output.emit(torch.from_numpy(np.ascontiguousarray(merged_orig)), self.OUTPUT_MERGED)
            op_output.emit(seg_writer, self.OUTPUT_IMAGE)
            op_output.emit(seg_dhw, self.OUTPUT_DHW)

            record = timing.stop()
            record["study"] = get_study_id(self.fragment)
            record["shape"] = list(merged_orig.shape)
            record["max_label"] = minfo["max_label"]
            StudyTimingCollector.record(self.fragment, record)
            self._logger.info("timing: %s", json.dumps(record))
