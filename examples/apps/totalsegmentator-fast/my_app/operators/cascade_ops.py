# Copyright 2021-2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

"""cascade_ops.py — the four operators that materialize the `liver_segments`
crop-cascade topology (Phase 18, plan 18-02; TS 2.18-faithful,
18-RESEARCH §TS 2.18 Pipeline + §Critical Correction).

DAG (non-empty path; the empty path is flag-gated — see below):

    series_to_vol ──┬───────────────────────────────────────────────┐
                    │                                                │ image
    CascadePrepOp#1 ─ swin_total_6mm ─ post6 ─┬── CropMaskOp ─┬─ CropOp ─┬─ CascadePrepOp#2
    (crop_prep_6mm)   (total_6mm, 6mm)  (6mm) │   (crop_mask)  │ (crop_to_mask) (crop_prep_taskres)
                    │                          │ seg_argmax_   │        │
                    │                          │ dicom + meta  │        ▼
                    │                          │        gate ──┤─ swin_ct_liver_segments ─ post_m
                    │                          │        (prev_part_seg,
                    │                          │         gated — 6mm released first)
                    └── meta ── emit ◄─ PasteOp (paste) ◄───────┘  seg_cropped

NATIVE-RESOLUTION CROP (TS-faithful): CropOp runs on the native volume with
addon (20,20,20) mm → voxels TRUNCATED at native zooms (crop_core, verbatim
TS cropping.py port); the 6mm crop-model labelmap is back-resampled to
native (order 0) in CropMaskOp (TS nnunet.py:758+); the task-res order-1
resample (CascadePrepOp#2) runs AFTER the crop (TS nnunet.py:518-527).

EMPTY-MASK PATH (LIV-03, TS nnunet.py:493 — the REQUIRED flag-gated skip):
CropMaskOp detects the empty mask and (a) emits the exact TS log message,
(b) emits an all-zero native-shape canvas on the SAME `seg_cropped` port the
main chain would use (fan-in on PasteOp), (c) CropOp emits a 1-voxel marker
+ bbox-empty dict. CascadePrepOp#2, the gated main SwIn, and the main
PostResample then NEVER fire (their required inputs never arrive) — so the
`crop_prep_taskres`, `inference_3d_fullres_ct_liver_segments`, and
`postresample_3d_fullres_ct_liver_segments` spans are ABSENT, exactly the
p18_run_study.sh --empty 10-present/3-absent contract. PasteOp fires on its
five always-flowing inputs and emits the all-zero native canvas.

Every declared input port of every operator here is fireable on BOTH paths
(18-RESEARCH Pitfall 5 — unwired/never-firing ports = silent hang).

Span contract (p18_run_study.sh): house `timing:` lines with NVTX/timing
labels EXACTLY `crop_prep_6mm`, `crop_mask`, `crop_to_mask`,
`crop_prep_taskres`, `paste`. The `crop_prep_taskres` span is logged ONLY
when the task-res resample actually runs.

Stage evidence (app output dir, CPU-side, SAR array order — see
sar_flip_axes; 18-03's comparator transposes SAR→DHW for comparison with the
TS nibabel DHW dumps):
  * liver_6mm_labelmap_native.npy  (CropMaskOp — 6mm labelmap @ native, uint8)
  * liver_crop_mask_native.npy     (CropMaskOp — binary liver mask @ native, uint8)
  * liver_crop_bbox.json           (CropOp — bbox/addon/zooms/shape/offset)
  * liver_cropped_native.npy       (CropOp — cropped native volume, int32)
  * liver_taskres_shape.json       (CascadePrepOp#2 — shape/spacing or skip record)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from monai.deploy.conditions import CountCondition
from monai.deploy.core import Image, Operator, OperatorSpec

# 24 pre-closeout (user-directed, 2026-09-07): gate for the debug .npy dumps
# written below. Default "1" (or any non-"0") keeps current behavior; "0"
# disables. All dumps are pure disk writes — tensor data flows via ports,
# nothing downstream reads these files. Shipped MAPs bake HOLOSCAN_EMIT_NPY=0.
_EMIT_NPY_ENABLED = os.environ.get("HOLOSCAN_EMIT_NPY", "1") != "0"

try:  # package-style import (my_app.*)
    from my_app.operators.crop_core import (
        back_resample_6mm_labelmap,
        crop_to_mask,
        get_bbox_from_mask,
        resolve_crop_label_ids,
        select_crop_labels,
        zooms_from_affine,
    )
    from my_app.operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        assert_on_gpu,
        get_study_id,
        nvtx_range,
    )
    from my_app.operators.merge_remap import flip_sar_to_dhw, flip_sar_to_writer
    from my_app.operators.preprocess_operator import (
        _create_nonzero_mask,
    )
    from my_app.operators.preprocess_operator import _get_bbox_from_mask as _preproc_get_bbox
    from my_app.operators.preprocess_operator import preprocess_reference, reorient_to_ras, to_holoscan_gpu_tensor
    from my_app.operators.resample_order1 import resample_order1_gpu
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from operators.crop_core import (
        back_resample_6mm_labelmap,
        crop_to_mask,
        get_bbox_from_mask,
        resolve_crop_label_ids,
        select_crop_labels,
        zooms_from_affine,
    )
    from operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        assert_on_gpu,
        get_study_id,
        nvtx_range,
    )
    from operators.merge_remap import flip_sar_to_dhw, flip_sar_to_writer
    from operators.preprocess_operator import (
        _create_nonzero_mask,
    )
    from operators.preprocess_operator import _get_bbox_from_mask as _preproc_get_bbox
    from operators.preprocess_operator import preprocess_reference, reorient_to_ras, to_holoscan_gpu_tensor
    from operators.resample_order1 import resample_order1_gpu

__all__ = ["CascadePrepOp", "CropMaskOp", "CropOp", "PasteOp"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sar_flip_axes(affine: np.ndarray) -> List[int]:
    """Array axes whose affine column carries a negative dominant sign —
    exactly the flip set of `reorient_to_ras` (DHW → RAS). Used to convert
    DHW evidence arrays to SAR for the 18-03 comparator and to flip
    cropped volumes back to RAS for the nnUNet layout chain."""
    cols = np.asarray(affine, dtype=np.float64)[:3, :3]
    dominant = np.argmax(np.abs(cols), axis=0)
    signs = np.sign(cols[dominant, np.arange(3)])
    return [int(i) for i in range(3) if signs[i] < 0]


def _meta_spacing_to_posttranspose(spacing_dhw: Sequence[float], tf: Sequence[int]) -> List[float]:
    """Replicates PreprocessOperator's spacing-domain bookkeeping:
    `spacing = tuple(reversed(spacing_dhw))` (W,H,D), then
    `original_spacing = [spacing[i] for i in tf]` (nnUNet post-transpose)."""
    whd = tuple(reversed([float(s) for s in spacing_dhw]))
    return [whd[i] for i in tf]


def _transposed_layout(vol_dhw: np.ndarray, tf: Sequence[int]) -> np.ndarray:
    """(D,H,W) → channel-first → (1,W,H,D) → tf transpose → (1,X,Y,Z), the
    exact layout chain of PreprocessOperator (its vol2/vol_out emission)."""
    v = np.asarray(vol_dhw)[np.newaxis, ...].transpose(0, 3, 2, 1)
    return np.ascontiguousarray(v.transpose(0, *[i + 1 for i in tf]))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _emit_uint8_3d(op_output, arr: np.ndarray, port: str) -> None:
    """Emit a 3D uint8 array as a zero-copy CUDA tensor (house pattern)."""
    t = torch.as_tensor(np.ascontiguousarray(arr))
    op_output.emit(to_holoscan_gpu_tensor(t.to(torch.device("cuda")).contiguous()), port)


def _emit_int32_4d(op_output, arr: np.ndarray, port: str) -> None:
    """Emit a 4D (1,X,Y,Z) int32 array as a zero-copy CUDA tensor."""
    t = torch.as_tensor(np.ascontiguousarray(arr.astype(np.int32, copy=False)))
    op_output.emit(to_holoscan_gpu_tensor(t.to(torch.device("cuda")).contiguous()), port)


def _flag_tensor(value: bool) -> Any:
    """One-element uint8 flag tensor factory (0-d DLPack is dicey; 1-elem is safe)."""
    return torch.tensor([1 if value else 0], dtype=torch.uint8)


def _receive_flag(op_input: Any, port: str) -> bool:
    holo = op_input.receive(port)
    if holo is None:
        raise ValueError(f"received no {port!r} flag input")
    t = torch.utils.dlpack.from_dlpack(holo)
    return int(t.reshape(-1)[0].item()) == 1


def _receive_uint8_3d(op_input: Any, port: str) -> np.ndarray:
    holo = op_input.receive(port)
    if holo is None:
        raise ValueError(f"received no {port!r} input")
    t = torch.utils.dlpack.from_dlpack(holo)
    assert_on_gpu(t)
    return t.detach().cpu().numpy().astype(np.uint8, copy=False)


# ---------------------------------------------------------------------------
# CascadePrepOp
# ---------------------------------------------------------------------------


class CascadePrepOp(Operator):
    """TS-semantic spacing prep for the cascade — TWO instances:

    #1 (crop_prep_6mm): native Image (DHW int16) -> 6.0 mm volume (order-1
       int32, TS change_spacing semantics via resample_order1_gpu), RAS
       reorient, nnUNet layout chain, crop_to_nonzero (TS's nnunet
       preprocessor crops at config spacing), emits `preprocessed`
       (int32 (1,X,Y,Z) GPU tensor at the 6mm model's config spacing) +
       `preprocessed_meta` (meta#1 — PreprocessOperator's contract keys,
       extended with affine/sar_flip_axes for the cascade).
    #2 (crop_prep_taskres): cropped native volume (DHW int32) -> task-res
       volume (order-1 int32) for the main model. FLAG-GATED: on an empty
       crop mask it logs a skip, writes the skip evidence JSON, and emits
       NOTHING (no `crop_prep_taskres` span — the --empty contract).

    Named Inputs:
        image: #1 — native `Image` from DICOMSeriesToVolumeOperator.
               #2 — `cropped_volume` from CropOp (CUDA int32 3D DHW tensor,
               or the 1-voxel marker on the empty path).
        mask_nonempty (#2 ONLY): 1-elem uint8 flag from CropMaskOp.
        crop_meta (#2 ONLY): dict from CropOp (cropped native shape +
               spacing + sar_flip_axes + bbox).
    Named Outputs:
        preprocessed: int32 (1,X,Y,Z) CUDA tensor at the target spacing.
        preprocessed_meta: dict (#1 = meta#1; #2 = meta#2 for post_m's revert).
    """

    INPUT_IMAGE = "image"
    INPUT_MASK_FLAG = "mask_nonempty"
    INPUT_CROP_META = "crop_meta"
    OUTPUT_PREPROCESSED = "preprocessed"
    OUTPUT_META = "preprocessed_meta"

    def __init__(
        self,
        fragment: Any,
        *args: Any,
        model_path: Optional[Union[str, Path]] = None,
        config_name: str = "3d_fullres",
        target_spacing_dhw: Optional[Sequence[float]] = None,
        task_res: bool = False,
        plans_semantic: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ):
        # flags-before-super() discipline (holoscan 4.2: Operator.__init__
        # invokes setup() before this constructor body finishes).
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._model_path = Path(model_path) if model_path is not None else None
        self._config_name = config_name
        self._target_dhw = tuple(float(s) for s in target_spacing_dhw) if target_spacing_dhw else None
        self._task_res = bool(task_res)
        self._plans_semantic = bool(plans_semantic)
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._tf: Optional[List[int]] = None
        self._params: Optional[Any] = None
        if self._task_res:
            # 3 required inputs, ALL must arrive (they always do — the empty
            # path delivers the marker + flag + crop_meta; see module doc).
            super().__init__(fragment, CountCondition(fragment, 3), *args, **kwargs)
        else:
            super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input(self.INPUT_IMAGE)
        if self._task_res:
            spec.input(self.INPUT_MASK_FLAG)
            spec.input(self.INPUT_CROP_META)
        spec.output(self.OUTPUT_PREPROCESSED)
        spec.output(self.OUTPUT_META)
        if self._model_path is None:
            raise RuntimeError("CascadePrepOp requires model_path (bundle root, for plans.json).")

    def _load_params(self) -> Any:
        if self._params is None:
            try:
                from my_app.config import load_preprocess_params
            except ImportError:
                from config import load_preprocess_params
            self._params = load_preprocess_params(self._model_path, self._config_name)
        return self._params

    def _load_tf(self) -> List[int]:
        return [int(i) for i in self._load_params().transpose_forward]

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        assert_cuda_available()
        if self._task_res:
            self._compute_task_res(op_input, op_output, context)
        else:
            self._compute_6mm(op_input, op_output, context)

    # -- #1: native -> 6mm (crop_prep_6mm) ----------------------------------

    def _compute_6mm(self, op_input: Any, op_output: Any, context: Any) -> None:
        import cupy as cp

        span = "crop_prep_6mm"
        with nvtx_range(span):
            timing = GpuTiming(span)
            timing.start()

            image: Image = op_input.receive(self.INPUT_IMAGE)
            if image is None:
                raise ValueError("CascadePrepOp(6mm) received no 'image' input.")
            arr = np.asarray(image.asnumpy())
            if arr.ndim != 3:
                raise ValueError(f"expected 3D native volume, got ndim={arr.ndim}")
            affine = np.asarray(image.metadata().get("nifti_affine_transform"), dtype=np.float64)
            if affine.shape != (4, 4):
                raise ValueError("image metadata missing 4x4 nifti_affine_transform")
            zooms_native_dhw = zooms_from_affine(affine)
            tf = self._load_tf()

            # TS pipeline step: native -> 6.0 mm via change_spacing(order=1,
            # dtype=int32). RAS reorient first (pure integer flips, exact —
            # mirrors PreprocessOperator's reorient_to_ras placement).
            arr_ras = reorient_to_ras(arr, affine)
            vol6_cp = resample_order1_gpu(
                cp.asarray(np.ascontiguousarray(arr_ras), dtype=cp.int16),
                zooms_native_dhw,
                (6.0, 6.0, 6.0),
            )
            vol6 = np.ascontiguousarray(vol6_cp.get())  # (D,H,W) RAS int32
            vol4 = _transposed_layout(vol6, tf)  # (1,X,Y,Z) post-transpose

            # crop_to_nonzero at config spacing (TS nnunet preprocessor).
            nz = _create_nonzero_mask(vol4)
            bbox = _preproc_get_bbox(nz)
            slicer = tuple(slice(lo, hi) for lo, hi in bbox)
            vol_c = np.ascontiguousarray(vol4[(slice(None),) + slicer])
            shape_before = tuple(int(s) for s in vol4.shape[1:])
            shape_after = tuple(int(s) for s in vol_c.shape)

            meta1: Dict[str, Any] = {
                "_pre_resample_spatial_shape": list(arr_ras.shape),  # native (D,H,W)
                "orig_spacing_xyz": [float(s) for s in zooms_native_dhw],
                "affine": affine.tolist(),
                "sar_flip_axes": _sar_flip_axes(affine),
                "pre_resample_skipped": False,
                "pre_resample_spacing": [6.0, 6.0, 6.0],
                "shape_before_cropping": list(shape_before),
                "bbox_used_for_cropping": [[int(v) for v in pair] for pair in bbox],
                "shape_after_cropping_and_before_resampling": list(shape_after),
                "new_shape": list(shape_after),  # config spacing == 6mm: no-op resample
                "original_spacing": _meta_spacing_to_posttranspose((6.0, 6.0, 6.0), tf),
                "target_spacing": _meta_spacing_to_posttranspose((6.0, 6.0, 6.0), tf),
                "transpose_forward": tf,
            }

            _emit_int32_4d(op_output, vol_c, self.OUTPUT_PREPROCESSED)
            op_output.emit(meta1, self.OUTPUT_META)

            record = timing.stop()
            record["study"] = get_study_id(self.fragment)
            record["shape"] = list(shape_after)
            StudyTimingCollector.record(self.fragment, record)
            self._logger.info("timing: %s", json.dumps(record))

    # -- #2: cropped native -> task-res (crop_prep_taskres, flag-gated) ------

    def _compute_task_res(self, op_input: Any, op_output: Any, context: Any) -> None:
        import cupy as cp

        nonempty = _receive_flag(op_input, self.INPUT_MASK_FLAG)
        crop_meta: Dict[str, Any] = op_input.receive(self.INPUT_CROP_META) or {}
        holo_vol = op_input.receive(self.INPUT_IMAGE)
        if holo_vol is None:
            raise ValueError("CascadePrepOp(taskres) received no 'image' (cropped_volume) input.")

        if not nonempty:
            # FLAG-GATED SKIP (LIV-03): no resample, no `crop_prep_taskres`
            # span (the --empty runner contract), no emits downstream.
            self._logger.info("CascadePrepOp(taskres): skipped — empty crop mask (main chain gated off)")
            if self._output_dir is not None:
                _write_json(
                    self._output_dir / "liver_taskres_shape.json",
                    {
                        "skipped": True,
                        "reason": "empty crop mask",
                        "array_order": "SAR",
                        "cropped_native_shape": crop_meta.get("cropped_native_shape"),
                    },
                )
            return

        span = "crop_prep_taskres"
        with nvtx_range(span):
            timing = GpuTiming(span)
            timing.start()

            t = torch.utils.dlpack.from_dlpack(holo_vol)
            assert_on_gpu(t)
            vol_ras = t.detach().cpu().numpy().astype(np.int32, copy=False)
            if vol_ras.ndim != 3:
                raise ValueError(f"expected 3D cropped volume, got ndim={vol_ras.ndim}")

            # 18-03 corrected: the received cropped volume is ALREADY in
            # (x, y, z) RAS order — CropOp crops the reoriented RAS native
            # volume and emits it untransformed. Flipping the SAR axes here
            # would double-transform (measured 2026-09-02: taskres input
            # mis-oriented, main-model seg 9x too small).
            flipped = list(crop_meta.get("sar_flip_axes", []))
            native_spacing_dhw = [float(s) for s in crop_meta["native_spacing_dhw"]]
            tf = self._load_tf()
            target_dhw = list(self._target_dhw)

            if self._plans_semantic:
                # 22-11 root-cause fix (22-stage-frame-audit.md S2): for
                # `resample is None` tasks, TS 2.18 passes the native crop
                # through UNRESAMPLED and the nnUNet preprocessor does
                # f32 -> crop_to_nonzero -> CTNormalization (BEFORE resample)
                # -> resampling_fn_data (plans order 3 / order_z 0 /
                # separate-z). The pre-fix path (order-1 int32 world resample
                # + in-swin normalize-after) diverged from that at 96–100 %
                # of voxels and dropped the thin effusion rims. The byte-
                # locked preprocess_reference port reproduces the live-
                # captured TS main-model input with MAD = 0 (audit S2), so
                # the model input becomes TS-bit-exact.
                vol_out, props = taskres_plans_semantic(vol_ras, native_spacing_dhw, self._load_params())
                vol_out = _np_from_array(vol_out)  # CuPy-safe (see helper doc)
                meta2: Dict[str, Any] = {
                    "_pre_resample_spatial_shape": list(crop_meta["cropped_native_shape"]),
                    "orig_spacing_xyz": list(native_spacing_dhw),
                    "affine": crop_meta.get("affine"),
                    "sar_flip_axes": flipped,
                    "pre_resample_skipped": True,  # TS pass-through; chain resamples
                    "pre_resample_spacing": [float(s) for s in props["target_spacing"]],
                    "shape_before_cropping": [int(s) for s in props["shape_before_cropping"]],
                    "bbox_used_for_cropping": [[int(v) for v in pair] for pair in props["bbox_used_for_cropping"]],
                    "shape_after_cropping_and_before_resampling": [
                        int(s) for s in props["shape_after_cropping_and_before_resampling"]
                    ],
                    "new_shape": [int(s) for s in props["new_shape"]],
                    "original_spacing": [float(s) for s in props["original_spacing"]],
                    "target_spacing": [float(s) for s in props["target_spacing"]],
                    "transpose_forward": [int(i) for i in props["transpose_forward"]],
                }
                _emit_f32_4d(op_output, vol_out, self.OUTPUT_PREPROCESSED)
                op_output.emit(meta2, self.OUTPUT_META)
                if self._output_dir is not None:
                    _write_json(
                        self._output_dir / "liver_taskres_shape.json",
                        {
                            "skipped": False,
                            "mode": "plans_semantic",
                            # vol_out is (1, X, Y, Z) post-transpose; the
                            # dump keeps the (x, y, z) RAS comparator space.
                            "shape_pretranspose": list(vol_out[0].transpose(2, 1, 0).shape),
                            "shape_posttranspose": list(vol_out.shape[1:]),
                            "spacing_native_dhw": list(native_spacing_dhw),
                            "spacing_target_dhw": [float(s) for s in props["target_spacing"]],
                            "dtype": "float32 (CT-normalized, pre-resample-normalized)",
                            "array_order": "SAR",
                        },
                    )
                    if _EMIT_NPY_ENABLED:
                        np.save(
                            self._output_dir / "liver_taskres_vol.npy",
                            np.ascontiguousarray(vol_out[0].transpose(2, 1, 0)).astype(np.float32),
                        )
                record = timing.stop()
                record["study"] = get_study_id(self.fragment)
                record["shape"] = list(vol_out.shape[1:])
                StudyTimingCollector.record(self.fragment, record)
                self._logger.info("timing: %s", json.dumps(record))
                return

            in_sp = _meta_spacing_to_posttranspose(native_spacing_dhw, tf)
            out_sp = _meta_spacing_to_posttranspose(target_dhw, tf)
            # 18-03 fix (Rule 1): resample the 3D RAS volume in world (x,y,z)
            # space with world-order spacings — the exact _compute_6mm pattern
            # (resample_order1_gpu is 3D-only). The 18-02 code transposed to
            # (1,X,Y,Z) FIRST and called the resampler on the 4D array with
            # post-transpose spacings (ndim=4 crash — surfaced by the first
            # real cascade run; the path was unreachable before CropOp fixed).
            out3d_cp = resample_order1_gpu(
                cp.asarray(np.ascontiguousarray(vol_ras), dtype=cp.int32),
                native_spacing_dhw,
                target_dhw,
            )
            out4 = _transposed_layout(np.ascontiguousarray(out3d_cp.get()), tf)  # (1,X,Y,Z)

            new_shape = tuple(int(s) for s in out4.shape[1:])
            meta2: Dict[str, Any] = {
                "_pre_resample_spatial_shape": list(crop_meta["cropped_native_shape"]),
                "orig_spacing_xyz": native_spacing_dhw,
                "affine": crop_meta.get("affine"),
                "sar_flip_axes": flipped,
                "pre_resample_skipped": False,
                "pre_resample_spacing": target_dhw,
                # The whole resampled volume IS the "crop" for post_m's
                # revert (shape_before_cropping == post-resample shape; the
                # bbox is the full extent — revert_crop_gpu becomes a pure
                # transpose-back, exactly the 5-part chain's no-crop case).
                "shape_before_cropping": list(new_shape),
                "bbox_used_for_cropping": [[0, int(s)] for s in new_shape],
                "shape_after_cropping_and_before_resampling": list(new_shape),
                "new_shape": list(new_shape),
                "original_spacing": in_sp,
                "target_spacing": out_sp,
                "transpose_forward": tf,
            }

            _emit_int32_4d(op_output, out4, self.OUTPUT_PREPROCESSED)
            op_output.emit(meta2, self.OUTPUT_META)

            if self._output_dir is not None:
                _write_json(
                    self._output_dir / "liver_taskres_shape.json",
                    {
                        "skipped": False,
                        "shape_pretranspose": list(
                            np.ascontiguousarray(out3d_cp.get()).shape
                        ),  # (x,y,z) world order (18-03 comparator space)
                        "shape_posttranspose": list(new_shape),
                        "spacing_native_dhw": native_spacing_dhw,
                        "spacing_target_dhw": target_dhw,
                        "array_order": "SAR",
                    },
                )
                # 18-03 Task 2 triage: dump the actual main-model INPUT volume
                # (RAS (x,y,z) order-1 resample of the crop) so it can be
                # byte-compared against the TS probe's captured taskres volume
                # (provenance test: input-bit-identical => any final delta is
                # inference-stack numerics, not pipeline defect).
                if _EMIT_NPY_ENABLED:
                    np.save(
                        self._output_dir / "liver_taskres_vol.npy",
                        np.ascontiguousarray(out3d_cp.get()).astype(np.int32),
                    )

            record = timing.stop()
            record["study"] = get_study_id(self.fragment)
            record["shape"] = list(new_shape)
            StudyTimingCollector.record(self.fragment, record)
            self._logger.info("timing: %s", json.dumps(record))


# ---------------------------------------------------------------------------
# CropMaskOp
# ---------------------------------------------------------------------------


class CropMaskOp(Operator):
    """Build the native-resolution crop mask from the 6mm crop-model labelmap
    (TS nnunet.py:758+ back-resample order 0, then the binary union over the
    task spec's crop labels).

    The crop label IDS are RESOLVED from the 6mm bundle's dataset.json at
    setup from the NAMES in `crop_labels` (the task spec's
    `TaskSpec.crop.labels` — the source of truth; no hard-coded label name
    or id). Single-label sets (e.g. the liver crop task, `labels=("liver",)`)
    are bit-identical to the historical single-label comparison.

    Named Inputs:
        seg_labelmap: 6mm PostResample `seg_argmax_dicom` (uint8 CUDA 3D DHW,
                      full 6mm grid — revert_crop_gpu zero-fills to
                      meta#1 shape_before_cropping).
        preprocessed_meta: meta#1 from CascadePrepOp#1.
    Named Outputs:
        crop_mask: uint8 CUDA 3D native binary mask.
        mask_nonempty: 1-elem uint8 flag.
        preprocessed_meta: meta#1 forwarded (CropOp evidence + PasteOp canvas).
        seg_cropped (CONDITIONAL — empty path ONLY): all-zero native-shape
                      uint8 CUDA canvas, fanned into PasteOp's seg_cropped
                      port (the non-empty path's message comes from the main
                      PostResample; only one ever flows per run).
    """

    INPUT_SEG = "seg_labelmap"
    INPUT_META = "preprocessed_meta"
    OUTPUT_MASK = "crop_mask"
    OUTPUT_FLAG = "mask_nonempty"
    OUTPUT_META = "preprocessed_meta"
    OUTPUT_SEG_CROPPED = "seg_cropped"

    def __init__(
        self,
        fragment: Any,
        *args: Any,
        dataset_json: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        crop_labels: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ):
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._dataset_json = Path(dataset_json) if dataset_json is not None else None
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._crop_labels = list(crop_labels) if crop_labels else None
        self._crop_ids: Optional[List[int]] = None
        super().__init__(fragment, CountCondition(fragment, 2), *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input(self.INPUT_SEG)
        spec.input(self.INPUT_META)
        spec.output(self.OUTPUT_MASK)
        spec.output(self.OUTPUT_FLAG)
        spec.output(self.OUTPUT_META)
        spec.output(self.OUTPUT_SEG_CROPPED)
        if self._dataset_json is None:
            raise RuntimeError("CropMaskOp requires dataset_json (6mm bundle jsonpkls/dataset.json).")
        if not self._crop_labels:
            raise RuntimeError("CropMaskOp requires crop_labels (the task spec's crop label names).")
        with open(self._dataset_json) as f:
            dataset = json.load(f)
        labels = dataset.get("labels")
        if not isinstance(labels, dict):
            raise RuntimeError(f"dataset.json {self._dataset_json} has no 'labels' dict")
        self._crop_ids = resolve_crop_label_ids(labels, self._crop_labels)
        self._logger.info(
            "CropMaskOp: crop labels %s -> ids %s resolved from dataset.json (%d-key label table)",
            list(self._crop_labels),
            self._crop_ids,
            len(labels),
        )

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        assert_cuda_available()
        with nvtx_range("crop_mask"):
            timing = GpuTiming("crop_mask")
            timing.start()

            seg6 = _receive_uint8_3d(op_input, self.INPUT_SEG)
            meta1: Dict[str, Any] = op_input.receive(self.INPUT_META) or {}
            native_shape = tuple(int(s) for s in meta1["_pre_resample_spatial_shape"])

            # 18-03 diagnostic: dump the RAW PostResample output + the meta the
            # layout decision depends on, so the correct orientation transform
            # can be determined empirically against the TS 6mm reference.
            if self._output_dir is not None:
                self._output_dir.mkdir(parents=True, exist_ok=True)
                if _EMIT_NPY_ENABLED:
                    np.save(self._output_dir / "diag_seg6_raw.npy", np.ascontiguousarray(seg6))
                _write_json(
                    self._output_dir / "diag_seg6_meta.json",
                    {
                        "seg6_shape": list(np.asarray(seg6).shape),
                        "native_shape": list(native_shape),
                        "transpose_forward": meta1.get("transpose_forward"),
                        "affine": meta1.get("affine"),
                        "sar_flip_axes": meta1.get("sar_flip_axes"),
                        "shape_before_cropping": meta1.get("shape_before_cropping"),
                        "bbox_used_for_cropping": meta1.get("bbox_used_for_cropping"),
                    },
                )

            # TS nnunet.py:758+ contract: order-0 (nearest) back-resample to
            # native, THEN the binary crop mask (union over the spec's crop
            # labels). scipy order 0 is bit-exact on integer label data.
            #
            # Array order (18-03 corrected, empirically verified 2026-09-02
            # against the TS 6mm reference with the diag_seg6_raw dump):
            # seg6 (PostResample seg_argmax_dicom) is in the SAR layout
            # (z, y, x) — revert_crop_gpu's permute is the identity for these
            # bundles (transpose_forward (0,1,2)), so the 4D (1,z,y,x) model
            # layout comes straight back — but its CONTENT already carries the
            # RAS signs (the preprocess reoriented the raw SDK array before
            # the 6mm resample; transposes preserved signs). So the ONLY step
            # needed to reach the (x,y,z) RAS frame of native_shape
            # (arr_ras.shape) is a pure transpose: NO sign flips.
            #   reorient_to_ras would transpose AND re-apply the SAR sign
            #   flips -> double flip -> mask misplaced by the full reorient
            #   offset (measured: crop window shifted ~157 mm in y, main
            #   model saw a liver-free slab, final seg 9x too small).
            #   No transpose at all is also wrong (z,y,x content zoomed into
            #   an x,y,z shape — axis-order mix).
            labelmap_native = back_resample_6mm_labelmap(seg6, native_shape)
            mask_native = select_crop_labels(labelmap_native, self._crop_ids)
            nonempty = bool(mask_native.sum() > 0)

            # Evidence (SAR-flipped of the (x,y,z) world arrays — the 18-03
            # comparator deterministically aligns these to TS nibabel space).
            if self._output_dir is not None:
                flip_axes = tuple(meta1.get("sar_flip_axes", []))
                sar_lmap = np.flip(labelmap_native, flip_axes) if flip_axes else labelmap_native
                sar_mask = np.flip(mask_native, flip_axes) if flip_axes else mask_native
                self._output_dir.mkdir(parents=True, exist_ok=True)
                if _EMIT_NPY_ENABLED:
                    np.save(self._output_dir / "liver_6mm_labelmap_native.npy", sar_lmap)
                    np.save(self._output_dir / "liver_crop_mask_native.npy", sar_mask)

            _emit_uint8_3d(op_output, mask_native, self.OUTPUT_MASK)
            op_output.emit(to_holoscan_gpu_tensor(_flag_tensor(nonempty).to(torch.device("cuda"))), self.OUTPUT_FLAG)
            op_output.emit(meta1, self.OUTPUT_META)

            if not nonempty:
                # TS nnunet.py:493-510: exact message; all-zero native canvas
                # fanned into PasteOp; the main chain is gated off (its ops'
                # required inputs never arrive).
                self._logger.info("Crop is empty. Returning empty segmentation.")
                zero = np.zeros(native_shape, dtype=np.uint8)
                _emit_uint8_3d(op_output, zero, self.OUTPUT_SEG_CROPPED)

            record = timing.stop()
            record["study"] = get_study_id(self.fragment)
            record["nonempty"] = nonempty
            record["native_shape"] = list(native_shape)
            StudyTimingCollector.record(self.fragment, record)
            self._logger.info("timing: %s", json.dumps(record))


# ---------------------------------------------------------------------------
# CropOp
# ---------------------------------------------------------------------------


class CropOp(Operator):
    """Native-resolution crop (TS nnunet.py:512 via crop_to_mask on the
    native volume — 18-RESEARCH §Critical Correction; crop_core is the
    verbatim TS cropping.py port: addon mm -> voxels TRUNCATED at native
    zooms, bbox clamped to image bounds, affine offset update verbatim).

    Named Inputs (all ALWAYS flowing — Pitfall 5):
        image: native `Image` from DICOMSeriesToVolumeOperator.
        crop_mask: uint8 CUDA 3D native mask from CropMaskOp.
        preprocessed_meta: meta#1 (forwarded on; also the affine source's
                           sibling — the affine itself comes from `image`).
    Named Outputs (all ALWAYS emitted — the empty path emits the marker):
        cropped_volume: int32 CUDA 3D (x,y,z) world-order tensor (1-voxel
                        zero marker when the mask is empty — carries the
                        flag-gated skip).
        crop_bbox: dict {empty, bbox, addon_vox, shape, affine_offset,
                        zooms, array_order:"DHW"}.
        crop_meta: dict {empty, cropped_native_shape, native_spacing_dhw,
                        sar_flip_axes, affine, bbox, addon_mm} for
                        CascadePrepOp#2 + PasteOp.
        preprocessed_meta: meta#1 forwarded (PasteOp canvas shape).
    """

    INPUT_IMAGE = "image"
    INPUT_MASK = "crop_mask"
    INPUT_META = "preprocessed_meta"
    OUTPUT_CROPPED = "cropped_volume"
    OUTPUT_BBOX = "crop_bbox"
    OUTPUT_CROP_META = "crop_meta"
    OUTPUT_META = "preprocessed_meta"

    def __init__(
        self,
        fragment: Any,
        *args: Any,
        addon_mm: Sequence[float] = (20, 20, 20),
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ):
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._addon_mm = [float(a) for a in addon_mm]
        self._output_dir = Path(output_dir) if output_dir is not None else None
        super().__init__(fragment, CountCondition(fragment, 3), *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input(self.INPUT_IMAGE)
        spec.input(self.INPUT_MASK)
        spec.input(self.INPUT_META)
        spec.output(self.OUTPUT_CROPPED)
        spec.output(self.OUTPUT_BBOX)
        spec.output(self.OUTPUT_CROP_META)
        spec.output(self.OUTPUT_META)

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        assert_cuda_available()
        with nvtx_range("crop_to_mask"):
            timing = GpuTiming("crop_to_mask")
            timing.start()

            image: Image = op_input.receive(self.INPUT_IMAGE)
            if image is None:
                raise ValueError("CropOp received no 'image' input.")
            mask = _receive_uint8_3d(op_input, self.INPUT_MASK)
            meta1: Dict[str, Any] = op_input.receive(self.INPUT_META) or {}

            arr = np.asarray(image.asnumpy())
            affine = np.asarray(image.metadata().get("nifti_affine_transform"), dtype=np.float64)
            zooms = [float(s) for s in zooms_from_affine(affine)]
            flip_axes = _sar_flip_axes(affine)

            # Array order (18-03 fix): `image` is the SDK original array
            # (z, y, x) — image.asnumpy() order. crop_core + zooms + affine
            # all use (x, y, z) world order (TS cropping.py convention; same
            # conversion _compute_6mm uses before the 6mm resample). Re-orient
            # exactly (transpose + sign flips) BEFORE the crop; the mask
            # arrives already in (x, y, z) from CropMaskOp.
            vol = reorient_to_ras(arr, affine)

            bbox, addon_vox = get_bbox_from_mask(mask, zooms, self._addon_mm)
            empty = bbox is None

            if empty:
                # nnunet.py:493 ordering: emptiness is decided BEFORE any
                # crop; the main chain never runs (flag-gated skip).
                self._logger.info("Crop is empty. Returning empty segmentation.")
                bbox_d: Dict[str, Any] = {
                    "empty": True,
                    "bbox": None,
                    "addon_vox": [int(v) for v in addon_vox],
                    "shape": None,
                    "affine_offset": None,
                    "zooms": zooms,
                    "addon_mm": self._addon_mm,
                    "array_order": "DHW",
                }
                crop_meta: Dict[str, Any] = {
                    "empty": True,
                    "cropped_native_shape": list(vol.shape),
                    "native_spacing_dhw": zooms,
                    "sar_flip_axes": flip_axes,
                    "affine": affine.tolist(),
                    "bbox": None,
                    "addon_mm": self._addon_mm,
                }
                marker = np.zeros((1, 1, 1), dtype=np.int32)
                _emit_int32_volume(op_output, marker, self.OUTPUT_CROPPED)
                if self._output_dir is not None:
                    self._output_dir.mkdir(parents=True, exist_ok=True)
                    _write_json(self._output_dir / "liver_crop_bbox.json", bbox_d)
                    if _EMIT_NPY_ENABLED:
                        np.save(self._output_dir / "liver_cropped_native.npy", np.empty((0,), dtype=np.int32))
            else:
                cropped, info = crop_to_mask(vol.astype(np.int32), mask, zooms, affine, self._addon_mm)
                bbox_d = {
                    "empty": False,
                    "bbox": info["bbox"],
                    "addon_vox": info["addon_vox"],
                    "shape": info["shape"],
                    "affine_offset": info["affine_offset"].tolist(),
                    "zooms": zooms,
                    "addon_mm": self._addon_mm,
                    "array_order": "DHW",
                }
                crop_meta = {
                    "empty": False,
                    "cropped_native_shape": list(info["shape"]),
                    "native_spacing_dhw": zooms,
                    "sar_flip_axes": flip_axes,
                    "affine": affine.tolist(),
                    "bbox": info["bbox"],
                    "addon_mm": self._addon_mm,
                }
                _emit_int32_volume(op_output, cropped, self.OUTPUT_CROPPED)
                if self._output_dir is not None:
                    self._output_dir.mkdir(parents=True, exist_ok=True)
                    _write_json(self._output_dir / "liver_crop_bbox.json", bbox_d)
                    sar_crop = np.flip(cropped, tuple(flip_axes)) if flip_axes else cropped
                    if _EMIT_NPY_ENABLED:
                        np.save(self._output_dir / "liver_cropped_native.npy", np.ascontiguousarray(sar_crop))

            op_output.emit(bbox_d, self.OUTPUT_BBOX)
            op_output.emit(crop_meta, self.OUTPUT_CROP_META)
            op_output.emit(meta1, self.OUTPUT_META)

            record = timing.stop()
            record["study"] = get_study_id(self.fragment)
            record["empty"] = empty
            record["shape"] = bbox_d["shape"]
            StudyTimingCollector.record(self.fragment, record)
            self._logger.info("timing: %s", json.dumps(record))


def _emit_int32_volume(op_output: Any, arr: np.ndarray, port: str) -> None:
    """Emit a 3D int32 volume as a zero-copy CUDA tensor."""
    t = torch.as_tensor(np.ascontiguousarray(arr.astype(np.int32, copy=False)))
    op_output.emit(to_holoscan_gpu_tensor(t.to(torch.device("cuda")).contiguous()), port)


def _np_from_array(arr) -> np.ndarray:
    """Coerce a numpy OR CuPy array to a C-contiguous numpy array.

    Needed because _resample_to_shape (the byte-locked reference resampler)
    can return a CuPy array on the non-separate-z branch when the HOLOSCAN
    GPU_RESAMPLE flag is on — and the plans-semantic path must never let a
    device array cross into numpy/torch without an explicit .get() (CuPy
    forbids implicit __array__ conversion)."""
    if isinstance(arr, np.ndarray):
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(arr.get())


def _emit_f32_4d(op_output: Any, arr: np.ndarray, port: str) -> None:
    """Emit a float32 (1,X,Y,Z) volume as a zero-copy CUDA tensor (22-11:
    the plans-semantic task-res path emits the pre-normalized f32 model
    input — the int32 helper would truncate the normalized floats)."""
    t = torch.as_tensor(_np_from_array(arr).astype(np.float32, copy=False))
    op_output.emit(to_holoscan_gpu_tensor(t.to(torch.device("cuda")).contiguous()), port)


def taskres_plans_semantic(
    vol_ras: np.ndarray,
    spacing_xyz: Sequence[float],
    params: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """TS 2.18-faithful main-model input prep for `resample is None`
    cascade tasks (22-stage-frame-audit.md S2 root cause).

    TS 2.18 passes the native crop UNRESAMPLED to ``nnUNet_predict_image``
    (``totalsegmentator/nnunet.py`` ``else: img_in_rsp = img_in``); the
    nnUNet ``DefaultPreprocessor.run_case_npy`` then runs, in order:
    f32 cast -> crop_to_nonzero -> CTNormalization (BEFORE resample —
    the clip is non-linear, so the order is load-bearing) ->
    ``resampling_fn_data`` to the bundle plans spacing with the plans
    ``order`` / ``order_z`` / ``force_separate_z`` kwargs (Dataset315:
    order 3 in-plane cubic, order_z 0, separate-z auto at
    ANISO_THRESHOLD 3). This reuses ``preprocess_reference`` — the
    byte-locked verbatim port of that chain shared with the total path
    (22-11 audit: reproduces the live-captured TS main-model input with
    MAD = 0 on 64199).

    Args:
        vol_ras: int32 (x, y, z) RAS-order cropped native volume (CropOp output).
        spacing_xyz: physical (x, y, z) voxel spacing (affine column norms).
        params: the main model bundle's ``PreprocessParams``.

    Returns:
        ``(vol_out, props)`` — the normalized float32 (1, X, Y, Z) model
        input in nnUNet post-transpose order, and the preprocess
        properties dict (bbox/shape/spacing bookkeeping for the
        back-resample meta).
    """
    data4 = np.ascontiguousarray(vol_ras[None].transpose(0, 3, 2, 1))  # (1, z, y, x) app-transposed
    return preprocess_reference(data4, tuple(float(s) for s in spacing_xyz), params)


# ---------------------------------------------------------------------------
# PasteOp
# ---------------------------------------------------------------------------


class PasteOp(Operator):
    """undo_crop (TS cropping.py) + the order-0 cropped-seg back-resample
    (TS nnunet.py:758-810, main-model prediction -> cropped native shape) +
    the P10 orientation contract (SAR metrics / flip(1,2) SEG-writer / DHW
    emit — merge_remap's flip functions are the single source of truth).

    Named Inputs (ALL five always flow on BOTH paths — Pitfall 5):
        seg_cropped: uint8 CUDA 3D DHW — the main PostResample's
                     seg_argmax_dicom at TASK resolution (non-empty path) OR
                     CropMaskOp's all-zero native canvas (empty path).
        crop_bbox: dict from CropOp.
        mask_nonempty: 1-elem uint8 flag from CropMaskOp.
        crop_meta: dict from CropOp (cropped native shape + spacing).
        preprocessed_meta: meta#1 (full native shape).
    Named Outputs (mirror MergeRemapOperator's contracts):
        seg_merged: SAR (flip of DHW — flip_sar_to_dhw is an involution),
                    CPU torch uint8 (the metrics op requires a torch tensor).
        seg_image:  SEG-writer orientation = flip_sar_to_writer(seg_merged)
                    (in-plane flip(1,2) of SAR = flip(D, H, W) of DHW).
        seg_dhw:    spatial DHW (decoded SEG frame order).
    """

    INPUT_SEG = "seg_cropped"
    INPUT_BBOX = "crop_bbox"
    INPUT_FLAG = "mask_nonempty"
    INPUT_CROP_META = "crop_meta"
    INPUT_META = "preprocessed_meta"
    OUTPUT_MERGED = "seg_merged"
    OUTPUT_IMAGE = "seg_image"
    OUTPUT_DHW = "seg_dhw"

    def __init__(self, fragment: Any, *args: Any, **kwargs: Any):
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        super().__init__(fragment, CountCondition(fragment, 5), *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input(self.INPUT_SEG)
        spec.input(self.INPUT_BBOX)
        spec.input(self.INPUT_FLAG)
        spec.input(self.INPUT_CROP_META)
        spec.input(self.INPUT_META)
        spec.output(self.OUTPUT_MERGED)
        spec.output(self.OUTPUT_IMAGE)
        spec.output(self.OUTPUT_DHW)

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        with nvtx_range("paste"):
            timing = GpuTiming("paste")
            timing.start()

            nonempty = _receive_flag(op_input, self.INPUT_FLAG)
            bbox_d: Dict[str, Any] = op_input.receive(self.INPUT_BBOX) or {}
            crop_meta: Dict[str, Any] = op_input.receive(self.INPUT_CROP_META) or {}
            meta1: Dict[str, Any] = op_input.receive(self.INPUT_META) or {}
            seg_holo = op_input.receive(self.INPUT_SEG)
            if seg_holo is None:
                raise ValueError("PasteOp received no 'seg_cropped' input.")
            seg = torch.utils.dlpack.from_dlpack(seg_holo).detach().cpu().numpy().astype(np.uint8, copy=False)

            native_shape = tuple(int(s) for s in meta1["_pre_resample_spatial_shape"])

            if nonempty:
                # TS nnunet.py:758-810: order-0 back-resample of the main
                # prediction to the cropped native shape (force_affine
                # equivalent — the shape IS the target).
                #
                # Array order (18-03 frame audit, 2026-09-02 — same root
                # cause class as the CropMaskOp seg6 fix): the PostResample
                # seg_argmax_dicom is in the SAR layout (z, y, x) (run-log
                # proof: main part emits [108,196,242] = z,y,x while the
                # RAS taskres shape is [242,196,108]; the bundle's revert
                # permute is the identity, so the 4D (1,z,y,x) model layout
                # comes straight back), but cropped_native_shape is (x,y,z)
                # RAS (crop/paste space). The original code zoomed the SAR
                # axes with RAS-shape factors (e.g. (433/108, 351/196,
                # 81/242)) — an axis-order mix that displaced the liver
                # across the canvas (per-label volumes correct, placement
                # ~0% IoU). Transpose to (x,y,z) RAS FIRST (content signs
                # are already RAS — no flips), then zoom.
                from scipy.ndimage import zoom as ndi_zoom

                seg_ras = np.ascontiguousarray(np.transpose(seg, (2, 1, 0)))  # (z,y,x) SAR -> (x,y,z) RAS
                cropped_native_shape = tuple(int(s) for s in crop_meta["cropped_native_shape"])
                if seg_ras.shape == cropped_native_shape:
                    seg_native = seg_ras
                else:
                    factors = tuple(float(n) / float(c) for n, c in zip(cropped_native_shape, seg_ras.shape))
                    seg_native = (
                        ndi_zoom(seg_ras.astype(np.float64), factors, order=0, mode="nearest").round().astype(np.uint8)
                    )
                canvas = np.zeros(native_shape, dtype=np.uint8)
                bb = bbox_d["bbox"]
                canvas[bb[0][0] : bb[0][1], bb[1][0] : bb[1][1], bb[2][0] : bb[2][1]] = seg_native
            else:
                self._logger.info("Crop is empty. Returning empty segmentation.")
                canvas = np.zeros(native_shape, dtype=np.uint8)

            # P10 orientation contract (merge_remap single source of truth):
            # the canvas is (x, y, z) RAS (crop/paste space). The 5-part chain's
            # "SAR" frame is the (z, y, x) SDK layout (back_resample_merged /
            # flip_sar_to_* docstrings: SAR = [S,A,R] slice-major; seg_total_dhw.npy
            # is (217,512,512); the SEG writer requires source-image shape match).
            # Transpose the RAS canvas to (z,y,x) FIRST, then apply the canonical
            # flip chain exactly like MergeRemapOperator does to its merged
            # (z, y, x) segs (signs already RAS; transposes preserve signs):
            canvas_sar = np.ascontiguousarray(np.transpose(canvas, (2, 1, 0)))  # (x,y,z) RAS -> (z,y,x) SAR layout
            seg_sar = canvas_sar
            seg_writer = flip_sar_to_writer(seg_sar)  # SEG-writer flip(1,2) of SAR
            seg_dhw = flip_sar_to_dhw(seg_sar)  # app DHW convention (flip all 3 axes)

            # Emit contracts mirror MergeRemapOperator (merge_5part):
            # seg_merged as a CPU torch tensor (the metrics op's device
            # probe crashes on bare numpy under numpy 2.2 — house lesson).
            op_output.emit(torch.from_numpy(np.ascontiguousarray(seg_sar)), self.OUTPUT_MERGED)
            op_output.emit(seg_writer, self.OUTPUT_IMAGE)
            op_output.emit(seg_dhw, self.OUTPUT_DHW)

            record = timing.stop()
            record["study"] = get_study_id(self.fragment)
            record["shape"] = list(native_shape)
            record["nonempty"] = nonempty
            StudyTimingCollector.record(self.fragment, record)
            self._logger.info("timing: %s", json.dumps(record))
