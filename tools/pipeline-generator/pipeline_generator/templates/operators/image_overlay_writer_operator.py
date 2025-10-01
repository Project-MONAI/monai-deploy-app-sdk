# Copyright 2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from monai.deploy.core import Fragment, Image, Operator, OperatorSpec
from monai.deploy.utils.importutil import optional_import

PILImage, _ = optional_import("PIL", name="Image")


class ImageOverlayWriter(Operator):
    """
    Image Overlay Writer

    Blends a segmentation mask onto an RGB image and saves the result as a PNG.

    Named inputs:
    - image: original RGB frame as Image or ndarray (HWC, uint8/float)
    - pred: predicted mask as Image or ndarray (H x W or 1 x H x W). If multi-channel
            probability tensor is provided, you may pre-argmax before this operator.
    - filename: base name (stem) for output file
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        output_folder: Path,
        alpha: float = 0.4,
        color: Tuple[int, int, int] = (255, 0, 0),
        threshold: Optional[float] = 0.5,
        **kwargs,
    ) -> None:
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._output_folder = Path(output_folder)
        self._alpha = float(alpha)
        self._color = tuple(int(c) for c in color)
        self._threshold = threshold
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("image")
        spec.input("pred")
        spec.input("filename")

    def compute(self, op_input, op_output, context):
        image_in = op_input.receive("image")
        pred_in = op_input.receive("pred")
        fname_stem = op_input.receive("filename")

        img = self._to_hwc_uint8(image_in)
        mask = self._to_mask_uint8(pred_in)

        # Blend
        overlay = self._blend_overlay(img, mask, self._alpha, self._color)

        self._output_folder.mkdir(parents=True, exist_ok=True)
        out_path = self._output_folder / f"{fname_stem}_overlay.png"
        PILImage.fromarray(overlay).save(out_path)
        self._logger.info(f"Saved overlay PNG: {out_path}")

    def _to_hwc_uint8(self, image) -> np.ndarray:
        if isinstance(image, Image):
            arr: np.ndarray = image.asnumpy()
        else:
            arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f"Expected HWC image with 3 or 4 channels, got shape {arr.shape}")
        # Drop alpha if present
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        # Scale/clip and convert
        if not np.issubdtype(arr.dtype, np.uint8):
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _to_mask_uint8(self, pred) -> np.ndarray:
        if isinstance(pred, Image):
            arr: np.ndarray = pred.asnumpy()
        else:
            arr = np.asarray(pred)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D mask after squeeze, got shape {arr.shape}")
        if self._threshold is not None and not np.issubdtype(arr.dtype, np.uint8):
            arr = (arr > float(self._threshold)).astype(np.uint8) * 255
        elif arr.dtype != np.uint8:
            # Assume already {0,1}
            arr = (arr != 0).astype(np.uint8) * 255
        return arr

    @staticmethod
    def _blend_overlay(img: np.ndarray, mask_u8: np.ndarray, alpha: float, color: Tuple[int, int, int]) -> np.ndarray:
        # img: HWC uint8, mask_u8: HW uint8
        mask = (mask_u8 > 0).astype(np.float32)[..., None]
        color_img = np.zeros_like(img, dtype=np.uint8)
        color_img[..., 0] = color[0]
        color_img[..., 1] = color[1]
        color_img[..., 2] = color[2]
        blended = (
            img.astype(np.float32) * (1.0 - alpha * mask) + color_img.astype(np.float32) * (alpha * mask)
        ).astype(np.uint8)
        return blended
