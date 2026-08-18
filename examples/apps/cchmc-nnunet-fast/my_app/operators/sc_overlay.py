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

"""DICOM Secondary Capture overlay generation for the cchmc-nnunet-fast app.

Reproduces the reference app's SC side-output math 1:1 (see
``cchmc_nnunet_fifteen_ckpt_app/my_app/post_transforms.py``):

* ``LabelToContourd`` — per-label 2-D-plane contours of the segmentation
  (MONAI ``LabelToContour`` on the full 3-D binary mask, kept where ``> 0``);
* ``OverlayImageLabeld`` — min-max normalized RGB image, jet colormap (256
  entries) on the contour volume, alpha blend (``alpha=0.7``) where the
  contour is present, uint8;
* ``SaveImaged(output_ext=".dcm", output_dtype=int16)`` — the ITK-based
  multi-frame DICOM write into the temp SC directory that the custom
  ``DICOMSCWriterOperator`` consumes (it expects exactly one ``.dcm``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Union

import matplotlib.cm as cm
import numpy as np
from numpy import int16

from monai.data import MetaTensor
from monai.transforms import LabelToContour, SaveImaged

__all__ = ["generate_contour", "create_overlay", "save_sc_dicom", "write_sc_overlay"]

_logger = logging.getLogger(f"{__name__}")

# The reference OverlayImageLabeld builds the jet colormap once (256 entries).
_JET_COLORMAP = cm.get_cmap("jet", 256)


def generate_contour(seg: np.ndarray, output_labels: Sequence[int]) -> np.ndarray:
    """Per-label contour volume (reference ``LabelToContourd`` math).

    Args:
        seg: uint8 segmentation ``(Z, Y, X)`` in original DICOM orientation.
        output_labels: labels to contour (reference: ``[1]``).

    Returns:
        uint8 volume of the same shape; 0 in the background, the label value
        on the (Laplace-thickened) contour of that label.
    """
    contour = np.zeros(seg.shape, dtype=np.uint8)
    for label in output_labels:
        binary_mask = (seg == label).astype(np.float32)
        if not binary_mask.any():
            continue
        thick_edges = LabelToContour()(binary_mask)
        contour[np.asarray(thick_edges) > 0] = label
    return contour


def create_overlay(image_volume: np.ndarray, label_volume: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """Alpha-blended jet overlay (reference ``OverlayImageLabeld`` math).

    Args:
        image_volume: single-channel grayscale volume ``(Z, Y, X)``.
        label_volume: contour volume ``(Z, Y, X)`` (0 / label values).
        alpha: blend weight of the colored label (reference: 0.7).

    Returns:
        uint8 ``(3, Z, Y, X)`` RGB overlay.
    """
    if image_volume.shape != label_volume.shape:
        raise ValueError(
            f"image and label shapes must match: {image_volume.shape} vs {label_volume.shape}."
        )
    if image_volume.ndim != 3:
        raise ValueError(f"expected 3-D (Z, Y, X) volumes, got ndim={image_volume.ndim}.")

    # convert_to_rgb: min-max normalize the grayscale volume, replicate 3x.
    image_min = float(image_volume.min())
    image_max = float(image_volume.max())
    denom = image_max - image_min
    image_normalized = (
        np.zeros_like(image_volume, dtype=np.float64)
        if denom == 0
        else (image_volume - image_min) / denom
    )
    image_rgb = np.stack([image_normalized] * 3, axis=0)
    image_rgb = (image_rgb * 255).astype(np.uint8)

    # apply_jet_colormap: normalize the label volume to 0..255 and look up jet.
    label_max = int(label_volume.max())
    if label_max <= 0:
        # No foreground in the contour: pure image (the reference math would
        # divide by zero; a blank contour carries no color to blend).
        return image_rgb
    label_uint8 = ((label_volume / label_max) * 255.0).astype(np.uint8)
    label_rgb = _JET_COLORMAP(label_uint8)[:, :, :, :3]  # (Z, Y, X, 3) RGBA -> RGB
    label_rgb = (label_rgb * 255).astype(np.uint8)
    label_rgb = np.transpose(label_rgb, (3, 0, 1, 2))  # (3, Z, Y, X)

    # alpha blend where the contour is present.
    overlay = image_rgb.copy()
    mask = label_volume > 0
    for i in range(3):
        overlay[i, mask] = (alpha * label_rgb[i, mask] + (1 - alpha) * overlay[i, mask]).astype(np.uint8)
    return overlay


def save_sc_dicom(overlay: np.ndarray, affine: np.ndarray, output_dir: Union[str, Path]) -> Path:
    """Write the ``(3, Z, Y, X)`` overlay as a multi-frame DICOM into ``output_dir``.

    Mirrors the reference ``SaveImaged(keys="overlay", output_ext=".dcm",
    output_dtype=int16, separate_folder=False)`` call. ``output_dir`` is
    created if missing; the previous frame file (if any) is removed so the
    directory always contains exactly one ``.dcm`` for the SC writer.

    Returns:
        The directory path containing the written ``.dcm`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.dcm"):
        stale.unlink()

    mt = MetaTensor(
        overlay,
        meta={"filename_or_obj": "Img_in_context"},
        affine=affine,
    )
    SaveImaged(
        keys="overlay",
        output_ext=".dcm",
        output_dir=output_dir,
        separate_folder=False,
        output_dtype=int16,
    )({"overlay": mt})
    written = sorted(output_dir.glob("*.dcm"))
    if not written:
        raise RuntimeError(f"SC DICOM write produced no .dcm file in {output_dir}.")
    _logger.info("Saved SC overlay DICOM: %s", written[0])
    return output_dir


def write_sc_overlay(
    seg: np.ndarray,
    image_volume: np.ndarray,
    output_labels: Sequence[int],
    affine: np.ndarray,
    output_dir: Union[str, Path],
    alpha: float = 0.7,
) -> Path:
    """One-call SC generation: contour -> overlay -> DICOM (temp SC dir)."""
    contour = generate_contour(seg, output_labels)
    overlay = create_overlay(image_volume, contour, alpha=alpha)
    return save_sc_dicom(overlay, np.asarray(affine), output_dir)
