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

* ``LabelToContourd`` — the reference's own dict transform: the label image
  is processed **slice-by-slice along its last axis**, each 2-D slice's
  binary mask goes through MONAI's ``LabelToContour`` (2-D Laplace kernel,
  kept where ``> 0``), and the label value is written at the edge voxels.
  The reference runs it on the label array in its internal orientation
  (``seg.transpose(2, 1, 0)`` of the original (Z, Y, X) volume);
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

__all__ = [
    "reference_label_to_contour",
    "generate_contour",
    "create_overlay",
    "save_sc_dicom",
    "write_sc_overlay",
]

_logger = logging.getLogger(f"{__name__}")

# The reference OverlayImageLabeld builds the jet colormap once (256 entries).
_JET_COLORMAP = cm.get_cmap("jet", 256)


def reference_label_to_contour(seg: np.ndarray, output_labels: Sequence[int]) -> np.ndarray:
    """Reference ``LabelToContourd`` math applied to ``seg`` **in the given
    layout** (reference app ``post_transforms.py``, verbatim semantics):

    the label volume is processed slice-by-slice along its last axis; for
    each slice and each non-background label in ``output_labels`` a binary
    mask (float32) goes through MONAI's 2-D ``LabelToContour`` (Laplace
    kernel, padded to keep the shape) and the label value is written at the
    voxels where the filtered slice is ``> 0``.

    The reference app invokes this on the label array in its internal
    orientation (``seg.transpose(2, 1, 0)`` of the original (Z, Y, X) volume);
    callers choose the layout.

    Args:
        seg: uint8 segmentation ``(Z, Y, X)`` (or the reference-internal
            orientation — this function is layout-agnostic).
        output_labels: labels to contour (reference: ``[1]``).

    Returns:
        uint8 volume of the same shape; 0 in the background, the label value
        on the (Laplace-thickened) per-slice contour of that label.
    """
    contour = np.zeros_like(seg, dtype=np.uint8)
    for i in range(seg.shape[-1]):
        slice_image = seg[:, :, i][None]  # (1, a, b) — channel-first 2-D slice
        unique_labels = np.unique(slice_image)
        unique_labels = unique_labels[unique_labels != 0]
        for label in unique_labels:
            # skip contour generation for labels that are not in output_labels
            if label not in output_labels:
                continue
            binary_mask = np.zeros_like(slice_image)
            binary_mask[slice_image == label] = 1.0
            binary_mask = binary_mask.astype(np.float32)
            thick_edges = LabelToContour()(binary_mask)
            thick_edges = (np.asarray(thick_edges) > 0).astype(np.uint8)
            contour[:, :, i] = np.where(thick_edges[0] > 0, label, contour[:, :, i])
    return contour


def generate_contour(seg: np.ndarray, output_labels: Sequence[int]) -> np.ndarray:
    """Reference-parity contour in the image's (Z, Y, X) orientation.

    The reference applies ``LabelToContourd`` to the label array in its
    internal orientation (``seg.transpose(2, 1, 0)``); the resulting contour
    is mapped back to (Z, Y, X) so it aligns with the original image volume
    for the SC overlay.

    Args:
        seg: uint8 segmentation ``(Z, Y, X)`` in original DICOM orientation.
        output_labels: labels to contour (reference: ``[1]``).

    Returns:
        uint8 volume of the same shape; 0 in the background, the label value
        on the (Laplace-thickened) contour of that label.
    """
    internal = np.ascontiguousarray(seg.transpose(2, 1, 0))
    contour_internal = reference_label_to_contour(internal, output_labels)
    return np.ascontiguousarray(contour_internal.transpose(2, 1, 0))


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
