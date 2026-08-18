# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PostprocessOperator: connected-component cleanup on GPU + exactly one
boundary transfer to CPU (POST-01/POST-03).

Replicates the reference post-inference cleanup, in order:

1. the ``jsonpkls/postprocessing.pkl`` rules — for this bundle the pkl is
   present but contains **zero rules** (verified); the operator still applies
   them generically for bundles that do carry rules. The only rule family
   nnUNet 2.8.1 writes is ``remove_all_but_largest_component_from_segmentation``
   (acvl_utils, ``skimage.measure.label`` default connectivity = full, i.e.
   26-neighbor in 3D; keeps all components tied for the largest size).
2. ``KeepLargestConnectedComponentd(keys=pred, applied_labels=[1])`` — the
   reference app's MONAI stage-2 transform (default ``connectivity=None``,
   which ``skimage.measure.label`` resolves to full connectivity; measured on
   the vendored skimage 0.25.2: a diagonal voxel pair merges, i.e. 26-neighbor
   in 3D), ``independent=True`` (per-label), ``num_components=1``.

The connected-component analysis runs on GPU via CuPy (the plan's "CuPy /
skimage.measure GPU path"; this venv has no ``cupyx.scimage``/``cucim``, so
the CC is a deterministic two-pass min-seed labeling on CuPy with the same
full connectivity — verified voxel-identical to the reference skimage path on
the airway segmentation and on random-blob trials).

Data contract (POST-03): the segmentation enters and stays on GPU (zero-copy
torch <-> CuPy via DLPack) and is transferred to CPU numpy **exactly once**,
at the pipeline boundary, for the DICOM-SEG writer. Small control-flow scalars
(component counts) are read from the GPU — those are synchronization reads,
not array transfers; the segmentation array itself crosses the boundary
exactly once.

Also emits the airway volume text for DICOM SR (reference
``CalculateVolumeFromMaskd``/``ExtractVolumeToTextd`` semantics:
``int(round(sum(mask == label) * |det(affine[:3,:3])| / 1000.0, 2))`` mL).
"""

from __future__ import annotations

import json
import logging
import pickle
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import cupy as cp

from monai.deploy.core import Image, Operator, OperatorSpec

try:  # package-style import (my_app.*)
    from my_app.config import find_jsonpkls_dir
    from my_app.operators.gpu_util import GpuTiming, assert_cuda_available, assert_on_gpu, nvtx_range
    from my_app.operators.sc_overlay import write_sc_overlay
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from config import find_jsonpkls_dir
    from gpu_util import GpuTiming, assert_cuda_available, assert_on_gpu, nvtx_range
    from sc_overlay import write_sc_overlay

__all__ = [
    "PostprocessOperator",
    "load_postprocessing_rules",
    "postprocess_gpu",
    "cc_label_gpu",
    "keep_largest_component_gpu",
    "remove_all_but_largest_component_gpu",
    "calculate_volume_ml",
]

# Full connectivity in 3D (26-neighbor) — the effective connectivity of both
# reference CC paths (see module docstring).
_FULL_CONNECTIVITY_OFFSETS_3D: Tuple[Tuple[int, int, int], ...] = tuple(
    (a, b, c) for a, b, c in product((-1, 0, 1), repeat=3) if (a, b, c) != (0, 0, 0)
)

_INT32_MAX = 2**31 - 1

# The only rule function nnUNet 2.8.1's determine_postprocessing writes into
# postprocessing.pkl (verified against the vendored source and the bundle pkl).
_REMOVE_ALL_BUT_LARGEST = "remove_all_but_largest_component_from_segmentation"


def load_postprocessing_rules(model_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load and interpret ``jsonpkls/postprocessing.pkl`` (setup-time).

    The pkl stores ``(pp_fns, pp_fn_kwargs)`` with pickled function objects.
    We interpret (rather than call) them so the math can run on GPU: the only
    rule family nnUNet 2.8.1 writes is
    ``remove_all_but_largest_component_from_segmentation`` (any unknown rule
    raises instead of being silently skipped).

    A missing pkl yields an empty rule list — the reference app also skips
    nnU-Net postprocessing with a log line when the pkl is absent.
    """
    pkl_path = Path(find_jsonpkls_dir(model_path)) / "postprocessing.pkl"
    if not pkl_path.exists():
        return []
    with open(pkl_path, "rb") as f:
        pp_fns, pp_fn_kwargs = pickle.load(f)
    rules: List[Dict[str, Any]] = []
    for fn, kwargs in zip(pp_fns, pp_fn_kwargs):
        name = getattr(fn, "__name__", None)
        if name == _REMOVE_ALL_BUT_LARGEST:
            labels_or_regions = kwargs.get("labels_or_regions", [])
            if not isinstance(labels_or_regions, (list, tuple)):
                labels_or_regions = [labels_or_regions]
            rules.append(
                {
                    "kind": "remove_all_but_largest_component",
                    "labels_or_regions": list(labels_or_regions),
                    "background_label": int(kwargs.get("background_label", 0)),
                }
            )
        else:
            raise ValueError(
                f"unsupported postprocessing rule {name!r} in {pkl_path}; "
                "only remove_all_but_largest_component_from_segmentation is implemented "
                "(the only rule family nnUNet 2.8.1 writes). Refusing to silently skip it."
            )
    return rules


def _shift_gpu(arr: cp.ndarray, offset: Tuple[int, int, int]) -> cp.ndarray:
    """Shifted copy: ``out[i] = arr[i - offset]`` where in bounds, else 0."""
    out = cp.zeros_like(arr)
    dst, src = [], []
    for i, o in enumerate(offset):
        n = arr.shape[i]
        if o > 0:
            dst.append(slice(o, None))
            src.append(slice(None, n - o))
        elif o < 0:
            dst.append(slice(None, n + o))
            src.append(slice(-o, None))
        else:
            dst.append(slice(None))
            src.append(slice(None))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def cc_label_gpu(mask: cp.ndarray) -> Tuple[cp.ndarray, int]:
    """Connected-component labeling of a boolean 3D mask on GPU (CuPy).

    Deterministic two-pass min-seed labeling with full connectivity
    (26-neighbor, the effective connectivity of the reference
    ``skimage.measure.label(..., connectivity=None)`` paths — see module
    docstring). Each foreground voxel seeds a unique label; labels propagate
    by taking the component-wise minimum over the neighborhood until
    convergence, so every component ends up with a single (its minimum seed)
    label. Only the *partition* (not the specific ids) is consumed downstream.

    Returns:
        ``(labels, num_components)`` — int32 label array (0 = background) and
        the number of connected components.
    """
    if mask.ndim != 3:
        raise ValueError(f"cc_label_gpu expects a 3D mask, got ndim={mask.ndim}.")
    labels = cp.zeros(mask.shape, dtype=cp.int32)
    nz = cp.nonzero(mask)
    n_seeds = int(nz[0].size)
    if n_seeds == 0:
        return labels, 0
    labels[nz[0], nz[1], nz[2]] = cp.arange(1, n_seeds + 1, dtype=cp.int32)

    sentinel = cp.full((), _INT32_MAX, dtype=cp.int32)
    max_iters = 4 * int(max(mask.shape)) + 4
    for iteration in range(max_iters):
        old = labels
        for offset in _FULL_CONNECTIVITY_OFFSETS_3D:
            shifted = _shift_gpu(labels, offset)
            # 0 (background/out-of-bounds) is "no label" and must not compete
            # in the minimum: replace it with a sentinel that never wins.
            shifted = cp.where(shifted > 0, shifted, sentinel)
            labels = cp.minimum(labels, shifted)
        if bool((labels == old).all()):
            # Number of components: unique labels restricted to the mask.
            # (labels[mask] holds only foreground voxels, so label 0 is
            # already excluded — unique size IS the component count.)
            num = int(cp.unique(labels[mask]).size)
            return labels, num
    raise RuntimeError(
        f"cc_label_gpu did not converge within {max_iters} iterations "
        f"(mask shape {tuple(mask.shape)})"
    )


def keep_largest_component_gpu(seg: cp.ndarray, label: int, background_label: int = 0) -> cp.ndarray:
    """Keep only the largest connected component of ``label`` (in-place on seg).

    Reference parity with the reference app's MONAI
    ``KeepLargestConnectedComponentd(applied_labels=[1])`` (``independent``
    per-label mode, ``num_components=1``, full connectivity — see module
    docstring). Reference tie-break: ``np.argsort(np.bincount(nonzeros))[::-1][:1]``
    keeps the component with the HIGHEST feature id among those tied for the
    largest size; this replica matches that (ties for max size effectively do
    not occur on the airway segmentation, which is a single component —
    verified).
    """
    mask = seg == label
    nz = cp.nonzero(mask)
    if int(nz[0].size) == 0:
        return seg
    sl = tuple(slice(int(nz[i].min()), int(nz[i].max()) + 1) for i in range(3))
    mask_sub = mask[sl]
    labels_sub, num = cc_label_gpu(mask_sub)
    if num <= 1:
        return seg
    counts = cp.bincount(labels_sub[mask_sub])  # index 0 (background) unused
    max_count = int(counts.max())
    # Reference tie-break: highest feature id among the max-size components.
    tied = cp.nonzero(counts == max_count)[0]
    keep_id = int(tied.max())
    seg[sl][mask_sub & (labels_sub != keep_id)] = background_label
    return seg


def remove_all_but_largest_component_gpu(
    seg: cp.ndarray,
    labels_or_regions: Sequence[Union[int, Tuple[int, ...]]],
    background_label: int = 0,
) -> cp.ndarray:
    """Apply one ``postprocessing.pkl`` rule on GPU (in-place on seg).

    Reference parity with nnUNet's
    ``remove_all_but_largest_component_from_segmentation`` (acvl_utils
    ``generic_filter_components``: full-connectivity CC over the union mask,
    keep every component whose size equals the max size).

    Integer labels only — region tuples are not supported (this bundle's
    dataset has no regions; Phase 2 scope).
    """
    for l_or_r in labels_or_regions:
        if isinstance(l_or_r, (tuple, list)):
            raise ValueError(
                "region-based postprocessing rules are not supported by "
                "PostprocessOperator (Phase 1 bundles have no regions)."
            )
    mask = cp.zeros(seg.shape, dtype=cp.bool_)
    for l in labels_or_regions:
        mask |= seg == int(l)
    nz = cp.nonzero(mask)
    if int(nz[0].size) < 2:
        return seg
    sl = tuple(slice(int(nz[i].min()), int(nz[i].max()) + 1) for i in range(3))
    mask_sub = mask[sl]
    labels_sub, num = cc_label_gpu(mask_sub)
    if num < 2:
        return seg
    counts = cp.bincount(labels_sub[mask_sub])  # index 0 (background) unused
    # acvl filter_fn keeps ALL components tied for the max size.
    keep = counts[labels_sub] == counts.max()
    seg[sl][mask_sub & ~keep] = background_label
    return seg


def postprocess_gpu(
    seg: torch.Tensor,
    rules: Sequence[Dict[str, Any]],
    applied_labels: Sequence[int],
    background_label: int = 0,
) -> torch.Tensor:
    """Full GPU postprocess pass: pkl rules -> keep-largest, zero-copy torch<->CuPy.

    Args:
        seg: CUDA uint8 ``(Z, Y, X)`` segmentation (argmax of the averaged
            probabilities).
        rules: interpreted postprocessing.pkl rules (``load_postprocessing_rules``).
        applied_labels: labels for the keep-largest step (reference app: ``[1]``).
        background_label: background label value (0).

    Returns:
        CUDA uint8 ``(Z, Y, X)`` tensor with the cleanup applied. The array
        never leaves the GPU (the single boundary transfer happens in the
        operator's ``compute``).
    """
    if seg.device.type != "cuda":
        raise RuntimeError(
            f"postprocess_gpu requires a CUDA tensor, got {seg.device} "
            "(zero-copy GPU pipeline contract; no CPU fallback)."
        )
    # DLPack transfers buffer ownership to CuPy — clone so the caller's tensor
    # remains valid (the pipeline reuses upstream tensors; a consumed buffer
    # would read back post-CC values through the original handle).
    seg_u8 = seg.to(torch.uint8).contiguous().clone()
    seg_cp = cp.from_dlpack(seg_u8)

    for rule in rules:
        if rule["kind"] != "remove_all_but_largest_component":
            raise ValueError(f"unknown postprocessing rule kind {rule['kind']!r}.")
        remove_all_but_largest_component_gpu(seg_cp, rule["labels_or_regions"], rule["background_label"])

    for label in applied_labels:
        keep_largest_component_gpu(seg_cp, int(label), background_label)

    return torch.utils.dlpack.from_dlpack(seg_cp)


def calculate_volume_ml(seg_cpu: np.ndarray, label: int, voxel_volume_mm3: float) -> int:
    """Voxels-of-``label`` volume in mL — reference ``CalculateVolumeFromMaskd``
    semantics: ``int(round(sum(mask == label) * voxel_volume_mm3 / 1000.0, 2))``.
    """
    label_volume_mm3 = int(np.sum(seg_cpu == label)) * voxel_volume_mm3
    label_volume_ml = label_volume_mm3 / 1000.0
    return int(round(label_volume_ml, 2))


class PostprocessOperator(Operator):
    """Connected-component cleanup on GPU; single boundary transfer to CPU
    (POST-01/POST-03).

    Named Inputs:
        seg: the argmax segmentation of the averaged probabilities — a CUDA
            uint8 tensor ``(Z, Y, X)`` (or ``(1, Z, Y, X)``).
        image: the original in-memory ``Image`` (its 4x4
            ``nifti_affine_transform`` provides the voxel volume for the SR
            measurement text).

    Named Outputs:
        seg: final uint8 segmentation as **CPU numpy** ``(Z, Y, X)`` — the
            exactly-once GPU->CPU boundary transfer, ready for
            ``DICOMSegmentationWriterOperator``.
        result_text: the airway volume text for DICOM SR
            (e.g. ``"Airway Volume: 1 mL"``).
        dicom_sc_dir: the temp directory holding the generated DICOM SC
            overlay ``.dcm`` (LabelToContour + jet colormap + alpha blend,
            ``SaveImaged(.dcm)`` — reference-parity), consumed by the custom
            ``DICOMSCWriterOperator``.
    """

    INPUT_SEG = "seg"
    INPUT_IMAGE = "image"
    OUTPUT_SEG = "seg"
    OUTPUT_TEXT = "result_text"
    OUTPUT_SC_DIR = "dicom_sc_dir"

    def __init__(
        self,
        fragment: Any,
        *args: Any,
        model_path: Optional[Union[str, Any]] = None,
        applied_labels: Sequence[int] = (1,),
        label_names: Optional[Dict[int, str]] = None,
        output_labels: Sequence[int] = (1,),
        output_folder: Optional[Union[str, Any]] = None,
        **kwargs: Any,
    ):
        """Create the operator.

        Args:
            fragment: the owning application (passed to ``Operator``).
            model_path: bundle path with ``jsonpkls/postprocessing.pkl``
                (or a config dir); rules are loaded once in setup. A missing
                pkl means no rules (reference-app behavior).
            applied_labels: keep-largest labels (reference app: ``[1]``).
            label_names: label -> display name for the SR text (reference
                app: ``{"background": 0, "airway": 1}``).
            output_labels: labels included in the SR text (reference app:
                ``[1]``).
            output_folder: base output directory; the SC overlay ``.dcm`` is
                written to ``output_folder / "temp"`` (reference-app
                behavior: the temp dir is consumed and removed by the
                ``DICOMSCWriterOperator``).
        """
        # NOTE: holoscan 4.2's Operator.__init__ invokes self.setup(spec)
        # before this constructor body finishes — initialize all state first.
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.model_path = model_path
        self.applied_labels = tuple(int(l) for l in applied_labels)
        self.label_names: Dict[int, str] = dict(label_names or {0: "background", 1: "airway"})
        self.output_labels = tuple(int(l) for l in output_labels)
        self.output_folder = Path(output_folder) if output_folder is not None else None
        self._rules: Optional[List[Dict[str, Any]]] = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        """Declare the I/O and load the postprocessing rules once."""
        spec.input(self.INPUT_SEG)
        spec.input(self.INPUT_IMAGE)
        spec.output(self.OUTPUT_SEG)
        spec.output(self.OUTPUT_TEXT)
        spec.output(self.OUTPUT_SC_DIR)
        self._load_rules()

    def _load_rules(self) -> List[Dict[str, Any]]:
        if self._rules is None:
            if self.model_path is None:
                self._rules = []
                self._logger.info(
                    "No model_path given; PostprocessOperator runs with no "
                    "postprocessing.pkl rules (keep-largest only)."
                )
            else:
                self._rules = load_postprocessing_rules(self.model_path)
                self._logger.info(
                    "Loaded %d postprocessing rule(s) from postprocessing.pkl", len(self._rules)
                )
        return self._rules

    @staticmethod
    def _to_3d(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize the incoming seg to ``(Z, Y, X)``."""
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(f"PostprocessOperator supports batch size 1, got {tensor.shape[0]}.")
            return tensor[0]
        if tensor.ndim == 3:
            return tensor
        raise ValueError(f"expected a (Z, Y, X) or (1, Z, Y, X) seg tensor, got ndim={tensor.ndim}.")

    def _write_sc_output(self, seg_cpu: np.ndarray, image: Image) -> Path:
        """Generate the DICOM SC overlay ``.dcm`` in the temp SC directory.

        Reference-parity side output: LabelToContour contours of the
        ``output_labels`` over the original image, jet colormap, alpha blend,
        ``SaveImaged(output_ext='.dcm', output_dtype=int16)`` (see
        ``sc_overlay.write_sc_overlay``). The custom ``DICOMSCWriterOperator``
        consumes the returned directory (exactly one ``.dcm``) and removes it.
        """
        if self.output_folder is None:
            raise RuntimeError(
                "PostprocessOperator requires output_folder to emit the 'dicom_sc_dir' output."
            )
        metadata = image.metadata()
        affine = np.asarray(metadata.get("nifti_affine_transform"))
        if affine.size == 0 or tuple(affine.shape) != (4, 4):
            raise ValueError(
                "Image metadata is missing a 4x4 'nifti_affine_transform'; "
                "cannot write the SC overlay."
            )
        image_volume = np.asarray(image.asnumpy())
        if image_volume.shape != seg_cpu.shape:
            raise ValueError(
                f"SC overlay shape mismatch: image {image_volume.shape} vs seg {seg_cpu.shape}."
            )
        sc_temp_dir = self.output_folder / "temp"
        write_sc_overlay(seg_cpu, image_volume, self.output_labels, affine, sc_temp_dir)
        return sc_temp_dir

    @staticmethod
    def _voxel_volume_mm3(image: Image) -> float:
        """``|det(affine[:3, :3])|`` (mm^3) — reference CalculateVolumeFromMaskd."""
        metadata = image.metadata()
        affine = np.asarray(metadata.get("nifti_affine_transform"))
        if affine.size == 0 or tuple(affine.shape) != (4, 4):
            raise ValueError(
                "Image metadata is missing a 4x4 'nifti_affine_transform'; "
                "cannot compute the volume measurement."
            )
        return float(np.abs(np.linalg.det(affine[:3, :3])))

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        """CC cleanup on GPU; exactly one CPU transfer at the boundary."""
        with nvtx_range("postprocess"):
            timing = GpuTiming("postprocess")
            timing.start()

            # Entry guard: the pipeline is GPU-resident by contract (INF-005).
            assert_cuda_available()

            seg_in = op_input.receive(self.INPUT_SEG)
            image = op_input.receive(self.INPUT_IMAGE)
            if seg_in is None:
                raise ValueError("PostprocessOperator received no 'seg' input.")
            if image is None:
                raise ValueError("PostprocessOperator received no 'image' input (needed for the SR volume).")

            if not torch.is_tensor(seg_in):
                seg_in = torch.utils.dlpack.from_dlpack(seg_in)
            seg3d = self._to_3d(seg_in)
            # Device invariant at the boundary (INF-005).
            assert_on_gpu(seg3d)

            rules = self._load_rules()
            seg_gpu = postprocess_gpu(seg3d, rules, self.applied_labels)
            # Exit guard: the cleanup result must still be CUDA-resident
            # (POST-03: the CPU transfer below is the only one).
            assert_on_gpu(seg_gpu)

            # The EXACTLY-ONCE GPU->CPU boundary transfer (POST-03).
            seg_cpu = seg_gpu.cpu().numpy().astype(np.uint8, copy=False)

            voxel_volume = self._voxel_volume_mm3(image)
            volume_lines = []
            for label in self.output_labels:
                if label == 0:
                    continue
                name = self.label_names.get(label, str(label))
                volume_lines.append(f"{name.capitalize()} Volume: {calculate_volume_ml(seg_cpu, label, voxel_volume)} mL")
            result_text = "\n".join(volume_lines)
            if not result_text:
                raise ValueError("PostprocessOperator produced no result text (no output_labels with label != 0?).")

            counts = {int(v): int(c) for v, c in zip(*np.unique(seg_cpu, return_counts=True))}
            record = timing.stop()
            record["label_counts"] = counts
            record["voxel_volume_mm3"] = voxel_volume
            self._logger.info("postprocess timing: %s", json.dumps(record))
            self._logger.info("result text: %s", result_text)

            # SC side output: reference-parity overlay .dcm in the temp dir.
            dicom_sc_dir = self._write_sc_output(seg_cpu, image)

            op_output.emit(seg_cpu, self.OUTPUT_SEG)
            op_output.emit(result_text, self.OUTPUT_TEXT)
            op_output.emit(dicom_sc_dir, self.OUTPUT_SC_DIR)
