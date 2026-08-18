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

"""EnsembleAverageOperator: in-memory GPU probability averaging — no disk I/O
(INF-009/INF-010).

The reference round-trips per-config probability maps through ``.npz`` files
on disk (``resample_and_save`` writes them, ``nnunetv2.ensembling.ensemble.
average_probabilities`` loads and averages them). This operator replaces that
with an in-GPU-memory element-wise mean of the per-config probability tensors
emitted by PostResampleOperator:

* same math as the reference ``average_probabilities``: float32, the first
  volume is the accumulator base, the rest are added sequentially with in-place
  ``+=``, then ``/= n_configs`` — so a single config passes through unchanged
  and the result is bit-identical to the reference for the same inputs. The
  final division runs through CuPy because torch's CUDA scalar division is not
  bit-identical to the reference's numpy division for non power-of-2 n (see
  :func:`_divide_refparity`);
* **argmax happens AFTER averaging** (``argmax_to_segmentation``), matching the
  reference ``EnsembleProbabilitiesToSegmentation`` which converts the
  *averaged* probabilities to segmentation, not per-config segmentations;
* no temporary files are ever created (INF-010).

Phase 1 runs a single config (3D_fullres), so the mean is over one volume —
but the operator accepts a list/stack of per-config probability tensors, so
Phase 2 (2D / lowres / cascade) adds configs without redesign.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Sequence, Union

import torch

from monai.deploy.core import Operator, OperatorSpec

try:  # package-style import (my_app.*)
    from my_app.operators.gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        assert_on_gpu,
        get_study_id,
        nvtx_range,
    )
    from my_app.operators.preprocess_operator import to_holoscan_gpu_tensor
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from gpu_util import (
        GpuTiming,
        StudyTimingCollector,
        assert_cuda_available,
        assert_on_gpu,
        get_study_id,
        nvtx_range,
    )
    from preprocess_operator import to_holoscan_gpu_tensor

__all__ = ["EnsembleAverageOperator", "average_probabilities", "argmax_to_segmentation"]


def _to_tensor_list(probabilities: Any) -> List[torch.Tensor]:
    """Normalize the incoming probabilities to a list of (C, Z, Y, X) torch tensors.

    Accepts:
    * a single torch tensor of shape (C, Z, Y, X) — one config;
    * a stacked torch tensor of shape (N, C, Z, Y, X) — N configs;
    * a list/tuple of torch tensors or holoscan ``Tensor`` objects (DLPack),
      each (C, Z, Y, X).
    """
    if isinstance(probabilities, (list, tuple)):
        tensors: List[torch.Tensor] = []
        for item in probabilities:
            if not torch.is_tensor(item):
                # holoscan.core.Tensor (DLPack-backed) or other DLPack provider
                tensors.append(torch.utils.dlpack.from_dlpack(item))
            else:
                tensors.append(item)
    elif torch.is_tensor(probabilities):
        volume = probabilities
    else:
        # single holoscan Tensor (or other DLPack provider)
        try:
            volume = torch.utils.dlpack.from_dlpack(probabilities)
        except Exception as e:
            raise ValueError(
                f"unsupported probabilities input type {type(probabilities)!r}"
            ) from e
        tensors = None
    if not isinstance(probabilities, (list, tuple)):
        if volume.ndim == 4:
            tensors = [volume]
        elif volume.ndim == 5:
            # (N, C, Z, Y, X) stack of per-config volumes
            tensors = [volume[i] for i in range(volume.shape[0])]
        else:
            raise ValueError(
                "expected (C, Z, Y, X) or (N, C, Z, Y, X) probability tensors, "
                f"got ndim={volume.ndim}."
            )
    for t in tensors:
        if t.ndim != 4:
            raise ValueError(f"each per-config probability volume must be (C, Z, Y, X), got {tuple(t.shape)}.")
    return tensors


def _divide_refparity(avg: torch.Tensor, n: int) -> torch.Tensor:
    """Final normalization ``avg /= n`` with the reference's numpy float32 rounding.

    The reference ``average_probabilities`` finishes with a numpy ``avg /= n``
    (IEEE float32 division). torch's CUDA scalar division is not bit-identical
    to it for non power-of-2 n (measured: 1-ulp flips on ~10% of voxels),
    while CuPy's in-place division matches numpy bit-for-bit (verified for
    n in {2, 3, 4, 5, 7}). CuPy is a pinned cu13 dependency of this app
    (cupy-cuda13x), so the division runs through CuPy on GPU — zero-copy, no
    device change.
    """
    try:
        import cupy as cp
    except ImportError as e:
        raise RuntimeError(
            "average_probabilities requires CuPy for the reference-parity final "
            "division (n>1); the app pins cupy-cuda13x. "
            "Refusing to silently fall back to torch's non-bit-identical division."
        ) from e
    arr = cp.from_dlpack(avg)
    arr /= n
    return torch.utils.dlpack.from_dlpack(arr)


def average_probabilities(probabilities: Any) -> torch.Tensor:
    """Element-wise float32 mean of per-config probability volumes, on GPU.

    Reference parity with ``nnunetv2.ensembling.ensemble.average_probabilities``
    (float32 mean of the per-config probability maps): the first volume is the
    base, later volumes are added with sequential in-place ``+=``, then the sum
    is divided by the number of configs. All arithmetic stays in GPU memory
    (INF-009); no ``.npz`` or any other file I/O (INF-010).

    Args:
        probabilities: one or more per-config probability volumes — a
            (C, Z, Y, X) tensor, an (N, C, Z, Y, X) stack, or a list of
            (C, Z, Y, X) tensors (see ``_to_tensor_list``).

    Returns:
        The averaged probabilities ``(C, Z, Y, X)`` float32, on the same CUDA
        device. For a single input the output is a bit-exact copy.
    """
    tensors = _to_tensor_list(probabilities)
    if not tensors:
        raise ValueError("average_probabilities requires at least one probability volume.")

    reference_shape = tuple(tensors[0].shape)
    for t in tensors[1:]:
        if tuple(t.shape) != reference_shape:
            raise ValueError(
                f"all per-config probability volumes must share shape {reference_shape}, "
                f"got {tuple(t.shape)}."
            )

    # Reference order: avg = first; avg += each subsequent; avg /= n — all FP32.
    avg = tensors[0].float().clone()
    for t in tensors[1:]:
        avg += t.float()
    if len(tensors) > 1:
        avg = _divide_refparity(avg, len(tensors))
    return avg


def argmax_to_segmentation(probabilities: torch.Tensor) -> torch.Tensor:
    """argmax of the (already averaged) probabilities over the class axis.

    Reference parity with ``LabelManager.convert_logits_to_segmentation`` for
    non-region training, which reduces to argmax over the channel axis (the
    reference notes "Argmax is the same between logits or probabilities"; ties
    resolve to the lowest class index in both the reference numpy path and
    torch's ``argmax``).

    The argmax runs on the *averaged* probabilities — never per-config before
    averaging (INF-009).

    Returns:
        ``uint8`` CUDA tensor with the same spatial shape.
    """
    if not torch.is_tensor(probabilities):
        raise TypeError(f"argmax_to_segmentation expects a torch tensor, got {type(probabilities)!r}.")
    if probabilities.ndim != 4:
        raise ValueError(f"expected (C, Z, Y, X) probabilities, got {tuple(probabilities.shape)}.")
    seg = torch.argmax(probabilities, dim=0)
    return seg.to(torch.uint8)


class EnsembleAverageOperator(Operator):
    """In-memory GPU ensemble averaging of per-config probabilities (INF-009/010).

    Named Inputs:
        probabilities: per-config post-softmax probability volume(s) in
            original DICOM orientation — a single GPU tensor
            ``(C, Z, Y, X)`` (Phase 1, single config) or a stacked
            ``(N, C, Z, Y, X)`` tensor / list of per-config volumes (Phase 2+
            multi-config).

    Named Outputs:
        averaged_probabilities: zero-copy GPU tensor (``holoscan.core.Tensor``)
            with the element-wise float32 mean ``(C, Z, Y, X)``.
        seg: zero-copy GPU tensor with the uint8 segmentation obtained by
            :func:`argmax_to_segmentation` on the averaged probabilities
            (argmax-after-average, matching the reference
            ``EnsembleProbabilitiesToSegmentation``). This is the output the
            DAG wires into ``PostprocessOperator``.
    """

    INPUT_PROBABILITIES = "probabilities"
    OUTPUT_AVERAGED = "averaged_probabilities"
    OUTPUT_SEG = "seg"

    def __init__(
        self,
        fragment: Any,
        *args: Any,
        emit_averaged_probabilities: bool = True,
        **kwargs: Any,
    ):
        # NOTE: holoscan 4.2's Operator.__init__ invokes self.setup(spec)
        # before this constructor body finishes — initialize all state first.
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._emit_averaged = bool(emit_averaged_probabilities)
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        """Declare the operator's I/O: probabilities in, averaged + seg out.

        ``averaged_probabilities`` is declared only when
        ``emit_averaged_probabilities`` is True: a declared output with no
        downstream receiver makes the GXF scheduler reject the entity (no
        receiver connected to the transmitter), so the DAG wires only ``seg``.
        """
        spec.input(self.INPUT_PROBABILITIES)
        if self._emit_averaged:
            spec.output(self.OUTPUT_AVERAGED)
        spec.output(self.OUTPUT_SEG)

    def compute(self, op_input: Any, op_output: Any, context: Any) -> None:
        """Mean the per-config probability volumes in GPU memory (no disk)."""
        with nvtx_range("ensemble_average"):
            timing = GpuTiming("ensemble_average")
            timing.start()

            # Entry guard: the pipeline is GPU-resident by contract (INF-005).
            assert_cuda_available()

            received = op_input.receive(self.INPUT_PROBABILITIES)
            if received is None:
                raise ValueError("EnsembleAverageOperator received no 'probabilities' input.")

            tensors = _to_tensor_list(received)
            for t in tensors:
                # Device invariant at the boundary (INF-005): silent CPU
                # averaging is not allowed.
                assert_on_gpu(t)

            averaged = average_probabilities(tensors)

            # Exit guard: the emitted buffers must be CUDA-resident FP32/uint8.
            assert_on_gpu(averaged)
            if self._emit_averaged:
                op_output.emit(to_holoscan_gpu_tensor(averaged), self.OUTPUT_AVERAGED)

            # Argmax AFTER averaging (INF-009) — the uint8 segmentation the
            # DAG wires into PostprocessOperator.
            seg = argmax_to_segmentation(averaged)
            assert_on_gpu(seg)
            op_output.emit(to_holoscan_gpu_tensor(seg), self.OUTPUT_SEG)

            record = timing.stop()
            record["study"] = get_study_id(self.fragment)
            record["n_configs"] = len(tensors)
            record["averaged_shape"] = list(averaged.shape)
            StudyTimingCollector.record(self.fragment, record)
            self._logger.info("timing: %s", json.dumps(record))
