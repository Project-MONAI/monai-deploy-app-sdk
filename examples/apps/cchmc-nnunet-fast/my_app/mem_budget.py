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

"""Memory-budget calculator (INFR-03, D-15).

Before full-volume logits/probability tensors are allocated during inference,
the app estimates how much VRAM the per-config volumes need and compares it
against the free VRAM. The estimate is pure arithmetic — the only external
input is the free-VRAM probe (``torch.cuda.mem_get_info()[0]``), which is
optional so the calculator unit-tests headless with an explicit
``free_vram_bytes`` argument (D-15: unit-tested with synthetic large-volume
sizes that force the defer branch).

The real OOM path (a study large enough to actually exceed the 40 GB A100)
is UNEXERCISED on the dev airway study (D-15) — the defer branch is
reachable in code (``EnsembleAverageOperator(defer_strategy=True)``) but is
never triggered there; this is documented, not faked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

_FP32_BYTES_PER_ELEMENT = 4


@dataclass(frozen=True)
class BudgetPlan:
    """Result of :func:`compute_memory_budget`.

    Attributes:
        per_config_mb: estimated VRAM footprint per config (MB, decimal
            1e6), keyed by ``config_name``.
        total_bytes: sum of per-config estimates × ``safety_factor``.
        free_vram_bytes: the free VRAM the total was compared against.
        safety_factor: multiplier applied to the raw estimate.
        strategy: ``"full_volume"`` when ``total_bytes <= free_vram_bytes``
            (inclusive ``<=`` semantics: exactly-equal fits), otherwise
            ``"defer_to_incremental"`` (ensemble frees each consumed
            per-config probability tensor as it accumulates).
    """

    per_config_mb: Dict[str, float]
    total_bytes: int
    free_vram_bytes: int
    safety_factor: float
    strategy: str  # "full_volume" | "defer_to_incremental"


def _config_bytes(cfg: Mapping[str, Any]) -> int:
    """Per-config fp32 VRAM estimate (bytes) for one plans configuration.

    ``cfg`` mapping keys: ``config_name`` (str), ``num_input_channels``
    (int), ``num_segmentation_heads`` (int), ``preprocessed_shape``
    (tuple, C×H×D×W after resample), ``cropped_shape`` (tuple, C×H×D×W
    pre-resample). Components (fp32 = 4 bytes/element):

    * preprocessed volume:  ``prod(preprocessed_shape) × 4``
    * logits (post-crop):   ``heads × prod(cropped_shape[1:]) × 4``
    * probabilities:        ``heads × prod(preprocessed_shape[1:]) × 4``
    """
    preprocessed = tuple(int(s) for s in cfg["preprocessed_shape"])
    cropped = tuple(int(s) for s in cfg["cropped_shape"])
    heads = int(cfg["num_segmentation_heads"])

    def _prod(shape: Sequence[int]) -> int:
        out = 1
        for s in shape:
            out *= s
        return out

    preprocessed_bytes = _prod(preprocessed) * _FP32_BYTES_PER_ELEMENT
    logits_bytes = heads * _prod(cropped[1:]) * _FP32_BYTES_PER_ELEMENT
    probabilities_bytes = heads * _prod(preprocessed[1:]) * _FP32_BYTES_PER_ELEMENT
    return preprocessed_bytes + logits_bytes + probabilities_bytes


def compute_memory_budget(
    cfgs: Sequence[Mapping[str, Any]],
    free_vram_bytes: Optional[int] = None,
    safety_factor: float = 1.15,
) -> BudgetPlan:
    """Estimate VRAM for the per-config full-volume allocations.

    Args:
        cfgs: one mapping per selected configuration (see
            :func:`_config_bytes` for the keys).
        free_vram_bytes: free VRAM to budget against. Defaults to
            ``torch.cuda.mem_get_info()[0]`` (torch is imported lazily, only
            for this probe — the arithmetic itself is pure Python and the
            function is unit-testable headless by passing the argument
            explicitly).
        safety_factor: multiplier on the raw sum (default 1.15).

    Returns:
        A :class:`BudgetPlan`. ``strategy`` is ``"full_volume"`` when
        ``total_bytes <= free_vram_bytes`` (the comparison is inclusive — a
        total exactly equal to the free VRAM is treated as fitting), else
        ``"defer_to_incremental"``.
    """
    if not cfgs:
        raise ValueError("compute_memory_budget requires at least one configuration.")

    if free_vram_bytes is None:
        import torch

        free_vram_bytes = int(torch.cuda.mem_get_info()[0])

    per_config_bytes = {str(cfg["config_name"]): _config_bytes(cfg) for cfg in cfgs}
    total_bytes = int(sum(per_config_bytes.values()) * float(safety_factor))

    strategy = (
        "full_volume"
        if total_bytes <= int(free_vram_bytes)
        else "defer_to_incremental"
    )

    return BudgetPlan(
        per_config_mb={name: b / 1e6 for name, b in per_config_bytes.items()},
        total_bytes=total_bytes,
        free_vram_bytes=int(free_vram_bytes),
        safety_factor=float(safety_factor),
        strategy=strategy,
    )
