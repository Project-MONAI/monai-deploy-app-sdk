"""Spec-driven topology planning (Phase 17, plan 17-03, TASK-02).

Pure data: stdlib-only plus the sibling registry module. No holoscan, no
torch, no nnunetv2, no my_app.config — headless-import safe like
``task_specs`` itself. No task/part name literals anywhere in this module:
every name arrives from the TaskSpec at call time, which is what makes the
builder topology-driven rather than task-driven.

``build_topology(spec, active_part_names)`` returns a frozen
:class:`TopologyPlan` for one of the three topologies:

* ``single_model``  — one active part (the P8 debug wiring; plain span names)
* ``multi_part``    — ALL of the spec's parts (the serialized chain; the plan
                      encodes the VRAM serialization edge set via
                      ``serialized=True``)
* ``crop_cascade``  — the spec carries a crop spec; construction is complete
                      (``crop_stages`` holds the 8-stage tuple), but
                      *materialization* (the crop operators) lands in
                      Phase 18 — compose() raises NotImplementedError for it.

``resolve_part(spec, name)`` is the single sanctioned lookup the app
materializer uses to map a selected active part name to its PartSpec
(label_offset, config_name, max_local_label). It returns the part whose
``name`` matches — never ``spec.parts[0]`` — so a non-first single-part
selection wires the correct model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

try:  # package-style import (my_app.*)
    from my_app.task_specs import PartSpec, TaskSpec
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from task_specs import PartSpec, TaskSpec

__all__ = ["CROP_STAGES", "TopologyPlan", "build_topology", "resolve_part"]

# The 8 crop-cascade stages in execution order (TS 2.18 semantics:
# native-resolution crop per 18-RESEARCH §Critical Correction; task-res
# resample occurs AFTER the crop; empty-mask -> empty segmentation).
# Construction complete in Phase 17; the operators that materialize them
# landed in Phase 18 (18-02).
CROP_STAGES: Tuple[str, ...] = (
    "downsample",
    "crop_infer",
    "crop_mask",
    "crop",
    "main_infer",
    "back_resample",
    "paste",
    "emit",
)


@dataclass(frozen=True)
class TopologyPlan:
    """Pure-data construction of one inference topology.

    The materializer (app.py compose()) dispatches on ``family`` and builds
    the EXACT v2.1 wiring for each family — same op classes, kwargs, port
    names, edges, flip contracts, scheduler.
    """

    family: str  # "single_model" | "multi_part" | "crop_cascade"
    parts: tuple  # ordered active part names (execution / later-part-wins order)
    serialized: bool  # True iff multi_part: builder must emit the VRAM gate edges
    # (post_{i-1} -> swin_i, port ("seg_argmax", "prev_part_seg"),
    #  for i in 1..N-1 — the P9 OOM fix)
    crop_stages: tuple  # () for single_model/multi_part; CROP_STAGES for crop_cascade


def build_topology(spec: TaskSpec, active_part_names: Sequence[str]) -> TopologyPlan:
    """Construct a TopologyPlan for the given spec and active-part selection.

    ``active_part_names`` must be an order-preserving subset of the spec's
    part names. Dispatch (order matters):

    1. crop spec present            -> crop_cascade (construction only in
                                       Phase 17; materialization is Phase 18)
    2. all parts, N >= 2            -> multi_part, serialized (VRAM edges)
    3. exactly one part             -> single_model (P8 plain-span wiring)
    4. any other subset size        -> NotImplementedError (preserves the
                                       pre-refactor 2-4-part guard)
    """
    spec_names = [p.name for p in spec.parts]
    active = list(active_part_names)

    # --- validate: order-preserving subset of the spec's part names ---
    positions: List[int] = []
    for name in active:
        if name not in spec_names:
            raise ValueError(f"active parts {active} are not a subset of spec parts {spec_names}")
        idx = spec_names.index(name)
        if positions and idx <= positions[-1]:
            raise ValueError(f"active parts {active} do not preserve spec part order {spec_names}")
        positions.append(idx)

    # --- dispatch (order matters) ---
    if spec.crop is not None:
        # Construction complete in Phase 17; materialization (the crop
        # operators) lands in Phase 18. Crop composes with any family —
        # the spec's own family is the post-crop inference topology.
        return TopologyPlan(
            family="crop_cascade",
            parts=tuple(active),
            serialized=False,
            crop_stages=CROP_STAGES,
        )
    if len(active) == len(spec_names) >= 2:
        return TopologyPlan(
            family="multi_part",
            parts=tuple(active),
            serialized=True,
            crop_stages=(),
        )
    if len(active) == 1:
        return TopologyPlan(
            family="single_model",
            parts=tuple(active),
            serialized=False,
            crop_stages=(),
        )
    raise NotImplementedError(
        f"{len(active)}-part subsets are not supported "
        f"(got {active}); supported: single-part debug or all "
        f"{len(spec_names)} parts"
    )


def resolve_part(spec: TaskSpec, name: str) -> PartSpec:
    """Return the PartSpec whose ``name`` matches; KeyError otherwise.

    The single sanctioned lookup for materialization: a selected active part
    name maps to THAT part's PartSpec (offset, config_name, max_local_label)
    — never a silent fallback to the first part.
    """
    for part in spec.parts:
        if part.name == name:
            return part
    raise KeyError(f"no part named {name!r} in spec; valid parts: {[p.name for p in spec.parts]}")
