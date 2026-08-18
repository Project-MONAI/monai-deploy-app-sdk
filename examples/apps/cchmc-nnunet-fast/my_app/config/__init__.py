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

"""Config-driven preprocessing parameters for the cchmc-nnunet-fast app.

Parameters are keyed by nnUNet configuration name (e.g. ``3d_fullres``) and read
from the model bundle ``jsonpkls/plans.json`` (+ ``dataset.json`` for labels),
mirroring how the reference bundle resolves them via ``PlansManager``. Nothing
is hard-coded in the operators: changing ``config_name`` (or the bundle) changes
the loaded parameters (PREP-01..04).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

__all__ = ["PreprocessParams", "load_preprocess_params", "find_jsonpkls_dir"]

ModelPath = Union[str, Path]


@dataclass(frozen=True)
class PreprocessParams:
    """Per-configuration preprocessing parameters loaded from the bundle.

    Attributes:
        config_name: the plans.json configuration key (e.g. ``3d_fullres``).
        spacing: target spacing in post-transpose axis order (H, D, W).
        transpose_forward: nnUNet ``transpose_forward`` applied to ``(C, ...)`` data.
        normalization_schemes: per-channel normalization class names (e.g.
            ``ZScoreNormalization``).
        use_mask_for_norm: per-channel flag for mask-based normalization.
        intensity_properties: per-channel intensity properties (plans.json
            ``foreground_intensity_properties_per_channel``).
        resample_order: skimage resize ``order`` for data resampling.
        resample_order_z: resize order used for the separate-z resample pass.
        resample_force_separate_z: force/forbid separate-z resampling
            (``None`` = auto from anisotropy, matching the reference default).
        resample_is_seg: whether the resampling kwargs describe a segmentation.
        labels: dataset.json label table (for downstream revert/metadata).
    """

    config_name: str
    spacing: Tuple[float, ...]
    transpose_forward: Tuple[int, ...]
    normalization_schemes: Tuple[str, ...]
    use_mask_for_norm: Tuple[bool, ...]
    intensity_properties: Mapping[str, Mapping[str, float]]
    resample_order: int
    resample_order_z: int
    resample_force_separate_z: Optional[bool]
    resample_is_seg: bool
    labels: Mapping[str, int] = field(default_factory=dict)


def find_jsonpkls_dir(model_path: ModelPath) -> Path:
    """Locate the bundle ``jsonpkls`` directory.

    Accepts either the bundle root (``.../models`` with ``jsonpkls/`` inside) or
    the repository layout where ``models`` itself is the path containing
    ``jsonpkls/``.
    """
    base = Path(model_path)
    for root in (base, base / "models"):
        candidate = root / "jsonpkls"
        if (candidate / "plans.json").is_file():
            return candidate
    raise FileNotFoundError(
        f"jsonpkls/plans.json not found under {model_path} (tried {base}/jsonpkls and "
        f"{base}/models/jsonpkls)."
    )


def load_preprocess_params(
    model_path: ModelPath,
    config_name: str = "3d_fullres",
) -> PreprocessParams:
    """Load per-config preprocessing parameters from ``plans.json``/``dataset.json``.

    Args:
        model_path: path to the model bundle (the directory that contains, or
            contains a ``models/``, folder with ``jsonpkls/plans.json``).
        config_name: plans.json configuration key (default ``3d_fullres``).

    Raises:
        FileNotFoundError: when ``jsonpkls/plans.json`` cannot be located.
        KeyError: when ``config_name`` is not a configuration in plans.json.
    """
    jsonpkls = find_jsonpkls_dir(model_path)
    plans: Dict[str, Any] = json.loads((jsonpkls / "plans.json").read_text())

    configurations = plans.get("configurations", {})
    if config_name not in configurations:
        raise KeyError(
            f"configuration {config_name!r} not found in {jsonpkls / 'plans.json'} "
            f"(available: {sorted(configurations)})."
        )
    cfg = configurations[config_name]
    resampling_kwargs = cfg.get("resampling_fn_data_kwargs", {}) or {}

    force_separate_z = resampling_kwargs.get("force_separate_z", None)

    labels: Mapping[str, int] = {}
    dataset_json_path = jsonpkls / "dataset.json"
    if dataset_json_path.is_file():
        labels = json.loads(dataset_json_path.read_text()).get("labels", {}) or {}

    return PreprocessParams(
        config_name=config_name,
        spacing=tuple(float(s) for s in cfg["spacing"]),
        transpose_forward=tuple(int(i) for i in plans["transpose_forward"]),
        normalization_schemes=tuple(cfg["normalization_schemes"]),
        use_mask_for_norm=tuple(bool(m) for m in cfg["use_mask_for_norm"]),
        intensity_properties=plans.get("foreground_intensity_properties_per_channel", {}) or {},
        resample_order=int(resampling_kwargs.get("order", 3)),
        resample_order_z=int(resampling_kwargs.get("order_z", 0)),
        resample_force_separate_z=None if force_separate_z is None else bool(force_separate_z),
        resample_is_seg=bool(resampling_kwargs.get("is_seg", False)),
        labels=labels,
    )
