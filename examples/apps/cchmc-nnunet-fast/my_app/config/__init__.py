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
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

__all__ = [
    "PreprocessParams",
    "load_preprocess_params",
    "find_jsonpkls_dir",
    "InferenceParams",
    "load_inference_params",
    "resolve_checkpoint_name",
    "detect_available_folds",
]

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


# ---------------------------------------------------------------------------
# Inference parameters (Phase 1 plan 02): config- and checkpoint-driven
# ---------------------------------------------------------------------------

NNUNET_CHECKPOINT_FILENAME = "nnunet_checkpoint.pth"

# Reference checkpoint auto-order (nnunet_bundle.py DEFAULT_MODEL_FILENAMES).
DEFAULT_CHECKPOINT_ORDER: Tuple[str, ...] = ("final_model.pt", "best_model.pt", "model.pt")


@dataclass(frozen=True)
class InferenceParams:
    """Per-configuration inference parameters loaded from the bundle.

    Everything the inference core needs that is *data*, not *code* (INF-006/
    INF-007): the patch size and network architecture come from
    ``plans.json``, the TTA mirror axes and trainer identity from
    ``nnunet_checkpoint.pth`` metadata (recorded, never hard-coded), and the
    weights come from the resolved checkpoint paths — so custom trainer
    variants load purely through their checkpoint path.
    """

    config_name: str
    trainer_name: str
    config_dir: Path
    jsonpkls_dir: Path
    patch_size: Tuple[int, ...]
    mirror_axes: Optional[Tuple[int, ...]]
    checkpoint_name: str
    folds: Tuple[int, ...]
    fold_paths: Tuple[Path, ...]
    num_input_channels: int
    num_segmentation_heads: int
    network_class_name: str
    network_init_kwargs: Mapping[str, Any]
    network_init_kwargs_req_import: Tuple[str, ...]


def resolve_checkpoint_name(config_dir: Union[str, Path], checkpoint_name: Optional[str] = None) -> str:
    """Resolve the concrete checkpoint filename for a config directory.

    Mirrors the reference ``_resolve_checkpoint_name``: an explicit name must
    exist (in the config dir or any ``fold_*/``); otherwise the first
    available of ``final_model.pt > best_model.pt > model.pt`` wins.
    """
    config_dir = Path(config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"Model folder not found: {config_dir}")

    fold_dirs = [p for p in config_dir.glob("fold_*") if p.is_dir()]

    def checkpoint_exists(filename: str) -> bool:
        if (config_dir / filename).exists():
            return True
        return any((d / filename).exists() for d in fold_dirs)

    available = sorted(
        {p.name for p in config_dir.glob("*.pt")}
        | {p.name for p in config_dir.glob("fold_*/*.pt")}
    )

    if checkpoint_name:
        if checkpoint_exists(checkpoint_name):
            return checkpoint_name
        raise FileNotFoundError(
            f"Checkpoint file not found in {config_dir}: {checkpoint_name}. "
            f"Available model files: {', '.join(available) or 'none found'}."
        )

    for candidate in DEFAULT_CHECKPOINT_ORDER:
        if checkpoint_exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"No supported checkpoint file was found in {config_dir}. "
        f"Expected one of {DEFAULT_CHECKPOINT_ORDER}. "
        f"Available model files: {', '.join(available) or 'none found'}."
    )


def detect_available_folds(config_dir: Union[str, Path], checkpoint_name: str) -> List[int]:
    """Reference ``auto_detect_available_folds``: sorted folds holding the checkpoint."""
    config_dir = Path(config_dir)
    return sorted(
        int(d.name.split("_")[-1])
        for d in config_dir.glob("fold_*")
        if d.is_dir() and (d / checkpoint_name).exists()
    )


def _resolve_config_dir_and_jsonpkls(
    model_path: Union[str, Path], config_name: str
) -> Tuple[Path, Path]:
    """Locate the per-config model dir and the bundle ``jsonpkls`` dir.

    Accepts the bundle root (``.../models``), the app root (containing
    ``models/``), or the per-config dir itself (containing
    ``nnunet_checkpoint.pth``).
    """
    base = Path(model_path)
    for root in (base, base / "models"):
        candidate = root / "jsonpkls"
        if (candidate / "plans.json").is_file():
            config_dir = root / config_name
            if (config_dir / NNUNET_CHECKPOINT_FILENAME).exists() or any(config_dir.glob("fold_*")):
                return config_dir, candidate
    if (base / NNUNET_CHECKPOINT_FILENAME).exists():
        jsonpkls = base.parent / "jsonpkls"
        if (jsonpkls / "plans.json").is_file():
            return base, jsonpkls
    raise FileNotFoundError(
        f"Could not locate config {config_name!r} (with {NNUNET_CHECKPOINT_FILENAME} or "
        f"fold_*/ dirs) and jsonpkls/plans.json under {model_path}."
    )


def load_inference_params(
    model_path: ModelPath,
    config_name: str = "3d_fullres",
    checkpoint_name: Optional[str] = None,
) -> InferenceParams:
    """Load per-config inference parameters from the bundle.

    Args:
        model_path: bundle root (with ``jsonpkls/`` + config dirs), app root,
            or the per-config dir itself.
        config_name: plans.json configuration key (default ``3d_fullres``).
        checkpoint_name: explicit checkpoint filename; ``None`` = reference
            auto-order ``final_model.pt > best_model.pt > model.pt``.

    Raises:
        FileNotFoundError: when the bundle, config, or checkpoint cannot be
            resolved.
        ValueError: when the checkpoint metadata names a different
            configuration than ``config_name`` (plans/weights mismatch).
    """
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    import torch

    config_dir, jsonpkls = _resolve_config_dir_and_jsonpkls(model_path, config_name)
    plans: Dict[str, Any] = json.loads((jsonpkls / "plans.json").read_text())
    dataset_json: Dict[str, Any] = json.loads((jsonpkls / "dataset.json").read_text())

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(config_name)

    meta_path = config_dir / NNUNET_CHECKPOINT_FILENAME
    if meta_path.exists():
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        trainer_name = str(meta.get("trainer_name", "unknown"))
        meta_config = (meta.get("init_args") or {}).get("configuration")
        if meta_config is not None and meta_config != config_name:
            raise ValueError(
                f"nnunet_checkpoint.pth for {config_dir} was trained for configuration "
                f"{meta_config!r}, not {config_name!r}; refusing to mix plans and weights."
            )
        mirror_axes = meta.get("inference_allowed_mirroring_axes")
        mirror_axes = None if mirror_axes is None else tuple(int(m) for m in mirror_axes)
    else:
        trainer_name, mirror_axes = "unknown", None

    checkpoint_name = resolve_checkpoint_name(config_dir, checkpoint_name)
    folds = detect_available_folds(config_dir, checkpoint_name)
    if not folds:
        raise FileNotFoundError(
            f"No fold_* directories with {checkpoint_name!r} found under {config_dir}."
        )

    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    num_heads = plans_manager.get_label_manager(dataset_json).num_segmentation_heads

    return InferenceParams(
        config_name=config_name,
        trainer_name=trainer_name,
        config_dir=config_dir,
        jsonpkls_dir=jsonpkls,
        patch_size=tuple(int(s) for s in configuration_manager.patch_size),
        mirror_axes=mirror_axes,
        checkpoint_name=checkpoint_name,
        folds=tuple(folds),
        fold_paths=tuple(config_dir / f"fold_{f}" / checkpoint_name for f in folds),
        num_input_channels=int(num_input_channels),
        num_segmentation_heads=int(num_heads),
        network_class_name=configuration_manager.network_arch_class_name,
        network_init_kwargs=dict(configuration_manager.network_arch_init_kwargs),
        network_init_kwargs_req_import=tuple(configuration_manager.network_arch_init_kwargs_req_import),
    )
