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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "PreprocessParams",
    "load_preprocess_params",
    "resolve_run_model_list",
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
        previous_stage: raw plans.json ``previous_stage`` of this config (the
            cascade producer, e.g. ``3d_lowres``); ``None`` for non-cascade
            configs. Drives the optional ``lowres_seg`` preprocess input.
        resample_seg_order: ``resampling_fn_seg_kwargs.order`` — the order of
            the cascade seg resample (1).
        resample_seg_order_z: ``resampling_fn_seg_kwargs.order_z`` (0).
        resample_seg_force_separate_z: ``resampling_fn_seg_kwargs.
            force_separate_z`` (``None`` = auto from anisotropy).
        foreground_labels: sorted non-zero dataset.json label values — the
            one-hot channel order for the cascade input (mirrors the
            reference ``label_manager.foreground_labels`` used by
            ``convert_labelmap_to_one_hot``).
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
    previous_stage: Optional[str] = None
    resample_seg_order: int = 1
    resample_seg_order_z: int = 0
    resample_seg_force_separate_z: Optional[bool] = None
    foreground_labels: Tuple[int, ...] = ()


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

    # Read every field from the PlansManager-RESOLVED configuration: the raw
    # entry of a cascading config carries only ``inherits_from`` +
    # ``previous_stage`` (spacing/normalization/resampling kwargs are
    # inherited), so the resolved configuration is the single venv-verified
    # source. ``previous_stage`` is present there too (verified on the real
    # bundle).
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    cfg = PlansManager(plans).get_configuration(config_name).configuration
    resampling_kwargs = cfg.get("resampling_fn_data_kwargs", {}) or {}

    force_separate_z = resampling_kwargs.get("force_separate_z", None)

    labels: Mapping[str, int] = {}
    dataset_json_path = jsonpkls / "dataset.json"
    if dataset_json_path.is_file():
        labels = json.loads(dataset_json_path.read_text()).get("labels", {}) or {}

    # Cascade (PIPE-04): the previous stage drives the optional lowres_seg
    # preprocess input; the seg resample kwargs (is_seg=True, order=1,
    # order_z=0, force_separate_z=None for the airway bundle) come from the
    # same resolved configuration.
    previous_stage = cfg.get("previous_stage")
    seg_kwargs = cfg.get("resampling_fn_seg_kwargs") or {}
    seg_force = seg_kwargs.get("force_separate_z", None)

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
        previous_stage=previous_stage,
        resample_seg_order=int(seg_kwargs.get("order", 1)),
        resample_seg_order_z=int(seg_kwargs.get("order_z", 0)),
        resample_seg_force_separate_z=None if seg_force is None else bool(seg_force),
        foreground_labels=tuple(sorted(int(v) for v in labels.values() if v != 0)),
    )


def resolve_run_model_list(
    model_list_arg: Optional[Sequence[str]],
    plans: Mapping[str, Any],
    model_root: ModelPath,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Resolve ``(run_model_list, ensemble_model_list)`` — reference semantics.

    Replicates the reference app's model-list logic
    (``nnunet_seg_operator.py:91-99``):

    * ``model_list_arg is None`` → the plans.json ``configurations`` dict
      order, filtered to configs with an existing model dir under
      ``model_root`` (reference ``_get_model_list_from_plans``);
    * an explicit ``model_list_arg`` is used as given (no filtering);
    * reference reorder (``nnunet_seg_operator.py:92-95``): if BOTH
      ``3d_lowres`` and ``3d_cascade_fullres`` are present, ``3d_lowres`` is
      removed and re-inserted immediately before ``3d_cascade_fullres``;
    * ``ensemble = run list minus ``3d_lowres```` (reference lines 96-98).
      Documented fast-app extension (Phase 2 Plan 04): when that leaves the
      ensemble empty but the run list is NOT (run == auxiliary stage only,
      e.g. ``['3d_lowres']``), the ensemble falls back to the run list —
      the reference raises ``ValueError`` there (it cannot read a lowres-only
      probability map back from disk) while the fast app's in-memory DAG can
      ensemble it (D-07: lowres runs standalone and is gated against its
      per-config reference). A truly empty run list still raises the
      reference's exact ``ValueError``.

    One fast-app extension (documented divergence from the reference): for
    each config whose RAW plans entry has ``previous_stage`` = ``p`` where
    ``p`` is not already in the list and HAS a model dir under
    ``model_root``, ``p`` is auto-inserted immediately before it. The
    reference app CRASHES when a cascade config is requested without its
    previous stage (it reads the previous stage's exported .nii.gz, which
    only exists if that stage actually ran — the reference list logic only
    reorders, it never auto-inserts). The in-memory cascade DAG requires the
    previous stage to be present, and the insertion is data-driven off
    plans.json ``previous_stage`` — a future 2d or other cascade stage needs
    no code change (D-02 config-genericity: no config names are hard-coded
    in this step; the only literals are the reference's own
    ``3d_lowres``/``3d_cascade_fullres`` in the reference-reorder step).

    Args:
        model_list_arg: explicit config list, or ``None`` for the default
            (plans.json order, filtered to existing model dirs).
        plans: the parsed ``jsonpkls/plans.json`` mapping.
        model_root: bundle model root containing the per-config dirs.

    Returns:
        ``(run, ensemble)`` tuples of config names.

    Raises:
        ValueError: when plans.json has no configurations.
        FileNotFoundError: when no config has a model dir under ``model_root``
            (default path only), mirroring the reference message.
    """
    configurations = plans.get("configurations", {}) or {}
    model_root = Path(model_root)

    if model_list_arg is None:
        configs = list(configurations.keys())
        if not configs:
            raise ValueError("No configurations found in plans.json.")
        run = [c for c in configs if (model_root / c).is_dir()]
        if not run:
            raise FileNotFoundError(
                f"No configured nnU-Net model directories were found under {model_root}. "
                f"Configured in plans.json: {configs}"
            )
    else:
        run = list(model_list_arg)

    # Fast-app extension: auto-insert each previous stage that a config in
    # the run list needs but is missing (data-driven off the raw plans
    # ``previous_stage`` field — no config names hard-coded here, D-02).
    extended: List[str] = []
    inserted: set = set()
    for c in run:
        cfg = configurations.get(c, {}) or {}
        previous = cfg.get("previous_stage")
        if (
            previous is not None
            and previous not in run
            and previous not in inserted
            and (model_root / previous).is_dir()
        ):
            extended.append(previous)
            inserted.add(previous)
        extended.append(c)
    run = extended

    # Reference reorder (nnunet_seg_operator.py:92-95): the reference app's
    # own semantics — 3d_lowres must run immediately before its cascade
    # consumer 3d_cascade_fullres in run order.
    if "3d_lowres" in run and "3d_cascade_fullres" in run:
        run.remove("3d_lowres")
        run.insert(run.index("3d_cascade_fullres"), "3d_lowres")

    ensemble = tuple(m for m in run if m != "3d_lowres")
    if not ensemble:
        if run:
            # Documented fast-app extension (Phase 2 Plan 04): a run list
            # that contains ONLY the auxiliary stage (e.g.
            # HOLOSCAN_MODEL_LIST=3d_lowres) would leave the ensemble empty.
            # The reference app raises here (nnunet_seg_operator.py:97-98 —
            # it cannot ensemble a stage whose probability maps it cannot
            # read back from disk); the fast app's in-memory DAG CAN, so the
            # ensemble falls back to the run list (D-07: lowres runs
            # standalone, gated against its per-config reference).
            ensemble = tuple(run)
        else:
            # Reference's exact error message (nnunet_seg_operator.py:97-98).
            raise ValueError(
                "At least one non-auxiliary model configuration is required for ensemble inference."
            )
    return tuple(run), ensemble


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
