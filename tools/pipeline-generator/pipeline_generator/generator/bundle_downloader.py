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

"""Download MONAI Bundles from HuggingFace."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from huggingface_hub import HfApi, snapshot_download

logger = logging.getLogger(__name__)


class BundleDownloader:
    """Downloads MONAI Bundle files from HuggingFace."""

    def __init__(self) -> None:
        """Initialize the downloader."""
        self.api = HfApi()

    def download_bundle(self, model_id: str, output_dir: Path, cache_dir: Optional[Path] = None) -> Path:
        """Download all files from a MONAI Bundle repository.

        Args:
            model_id: HuggingFace model ID (e.g., 'MONAI/spleen_ct_segmentation')
            output_dir: Directory to save the downloaded files
            cache_dir: Optional cache directory for HuggingFace downloads

        Returns:
            Path to the downloaded bundle directory
        """
        logger.info(f"Downloading bundle: {model_id}")

        # Create output directory
        bundle_dir = output_dir / "model"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download all files from the repository
            local_path = snapshot_download(
                repo_id=model_id,
                local_dir=bundle_dir,
                cache_dir=cache_dir,
            )

            logger.info(f"Bundle downloaded to: {local_path}")
            return Path(local_path)

        except Exception as e:
            logger.error(f"Failed to download bundle {model_id}: {e}")
            raise

    def get_bundle_metadata(self, bundle_path: Path) -> Optional[Dict[str, Any]]:
        """Read metadata.json from downloaded bundle.

        Args:
            bundle_path: Path to the downloaded bundle

        Returns:
            Dictionary containing bundle metadata or None if not found
        """
        metadata_paths = [
            bundle_path / "metadata.json",
            bundle_path / "configs" / "metadata.json",
        ]

        for metadata_path in metadata_paths:
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        data: Dict[str, Any] = json.load(f)
                        return data
                except Exception as e:
                    logger.error(f"Failed to read metadata from {metadata_path}: {e}")

        return None

    def get_inference_config(self, bundle_path: Path) -> Optional[Dict[str, Any]]:
        """Read inference.json from downloaded bundle.

        Args:
            bundle_path: Path to the downloaded bundle

        Returns:
            Dictionary containing inference configuration or None if not found
        """
        inference_paths = [
            bundle_path / "inference.json",
            bundle_path / "configs" / "inference.json",
        ]

        for inference_path in inference_paths:
            if inference_path.exists():
                try:
                    with open(inference_path, "r") as f:
                        data: Dict[str, Any] = json.load(f)
                        return data
                except Exception as e:
                    logger.error(f"Failed to read inference config from {inference_path}: {e}")

        return None

    def detect_model_file(self, bundle_path: Path) -> Optional[Path]:
        """Detect the model file in the bundle.

        Args:
            bundle_path: Path to the downloaded bundle

        Returns:
            Path to the model file or None if not found
        """
        # Common model file patterns
        model_patterns = [
            "models/model.ts",  # TorchScript
            "models/model.pt",  # PyTorch
            "models/model.onnx",  # ONNX
            "model.ts",
            "model.pt",
            "model.onnx",
        ]

        for pattern in model_patterns:
            model_path = bundle_path / pattern
            if model_path.exists():
                logger.info(f"Found model file: {model_path}")
                return model_path

        # If no standard pattern found, search for any model file
        for ext in [".ts", ".pt", ".onnx"]:
            model_files = list(bundle_path.glob(f"**/*{ext}"))
            if model_files:
                logger.info(f"Found model file: {model_files[0]}")
                return model_files[0]

        logger.warning(f"No model file found in bundle: {bundle_path}")
        return None

    def organize_bundle_structure(self, bundle_path: Path) -> None:
        """Organize bundle files into the expected MONAI Bundle structure.

        Creates the standard structure if files are in the root directory:
        bundle_root/
          configs/
            metadata.json
            inference.json
          models/
            model.pt
            model.ts

        Args:
            bundle_path: Path to the downloaded bundle
        """
        configs_dir = bundle_path / "configs"
        models_dir = bundle_path / "models"

        # Check if structure already exists
        has_configs_structure = configs_dir.exists() and (configs_dir / "metadata.json").exists()
        has_models_structure = models_dir.exists() and any(models_dir.glob("model.*"))

        if has_configs_structure and has_models_structure:
            logger.debug("Bundle already has proper structure")
            return

        logger.info("Organizing bundle into standard structure")

        # Create directories
        configs_dir.mkdir(exist_ok=True)
        models_dir.mkdir(exist_ok=True)

        # Move config files to configs/
        config_files = ["metadata.json", "inference.json"]
        for config_file in config_files:
            src_path = bundle_path / config_file
            if src_path.exists() and not (configs_dir / config_file).exists():
                src_path.rename(configs_dir / config_file)
                logger.debug(f"Moved {config_file} to configs/")

        # Move model files to models/
        # Prefer PyTorch (.pt) > ONNX (.onnx) > TorchScript (.ts) for better compatibility
        model_extensions = [".pt", ".onnx", ".ts"]

        # First move model files from root directory
        for ext in model_extensions:
            for model_file in bundle_path.glob(f"*{ext}"):
                if model_file.is_file() and not (models_dir / model_file.name).exists():
                    model_file.rename(models_dir / model_file.name)
                    logger.debug(f"Moved {model_file.name} to models/")

        # Check if we already have a suitable model in the main directory
        # Prefer .pt files, then .onnx, then .ts
        has_suitable_model = False
        for ext in model_extensions:
            if any(models_dir.glob(f"*{ext}")):
                has_suitable_model = True
                break

        # If no suitable model in main directory, move from subdirectories
        if not has_suitable_model:
            # Also move model files from subdirectories to the main models/ directory
            # This handles cases where models are in subdirectories like models/A100/
            # Prefer PyTorch models over TensorRT models for better compatibility
            for ext in model_extensions:
                model_files = list(models_dir.glob(f"**/*{ext}"))
                if not model_files:
                    continue

                # Filter files that are not in the main models directory
                subdirectory_files = [f for f in model_files if f.parent != models_dir]
                if not subdirectory_files:
                    continue

                target_name = f"model{ext}"
                target_path = models_dir / target_name
                if target_path.exists():
                    continue  # Target already exists

                # Prefer non-TensorRT models for better compatibility
                # TensorRT models often have "_trt" in their name
                preferred_file = None
                for model_file in subdirectory_files:
                    if "_trt" not in model_file.name.lower():
                        preferred_file = model_file
                        break

                # If no non-TensorRT model found, use the first available
                if preferred_file is None:
                    preferred_file = subdirectory_files[0]

                # Move the preferred model file
                preferred_file.rename(target_path)
                logger.debug(f"Moved {preferred_file.name} from {preferred_file.parent.name}/ to models/{target_name}")

                # Clean up empty subdirectory if it exists
                try:
                    if preferred_file.parent.exists() and not any(preferred_file.parent.iterdir()):
                        preferred_file.parent.rmdir()
                        logger.debug(f"Removed empty directory {preferred_file.parent}")
                except OSError:
                    pass  # Directory not empty or other issue
                break  # Only move one model file total

        # Ensure we have model.pt or model.ts in the main directory for MONAI Deploy
        # Create symlinks with standard names if needed
        standard_model_path = models_dir / "model.pt"
        if not standard_model_path.exists():
            # Look for any .pt file to link to model.pt
            pt_files = list(models_dir.glob("*.pt"))
            if pt_files:
                # Create a copy with the standard name
                pt_files[0].rename(standard_model_path)
                logger.debug(f"Renamed {pt_files[0].name} to model.pt")
            else:
                # No .pt file found, look for .ts file and create model.ts instead
                standard_ts_path = models_dir / "model.ts"
                if not standard_ts_path.exists():
                    ts_files = list(models_dir.glob("*.ts"))
                    if ts_files:
                        ts_files[0].rename(standard_ts_path)
                        logger.debug(f"Renamed {ts_files[0].name} to model.ts")
