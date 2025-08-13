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

"""Generate MONAI Deploy applications from MONAI Bundles."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader

from .bundle_downloader import BundleDownloader
from ..config.settings import Settings, load_config

logger = logging.getLogger(__name__)


class AppGenerator:
    """Generates MONAI Deploy applications from MONAI Bundles."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the generator.

        Args:
            settings: Configuration settings (loads default if None)
        """
        self.downloader = BundleDownloader()
        self.settings = settings or load_config()

        # Set up Jinja2 template environment
        template_dir = Path(__file__).parent.parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate_app(
        self,
        model_id: str,
        output_dir: Path,
        app_name: Optional[str] = None,
        data_format: str = "auto",
    ) -> Path:
        """Generate a MONAI Deploy application from a HuggingFace model.

        Args:
            model_id: HuggingFace model ID (e.g., 'MONAI/spleen_ct_segmentation')
            output_dir: Directory to generate the application in
            app_name: Optional custom application name
            data_format: Data format - 'auto', 'dicom', or 'nifti'

        Returns:
            Path to the generated application directory
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download the bundle
        logger.info(f"Downloading bundle: {model_id}")
        bundle_path = self.downloader.download_bundle(model_id, output_dir)

        # Read bundle metadata and config
        metadata = self.downloader.get_bundle_metadata(bundle_path)
        inference_config = self.downloader.get_inference_config(bundle_path)

        if not metadata:
            logger.warning("No metadata.json found in bundle, using defaults")
            metadata = self._get_default_metadata(model_id)

        if not inference_config:
            logger.warning("No inference.json found in bundle, using defaults")
            inference_config = {}

        # Detect model file
        model_file = self.downloader.detect_model_file(bundle_path)
        if model_file:
            # Make path relative to bundle directory
            model_file = model_file.relative_to(bundle_path)

        # Detect model type from model_id or metadata
        model_type = self._detect_model_type(model_id, metadata)

        # Get model configuration if available
        model_config = self.settings.get_model_config(model_id)
        if model_config and data_format == "auto":
            # Use data types from configuration
            input_type = model_config.input_type
            output_type = model_config.output_type
        else:
            # Fall back to detection
            input_type = None
            output_type = None

        # Prepare template context
        context = self._prepare_context(
            model_id=model_id,
            metadata=metadata,
            inference_config=inference_config,
            model_file=model_file,
            app_name=app_name,
            data_format=data_format,
            model_type=model_type,
            input_type=input_type,
            output_type=output_type,
            model_config=model_config,
        )

        # Generate app.py
        self._generate_app_py(output_dir, context)

        # Generate app.yaml
        self._generate_app_yaml(output_dir, context)

        # Copy additional files if needed
        self._copy_additional_files(output_dir, context)

        logger.info(f"Application generated successfully in: {output_dir}")
        return output_dir

    def _prepare_context(
        self,
        model_id: str,
        metadata: Dict[str, Any],
        inference_config: Dict[str, Any],
        model_file: Optional[Path],
        app_name: Optional[str],
        data_format: str = "auto",
        model_type: str = "segmentation",
        input_type: Optional[str] = None,
        output_type: Optional[str] = None,
        model_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Prepare context for template rendering.

        Args:
            model_id: HuggingFace model ID
            metadata: Bundle metadata
            inference_config: Inference configuration
            model_file: Path to model file relative to bundle
            app_name: Optional custom application name
            data_format: Data format - 'auto', 'dicom', or 'nifti'

        Returns:
            Context dictionary for templates
        """
        # Extract model name from ID
        model_short_name = model_id.split("/")[-1]

        # Determine app name
        if not app_name:
            # Sanitize name to ensure valid Python identifier
            sanitized_name = "".join(
                c if c.isalnum() else "" for c in model_short_name.title()
            )
            app_name = f"{sanitized_name}App" if sanitized_name else "GeneratedApp"

        # Determine task type from metadata
        task = metadata.get("task", "segmentation").lower()
        modality = metadata.get("modality", "CT").upper()

        # Extract network data format
        network_data_format = metadata.get("network_data_format", {})
        inputs = network_data_format.get("inputs", {})
        outputs = network_data_format.get("outputs", {})

        # Determine if this is DICOM or NIfTI based
        if input_type:
            # Use provided input type
            use_dicom = input_type == "dicom"
            use_image = input_type == "image"
        elif data_format == "auto":
            # Try to detect from inference config
            use_dicom = self._detect_data_format(inference_config, modality)
            use_image = False
        elif data_format == "dicom":
            use_dicom = True
            use_image = False
        else:  # nifti
            use_dicom = False
            use_image = False

        # Extract organ/structure name
        organ = self._extract_organ_name(model_short_name, metadata)

        # Get output postfix from inference config
        output_postfix = "seg"  # Default postfix
        if "output_postfix" in inference_config:
            postfix_value = inference_config["output_postfix"]
            if isinstance(postfix_value, str) and not postfix_value.startswith("@"):
                output_postfix = postfix_value

        # Resolve generator-level overrides/configs
        resolved_channel_first = None
        if model_config and getattr(model_config, "configs", None) is not None:
            cfgs = model_config.configs
            if isinstance(cfgs, list):
                # Merge list of dicts; last one wins
                merged = {}
                for item in cfgs:
                    if isinstance(item, dict):
                        merged.update(item)
                resolved_channel_first = merged.get("channel_first", None)
            elif isinstance(cfgs, dict):
                resolved_channel_first = cfgs.get("channel_first", None)

        # Collect dependency hints from metadata.json
        required_packages_version = (
            metadata.get("required_packages_version", {}) if metadata else {}
        )
        extra_dependencies = (
            getattr(model_config, "dependencies", []) if model_config else []
        )
        if metadata and "numpy_version" in metadata:
            extra_dependencies.append(f"numpy=={metadata['numpy_version']}")
        if metadata and "pytorch_version" in metadata:
            extra_dependencies.append(f"torch=={metadata['pytorch_version']}")

        return {
            "model_id": model_id,
            "model_short_name": model_short_name,
            "app_name": app_name,
            "app_title": metadata.get("name", f"{organ} {task.title()} Inference"),
            "app_description": metadata.get("description", ""),
            "task": task,
            "modality": modality,
            "organ": organ,
            "use_dicom": use_dicom,
            "use_image": use_image,
            "input_type": input_type or ("dicom" if use_dicom else "nifti"),
            "output_type": output_type
            or ("json" if task == "classification" else "nifti"),
            "model_file": str(model_file) if model_file else "models/model.ts",
            "inference_config": inference_config,
            "metadata": metadata,
            "inputs": inputs,
            "outputs": outputs,
            "version": metadata.get("version", "1.0"),
            "authors": metadata.get("authors", "MONAI"),
            "output_postfix": output_postfix,
            "model_type": model_type,
            "channel_first_override": resolved_channel_first,
            "required_packages_version": required_packages_version,
            "extra_dependencies": extra_dependencies,
        }

    def _detect_data_format(
        self, inference_config: Dict[str, Any], modality: str
    ) -> bool:
        """Detect whether to use DICOM or NIfTI based on inference config and modality.

        Args:
            inference_config: Inference configuration
            modality: Image modality

        Returns:
            True for DICOM, False for NIfTI
        """
        # Check preprocessing transforms for hints
        if "preprocessing" in inference_config:
            transforms = inference_config["preprocessing"].get("transforms", [])
            for transform in transforms:
                target = transform.get("_target_", "")
                if "LoadImaged" in target or "LoadImage" in target:
                    # This suggests NIfTI format
                    return False

        # Default based on modality
        return modality in ["CT", "MR", "MRI"]

    def _extract_organ_name(self, model_name: str, metadata: Dict[str, Any]) -> str:
        """Extract organ/structure name from model name or metadata.

        Args:
            model_name: Short model name
            metadata: Bundle metadata

        Returns:
            Organ/structure name
        """
        # Try to get from metadata first
        if "organ" in metadata:
            return str(metadata["organ"])

        # Common organ names to extract
        organs = [
            "spleen",
            "liver",
            "kidney",
            "lung",
            "brain",
            "heart",
            "pancreas",
            "prostate",
            "breast",
            "colon",
        ]

        model_lower = model_name.lower()
        for organ in organs:
            if organ in model_lower:
                return organ.title()

        # Default
        return "Organ"

    def _detect_model_type(self, model_id: str, metadata: Dict[str, Any]) -> str:
        """Detect the model type based on model ID and metadata.

        Args:
            model_id: HuggingFace model ID
            metadata: Bundle metadata

        Returns:
            Model type: segmentation, pathology, multimodal, multimodal_llm
        """
        model_lower = model_id.lower()

        # Check for pathology models
        if "exaonepath" in model_lower or "pathology" in model_lower:
            return "pathology"

        # Check for multimodal LLMs
        if "llama" in model_lower or "vila" in model_lower:
            return "multimodal_llm"

        # Check for multimodal models
        if "chat" in model_lower or "multimodal" in model_lower:
            return "multimodal"

        # Check metadata for hints
        if metadata:
            task = metadata.get("task", "").lower()
            if "pathology" in task:
                return "pathology"
            elif "chat" in task or "qa" in task:
                return "multimodal"

        # Default to segmentation
        return "segmentation"

    def _generate_app_py(self, output_dir: Path, context: Dict[str, Any]) -> None:
        """Generate app.py file.

        Args:
            output_dir: Output directory
            context: Template context
        """
        # Select template based on model type and input/output types
        model_type = context.get("model_type", "segmentation")
        input_type = context.get("input_type", "nifti")
        output_type = context.get("output_type", "nifti")

        # Use the unified template for all cases
        template = self.env.get_template("app.py.j2")

        app_content = template.render(**context)
        app_path = output_dir / "app.py"

        with open(app_path, "w") as f:
            f.write(app_content)

        # Make executable
        app_path.chmod(0o755)

        logger.info(f"Generated app.py: {app_path}")

    def _generate_app_yaml(self, output_dir: Path, context: Dict[str, Any]) -> None:
        """Generate app.yaml file.

        Args:
            output_dir: Output directory
            context: Template context
        """
        template = self.env.get_template("app.yaml.j2")
        yaml_content = template.render(**context)

        yaml_path = output_dir / "app.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        logger.info(f"Generated app.yaml: {yaml_path}")

    def _copy_additional_files(self, output_dir: Path, context: Dict[str, Any]) -> None:
        """Copy additional required files.

        Args:
            output_dir: Output directory
            context: Template context
        """
        # No need for custom operators anymore - using SDK operators

        # Generate requirements.txt
        self._generate_requirements(output_dir, context)

        # Generate README.md
        self._generate_readme(output_dir, context)

    def _generate_requirements(self, output_dir: Path, context: Dict[str, Any]) -> None:
        """Generate requirements.txt file.

        Args:
            output_dir: Output directory
            context: Template context
        """
        template = self.env.get_template("requirements.txt.j2")
        requirements_content = template.render(**context)

        requirements_path = output_dir / "requirements.txt"
        with open(requirements_path, "w") as f:
            f.write(requirements_content)

        logger.info(f"Generated requirements.txt: {requirements_path}")

    def _generate_readme(self, output_dir: Path, context: Dict[str, Any]) -> None:
        """Generate README.md file.

        Args:
            output_dir: Output directory
            context: Template context
        """
        template = self.env.get_template("README.md.j2")
        readme_content = template.render(**context)

        readme_path = output_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)

        logger.info(f"Generated README.md: {readme_path}")

    def _get_default_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get default metadata when none is provided.

        Args:
            model_id: HuggingFace model ID

        Returns:
            Default metadata dictionary
        """
        model_name = model_id.split("/")[-1]
        return {
            "name": model_name.replace("_", " ").title(),
            "version": "1.0",
            "task": "segmentation",
            "modality": "CT",
            "description": f"MONAI Deploy application for {model_name}",
        }
