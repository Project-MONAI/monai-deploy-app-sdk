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

"""Tests for the app generator."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pipeline_generator.generator import AppGenerator, BundleDownloader


class TestBundleDownloader:
    """Test BundleDownloader class."""

    def test_init(self):
        """Test BundleDownloader initialization."""
        downloader = BundleDownloader()
        assert downloader.api is not None

    @patch("pipeline_generator.generator.bundle_downloader.snapshot_download")
    def test_download_bundle(self, mock_snapshot_download):
        """Test downloading a bundle."""
        downloader = BundleDownloader()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_snapshot_download.return_value = str(temp_path / "model")

            result = downloader.download_bundle("MONAI/test_model", temp_path)

            assert result == temp_path / "model"
            mock_snapshot_download.assert_called_once()

    def test_get_bundle_metadata(self):
        """Test reading bundle metadata."""
        downloader = BundleDownloader()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test metadata
            metadata_path = temp_path / "configs" / "metadata.json"
            metadata_path.parent.mkdir(parents=True)
            metadata_path.write_text('{"name": "Test Model", "version": "1.0"}')

            metadata = downloader.get_bundle_metadata(temp_path)

            assert metadata is not None
            assert metadata["name"] == "Test Model"
            assert metadata["version"] == "1.0"

    def test_detect_model_file(self):
        """Test detecting model file in bundle."""
        downloader = BundleDownloader()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test model file
            models_dir = temp_path / "models"
            models_dir.mkdir()
            model_file = models_dir / "model.ts"
            model_file.touch()

            detected = downloader.detect_model_file(temp_path)

            assert detected is not None
            assert detected.name == "model.ts"


class TestAppGenerator:
    """Test AppGenerator class."""

    def test_init(self):
        """Test AppGenerator initialization."""
        generator = AppGenerator()
        assert generator.downloader is not None
        assert generator.env is not None

    def test_extract_organ_name(self):
        """Test organ name extraction."""
        generator = AppGenerator()

        # Test with known organ names
        assert generator._extract_organ_name("spleen_ct_segmentation", {}) == "Spleen"
        assert generator._extract_organ_name("liver_tumor_seg", {}) == "Liver"
        assert generator._extract_organ_name("kidney_segmentation", {}) == "Kidney"

        # Test with metadata
        assert generator._extract_organ_name("test_model", {"organ": "Heart"}) == "Heart"

        # Test default
        assert generator._extract_organ_name("unknown_model", {}) == "Organ"

    def test_prepare_context(self):
        """Test context preparation for templates."""
        generator = AppGenerator()

        metadata = {
            "name": "Test Model",
            "version": "1.0",
            "task": "segmentation",
            "modality": "CT",
        }

        context = generator._prepare_context(
            model_id="MONAI/test_model",
            metadata=metadata,
            inference_config={},
            model_file=Path("models/model.ts"),
            app_name=None,
        )

        assert context["model_id"] == "MONAI/test_model"
        assert context["app_name"] == "TestModelApp"
        assert context["task"] == "segmentation"
        assert context["modality"] == "CT"
        assert context["use_dicom"] is True
        assert context["model_file"] == "models/model.ts"

    @patch.object(BundleDownloader, "download_bundle")
    @patch.object(BundleDownloader, "get_bundle_metadata")
    @patch.object(BundleDownloader, "get_inference_config")
    @patch.object(BundleDownloader, "detect_model_file")
    def test_generate_app(self, mock_detect_model, mock_get_inference, mock_get_metadata, mock_download):
        """Test full app generation."""
        generator = AppGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"

            # Mock bundle download
            bundle_path = temp_path / "bundle"
            bundle_path.mkdir()
            mock_download.return_value = bundle_path

            # Mock metadata
            mock_get_metadata.return_value = {
                "name": "Test Model",
                "version": "1.0",
                "task": "segmentation",
                "modality": "CT",
            }

            # Mock inference config
            mock_get_inference.return_value = {}

            # Mock model file
            model_file = bundle_path / "models" / "model.ts"
            model_file.parent.mkdir(parents=True)
            model_file.touch()
            mock_detect_model.return_value = model_file

            # Generate app
            result = generator.generate_app("MONAI/test_model", output_dir)

            # Check generated files
            assert result == output_dir
            assert (output_dir / "app.py").exists()
            assert (output_dir / "app.yaml").exists()
            assert (output_dir / "requirements.txt").exists()

    def test_missing_metadata_uses_default(self):
        """Test that missing metadata triggers default metadata creation."""
        generator = AppGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"

            # Create a minimal bundle structure
            bundle_path = temp_path / "model"
            bundle_path.mkdir()

            # Mock the downloader to return bundle without metadata
            with patch.object(generator.downloader, "download_bundle") as mock_download:
                mock_download.return_value = bundle_path

                with patch.object(generator.downloader, "get_bundle_metadata") as mock_meta:
                    with patch.object(generator.downloader, "get_inference_config") as mock_inf:
                        with patch.object(generator.downloader, "detect_model_file") as mock_detect:
                            mock_meta.return_value = None  # No metadata
                            mock_inf.return_value = {}
                            mock_detect.return_value = None

                            with patch.object(generator, "_prepare_context") as mock_prepare:
                                with patch.object(generator, "_generate_app_py") as mock_app_py:
                                    with patch.object(generator, "_generate_app_yaml") as mock_yaml:
                                        with patch.object(generator, "_copy_additional_files") as mock_copy:
                                            # Return a valid context
                                            mock_prepare.return_value = {
                                                "model_id": "MONAI/test_model",
                                                "app_name": "TestApp",
                                                "task": "segmentation",
                                            }

                                            # This should trigger lines 73-74 and 438-439
                                            with patch(
                                                "pipeline_generator.generator.app_generator.logger"
                                            ) as mock_logger:
                                                generator.generate_app(
                                                    "MONAI/test_model",
                                                    output_dir,
                                                    data_format="auto",
                                                )

                                                # Verify warning was logged
                                                mock_logger.warning.assert_any_call(
                                                    "No metadata.json found in bundle, using defaults"
                                                )

    def test_inference_config_with_output_postfix(self):
        """Test inference config with output_postfix string value."""
        generator = AppGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"

            bundle_path = temp_path / "model"
            bundle_path.mkdir()

            # Create inference config with output_postfix
            inference_config = {"output_postfix": "_prediction"}  # String value, not @variable

            metadata = {"name": "Test Model"}

            with patch.object(generator.downloader, "download_bundle") as mock_download:
                mock_download.return_value = bundle_path

                with patch.object(generator.downloader, "get_bundle_metadata") as mock_meta:
                    with patch.object(generator.downloader, "get_inference_config") as mock_inf:
                        with patch.object(generator.downloader, "detect_model_file") as mock_detect:
                            mock_meta.return_value = metadata
                            mock_inf.return_value = inference_config  # This triggers lines 194-196
                            mock_detect.return_value = None

                            with patch.object(generator, "_generate_app_py") as mock_app_py:
                                with patch.object(generator, "_generate_app_yaml") as mock_yaml:
                                    with patch.object(generator, "_copy_additional_files") as mock_copy:
                                        result = generator.generate_app(
                                            "MONAI/test_model",
                                            output_dir,
                                            data_format="auto",
                                        )

                                        # Verify the output_postfix was extracted
                                        call_args = mock_app_py.call_args[0][1]
                                        assert call_args["output_postfix"] == "_prediction"

    def test_model_config_with_channel_first_override(self):
        """Test model config with channel_first override in configs list."""
        from pipeline_generator.config.settings import ModelConfig

        generator = AppGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"

            bundle_path = temp_path / "model"
            bundle_path.mkdir()

            # Create model config with configs list
            model_config = ModelConfig(
                model_id="MONAI/test_model",
                input_type="nifti",
                output_type="nifti",
                configs=[
                    {"channel_first": True, "other": "value"},
                    {"channel_first": False},  # Last one wins
                ],
            )

            # Mock settings.get_model_config using patch
            with patch("pipeline_generator.generator.app_generator.Settings.get_model_config") as mock_get_config:
                mock_get_config.return_value = model_config

                with patch.object(generator.downloader, "download_bundle") as mock_download:
                    mock_download.return_value = bundle_path

                    with patch.object(generator.downloader, "get_bundle_metadata") as mock_meta:
                        with patch.object(generator.downloader, "get_inference_config") as mock_inf:
                            with patch.object(generator.downloader, "detect_model_file") as mock_detect:
                                mock_meta.return_value = {"name": "Test"}
                                mock_inf.return_value = {}
                                mock_detect.return_value = None

                                with patch.object(generator, "_generate_app_py") as mock_app_py:
                                    with patch.object(generator, "_generate_app_yaml") as mock_yaml:
                                        with patch.object(generator, "_copy_additional_files") as mock_copy:
                                            generator.generate_app(
                                                "MONAI/test_model",
                                                output_dir,
                                                data_format="auto",
                                            )

                                            # Verify channel_first logic is computed correctly
                                            call_args = mock_app_py.call_args[0][1]
                                            assert call_args["channel_first"] is False

    def test_metadata_with_numpy_pytorch_versions(self):
        """Test metadata with numpy_version and pytorch_version."""
        generator = AppGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"

            bundle_path = temp_path / "model"
            bundle_path.mkdir()

            # Create metadata with version info
            metadata = {
                "name": "Test Model",
                "numpy_version": "1.21.0",
                "pytorch_version": "2.0.0",
            }

            with patch.object(generator.downloader, "download_bundle") as mock_download:
                mock_download.return_value = bundle_path

                with patch.object(generator.downloader, "get_bundle_metadata") as mock_meta:
                    with patch.object(generator.downloader, "get_inference_config") as mock_inf:
                        with patch.object(generator.downloader, "detect_model_file") as mock_detect:
                            with patch.object(generator.downloader, "organize_bundle_structure") as mock_organize:
                                mock_meta.return_value = metadata  # This triggers lines 216, 218
                                mock_inf.return_value = {}
                                mock_detect.return_value = None

                                with patch.object(generator, "_generate_app_py") as mock_app_py:
                                    with patch.object(generator, "_generate_app_yaml") as mock_yaml:
                                        with patch.object(generator, "_copy_additional_files") as mock_copy:
                                            generator.generate_app(
                                                "MONAI/test_model",
                                                output_dir,
                                                data_format="auto",
                                            )

                                        # Verify dependencies were added
                                        call_args = mock_copy.call_args[0][1]
                                        assert "numpy==1.21.0" in call_args["extra_dependencies"]
                                        assert "torch==2.0.0" in call_args["extra_dependencies"]

    def test_config_based_dependency_overrides(self):
        """Test config-based dependency overrides prevent metadata conflicts."""
        from pipeline_generator.config.settings import Endpoint, ModelConfig, Settings

        # Mock settings with config override for a model
        model_config = ModelConfig(
            model_id="MONAI/test_model",
            input_type="nifti",
            output_type="nifti",
            dependencies=["torch>=1.11.0", "numpy>=1.21.0", "monai>=1.3.0"],
        )

        endpoint = Endpoint(
            organization="MONAI", base_url="https://huggingface.co", description="Test", models=[model_config]
        )

        settings = Settings(endpoints=[endpoint])
        generator = AppGenerator(settings)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"
            bundle_path = temp_path / "model"
            bundle_path.mkdir()

            # Mock metadata with conflicting versions
            metadata = {
                "name": "Test Model",
                "numpy_version": "1.20.0",  # Older version
                "pytorch_version": "1.10.0",  # Incompatible version
                "monai_version": "0.8.0",  # Old MONAI version
            }

            with patch.object(generator.downloader, "download_bundle") as mock_download:
                mock_download.return_value = bundle_path

                with patch.object(generator.downloader, "get_bundle_metadata") as mock_meta:
                    with patch.object(generator.downloader, "get_inference_config") as mock_inf:
                        with patch.object(generator.downloader, "detect_model_file") as mock_detect:
                            with patch.object(generator.downloader, "organize_bundle_structure") as mock_organize:
                                mock_meta.return_value = metadata
                                mock_inf.return_value = {}
                                mock_detect.return_value = None

                                with patch.object(generator, "_generate_app_py") as mock_app_py:
                                    with patch.object(generator, "_generate_app_yaml") as mock_yaml:
                                        with patch.object(generator, "_copy_additional_files") as mock_copy:
                                            generator.generate_app(
                                                "MONAI/test_model",
                                                output_dir,
                                                data_format="auto",
                                            )

                                            call_args = mock_copy.call_args[0][1]

                                            # Config dependencies should be used instead of metadata
                                            assert "torch>=1.11.0" in call_args["extra_dependencies"]
                                            assert "numpy>=1.21.0" in call_args["extra_dependencies"]
                                            assert "monai>=1.3.0" in call_args["extra_dependencies"]

                                            # Old metadata versions should NOT be included
                                            assert "torch==1.10.0" not in call_args["extra_dependencies"]
                                            assert "numpy==1.20.0" not in call_args["extra_dependencies"]

                                            # MONAI version should be removed from metadata to prevent template conflict
                                            assert "monai_version" not in call_args["metadata"]

                                            # Verify bundle structure was organized
                                            mock_organize.assert_called_once()

    def test_dependency_conflict_resolution_no_config(self):
        """Test that without config overrides, metadata versions are used."""
        generator = AppGenerator()  # No settings, no config overrides

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"
            bundle_path = temp_path / "model"
            bundle_path.mkdir()

            metadata = {
                "name": "Test Model",
                "numpy_version": "1.21.0",
                "pytorch_version": "1.12.0",
                "monai_version": "1.0.0",
            }

            with patch.object(generator.downloader, "download_bundle") as mock_download:
                mock_download.return_value = bundle_path

                with patch.object(generator.downloader, "get_bundle_metadata") as mock_meta:
                    with patch.object(generator.downloader, "get_inference_config") as mock_inf:
                        with patch.object(generator.downloader, "detect_model_file") as mock_detect:
                            with patch.object(generator.downloader, "organize_bundle_structure") as mock_organize:
                                mock_meta.return_value = metadata
                                mock_inf.return_value = {}
                                mock_detect.return_value = None

                                with patch.object(generator, "_generate_app_py") as mock_app_py:
                                    with patch.object(generator, "_generate_app_yaml") as mock_yaml:
                                        with patch.object(generator, "_copy_additional_files") as mock_copy:
                                            generator.generate_app(
                                                "MONAI/test_model",
                                                output_dir,
                                                data_format="auto",
                                            )

                                            call_args = mock_copy.call_args[0][1]

                                            # Should use metadata versions when no config
                                            assert "numpy==1.21.0" in call_args["extra_dependencies"]
                                            assert "torch==1.12.0" in call_args["extra_dependencies"]

                                            # MONAI version should be moved from metadata to extra_dependencies
                                            assert "monai==1.0.0" in call_args["extra_dependencies"]
                                            assert "monai_version" not in call_args["metadata"]

    def test_monai_version_handling_in_app_generator(self):
        """Test that MONAI version logic is correctly handled in app generator (moved from template)."""
        generator = AppGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"
            bundle_path = temp_path / "model"
            bundle_path.mkdir()

            # Test case 1: Config has MONAI - should not add metadata version
            with patch.object(generator.downloader, "download_bundle") as mock_download:
                mock_download.return_value = bundle_path

                with patch.object(generator.downloader, "get_bundle_metadata") as mock_meta:
                    with patch.object(generator.downloader, "get_inference_config") as mock_inf:
                        with patch.object(generator.downloader, "detect_model_file") as mock_detect:
                            with patch.object(generator.downloader, "organize_bundle_structure") as mock_organize:
                                # Mock model config with MONAI dependency
                                from pipeline_generator.config.settings import Endpoint, ModelConfig, Settings

                                model_config = ModelConfig(
                                    model_id="MONAI/test_model",
                                    input_type="nifti",
                                    output_type="nifti",
                                    dependencies=["monai>=1.3.0"],
                                )
                                endpoint = Endpoint(
                                    organization="MONAI",
                                    base_url="https://huggingface.co",
                                    description="Test",
                                    models=[model_config],
                                )
                                settings = Settings(endpoints=[endpoint])
                                generator_with_config = AppGenerator(settings)

                                mock_meta.return_value = {"monai_version": "0.8.0"}
                                mock_inf.return_value = {}
                                mock_detect.return_value = None

                                context = generator_with_config._prepare_context(
                                    "MONAI/test_model",
                                    {"monai_version": "0.8.0"},
                                    {},
                                    None,
                                    None,
                                    "auto",
                                    "segmentation",
                                    None,
                                    None,
                                    model_config,  # Pass the model config
                                )

                                # Should have config MONAI but not metadata MONAI
                                assert "monai>=1.3.0" in context["extra_dependencies"]
                                assert "monai==0.8.0" not in context["extra_dependencies"]
                                assert "monai_version" not in context["metadata"]

            # Test case 2: No config MONAI - should add metadata version
            generator_no_config = AppGenerator()  # No settings
            context2 = generator_no_config._prepare_context(
                "MONAI/test_model",
                {"monai_version": "1.0.0"},
                {},
                None,
                None,
                "auto",
                "segmentation",
                None,
                None,
                None,  # No model config
            )

            # Should add metadata MONAI version to extra_dependencies
            assert "monai==1.0.0" in context2["extra_dependencies"]
            assert "monai_version" not in context2["metadata"]

            # Test case 3: No config and no metadata - should add fallback
            context3 = generator_no_config._prepare_context(
                "MONAI/test_model", {}, {}, None, None, "auto", "segmentation", None, None, None  # No model config
            )

            # Should add fallback MONAI version
            assert "monai<=1.5.0" in context3["extra_dependencies"]

    def test_inference_config_with_loadimage_transform(self):
        """Test _detect_data_format with LoadImaged transform."""
        generator = AppGenerator()

        # Create inference config with LoadImaged transform
        inference_config = {
            "preprocessing": {
                "transforms": [
                    {"_target_": "monai.transforms.LoadImaged", "keys": ["image"]},
                    {"_target_": "monai.transforms.EnsureChannelFirstd"},
                ]
            }
        }

        # This should return False (NIfTI format) - covers lines 259-264
        result = generator._detect_data_format(inference_config, "CT")
        assert result is False

    def test_inference_config_with_string_transforms(self):
        """Test _detect_data_format with string transforms expression."""
        generator = AppGenerator()

        # Create inference config with string transforms (like spleen_deepedit_annotation)
        inference_config = {
            "preprocessing": {
                "_target_": "Compose",
                "transforms": "$@preprocessing_transforms + @deepedit_transforms + @extra_transforms",
            },
            "preprocessing_transforms": [
                {"_target_": "LoadImaged", "keys": "image"},
                {"_target_": "EnsureChannelFirstd", "keys": "image"},
            ],
        }

        # This should return False (NIfTI format) because LoadImaged is found in config string
        result = generator._detect_data_format(inference_config, "CT")
        assert result is False

    def test_inference_config_with_string_transforms_no_loadimage(self):
        """Test _detect_data_format with string transforms expression without LoadImaged."""
        generator = AppGenerator()

        # Create inference config with string transforms but no LoadImaged
        inference_config = {
            "preprocessing": {"_target_": "Compose", "transforms": "$@preprocessing_transforms + @other_transforms"},
            "preprocessing_transforms": [
                {"_target_": "SomeOtherTransform", "keys": "image"},
                {"_target_": "EnsureChannelFirstd", "keys": "image"},
            ],
        }

        # This should return True (DICOM format) for CT modality when no LoadImaged found
        result = generator._detect_data_format(inference_config, "CT")
        assert result is True

    def test_detect_model_type_pathology(self):
        """Test _detect_model_type for pathology models."""
        generator = AppGenerator()

        # Test pathology detection by model ID - covers line 319
        assert generator._detect_model_type("LGAI-EXAONE/EXAONEPath", {}) == "pathology"
        assert generator._detect_model_type("MONAI/pathology_model", {}) == "pathology"

        # Test pathology detection by metadata - covers line 333
        metadata = {"task": "pathology classification"}
        assert generator._detect_model_type("MONAI/some_model", metadata) == "pathology"

    def test_detect_model_type_multimodal_llm(self):
        """Test _detect_model_type for multimodal LLM models."""
        generator = AppGenerator()

        # Test LLM detection - covers line 323
        assert generator._detect_model_type("MONAI/Llama3-VILA-M3-3B", {}) == "multimodal_llm"
        assert generator._detect_model_type("MONAI/vila_model", {}) == "multimodal_llm"

    def test_detect_model_type_multimodal(self):
        """Test _detect_model_type for multimodal models."""
        generator = AppGenerator()

        # Test multimodal detection by model ID - covers line 327
        assert generator._detect_model_type("MONAI/chat_model", {}) == "multimodal"
        assert generator._detect_model_type("MONAI/multimodal_seg", {}) == "multimodal"

        # Test multimodal detection by metadata - covers line 335
        metadata = {"task": "medical chat"}
        assert generator._detect_model_type("MONAI/some_model", metadata) == "multimodal"

        metadata = {"task": "visual qa"}
        assert generator._detect_model_type("MONAI/some_model", metadata) == "multimodal"

    def test_model_config_with_dict_configs(self):
        """Test model config with configs as dict instead of list."""
        from pipeline_generator.config.settings import ModelConfig

        generator = AppGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"

            bundle_path = temp_path / "model"
            bundle_path.mkdir()

            # Create model config with configs dict - covers line 210
            model_config = ModelConfig(
                model_id="MONAI/test_model",
                input_type="nifti",
                output_type="nifti",
                configs={"channel_first": True},  # Dict instead of list
            )

            # Mock settings.get_model_config using patch
            with patch("pipeline_generator.generator.app_generator.Settings.get_model_config") as mock_get_config:
                mock_get_config.return_value = model_config

                with patch.object(generator.downloader, "download_bundle") as mock_download:
                    mock_download.return_value = bundle_path

                    with patch.object(generator.downloader, "get_bundle_metadata") as mock_meta:
                        with patch.object(generator.downloader, "get_inference_config") as mock_inf:
                            with patch.object(generator.downloader, "detect_model_file") as mock_detect:
                                mock_meta.return_value = {"name": "Test"}
                                mock_inf.return_value = {}
                                mock_detect.return_value = None

                                with patch.object(generator, "_generate_app_py") as mock_app_py:
                                    with patch.object(generator, "_generate_app_yaml") as mock_yaml:
                                        with patch.object(generator, "_copy_additional_files") as mock_copy:
                                            generator.generate_app(
                                                "MONAI/test_model",
                                                output_dir,
                                                data_format="auto",
                                            )

                                            call_args = mock_app_py.call_args[0][1]
                                            assert call_args["channel_first"] is True

    def test_channel_first_logic_refactoring(self):
        """Test the refactored channel_first logic works correctly."""
        generator = AppGenerator()

        # Test case 1: image input, non-classification task -> should be False
        context1 = generator._prepare_context(
            model_id="test/model",
            metadata={"task": "segmentation", "name": "Test Model"},
            inference_config={},
            model_file=None,
            app_name="TestApp",
            input_type="image",
            output_type="nifti",
        )
        assert context1["channel_first"] is False

        # Test case 2: image input, classification task -> should be True
        context2 = generator._prepare_context(
            model_id="test/model",
            metadata={"task": "classification", "name": "Test Model"},
            inference_config={},
            model_file=None,
            app_name="TestApp",
            input_type="image",
            output_type="json",
        )
        assert context2["channel_first"] is True

        # Test case 3: dicom input -> should be True
        context3 = generator._prepare_context(
            model_id="test/model",
            metadata={"task": "segmentation", "name": "Test Model"},
            inference_config={},
            model_file=None,
            app_name="TestApp",
            input_type="dicom",
            output_type="nifti",
        )
        assert context3["channel_first"] is True

        # Test case 4: nifti input -> should be True
        context4 = generator._prepare_context(
            model_id="test/model",
            metadata={"task": "segmentation", "name": "Test Model"},
            inference_config={},
            model_file=None,
            app_name="TestApp",
            input_type="nifti",
            output_type="nifti",
        )
        assert context4["channel_first"] is True

    def test_get_default_metadata(self):
        """Test _get_default_metadata method directly."""
        generator = AppGenerator()

        # Test default metadata generation - covers lines 438-439
        metadata = generator._get_default_metadata("MONAI/spleen_ct_segmentation")

        assert metadata["name"] == "Spleen Ct Segmentation"
        assert metadata["version"] == "1.0"
        assert metadata["task"] == "segmentation"
        assert metadata["modality"] == "CT"
        assert "spleen_ct_segmentation" in metadata["description"]

    @patch.object(BundleDownloader, "download_bundle")
    @patch.object(BundleDownloader, "get_bundle_metadata")
    @patch.object(BundleDownloader, "get_inference_config")
    @patch.object(BundleDownloader, "detect_model_file")
    def test_nifti_segmentation_imports(self, mock_detect_model, mock_get_inference, mock_get_metadata, mock_download):
        """Test that NIfTI segmentation apps have required imports."""
        generator = AppGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"

            # Mock bundle download
            bundle_path = temp_path / "bundle"
            bundle_path.mkdir()
            mock_download.return_value = bundle_path

            # Mock metadata for NIfTI segmentation
            mock_get_metadata.return_value = {
                "name": "Spleen CT Segmentation",
                "version": "1.0",
                "task": "segmentation",
                "modality": "CT",
            }

            # Mock inference config (minimal)
            mock_get_inference.return_value = {}

            # Mock model file (TorchScript)
            model_file = bundle_path / "models" / "model.ts"
            model_file.parent.mkdir(parents=True)
            model_file.touch()
            mock_detect_model.return_value = model_file

            # Generate app
            generator.generate_app("MONAI/spleen_ct_segmentation", output_dir)

            # Read generated app.py
            app_file = output_dir / "app.py"
            assert app_file.exists()
            app_content = app_file.read_text()

            # Check critical imports for MonaiBundleInferenceOperator
            assert (
                "from monai.deploy.core.domain import Image" in app_content
            ), "Image import missing - required for MonaiBundleInferenceOperator"
            assert (
                "from monai.deploy.core.io_type import IOType" in app_content
            ), "IOType import missing - required for MonaiBundleInferenceOperator"
            assert "IOMapping" in app_content, "IOMapping import missing - required for MonaiBundleInferenceOperator"

            # Check operator imports
            assert "from generic_directory_scanner_operator import GenericDirectoryScanner" in app_content
            assert "from monai.deploy.operators.nii_data_loader_operator import NiftiDataLoader" in app_content
            assert "from nifti_writer_operator import NiftiWriter" in app_content
            assert "from monai.deploy.operators.monai_bundle_inference_operator import" in app_content

            # Check that the required operator files are physically copied (Phase 7 verification)
            assert (
                output_dir / "generic_directory_scanner_operator.py"
            ).exists(), "GenericDirectoryScanner operator file not copied"
            assert (output_dir / "nifti_writer_operator.py").exists(), "NiftiWriter operator file not copied"

    @patch.object(BundleDownloader, "download_bundle")
    @patch.object(BundleDownloader, "get_bundle_metadata")
    @patch.object(BundleDownloader, "get_inference_config")
    @patch.object(BundleDownloader, "detect_model_file")
    def test_image_classification_imports(
        self, mock_detect_model, mock_get_inference, mock_get_metadata, mock_download
    ):
        """Test that image classification apps have required imports."""
        generator = AppGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"

            # Mock bundle download
            bundle_path = temp_path / "bundle"
            bundle_path.mkdir()
            mock_download.return_value = bundle_path

            # Mock metadata for classification
            mock_get_metadata.return_value = {
                "name": "Breast Density Classification",
                "version": "1.0",
                "task": "Mammographic Breast Density Classification (BI-RADS)",
                "modality": "MG",
                "data_type": "jpeg",
            }

            # Mock inference config
            mock_get_inference.return_value = {}

            # Mock model file (PyTorch)
            model_file = bundle_path / "models" / "model.pt"
            model_file.parent.mkdir(parents=True)
            model_file.touch()
            mock_detect_model.return_value = model_file

            # Generate app with detected image/json format
            generator.generate_app("MONAI/breast_density_classification", output_dir)

            # Read generated app.py
            app_file = output_dir / "app.py"
            assert app_file.exists()
            app_content = app_file.read_text()

            # Check critical imports
            assert "from monai.deploy.core.domain import Image" in app_content, "Image import missing"
            assert "from monai.deploy.core.io_type import IOType" in app_content, "IOType import missing"

            # Check operator imports
            assert "from generic_directory_scanner_operator import GenericDirectoryScanner" in app_content
            assert "from image_file_loader_operator import ImageFileLoader" in app_content
            assert "from json_results_writer_operator import JSONResultsWriter" in app_content
            assert "from monai_classification_operator import MonaiClassificationOperator" in app_content

            # Check that the required operator files are physically copied (Phase 7 verification)
            assert (
                output_dir / "generic_directory_scanner_operator.py"
            ).exists(), "GenericDirectoryScanner operator file not copied"
            assert (output_dir / "image_file_loader_operator.py").exists(), "ImageFileLoader operator file not copied"
            assert (
                output_dir / "json_results_writer_operator.py"
            ).exists(), "JSONResultsWriter operator file not copied"
            assert (
                output_dir / "monai_classification_operator.py"
            ).exists(), "MonaiClassificationOperator operator file not copied"

    @patch.object(BundleDownloader, "download_bundle")
    @patch.object(BundleDownloader, "get_bundle_metadata")
    @patch.object(BundleDownloader, "get_inference_config")
    @patch.object(BundleDownloader, "detect_model_file")
    def test_vlm_model_imports_and_operators(
        self, mock_detect_model, mock_get_inference, mock_get_metadata, mock_download
    ):
        """Test that VLM apps have required imports and operators copied (Phase 7 verification)."""
        generator = AppGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"

            # Mock bundle download
            bundle_path = temp_path / "bundle"
            bundle_path.mkdir()
            mock_download.return_value = bundle_path

            # Mock metadata for VLM model
            mock_get_metadata.return_value = {
                "name": "Llama3-VILA-M3-3B",
                "version": "1.0",
                "task": "vlm",
                "modality": "multimodal",
            }

            # Mock inference config (VLM doesn't have traditional inference config)
            mock_get_inference.return_value = {}

            # Mock: No traditional model file for VLM
            mock_detect_model.return_value = None

            # Generate app
            generator.generate_app("MONAI/Llama3-VILA-M3-3B", output_dir)

            # Read generated app.py
            app_file = output_dir / "app.py"
            assert app_file.exists()
            app_content = app_file.read_text()

            # Check VLM-specific imports
            assert "from prompts_loader_operator import PromptsLoaderOperator" in app_content
            assert "from llama3_vila_inference_operator import Llama3VILAInferenceOperator" in app_content
            assert "from vlm_results_writer_operator import VLMResultsWriterOperator" in app_content

            # Check that the VLM operator files are physically copied (Phase 7 verification)
            assert (
                output_dir / "prompts_loader_operator.py"
            ).exists(), "PromptsLoaderOperator operator file not copied"
            assert (
                output_dir / "llama3_vila_inference_operator.py"
            ).exists(), "Llama3VILAInferenceOperator operator file not copied"
            assert (
                output_dir / "vlm_results_writer_operator.py"
            ).exists(), "VLMResultsWriterOperator operator file not copied"

            # Verify that non-VLM operators are NOT copied for VLM models
            assert not (
                output_dir / "nifti_writer_operator.py"
            ).exists(), "NiftiWriter should not be copied for VLM models"
            assert not (
                output_dir / "monai_classification_operator.py"
            ).exists(), "MonaiClassificationOperator should not be copied for VLM models"

    @patch.object(BundleDownloader, "download_bundle")
    @patch.object(BundleDownloader, "get_bundle_metadata")
    @patch.object(BundleDownloader, "get_inference_config")
    @patch.object(BundleDownloader, "detect_model_file")
    def test_dicom_segmentation_imports(self, mock_detect_model, mock_get_inference, mock_get_metadata, mock_download):
        """Test that DICOM segmentation apps have required imports."""
        generator = AppGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"

            # Mock bundle download
            bundle_path = temp_path / "bundle"
            bundle_path.mkdir()
            mock_download.return_value = bundle_path

            # Mock metadata for DICOM segmentation
            mock_get_metadata.return_value = {
                "name": "Spleen CT Segmentation",
                "version": "1.0",
                "task": "Automated Spleen Segmentation in CT Images",
                "modality": "CT",
            }

            # Mock inference config
            mock_get_inference.return_value = {}

            # Mock model file
            model_file = bundle_path / "models" / "model.ts"
            model_file.parent.mkdir(parents=True)
            model_file.touch()
            mock_detect_model.return_value = model_file

            # Generate app with DICOM format
            generator.generate_app("MONAI/spleen_ct_segmentation", output_dir, data_format="dicom")

            # Read generated app.py
            app_file = output_dir / "app.py"
            assert app_file.exists()
            app_content = app_file.read_text()

            # Check critical imports
            assert (
                "from monai.deploy.core.domain import Image" in app_content
            ), "Image import missing - required for MonaiBundleInferenceOperator"
            assert (
                "from monai.deploy.core.io_type import IOType" in app_content
            ), "IOType import missing - required for MonaiBundleInferenceOperator"

            # Check DICOM-specific imports
            assert "from pydicom.sr.codedict import codes" in app_content
            assert "from monai.deploy.conditions import CountCondition" in app_content
            assert (
                "from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator" in app_content
            )
            assert (
                "from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator"
                in app_content
            )
            assert "from monai.deploy.operators.stl_conversion_operator import STLConversionOperator" in app_content

    def test_imports_syntax_validation(self):
        """Test that generated apps have valid Python syntax."""
        generator = AppGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"

            # Create a minimal test by mocking all dependencies
            with (
                patch.object(BundleDownloader, "download_bundle") as mock_download,
                patch.object(BundleDownloader, "get_bundle_metadata") as mock_metadata,
                patch.object(BundleDownloader, "get_inference_config") as mock_config,
                patch.object(BundleDownloader, "detect_model_file") as mock_detect,
            ):
                bundle_path = temp_path / "bundle"
                bundle_path.mkdir()
                mock_download.return_value = bundle_path
                mock_metadata.return_value = {"name": "Test", "task": "segmentation"}
                mock_config.return_value = {}
                model_file = bundle_path / "models" / "model.ts"
                model_file.parent.mkdir(parents=True)
                model_file.touch()
                mock_detect.return_value = model_file

                generator.generate_app("MONAI/test", output_dir)

                # Try to compile the generated Python file
                app_file = output_dir / "app.py"
                app_content = app_file.read_text()

                try:
                    compile(app_content, str(app_file), "exec")
                except SyntaxError as e:
                    pytest.fail(f"Generated app.py has syntax error: {e}")

    def test_monai_bundle_inference_operator_requirements(self):
        """Test that apps using MonaiBundleInferenceOperator have all required imports."""
        generator = AppGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"

            # Test different scenarios that use MonaiBundleInferenceOperator
            test_cases = [
                # NIfTI segmentation (original failing case)
                {
                    "metadata": {
                        "name": "Test Segmentation",
                        "task": "segmentation",
                        "modality": "CT",
                    },
                    "model_file": "model.ts",
                    "format": "auto",
                },
                # NIfTI with different task description
                {
                    "metadata": {
                        "name": "Organ Detection",
                        "task": "detection",
                        "modality": "MR",
                    },
                    "model_file": "model.ts",
                    "format": "nifti",
                },
            ]

            for test_case in test_cases:
                with (
                    patch.object(BundleDownloader, "download_bundle") as mock_download,
                    patch.object(BundleDownloader, "get_bundle_metadata") as mock_metadata,
                    patch.object(BundleDownloader, "get_inference_config") as mock_config,
                    patch.object(BundleDownloader, "detect_model_file") as mock_detect,
                ):
                    bundle_path = temp_path / f"bundle_{test_case['format']}"
                    bundle_path.mkdir()
                    mock_download.return_value = bundle_path
                    mock_metadata.return_value = test_case["metadata"]
                    mock_config.return_value = {}

                    model_file = bundle_path / "models" / test_case["model_file"]
                    model_file.parent.mkdir(parents=True)
                    model_file.touch()
                    mock_detect.return_value = model_file

                    output_subdir = output_dir / f"test_{test_case['format']}"
                    generator.generate_app("MONAI/test", output_subdir, data_format=test_case["format"])

                    # Read and check generated app
                    app_file = output_subdir / "app.py"
                    app_content = app_file.read_text()

                    # If MonaiBundleInferenceOperator is used, these imports must be present
                    if "MonaiBundleInferenceOperator" in app_content:
                        assert (
                            "from monai.deploy.core.domain import Image" in app_content
                        ), f"Image import missing for {test_case['format']} format"
                        assert (
                            "from monai.deploy.core.io_type import IOType" in app_content
                        ), f"IOType import missing for {test_case['format']} format"
                        assert (
                            "IOMapping" in app_content
                        ), "IOMapping must be imported when using MonaiBundleInferenceOperator"
