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

"""Tests for bundle downloader."""

import json
from unittest.mock import patch

import pytest
from pipeline_generator.generator.bundle_downloader import BundleDownloader


class TestBundleDownloader:
    """Test bundle downloader functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.downloader = BundleDownloader()

    @patch("pipeline_generator.generator.bundle_downloader.snapshot_download")
    def test_download_bundle_success(self, mock_snapshot_download, tmp_path):
        """Test successful bundle download."""
        output_dir = tmp_path / "output"
        cache_dir = tmp_path / "cache"

        # Mock successful download
        mock_snapshot_download.return_value = str(output_dir / "model")

        result = self.downloader.download_bundle("MONAI/spleen_ct_segmentation", output_dir, cache_dir)

        assert result == output_dir / "model"
        mock_snapshot_download.assert_called_once_with(
            repo_id="MONAI/spleen_ct_segmentation",
            local_dir=output_dir / "model",
            cache_dir=cache_dir,
        )

    @patch("pipeline_generator.generator.bundle_downloader.snapshot_download")
    def test_download_bundle_failure(self, mock_snapshot_download, tmp_path):
        """Test bundle download failure."""
        output_dir = tmp_path / "output"

        # Mock download failure
        mock_snapshot_download.side_effect = Exception("Download failed")

        with pytest.raises(Exception, match="Download failed"):
            self.downloader.download_bundle("MONAI/nonexistent", output_dir)

    def test_get_bundle_metadata_from_configs(self, tmp_path):
        """Test getting bundle metadata from configs directory."""
        bundle_path = tmp_path / "bundle"
        configs_dir = bundle_path / "configs"
        configs_dir.mkdir(parents=True)

        # Create metadata.json
        metadata = {
            "name": "Test Model",
            "version": "1.0.0",
            "description": "Test description",
        }
        metadata_file = configs_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        result = self.downloader.get_bundle_metadata(bundle_path)

        assert result is not None
        assert result["name"] == "Test Model"
        assert result["version"] == "1.0.0"

    def test_get_bundle_metadata_from_root(self, tmp_path):
        """Test getting bundle metadata from root directory."""
        bundle_path = tmp_path / "bundle"
        bundle_path.mkdir()

        # Create metadata.json in root
        metadata = {"name": "Test Model", "version": "1.0.0"}
        metadata_file = bundle_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        result = self.downloader.get_bundle_metadata(bundle_path)

        assert result is not None
        assert result["name"] == "Test Model"

    def test_get_bundle_metadata_not_found(self, tmp_path):
        """Test getting bundle metadata when file doesn't exist."""
        bundle_path = tmp_path / "bundle"
        bundle_path.mkdir()

        result = self.downloader.get_bundle_metadata(bundle_path)

        assert result is None

    def test_get_bundle_metadata_invalid_json(self, tmp_path):
        """Test getting bundle metadata with invalid JSON."""
        bundle_path = tmp_path / "bundle"
        configs_dir = bundle_path / "configs"
        configs_dir.mkdir(parents=True)

        # Create invalid metadata.json
        metadata_file = configs_dir / "metadata.json"
        metadata_file.write_text("invalid json")

        result = self.downloader.get_bundle_metadata(bundle_path)

        assert result is None

    def test_get_inference_config_success(self, tmp_path):
        """Test getting inference configuration."""
        bundle_path = tmp_path / "bundle"
        configs_dir = bundle_path / "configs"
        configs_dir.mkdir(parents=True)

        # Create inference.json
        inference_config = {
            "preprocessing": {"transforms": [{"name": "LoadImaged"}, {"name": "EnsureChannelFirstd"}]},
            "postprocessing": {"transforms": [{"name": "Activationsd", "sigmoid": True}]},
        }
        inference_file = configs_dir / "inference.json"
        inference_file.write_text(json.dumps(inference_config))

        result = self.downloader.get_inference_config(bundle_path)

        assert result is not None
        assert "preprocessing" in result
        assert len(result["preprocessing"]["transforms"]) == 2

    def test_get_inference_config_not_found(self, tmp_path):
        """Test getting inference config when file doesn't exist."""
        bundle_path = tmp_path / "bundle"
        bundle_path.mkdir()

        result = self.downloader.get_inference_config(bundle_path)

        assert result is None

    def test_detect_model_file_torchscript(self, tmp_path):
        """Test detecting TorchScript model file."""
        bundle_path = tmp_path / "bundle"
        models_dir = bundle_path / "models"
        models_dir.mkdir(parents=True)

        # Create model.ts file
        model_file = models_dir / "model.ts"
        model_file.write_text("torchscript model")

        result = self.downloader.detect_model_file(bundle_path)

        assert result == models_dir / "model.ts"

    def test_detect_model_file_pytorch(self, tmp_path):
        """Test detecting PyTorch model file."""
        bundle_path = tmp_path / "bundle"
        models_dir = bundle_path / "models"
        models_dir.mkdir(parents=True)

        # Create model.pt file
        model_file = models_dir / "model.pt"
        model_file.write_bytes(b"pytorch model")

        result = self.downloader.detect_model_file(bundle_path)

        assert result == models_dir / "model.pt"

    def test_detect_model_file_onnx(self, tmp_path):
        """Test detecting ONNX model file."""
        bundle_path = tmp_path / "bundle"
        models_dir = bundle_path / "models"
        models_dir.mkdir(parents=True)

        # Create model.onnx file
        model_file = models_dir / "model.onnx"
        model_file.write_bytes(b"onnx model")

        result = self.downloader.detect_model_file(bundle_path)

        assert result == models_dir / "model.onnx"

    def test_detect_model_file_non_standard_location(self, tmp_path):
        """Test detecting model file in non-standard location."""
        bundle_path = tmp_path / "bundle"
        custom_dir = bundle_path / "custom" / "location"
        custom_dir.mkdir(parents=True)

        # Create model.pt file in custom location
        model_file = custom_dir / "model.pt"
        model_file.write_bytes(b"pytorch model")

        result = self.downloader.detect_model_file(bundle_path)

        assert result == custom_dir / "model.pt"

    def test_detect_model_file_in_root(self, tmp_path):
        """Test detecting model file in root directory."""
        bundle_path = tmp_path / "bundle"
        bundle_path.mkdir()

        # Create model.pt in root
        model_file = bundle_path / "model.pt"
        model_file.write_bytes(b"pytorch model")

        result = self.downloader.detect_model_file(bundle_path)

        assert result == bundle_path / "model.pt"

    def test_detect_model_file_not_found(self, tmp_path):
        """Test detecting model file when none exists."""
        bundle_path = tmp_path / "bundle"
        bundle_path.mkdir()

        result = self.downloader.detect_model_file(bundle_path)

        assert result is None

    def test_organize_bundle_structure_flat_to_structured(self, tmp_path):
        """Test organizing flat bundle structure into standard format."""
        bundle_path = tmp_path / "bundle"
        bundle_path.mkdir()

        # Create files in flat structure
        metadata_file = bundle_path / "metadata.json"
        inference_file = bundle_path / "inference.json"
        model_pt_file = bundle_path / "model.pt"
        model_ts_file = bundle_path / "model.ts"

        metadata_file.write_text('{"name": "Test"}')
        inference_file.write_text('{"config": "test"}')
        model_pt_file.touch()
        model_ts_file.touch()

        # Organize structure
        self.downloader.organize_bundle_structure(bundle_path)

        # Check that files were moved to proper locations
        assert (bundle_path / "configs" / "metadata.json").exists()
        assert (bundle_path / "configs" / "inference.json").exists()
        assert (bundle_path / "models" / "model.pt").exists()
        assert (bundle_path / "models" / "model.ts").exists()

        # Check that original files were moved (not copied)
        assert not metadata_file.exists()
        assert not inference_file.exists()
        assert not model_pt_file.exists()
        assert not model_ts_file.exists()

    def test_organize_bundle_structure_already_structured(self, tmp_path):
        """Test organizing bundle that already has proper structure."""
        bundle_path = tmp_path / "bundle"
        configs_dir = bundle_path / "configs"
        models_dir = bundle_path / "models"
        configs_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)

        # Create files in proper structure
        metadata_file = configs_dir / "metadata.json"
        model_file = models_dir / "model.pt"
        metadata_file.write_text('{"name": "Test"}')
        model_file.touch()

        # Should not change anything
        self.downloader.organize_bundle_structure(bundle_path)

        # Files should remain in place
        assert metadata_file.exists()
        assert model_file.exists()

    def test_organize_bundle_structure_partial_structure(self, tmp_path):
        """Test organizing bundle with partial structure."""
        bundle_path = tmp_path / "bundle"
        configs_dir = bundle_path / "configs"
        configs_dir.mkdir(parents=True)

        # Create metadata in configs but model in root
        metadata_file = configs_dir / "metadata.json"
        model_file = bundle_path / "model.pt"
        metadata_file.write_text('{"name": "Test"}')
        model_file.touch()

        # Organize structure
        self.downloader.organize_bundle_structure(bundle_path)

        # Metadata should stay, model should move
        assert metadata_file.exists()
        assert (bundle_path / "models" / "model.pt").exists()
        assert not model_file.exists()

    def test_detect_model_file_multiple_models(self, tmp_path):
        """Test detecting model file with multiple model files (returns first found)."""
        bundle_path = tmp_path / "bundle"
        models_dir = bundle_path / "models"
        models_dir.mkdir(parents=True)

        # Create multiple model files
        (models_dir / "model.ts").write_text("torchscript")
        (models_dir / "model.pt").write_bytes(b"pytorch")
        (models_dir / "model.onnx").write_bytes(b"onnx")

        result = self.downloader.detect_model_file(bundle_path)

        # Should return the first one found (model.ts in this case)
        assert result == models_dir / "model.ts"

    @patch("pipeline_generator.generator.bundle_downloader.logger")
    def test_get_bundle_metadata_logs_error(self, mock_logger, tmp_path):
        """Test that metadata reading errors are logged."""
        bundle_path = tmp_path / "bundle"
        configs_dir = bundle_path / "configs"
        configs_dir.mkdir(parents=True)

        # Create a file that will cause a read error
        metadata_file = configs_dir / "metadata.json"
        metadata_file.write_text("invalid json")

        result = self.downloader.get_bundle_metadata(bundle_path)

        assert result is None
        mock_logger.error.assert_called()

    @patch("pipeline_generator.generator.bundle_downloader.logger")
    def test_get_inference_config_logs_error(self, mock_logger, tmp_path):
        """Test that inference config reading errors are logged."""
        bundle_path = tmp_path / "bundle"
        configs_dir = bundle_path / "configs"
        configs_dir.mkdir(parents=True)

        # Create a file that will cause a read error
        inference_file = configs_dir / "inference.json"
        inference_file.write_text("invalid json")

        result = self.downloader.get_inference_config(bundle_path)

        assert result is None
        mock_logger.error.assert_called()

    def test_organize_bundle_structure_subdirectory_models(self, tmp_path):
        """Test organizing models from subdirectories to main models/ directory."""
        bundle_path = tmp_path / "bundle"
        models_dir = bundle_path / "models"
        subdir = models_dir / "A100"
        subdir.mkdir(parents=True)

        # Create model file in subdirectory
        subdir_model = subdir / "dynunet_FT_trt_16.ts"
        subdir_model.write_text("tensorrt model")

        # Organize structure
        self.downloader.organize_bundle_structure(bundle_path)

        # Model should be moved to main models/ directory with standard name
        assert (models_dir / "model.ts").exists()
        assert not subdir_model.exists()
        assert not subdir.exists()  # Empty subdirectory should be removed

    def test_organize_bundle_structure_prefers_pytorch_over_tensorrt(self, tmp_path):
        """Test that PyTorch models are preferred over TensorRT models."""
        bundle_path = tmp_path / "bundle"
        models_dir = bundle_path / "models"
        subdir = models_dir / "A100"
        subdir.mkdir(parents=True)

        # Create both PyTorch and TensorRT models in subdirectory
        pytorch_model = subdir / "dynunet_FT.pt"
        tensorrt_model = subdir / "dynunet_FT_trt_16.ts"
        pytorch_model.write_bytes(b"pytorch model")
        tensorrt_model.write_text("tensorrt model")

        # Organize structure
        self.downloader.organize_bundle_structure(bundle_path)

        # PyTorch model should be preferred and moved
        assert (models_dir / "model.pt").exists()
        assert not (models_dir / "model.ts").exists()
        assert not pytorch_model.exists()
        # TensorRT model should remain in subdirectory
        assert tensorrt_model.exists()

    def test_organize_bundle_structure_standard_naming_pytorch(self, tmp_path):
        """Test renaming PyTorch models to standard names."""
        bundle_path = tmp_path / "bundle"
        models_dir = bundle_path / "models"
        models_dir.mkdir(parents=True)

        # Create PyTorch model with custom name
        custom_model = models_dir / "dynunet_FT.pt"
        custom_model.write_bytes(b"pytorch model")

        # Organize structure
        self.downloader.organize_bundle_structure(bundle_path)

        # Model should be renamed to standard name
        assert (models_dir / "model.pt").exists()
        assert not custom_model.exists()

    def test_organize_bundle_structure_standard_naming_torchscript(self, tmp_path):
        """Test renaming TorchScript models to standard names when no PyTorch model exists."""
        bundle_path = tmp_path / "bundle"
        models_dir = bundle_path / "models"
        models_dir.mkdir(parents=True)

        # Create only TorchScript model with custom name
        custom_model = models_dir / "custom_model.ts"
        custom_model.write_text("torchscript model")

        # Organize structure
        self.downloader.organize_bundle_structure(bundle_path)

        # Model should be renamed to standard name
        assert (models_dir / "model.ts").exists()
        assert not custom_model.exists()

    def test_organize_bundle_structure_skips_when_suitable_model_exists(self, tmp_path):
        """Test that subdirectory organization is skipped when suitable model already exists."""
        bundle_path = tmp_path / "bundle"
        models_dir = bundle_path / "models"
        subdir = models_dir / "A100"
        subdir.mkdir(parents=True)

        # Create model in main directory
        main_model = models_dir / "existing_model.pt"
        main_model.write_bytes(b"existing pytorch model")

        # Create model in subdirectory
        subdir_model = subdir / "dynunet_FT_trt_16.ts"
        subdir_model.write_text("tensorrt model")

        # Organize structure
        self.downloader.organize_bundle_structure(bundle_path)

        # Main model should be renamed to standard name
        assert (models_dir / "model.pt").exists()
        assert not main_model.exists()

        # Subdirectory model should remain untouched
        assert subdir_model.exists()
        assert subdir.exists()

    def test_organize_bundle_structure_multiple_extensions_preference(self, tmp_path):
        """Test extension preference order: .pt > .onnx > .ts."""
        bundle_path = tmp_path / "bundle"
        models_dir = bundle_path / "models"
        subdir = models_dir / "A100"
        subdir.mkdir(parents=True)

        # Create models with different extensions in subdirectory
        pt_model = subdir / "model.pt"
        onnx_model = subdir / "model.onnx"
        ts_model = subdir / "model.ts"

        pt_model.write_bytes(b"pytorch model")
        onnx_model.write_bytes(b"onnx model")
        ts_model.write_text("torchscript model")

        # Organize structure
        self.downloader.organize_bundle_structure(bundle_path)

        # Should prefer .pt model
        assert (models_dir / "model.pt").exists()
        assert not (models_dir / "model.onnx").exists()
        assert not (models_dir / "model.ts").exists()
        assert not pt_model.exists()

        # Other models should remain in subdirectory
        assert onnx_model.exists()
        assert ts_model.exists()
