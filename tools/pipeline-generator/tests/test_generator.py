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

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from pipeline_generator.generator import AppGenerator, BundleDownloader


class TestBundleDownloader:
    """Test BundleDownloader class."""
    
    def test_init(self):
        """Test BundleDownloader initialization."""
        downloader = BundleDownloader()
        assert downloader.api is not None
    
    @patch('pipeline_generator.generator.bundle_downloader.snapshot_download')
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
            "modality": "CT"
        }
        
        context = generator._prepare_context(
            model_id="MONAI/test_model",
            metadata=metadata,
            inference_config={},
            model_file=Path("models/model.ts"),
            app_name=None
        )
        
        assert context["model_id"] == "MONAI/test_model"
        assert context["app_name"] == "TestModelApp"
        assert context["task"] == "segmentation"
        assert context["modality"] == "CT"
        assert context["use_dicom"] is True
        assert context["model_file"] == "models/model.ts"
    
    @patch.object(BundleDownloader, 'download_bundle')
    @patch.object(BundleDownloader, 'get_bundle_metadata')
    @patch.object(BundleDownloader, 'get_inference_config')
    @patch.object(BundleDownloader, 'detect_model_file')
    def test_generate_app(self, mock_detect_model, mock_get_inference, 
                         mock_get_metadata, mock_download):
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
                "modality": "CT"
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
            assert (output_dir / "README.md").exists() 