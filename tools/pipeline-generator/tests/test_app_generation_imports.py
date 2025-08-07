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

"""Tests for validating imports in generated applications."""

import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

import pytest

from pipeline_generator.generator.app_generator import AppGenerator
from pipeline_generator.generator.bundle_downloader import BundleDownloader


class TestAppGenerationImports:
    """Test that generated apps have correct imports."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = AppGenerator()
    
    @patch.object(BundleDownloader, 'download_bundle')
    @patch.object(BundleDownloader, 'get_bundle_metadata')
    @patch.object(BundleDownloader, 'get_inference_config')
    @patch.object(BundleDownloader, 'detect_model_file')
    def test_nifti_segmentation_imports(self, mock_detect_model, mock_get_inference, 
                                       mock_get_metadata, mock_download):
        """Test that NIfTI segmentation apps have required imports."""
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
                "modality": "CT"
            }
            
            # Mock inference config (minimal)
            mock_get_inference.return_value = {}
            
            # Mock model file (TorchScript)
            model_file = bundle_path / "models" / "model.ts"
            model_file.parent.mkdir(parents=True)
            model_file.touch()
            mock_detect_model.return_value = model_file
            
            # Generate app
            self.generator.generate_app("MONAI/spleen_ct_segmentation", output_dir)
            
            # Read generated app.py
            app_file = output_dir / "app.py"
            assert app_file.exists()
            app_content = app_file.read_text()
            
            # Check critical imports for MonaiBundleInferenceOperator
            assert "from monai.deploy.core.domain import Image" in app_content, \
                "Image import missing - required for MonaiBundleInferenceOperator"
            assert "from monai.deploy.core.io_type import IOType" in app_content, \
                "IOType import missing - required for MonaiBundleInferenceOperator"
            assert "IOMapping" in app_content, \
                "IOMapping import missing - required for MonaiBundleInferenceOperator"
            
            # Check operator imports
            assert "from monai.deploy.operators.nifti_directory_loader_operator import NiftiDirectoryLoader" in app_content
            assert "from monai.deploy.operators.nifti_writer_operator import NiftiWriter" in app_content
            assert "from monai.deploy.operators.monai_bundle_inference_operator import" in app_content
    
    @patch.object(BundleDownloader, 'download_bundle')
    @patch.object(BundleDownloader, 'get_bundle_metadata')
    @patch.object(BundleDownloader, 'get_inference_config')
    @patch.object(BundleDownloader, 'detect_model_file')
    def test_image_classification_imports(self, mock_detect_model, mock_get_inference, 
                                         mock_get_metadata, mock_download):
        """Test that image classification apps have required imports."""
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
                "data_type": "jpeg"
            }
            
            # Mock inference config
            mock_get_inference.return_value = {}
            
            # Mock model file (PyTorch)
            model_file = bundle_path / "models" / "model.pt"
            model_file.parent.mkdir(parents=True)
            model_file.touch()
            mock_detect_model.return_value = model_file
            
            # Generate app with detected image/json format
            self.generator.generate_app("MONAI/breast_density_classification", output_dir)
            
            # Read generated app.py
            app_file = output_dir / "app.py"
            assert app_file.exists()
            app_content = app_file.read_text()
            
            # Check critical imports
            assert "from monai.deploy.core.domain import Image" in app_content, \
                "Image import missing"
            assert "from monai.deploy.core.io_type import IOType" in app_content, \
                "IOType import missing"
            
            # Check operator imports
            assert "from monai.deploy.operators.image_directory_loader_operator import ImageDirectoryLoader" in app_content
            assert "from monai.deploy.operators.json_results_writer_operator import JSONResultsWriter" in app_content
            assert "from monai.deploy.operators.monai_classification_operator import MonaiClassificationOperator" in app_content
    
    @patch.object(BundleDownloader, 'download_bundle')
    @patch.object(BundleDownloader, 'get_bundle_metadata')
    @patch.object(BundleDownloader, 'get_inference_config')
    @patch.object(BundleDownloader, 'detect_model_file')
    def test_dicom_segmentation_imports(self, mock_detect_model, mock_get_inference, 
                                       mock_get_metadata, mock_download):
        """Test that DICOM segmentation apps have required imports."""
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
                "modality": "CT"
            }
            
            # Mock inference config
            mock_get_inference.return_value = {}
            
            # Mock model file
            model_file = bundle_path / "models" / "model.ts"
            model_file.parent.mkdir(parents=True)
            model_file.touch()
            mock_detect_model.return_value = model_file
            
            # Generate app with DICOM format
            self.generator.generate_app("MONAI/spleen_ct_segmentation", output_dir, data_format="dicom")
            
            # Read generated app.py
            app_file = output_dir / "app.py"
            assert app_file.exists()
            app_content = app_file.read_text()
            
            # Check critical imports
            assert "from monai.deploy.core.domain import Image" in app_content, \
                "Image import missing - required for MonaiBundleInferenceOperator"
            assert "from monai.deploy.core.io_type import IOType" in app_content, \
                "IOType import missing - required for MonaiBundleInferenceOperator"
            
            # Check DICOM-specific imports
            assert "from pydicom.sr.codedict import codes" in app_content
            assert "from monai.deploy.conditions import CountCondition" in app_content
            assert "from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator" in app_content
            assert "from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator" in app_content
            assert "from monai.deploy.operators.stl_conversion_operator import STLConversionOperator" in app_content
    
    def test_imports_syntax_validation(self):
        """Test that generated apps have valid Python syntax."""
        # This is implicitly tested by the other tests since reading/parsing 
        # the file would fail if syntax is invalid, but we can make it explicit
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"
            
            # Create a minimal test by mocking all dependencies
            with patch.object(BundleDownloader, 'download_bundle') as mock_download, \
                 patch.object(BundleDownloader, 'get_bundle_metadata') as mock_metadata, \
                 patch.object(BundleDownloader, 'get_inference_config') as mock_config, \
                 patch.object(BundleDownloader, 'detect_model_file') as mock_detect:
                
                bundle_path = temp_path / "bundle"
                bundle_path.mkdir()
                mock_download.return_value = bundle_path
                mock_metadata.return_value = {"name": "Test", "task": "segmentation"}
                mock_config.return_value = {}
                model_file = bundle_path / "models" / "model.ts"
                model_file.parent.mkdir(parents=True)
                model_file.touch()
                mock_detect.return_value = model_file
                
                self.generator.generate_app("MONAI/test", output_dir)
                
                # Try to compile the generated Python file
                app_file = output_dir / "app.py"
                app_content = app_file.read_text()
                
                try:
                    compile(app_content, str(app_file), 'exec')
                except SyntaxError as e:
                    pytest.fail(f"Generated app.py has syntax error: {e}")
    
    def test_monai_bundle_inference_operator_requirements(self):
        """Test that apps using MonaiBundleInferenceOperator have all required imports."""
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
                        "modality": "CT"
                    },
                    "model_file": "model.ts",
                    "format": "auto"
                },
                # NIfTI with different task description
                {
                    "metadata": {
                        "name": "Organ Detection",
                        "task": "detection",
                        "modality": "MR"  
                    },
                    "model_file": "model.ts",
                    "format": "nifti"
                }
            ]
            
            for test_case in test_cases:
                with patch.object(BundleDownloader, 'download_bundle') as mock_download, \
                     patch.object(BundleDownloader, 'get_bundle_metadata') as mock_metadata, \
                     patch.object(BundleDownloader, 'get_inference_config') as mock_config, \
                     patch.object(BundleDownloader, 'detect_model_file') as mock_detect:
                    
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
                    self.generator.generate_app("MONAI/test", output_subdir, data_format=test_case["format"])
                    
                    # Read and check generated app
                    app_file = output_subdir / "app.py"
                    app_content = app_file.read_text()
                    
                    # If MonaiBundleInferenceOperator is used, these imports must be present
                    if "MonaiBundleInferenceOperator" in app_content:
                        assert "from monai.deploy.core.domain import Image" in app_content, \
                            f"Image import missing for {test_case['format']} format"
                        assert "from monai.deploy.core.io_type import IOType" in app_content, \
                            f"IOType import missing for {test_case['format']} format"
                        assert "IOMapping" in app_content, \
                            f"IOMapping must be imported when using MonaiBundleInferenceOperator"