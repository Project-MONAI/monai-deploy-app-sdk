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

"""Tests for VLM model generation in pipeline generator."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestVLMGeneration:
    """Test VLM model generation functionality."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_vlm_config_identification(self):
        """Test that custom input/output types are correctly identified."""
        from pipeline_generator.config.settings import load_config

        settings = load_config()

        # Find VLM model in config
        vlm_models = []
        for endpoint in settings.endpoints:
            for model in endpoint.models:
                if model.input_type == "custom" and model.output_type == "custom":
                    vlm_models.append(model)

        # Should have at least the Llama3-VILA-M3-3B model
        assert len(vlm_models) > 0
        assert any(m.model_id == "MONAI/Llama3-VILA-M3-3B" for m in vlm_models)

    def test_vlm_template_rendering(self, temp_output_dir):
        """Test that VLM models use correct operators in template."""
        from jinja2 import Environment, FileSystemLoader

        # Set up template environment
        template_dir = Path(__file__).parent.parent / "pipeline_generator" / "templates"
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=False,
        )

        # Render template with VLM config
        template = env.get_template("app.py.j2")

        # Test data for VLM model
        context = {
            "model_id": "MONAI/Llama3-VILA-M3-3B",
            "app_name": "TestVLMApp",
            "input_type": "custom",
            "output_type": "custom",
            "use_dicom": False,
            "task": "Vision-Language Understanding",
            "description": "Test VLM model",
            "model_file": "model.safetensors",
            "bundles": [],
            "configs": [],
            "preprocessing": {},
            "postprocessing": {},
            "output_postfix": "_pred",
            "modality": "MR",
        }

        rendered = template.render(**context)

        # Verify VLM operators are used
        assert "PromptsLoaderOperator" in rendered
        assert "Llama3VILAInferenceOperator" in rendered
        assert "VLMResultsWriterOperator" in rendered

        # Verify standard operators are NOT used
        assert "GenericDirectoryScanner" not in rendered
        assert "NiftiDataLoader" not in rendered
        assert "ImageFileLoader" not in rendered
        assert "MonaiBundleInferenceOperator" not in rendered

        # Verify operator connections
        assert "prompts_loader" in rendered
        assert "vlm_inference" in rendered
        assert "vlm_writer" in rendered

        # Verify port connections
        assert '("prompt", "prompt")' in rendered
        assert '("output_type", "output_type")' in rendered
        assert '("request_id", "request_id")' in rendered

    def test_vlm_requirements_template(self):
        """Test requirements.txt generation for VLM models."""
        from jinja2 import Environment, FileSystemLoader

        template_dir = Path(__file__).parent.parent / "pipeline_generator" / "templates"
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=False,
        )

        template = env.get_template("requirements.txt.j2")

        context = {
            "bundles": [],
            "input_type": "custom",
            "output_type": "custom",
            "metadata": {},
        }

        rendered = template.render(**context)

        # Should include basic dependencies
        assert "monai-deploy-app-sdk" in rendered.lower()
        # VLM-specific deps are handled by operator optional imports

    def test_vlm_readme_template(self):
        """Test README generation for VLM models."""
        from jinja2 import Environment, FileSystemLoader

        template_dir = Path(__file__).parent.parent / "pipeline_generator" / "templates"
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=False,
        )

        template = env.get_template("README.md.j2")

        context = {
            "model_id": "MONAI/Llama3-VILA-M3-3B",
            "app_name": "Llama3VilaM33BApp",
            "task": "Vision-Language Understanding",
            "description": "VLM for medical image analysis",
            "input_type": "custom",
            "output_type": "custom",
            "use_dicom": False,
            "metadata": {"network_data_format": {"network": "Llama3-VILA-M3-3B"}},
        }

        rendered = template.render(**context)

        # Should mention VLM-specific usage
        assert "MONAI/Llama3-VILA-M3-3B" in rendered
        assert context["task"] in rendered

    @patch("pipeline_generator.core.hub_client.list_models")
    def test_vlm_model_listing(self, mock_list_models):
        """Test that VLM models appear correctly in listings."""
        from types import SimpleNamespace

        from pipeline_generator.core.hub_client import HuggingFaceClient

        # Mock the list_models response
        mock_model = SimpleNamespace(
            modelId="MONAI/Llama3-VILA-M3-3B",
            tags=["medical", "vision-language"],
            downloads=100,
            likes=10,
            name="Llama3-VILA-M3-3B",
            author="MONAI",
            description="VLM for medical imaging",
            created_at=None,
            lastModified=None,
            siblings=[],
        )

        mock_list_models.return_value = [mock_model]

        client = HuggingFaceClient()
        models = client.list_models_from_organization("MONAI")

        assert len(models) == 1
        assert models[0].model_id == "MONAI/Llama3-VILA-M3-3B"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
