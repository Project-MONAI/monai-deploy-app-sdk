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

"""Tests for ModelInfo data model."""

from datetime import datetime

from pipeline_generator.core.models import ModelInfo


class TestModelInfo:
    """Test ModelInfo data model."""

    def test_basic_model_creation(self):
        """Test creating a basic ModelInfo object."""
        model = ModelInfo(model_id="MONAI/spleen_ct_segmentation", name="Spleen CT Segmentation")

        assert model.model_id == "MONAI/spleen_ct_segmentation"
        assert model.name == "Spleen CT Segmentation"
        assert model.is_monai_bundle is False
        assert model.tags == []

    def test_display_name_with_name(self):
        """Test display_name property when name is provided."""
        model = ModelInfo(model_id="MONAI/test_model", name="Test Model")

        assert model.display_name == "Test Model"

    def test_display_name_without_name(self):
        """Test display_name property when name is not provided."""
        model = ModelInfo(model_id="MONAI/spleen_ct_segmentation", name="")

        assert model.display_name == "Spleen Ct Segmentation"

    def test_short_id(self):
        """Test short_id property."""
        model = ModelInfo(model_id="MONAI/spleen_ct_segmentation", name="Test")

        assert model.short_id == "spleen_ct_segmentation"

    def test_full_model_creation(self):
        """Test creating a ModelInfo with all fields."""
        now = datetime.now()
        model = ModelInfo(
            model_id="MONAI/test_model",
            name="Test Model",
            author="MONAI",
            description="A test model",
            downloads=100,
            likes=10,
            created_at=now,
            updated_at=now,
            tags=["medical", "segmentation"],
            is_monai_bundle=True,
            bundle_metadata={"version": "1.0"},
        )

        assert model.author == "MONAI"
        assert model.description == "A test model"
        assert model.downloads == 100
        assert model.likes == 10
        assert model.created_at == now
        assert model.updated_at == now
        assert model.tags == ["medical", "segmentation"]
        assert model.is_monai_bundle is True
        assert model.bundle_metadata == {"version": "1.0"}
