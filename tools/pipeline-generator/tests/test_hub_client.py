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

"""Tests for HuggingFace Hub client."""

from datetime import datetime
from unittest.mock import Mock, patch

from huggingface_hub.utils import HfHubHTTPError
from pipeline_generator.core.hub_client import HuggingFaceClient


class SimpleModelData:
    """Simple class to simulate HuggingFace model data."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestHuggingFaceClient:
    """Test HuggingFace client functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = HuggingFaceClient()

    @patch("pipeline_generator.core.hub_client.list_models")
    def test_list_models_from_organization_success(self, mock_list_models):
        """Test successfully listing models from organization."""
        # Mock model data
        mock_model1 = SimpleModelData(
            modelId="MONAI/spleen_ct_segmentation",
            author="MONAI",
            downloads=100,
            likes=10,
            created_at=datetime(2023, 1, 1),
            lastModified=datetime(2023, 12, 1),
            tags=["medical", "segmentation"],
            siblings=[Mock(rfilename="configs/metadata.json"), Mock(rfilename="models/model.ts")],
        )

        mock_model2 = SimpleModelData(
            modelId="MONAI/liver_segmentation",
            author="MONAI",
            downloads=50,
            likes=5,
            created_at=datetime(2023, 2, 1),
            lastModified=datetime(2023, 11, 1),
            tags=["medical"],
            siblings=[],
        )

        mock_list_models.return_value = [mock_model1, mock_model2]

        # Call the method
        models = self.client.list_models_from_organization("MONAI")

        # Verify results
        assert len(models) == 2
        assert models[0].model_id == "MONAI/spleen_ct_segmentation"
        assert models[0].is_monai_bundle is True  # Has .ts file
        assert models[1].model_id == "MONAI/liver_segmentation"
        assert models[1].is_monai_bundle is False  # No .ts file

    @patch("pipeline_generator.core.hub_client.list_models")
    def test_list_models_from_organization_empty(self, mock_list_models):
        """Test listing models from organization with no results."""
        mock_list_models.return_value = []

        models = self.client.list_models_from_organization("NonExistent")

        assert len(models) == 0

    @patch("pipeline_generator.core.hub_client.list_models")
    def test_list_models_from_organization_error(self, mock_list_models):
        """Test handling errors when listing models."""
        mock_list_models.side_effect = Exception("API Error")

        models = self.client.list_models_from_organization("MONAI")

        assert len(models) == 0  # Should return empty list on error

    @patch("pipeline_generator.core.hub_client.model_info")
    def test_get_model_info_success(self, mock_model_info):
        """Test successfully getting model info."""
        # Mock model data
        mock_model = SimpleModelData(
            modelId="MONAI/spleen_ct_segmentation",
            author="MONAI",
            downloads=100,
            likes=10,
            created_at=datetime(2023, 1, 1),
            lastModified=datetime(2023, 12, 1),
            tags=["medical", "segmentation"],
            siblings=[Mock(rfilename="configs/metadata.json"), Mock(rfilename="models/model.ts")],
            cardData={"description": "Spleen segmentation model"},
        )

        mock_model_info.return_value = mock_model

        # Call the method
        model = self.client.get_model_info("MONAI/spleen_ct_segmentation")

        # Verify results
        assert model is not None
        assert model.model_id == "MONAI/spleen_ct_segmentation"
        assert model.author == "MONAI"
        assert model.is_monai_bundle is True
        assert model.description == "Spleen segmentation model"

    @patch("pipeline_generator.core.hub_client.model_info")
    def test_get_model_info_not_found(self, mock_model_info):
        """Test getting model info for non-existent model."""
        mock_model_info.side_effect = HfHubHTTPError("Model not found", response=Mock(status_code=404))

        model = self.client.get_model_info("MONAI/nonexistent")

        assert model is None

    @patch("pipeline_generator.core.hub_client.model_info")
    def test_get_model_info_error(self, mock_model_info):
        """Test handling errors when getting model info."""
        mock_model_info.side_effect = Exception("API Error")

        model = self.client.get_model_info("MONAI/spleen_ct_segmentation")

        assert model is None

    def test_extract_model_info_with_name(self):
        """Test parsing model info with explicit name."""
        mock_model = SimpleModelData(
            modelId="MONAI/test_model",
            name="Test Model",
            author="MONAI",
            downloads=100,
            likes=10,
            created_at=datetime(2023, 1, 1),
            lastModified=datetime(2023, 12, 1),
            tags=["test"],
            siblings=[],
        )

        model = self.client._extract_model_info(mock_model)

        assert model.model_id == "MONAI/test_model"
        assert model.name == "Test Model"
        assert model.display_name == "Test Model"

    def test_extract_model_info_without_name(self):
        """Test parsing model info without explicit name."""
        mock_model = SimpleModelData(
            modelId="MONAI/test_model",
            author=None,
            downloads=None,
            likes=None,
            created_at=None,
            lastModified=None,
            tags=[],
            siblings=[],
        )

        model = self.client._extract_model_info(mock_model)

        assert model.model_id == "MONAI/test_model"
        assert model.name == "MONAI/test_model"  # Uses modelId as fallback
        assert model.author is None

    def test_extract_model_info_bundle_detection(self):
        """Test MONAI bundle detection during parsing."""
        # Test with TorchScript (.ts) file in siblings
        mock_model = SimpleModelData(
            modelId="MONAI/test_bundle",
            author="MONAI",
            downloads=100,
            likes=10,
            created_at=datetime(2023, 1, 1),
            lastModified=datetime(2023, 12, 1),
            tags=[],
            siblings=[
                Mock(rfilename="configs/metadata.json"),
                Mock(rfilename="models/model.pt"),
                Mock(rfilename="models/model.ts"),
            ],
        )
        model = self.client._extract_model_info(mock_model)
        assert model.is_monai_bundle is True

        # Test without TorchScript (.ts) file - only .pt file
        mock_model.siblings = [Mock(rfilename="configs/metadata.json"), Mock(rfilename="models/model.pt")]
        model = self.client._extract_model_info(mock_model)
        assert model.is_monai_bundle is False

    def test_extract_model_info_missing_siblings(self):
        """Test parsing model info when siblings attribute is missing."""
        mock_model = SimpleModelData(
            modelId="MONAI/test_model",
            author="MONAI",
            downloads=100,
            likes=10,
            created_at=datetime(2023, 1, 1),
            lastModified=datetime(2023, 12, 1),
            tags=[],
        )
        # Don't set siblings attribute

        model = self.client._extract_model_info(mock_model)

        assert model.is_monai_bundle is False  # Should default to False on error

    def test_extract_model_info_with_description(self):
        """Test parsing model info with description in cardData."""
        mock_model = SimpleModelData(
            modelId="MONAI/test_model",
            author="MONAI",
            downloads=100,
            likes=10,
            created_at=datetime(2023, 1, 1),
            lastModified=datetime(2023, 12, 1),
            tags=["medical"],
            siblings=[],
            cardData={"description": "This is a test model"},
        )

        model = self.client._extract_model_info(mock_model)

        assert model.description == "This is a test model"

    def test_extract_model_info_missing_optional_attributes(self):
        """Test parsing model info with missing optional attributes."""
        mock_model = SimpleModelData(modelId="MONAI/test_model", siblings=[])

        model = self.client._extract_model_info(mock_model)

        assert model.model_id == "MONAI/test_model"
        assert model.author is None
        assert model.downloads is None
        assert model.likes is None
        assert model.created_at is None
        assert model.updated_at is None
        assert model.tags == []

    def test_list_models_from_endpoints_with_organization(self):
        """Test listing models from endpoints with organization."""
        from pipeline_generator.config.settings import Endpoint

        # Create test endpoints
        endpoints = [
            Endpoint(
                organization="MONAI",
                base_url="https://huggingface.co",
                description="Test org",
                models=[],
            )
        ]

        # Mock the list_models_from_organization method
        with patch.object(self.client, "list_models_from_organization") as mock_list:
            mock_list.return_value = [Mock(model_id="MONAI/test_model")]

            result = self.client.list_models_from_endpoints(endpoints)

            assert len(result) == 1
            mock_list.assert_called_once_with("MONAI")

    def test_list_models_from_endpoints_with_model_id(self):
        """Test listing models from endpoints with specific model_id."""
        from pipeline_generator.config.settings import Endpoint

        # Create test endpoints with model_id
        endpoints = [
            Endpoint(
                model_id="MONAI/specific_model",
                base_url="https://huggingface.co",
                description="Test model",
                models=[],
            )
        ]

        # Mock the get_model_info method
        with patch.object(self.client, "get_model_info") as mock_get:
            mock_model = Mock(model_id="MONAI/specific_model")
            mock_get.return_value = mock_model

            result = self.client.list_models_from_endpoints(endpoints)

            assert len(result) == 1
            assert result[0] == mock_model
            mock_get.assert_called_once_with("MONAI/specific_model")

    def test_list_models_from_endpoints_model_not_found(self):
        """Test listing models when specific model is not found."""
        from pipeline_generator.config.settings import Endpoint

        endpoints = [
            Endpoint(
                model_id="MONAI/missing_model",
                base_url="https://huggingface.co",
                description="Missing model",
                models=[],
            )
        ]

        # Mock get_model_info to return None
        with patch.object(self.client, "get_model_info") as mock_get:
            mock_get.return_value = None

            result = self.client.list_models_from_endpoints(endpoints)

            assert len(result) == 0
            mock_get.assert_called_once_with("MONAI/missing_model")

    def test_extract_model_info_siblings_exception(self):
        """Test _extract_model_info handles exception in siblings check."""

        # Create a mock model that will raise exception when accessing siblings
        class MockModelWithException:
            def __init__(self):
                self.modelId = "test/model"
                self.tags = []
                self.downloads = 100
                self.likes = 10
                self.name = "Test Model"
                self.author = "test"
                self.description = None
                self.created_at = None
                self.lastModified = None

            @property
            def siblings(self):
                raise Exception("Test error")

        mock_model = MockModelWithException()

        # Should not raise, just catch and continue
        result = self.client._extract_model_info(mock_model)

        assert result.is_monai_bundle is False

    def test_extract_model_info_with_card_data_preference(self):
        """Test _extract_model_info prefers description from cardData."""
        mock_model = SimpleModelData(
            modelId="test/model",
            tags=[],
            downloads=100,
            likes=10,
            name="Test Model",
            author="test",
            description="Direct description",
            cardData={"description": "Card description"},
            created_at=None,
            lastModified=None,
            siblings=[],
        )

        result = self.client._extract_model_info(mock_model)

        # Should prefer cardData description
        assert result.description == "Card description"

    def test_detect_model_extensions_with_torchscript(self):
        """Test detecting TorchScript model extensions."""
        # Mock model with .ts file
        mock_model = SimpleModelData(
            modelId="MONAI/test_model",
            siblings=[
                Mock(rfilename="model.ts"),
                Mock(rfilename="config.yaml"),
                Mock(rfilename="README.md"),
            ],
        )

        extensions = self.client._detect_model_extensions(mock_model)

        assert ".ts" in extensions
        assert len(extensions) == 1

    def test_detect_model_extensions_multiple_formats(self):
        """Test detecting multiple model file extensions."""
        # Mock model with multiple extensions
        mock_model = SimpleModelData(
            modelId="MONAI/test_model",
            siblings=[
                Mock(rfilename="model.ts"),
                Mock(rfilename="model.pt"),
                Mock(rfilename="model.safetensors"),
                Mock(rfilename="config.yaml"),
            ],
        )

        extensions = self.client._detect_model_extensions(mock_model)

        assert ".ts" in extensions
        assert ".pt" in extensions
        assert ".safetensors" in extensions
        assert len(extensions) == 3

    def test_detect_model_extensions_no_model_files(self):
        """Test detecting extensions when no model files present."""
        # Mock model with no model files
        mock_model = SimpleModelData(
            modelId="MONAI/test_model",
            siblings=[
                Mock(rfilename="README.md"),
                Mock(rfilename="config.yaml"),
            ],
        )

        extensions = self.client._detect_model_extensions(mock_model)

        assert len(extensions) == 0

    def test_detect_model_extensions_no_siblings(self):
        """Test detecting extensions when model has no siblings attribute."""
        # Mock model without siblings
        mock_model = SimpleModelData(modelId="MONAI/test_model")

        extensions = self.client._detect_model_extensions(mock_model)

        assert len(extensions) == 0

    @patch("pipeline_generator.core.hub_client.list_models")
    def test_list_torchscript_models_filters_correctly(self, mock_list_models):
        """Test that list_torchscript_models only returns models with .ts files."""
        # Mock model data - one with .ts, one without
        mock_model_with_ts = SimpleModelData(
            modelId="MONAI/model_with_ts",
            author="MONAI",
            downloads=100,
            likes=10,
            created_at=datetime(2023, 1, 1),
            lastModified=datetime(2023, 12, 1),
            tags=["medical"],
            siblings=[Mock(rfilename="model.ts")],
        )

        mock_model_without_ts = SimpleModelData(
            modelId="MONAI/model_without_ts",
            author="MONAI",
            downloads=50,
            likes=5,
            created_at=datetime(2023, 2, 1),
            lastModified=datetime(2023, 11, 1),
            tags=["medical"],
            siblings=[Mock(rfilename="model.pt")],
        )

        mock_list_models.return_value = [mock_model_with_ts, mock_model_without_ts]

        # Mock get_model_info to return processed ModelInfo objects
        def mock_get_model_info(model_id):
            if model_id == "MONAI/model_with_ts":
                return self.client._extract_model_info(mock_model_with_ts)
            elif model_id == "MONAI/model_without_ts":
                return self.client._extract_model_info(mock_model_without_ts)
            return None

        with patch.object(self.client, "get_model_info", side_effect=mock_get_model_info):
            from pipeline_generator.config.settings import Endpoint

            endpoints = [Endpoint(organization="MONAI")]

            # Test the torchscript filtering
            torchscript_models = self.client.list_torchscript_models(endpoints)

            # Should only return the model with .ts file
            assert len(torchscript_models) == 1
            assert torchscript_models[0].model_id == "MONAI/model_with_ts"
            assert torchscript_models[0].has_torchscript is True

    def test_extract_model_info_includes_extensions(self):
        """Test that _extract_model_info includes model extensions."""
        mock_model = SimpleModelData(
            modelId="MONAI/test_model",
            author="MONAI",
            downloads=100,
            likes=10,
            created_at=datetime(2023, 1, 1),
            lastModified=datetime(2023, 12, 1),
            tags=["medical"],
            siblings=[
                Mock(rfilename="model.ts"),
                Mock(rfilename="model.pt"),
                Mock(rfilename="config.yaml"),
            ],
        )

        result = self.client._extract_model_info(mock_model)

        assert ".ts" in result.model_extensions
        assert ".pt" in result.model_extensions
        assert result.has_torchscript is True
        assert result.primary_extension == ".ts"  # Should prioritize .ts
