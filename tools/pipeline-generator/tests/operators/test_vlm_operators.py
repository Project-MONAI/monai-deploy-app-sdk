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

"""Unit tests for Vision-Language Model (VLM) operators."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import yaml

from monai.deploy.core import AppContext, Fragment, Image, OperatorSpec


class TestPromptsLoaderOperator(unittest.TestCase):
    """Test cases for PromptsLoaderOperator."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_prompts = {
            "defaults": {"max_new_tokens": 256, "temperature": 0.2, "top_p": 0.9},
            "prompts": [
                {"prompt": "Test prompt 1", "image": "test1.jpg", "output": "json"},
                {
                    "prompt": "Test prompt 2",
                    "image": "test2.jpg",
                    "output": "image_overlay",
                    "max_new_tokens": 128,
                },
            ],
        }

        # Create prompts.yaml
        self.prompts_file = Path(self.test_dir) / "prompts.yaml"
        with open(self.prompts_file, "w") as f:
            yaml.dump(self.test_prompts, f)

        # Create mock images
        for i in range(1, 3):
            img_path = Path(self.test_dir) / f"test{i}.jpg"
            # Create a simple RGB image
            img_array = np.ones((100, 100, 3), dtype=np.uint8) * (i * 50)
            # Mock PIL Image save
            img_path.touch()

    def tearDown(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("monai.deploy.operators.prompts_loader_operator.PILImage")
    def test_prompts_loading(self, mock_pil):
        """Test loading and parsing prompts.yaml."""
        from monai.deploy.operators.prompts_loader_operator import PromptsLoaderOperator

        # Mock PIL Image
        mock_image = Mock()
        mock_image.mode = "RGB"
        mock_array = np.ones((100, 100, 3), dtype=np.float32)
        mock_pil.open.return_value = mock_image
        mock_image.convert.return_value = mock_image

        # Use numpy's array function directly
        with patch("numpy.array", return_value=mock_array):
            # Create operator
            fragment = Mock(spec=Fragment)
            operator = PromptsLoaderOperator(fragment, input_folder=self.test_dir)

            # Setup
            spec = Mock(spec=OperatorSpec)
            operator.setup(spec)

            # Verify setup calls
            self.assertEqual(spec.output.call_count, 5)  # 5 output ports

            # Test compute
            mock_output = Mock()
            operator.compute(None, mock_output, None)

            # Verify first prompt emission
            self.assertEqual(mock_output.emit.call_count, 5)
            calls = mock_output.emit.call_args_list

            # Check emitted data
            self.assertEqual(calls[1][0][1], "prompt")  # Port name
            self.assertEqual(calls[1][0][0], "Test prompt 1")  # Prompt text

            self.assertEqual(calls[2][0][1], "output_type")
            self.assertEqual(calls[2][0][0], "json")

            # Check generation params include defaults
            gen_params = calls[4][0][0]  # generation_params
            self.assertEqual(gen_params["max_new_tokens"], 256)
            self.assertEqual(gen_params["temperature"], 0.2)

    def test_empty_prompts_file(self):
        """Test handling of empty prompts file."""
        from monai.deploy.operators.prompts_loader_operator import PromptsLoaderOperator

        # Create empty prompts file
        empty_file = Path(self.test_dir) / "empty_prompts.yaml"
        with open(empty_file, "w") as f:
            yaml.dump({"prompts": []}, f)

        fragment = Mock(spec=Fragment)
        operator = PromptsLoaderOperator(fragment, input_folder=empty_file.parent)

        # Rename file to prompts.yaml
        empty_file.rename(Path(self.test_dir) / "prompts.yaml")

        spec = Mock(spec=OperatorSpec)
        operator.setup(spec)

        # Should handle empty prompts gracefully
        self.assertEqual(len(operator._prompts_data), 0)

    def test_missing_prompts_file(self):
        """Test handling of missing prompts.yaml."""
        from monai.deploy.operators.prompts_loader_operator import PromptsLoaderOperator

        # Remove prompts file
        self.prompts_file.unlink()

        fragment = Mock(spec=Fragment)
        operator = PromptsLoaderOperator(fragment, input_folder=self.test_dir)

        spec = Mock(spec=OperatorSpec)
        operator.setup(spec)

        # Should handle missing file gracefully
        self.assertEqual(len(operator._prompts_data), 0)


class TestLlama3VILAInferenceOperator(unittest.TestCase):
    """Test cases for Llama3VILAInferenceOperator."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_path = tempfile.mkdtemp()
        Path(self.model_path).mkdir(exist_ok=True)

        # Create mock config file
        config = {"model_type": "llava_llama"}
        config_file = Path(self.model_path) / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

    def tearDown(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.model_path, ignore_errors=True)

    def test_inference_operator_init(self):
        """Test inference operator initialization."""
        from monai.deploy.operators.llama3_vila_inference_operator import (
            Llama3VILAInferenceOperator,
        )

        fragment = Mock(spec=Fragment)
        app_context = Mock(spec=AppContext)

        operator = Llama3VILAInferenceOperator(fragment, app_context=app_context, model_path=self.model_path)

        self.assertEqual(operator.model_path, Path(self.model_path))
        self.assertIsNotNone(operator.device)

    @patch("monai.deploy.operators.llama3_vila_inference_operator.AutoConfig")
    def test_mock_inference(self, mock_autoconfig):
        """Test mock inference mode."""
        from monai.deploy.operators.llama3_vila_inference_operator import (
            Llama3VILAInferenceOperator,
        )

        # Mock config loading failure to trigger mock mode
        mock_autoconfig.from_pretrained.side_effect = Exception("Test error")

        fragment = Mock(spec=Fragment)
        app_context = Mock(spec=AppContext)

        operator = Llama3VILAInferenceOperator(fragment, app_context=app_context, model_path=self.model_path)

        spec = Mock(spec=OperatorSpec)
        operator.setup(spec)

        # Verify mock mode is enabled
        self.assertTrue(operator._mock_mode)

        # Test inference
        mock_image = Mock(spec=Image)
        mock_image.asnumpy.return_value = np.ones((100, 100, 3), dtype=np.float32)
        mock_image.metadata.return_value = {"filename": "/test/image.jpg"}

        mock_input = Mock()
        mock_input.receive.side_effect = lambda x: {
            "image": mock_image,
            "prompt": "What is this image showing?",
            "output_type": "json",
            "request_id": "test-123",
            "generation_params": {"max_new_tokens": 256},
        }.get(x)

        mock_output = Mock()
        operator.compute(mock_input, mock_output, None)

        # Verify outputs
        self.assertEqual(mock_output.emit.call_count, 3)

        # Check JSON result
        result = mock_output.emit.call_args_list[0][0][0]
        self.assertIsInstance(result, dict)
        self.assertEqual(result["request_id"], "test-123")
        self.assertEqual(result["status"], "success")
        self.assertIn("prompt", result)
        self.assertEqual(result["prompt"], "What is this image showing?")
        self.assertIn("image", result)
        self.assertEqual(result["image"], "/test/image.jpg")
        self.assertIn("response", result)

    def test_json_result_creation(self):
        """Test JSON result creation with prompt and image metadata."""
        from monai.deploy.operators.llama3_vila_inference_operator import (
            Llama3VILAInferenceOperator,
        )

        fragment = Mock(spec=Fragment)
        app_context = Mock(spec=AppContext)

        operator = Llama3VILAInferenceOperator(fragment, app_context=app_context, model_path=self.model_path)

        # Test with all parameters
        result = operator._create_json_result(
            "Test response",
            "req-123",
            "Test prompt?",
            {"filename": "/path/to/image.jpg"},
        )

        self.assertEqual(result["request_id"], "req-123")
        self.assertEqual(result["response"], "Test response")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["prompt"], "Test prompt?")
        self.assertEqual(result["image"], "/path/to/image.jpg")

        # Test without optional parameters
        result2 = operator._create_json_result("Response only", "req-456")
        self.assertNotIn("prompt", result2)
        self.assertNotIn("image", result2)

    @patch("monai.deploy.operators.llama3_vila_inference_operator.PILImage")
    @patch("monai.deploy.operators.llama3_vila_inference_operator.ImageDraw")
    def test_image_overlay_creation(self, mock_draw, mock_pil):
        """Test image overlay creation."""
        from monai.deploy.operators.llama3_vila_inference_operator import (
            Llama3VILAInferenceOperator,
        )

        fragment = Mock(spec=Fragment)
        app_context = Mock(spec=AppContext)

        operator = Llama3VILAInferenceOperator(fragment, app_context=app_context, model_path=self.model_path)

        # Create mock image
        mock_image = Mock(spec=Image)
        image_array = np.ones((100, 100, 3), dtype=np.float32)
        mock_image.asnumpy.return_value = image_array
        mock_image.metadata.return_value = {"test": "metadata"}

        # Mock PIL
        mock_pil_image = Mock()
        mock_pil_image.width = 100
        mock_pil.fromarray.return_value = mock_pil_image

        mock_drawer = Mock()
        mock_draw.Draw.return_value = mock_drawer

        # Test overlay creation
        result = operator._create_image_overlay(mock_image, "Test overlay text")

        # Verify Image object returned
        self.assertIsInstance(result, Image)

        # Verify draw operations were called
        self.assertTrue(mock_drawer.rectangle.called)
        self.assertTrue(mock_drawer.text.called)


class TestVLMResultsWriterOperator(unittest.TestCase):
    """Test cases for VLMResultsWriterOperator."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_json_writing(self):
        """Test writing JSON results."""
        from monai.deploy.operators.vlm_results_writer_operator import (
            VLMResultsWriterOperator,
        )

        fragment = Mock(spec=Fragment)
        operator = VLMResultsWriterOperator(fragment, output_folder=self.output_dir)

        spec = Mock(spec=OperatorSpec)
        operator.setup(spec)

        # Test data
        result = {
            "request_id": "test-123",
            "prompt": "Test prompt",
            "response": "Test response",
            "status": "success",
        }

        mock_input = Mock()
        mock_input.receive.side_effect = lambda x: {
            "result": result,
            "output_type": "json",
            "request_id": "test-123",
        }.get(x)

        operator.compute(mock_input, None, None)

        # Verify file created
        output_file = Path(self.output_dir) / "test-123.json"
        self.assertTrue(output_file.exists())

        # Verify content
        with open(output_file) as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data["request_id"], "test-123")
        self.assertEqual(saved_data["prompt"], "Test prompt")
        self.assertEqual(saved_data["response"], "Test response")

    @patch("monai.deploy.operators.vlm_results_writer_operator.PILImage")
    def test_image_writing(self, mock_pil):
        """Test writing image results."""
        from monai.deploy.operators.vlm_results_writer_operator import (
            VLMResultsWriterOperator,
        )

        fragment = Mock(spec=Fragment)
        operator = VLMResultsWriterOperator(fragment, output_folder=self.output_dir)

        # Create mock image
        mock_image = Mock(spec=Image)
        image_array = np.ones((100, 100, 3), dtype=np.uint8)
        mock_image.asnumpy.return_value = image_array

        mock_pil_image = Mock()
        mock_pil.fromarray.return_value = mock_pil_image

        mock_input = Mock()
        mock_input.receive.side_effect = lambda x: {
            "result": mock_image,
            "output_type": "image",
            "request_id": "test-456",
        }.get(x)

        operator.compute(mock_input, None, None)

        # Verify save was called
        expected_path = Path(self.output_dir) / "test-456.png"
        mock_pil_image.save.assert_called_once()

        # Verify correct path
        save_path = mock_pil_image.save.call_args[0][0]
        self.assertEqual(save_path, expected_path)

    def test_error_handling(self):
        """Test error handling in results writer."""
        from monai.deploy.operators.vlm_results_writer_operator import (
            VLMResultsWriterOperator,
        )

        fragment = Mock(spec=Fragment)
        operator = VLMResultsWriterOperator(fragment, output_folder=self.output_dir)

        # Test with invalid output type
        mock_input = Mock()
        mock_input.receive.side_effect = lambda x: {
            "result": "Invalid data",
            "output_type": "image",  # Expects Image object
            "request_id": "test-error",
        }.get(x)

        # Should handle error gracefully
        operator.compute(mock_input, None, None)

        # Verify results counter still increments
        self.assertEqual(operator._results_written, 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for VLM operators working together."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()

        # Create test prompts
        self.prompts = {
            "defaults": {"max_new_tokens": 256},
            "prompts": [{"prompt": "Integration test", "image": "test.jpg", "output": "json"}],
        }

        with open(Path(self.test_dir) / "prompts.yaml", "w") as f:
            yaml.dump(self.prompts, f)

        # Create test image
        Path(self.test_dir, "test.jpg").touch()

    def tearDown(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)

    @patch("monai.deploy.operators.prompts_loader_operator.PILImage")
    @patch("monai.deploy.operators.llama3_vila_inference_operator.AutoConfig")
    def test_end_to_end_flow(self, mock_autoconfig, mock_pil):
        """Test end-to-end flow of VLM operators."""
        from monai.deploy.operators.llama3_vila_inference_operator import (
            Llama3VILAInferenceOperator,
        )
        from monai.deploy.operators.prompts_loader_operator import PromptsLoaderOperator
        from monai.deploy.operators.vlm_results_writer_operator import (
            VLMResultsWriterOperator,
        )

        # Mock PIL for loader
        mock_image = Mock()
        mock_image.mode = "RGB"
        mock_image.convert.return_value = mock_image
        mock_pil.open.return_value = mock_image

        with patch("numpy.array", return_value=np.ones((100, 100, 3), dtype=np.float32)):
            # Create operators
            fragment = Mock(spec=Fragment)
            app_context = Mock(spec=AppContext)

            loader = PromptsLoaderOperator(fragment, input_folder=self.test_dir)
            inference = Llama3VILAInferenceOperator(fragment, app_context=app_context, model_path=self.test_dir)
            writer = VLMResultsWriterOperator(fragment, output_folder=self.output_dir)

            # Setup all operators
            for op in [loader, inference, writer]:
                spec = Mock(spec=OperatorSpec)
                op.setup(spec)

            # Simulate data flow
            loader_output = Mock()
            emitted_data = {}

            def capture_emit(data, port):
                emitted_data[port] = data

            loader_output.emit = capture_emit

            # Run loader
            loader.compute(None, loader_output, None)

            # Pass data to inference
            inference_input = Mock()
            inference_input.receive = lambda x: emitted_data.get(x)

            inference_output = Mock()
            inference_emitted = {}
            inference_output.emit = lambda d, p: inference_emitted.update({p: d})

            inference.compute(inference_input, inference_output, None)

            # Verify inference output includes prompt
            result = inference_emitted.get("result")
            self.assertIsInstance(result, dict)
            self.assertIn("prompt", result)
            self.assertEqual(result["prompt"], "Integration test")


if __name__ == "__main__":
    unittest.main()
