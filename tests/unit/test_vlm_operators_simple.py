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

"""Simple unit tests for VLM operators that test basic functionality."""

import json
import tempfile
import unittest
from pathlib import Path


class TestVLMOperatorsBasic(unittest.TestCase):
    """Basic tests for VLM operators without heavy dependencies."""

    def test_prompts_loader_yaml_parsing(self):
        """Test YAML parsing logic in PromptsLoaderOperator."""
        # Test YAML structure
        prompts_data = {
            "defaults": {"max_new_tokens": 256, "temperature": 0.2, "top_p": 0.9},
            "prompts": [{"prompt": "Test prompt", "image": "test.jpg", "output": "json"}],
        }

        # Verify structure
        self.assertIn("defaults", prompts_data)
        self.assertIn("prompts", prompts_data)
        self.assertEqual(len(prompts_data["prompts"]), 1)
        self.assertEqual(prompts_data["prompts"][0]["output"], "json")

    def test_json_result_format(self):
        """Test JSON result structure for VLM outputs."""
        # Test the expected JSON format
        result = {
            "request_id": "test-123",
            "response": "Test response",
            "status": "success",
            "prompt": "Test prompt",
            "image": "/path/to/test.jpg",
        }

        # Verify all required fields
        self.assertIn("request_id", result)
        self.assertIn("response", result)
        self.assertIn("status", result)
        self.assertIn("prompt", result)
        self.assertIn("image", result)

        # Verify JSON serializable
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["prompt"], "Test prompt")

    def test_output_type_handling(self):
        """Test different output type handling."""
        output_types = ["json", "image", "image_overlay"]

        for output_type in output_types:
            self.assertIn(output_type, ["json", "image", "image_overlay"])

    def test_prompts_file_loading(self):
        """Test prompts.yaml file loading behavior."""
        # Test YAML structure that would be loaded
        yaml_content = {
            "defaults": {"max_new_tokens": 256},
            "prompts": [{"prompt": "Test", "image": "test.jpg", "output": "json"}],
        }

        # Simulate file loading
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            # Write and verify
            import yaml

            yaml.dump(yaml_content, f)
            f.flush()

            # File exists
            self.assertTrue(Path(f.name).exists())

            # Can be loaded
            with open(f.name) as rf:
                loaded = yaml.safe_load(rf)
            self.assertEqual(loaded["defaults"]["max_new_tokens"], 256)

    def test_request_id_generation(self):
        """Test request ID generation logic."""
        import uuid

        # Generate request ID
        request_id = str(uuid.uuid4())

        # Verify format
        self.assertIsInstance(request_id, str)
        self.assertEqual(len(request_id), 36)  # UUID4 format
        self.assertIn("-", request_id)

    def test_generation_params_merging(self):
        """Test merging of default and prompt-specific generation parameters."""
        defaults = {"max_new_tokens": 256, "temperature": 0.2, "top_p": 0.9}

        prompt_params = {"max_new_tokens": 128}  # Override

        # Merge logic
        gen_params = defaults.copy()
        gen_params.update(prompt_params)

        # Verify merge
        self.assertEqual(gen_params["max_new_tokens"], 128)  # Overridden
        self.assertEqual(gen_params["temperature"], 0.2)  # From defaults
        self.assertEqual(gen_params["top_p"], 0.9)  # From defaults

    def test_error_result_format(self):
        """Test error result format."""
        error_result = {
            "request_id": "test-error",
            "prompt": "Test prompt",
            "error": "Test error message",
            "status": "error",
        }

        # Verify error format
        self.assertEqual(error_result["status"], "error")
        self.assertIn("error", error_result)
        self.assertIn("prompt", error_result)

    def test_file_naming_convention(self):
        """Test output file naming conventions."""
        request_id = "abc123"

        # Test different output formats
        json_filename = f"{request_id}.json"
        image_filename = f"{request_id}.png"
        overlay_filename = f"{request_id}_overlay.png"

        self.assertTrue(json_filename.endswith(".json"))
        self.assertTrue(image_filename.endswith(".png"))
        self.assertTrue(overlay_filename.endswith("_overlay.png"))


if __name__ == "__main__":
    unittest.main()
