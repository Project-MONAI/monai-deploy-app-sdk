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

"""Test security features of the pipeline generator."""

from pathlib import Path

import pytest
from pipeline_generator.generator.app_generator import AppGenerator


class TestSecurity:
    """Test security measures in the app generator."""

    def test_model_id_validation(self):
        """Test that invalid model IDs are rejected."""
        generator = AppGenerator()
        output_dir = Path("/tmp/test")

        # Valid model IDs
        valid_ids = [
            "MONAI/spleen_ct_segmentation",
            "test-org/model_name",
            "user/model-with-dashes",
            "org/model_with_underscores",
        ]

        # Invalid model IDs that could cause code injection
        invalid_ids = [
            "test; rm -rf /",  # Shell command injection
            "test' OR '1'='1",  # SQL injection style
            "test<script>alert('xss')</script>",  # HTML/JS injection
            "test`echo hacked`",  # Command substitution
            "test$(rm -rf /)",  # Command substitution
            "test\" + __import__('os').system('ls') + \"",  # Python injection
            "",  # Empty
            None,  # None
        ]

        # Test valid IDs (should not raise)
        for model_id in valid_ids:
            # We're just testing validation, not full generation
            try:
                # This will fail at download stage, but validation should pass
                generator.generate_app(model_id, output_dir)
            except ValueError as e:
                if "Invalid model_id" in str(e):
                    pytest.fail(f"Valid model_id '{model_id!r}' was rejected: {e}")
                # Other errors are fine (e.g., download failures)

        # Test invalid IDs (should raise ValueError)
        for model_id in invalid_ids:
            if model_id is None:
                continue  # Skip None test as it would fail at type checking
            with pytest.raises(ValueError, match="Invalid model_id"):
                generator.generate_app(model_id, output_dir)

    def test_app_name_sanitization(self):
        """Test that app names are properly sanitized for Python identifiers."""
        # Test cases mapping input to expected sanitized output
        test_cases = [
            ("test; rm -rf /", "test__rm__rfApp"),  # Multiple special chars become underscores
            ("test-with-dashes", "test_with_dashesApp"),
            ("test.with.dots", "test_with_dotsApp"),
            ("test space", "test_spaceApp"),
            ("123test", "_123testApp"),  # Starting with digit
            ("Test", "TestApp"),  # Normal case
        ]

        for input_name, expected_class_name in test_cases:
            # The AppGenerator will sanitize the name internally
            # We test the sanitization function directly
            sanitized = AppGenerator._sanitize_for_python_identifier(input_name)
            result_with_app = f"{sanitized}App"
            assert (
                result_with_app == expected_class_name
            ), f"Failed for '{input_name!r}': got '{result_with_app!r}', expected '{expected_class_name!r}'"

    def test_sanitize_for_python_identifier(self):
        """Test the Python identifier sanitization method."""
        test_cases = [
            ("normal_name", "normal_name"),
            ("name-with-dashes", "name_with_dashes"),
            ("name.with.dots", "name_with_dots"),
            ("name with spaces", "name_with_spaces"),
            ("123name", "_123name"),  # Can't start with digit
            ("", "app"),  # Empty string
            ("!@#$%", "app"),  # All invalid chars
            ("name!@#valid", "name___valid"),
            ("CamelCase", "CamelCase"),  # Preserve case
        ]

        for input_str, expected in test_cases:
            result = AppGenerator._sanitize_for_python_identifier(input_str)
            assert result == expected, f"Failed for {input_str!r}: got {result!r}, expected {expected!r}"

    def test_no_autoescape_with_comment(self):
        """Test that autoescape is disabled with proper documentation."""
        generator = AppGenerator()

        # Verify autoescape is False
        assert generator.env.autoescape is False

        # The comment explaining why is in the source code
        # This test just verifies the runtime behavior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
