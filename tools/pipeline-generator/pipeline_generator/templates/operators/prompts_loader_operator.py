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

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml  # type: ignore

from monai.deploy.core import Fragment, Image, Operator, OperatorSpec
from monai.deploy.utils.importutil import optional_import

PILImage, _ = optional_import("PIL", name="Image")


class PromptsLoaderOperator(Operator):
    """Load prompts from a YAML file and emit them one at a time with associated images.

    This operator reads a prompts.yaml file with the following format:

    ```yaml
    defaults:
      max_new_tokens: 256
      temperature: 0.2
      top_p: 0.9
    prompts:
      - prompt: Summarize key findings.
        image: img1.png
        output: json
      - prompt: Is there a focal lesion?
        image: img2.png
        output: image
        max_new_tokens: 128
    ```

    For each prompt, it emits:
    - image: The loaded image as an Image object
    - prompt: The prompt text
    - output_type: The expected output type (json, image, or image_overlay)
    - request_id: A unique identifier for the request
    - generation_params: A dictionary of generation parameters

    The operator processes prompts sequentially and stops execution when all prompts
    have been processed.
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        input_folder: Path,
        **kwargs,
    ) -> None:
        """Initialize the PromptsLoaderOperator.

        Args:
            fragment: An instance of the Application class
            input_folder: Path to folder containing prompts.yaml and image files
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._input_folder = Path(input_folder)

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Define the operator outputs."""
        spec.output("image")
        spec.output("prompt")
        spec.output("output_type")
        spec.output("request_id")
        spec.output("generation_params")

        # Load and parse the prompts file
        self._prompts_data = self._load_prompts()
        self._current_index = 0

        if not self._prompts_data:
            self._logger.warning(f"No prompts found in {self._input_folder}/prompts.yaml")
        else:
            self._logger.info(f"Found {len(self._prompts_data)} prompts to process")

    def _load_prompts(self) -> List[Dict[str, Any]]:
        """Load and parse the prompts.yaml file."""
        prompts_file = self._input_folder / "prompts.yaml"

        if not prompts_file.exists():
            self._logger.error(f"prompts.yaml not found in {self._input_folder}")
            return []

        try:
            with open(prompts_file, "r") as f:
                data = yaml.safe_load(f)

            defaults = data.get("defaults", {})
            prompts = data.get("prompts", [])

            # Merge defaults with each prompt
            processed_prompts = []
            for prompt in prompts:
                # Create generation parameters by merging defaults with prompt-specific params
                gen_params = defaults.copy()

                # Override with prompt-specific parameters
                for key in ["max_new_tokens", "temperature", "top_p"]:
                    if key in prompt:
                        gen_params[key] = prompt[key]

                processed_prompts.append(
                    {
                        "prompt": prompt.get("prompt", ""),
                        "image": prompt.get("image", ""),
                        "output_type": prompt.get("output", "json"),
                        "generation_params": gen_params,
                    }
                )

            return processed_prompts

        except Exception as e:
            self._logger.error(f"Error loading prompts.yaml: {e}")
            return []

    def _load_image(self, image_filename: str) -> Optional[Image]:
        """Load an image file and convert it to an Image object."""
        image_path = self._input_folder / image_filename

        if not image_path.exists():
            self._logger.error(f"Image file not found: {image_path}")
            return None

        try:
            # Load image using PIL
            pil_image = PILImage.open(image_path)

            # Convert to RGB if necessary
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Convert to numpy array (HWC format, float32)
            # Note: For VLM models, we typically keep HWC format
            image_array = np.array(pil_image).astype(np.float32)

            # Create metadata
            metadata = {
                "filename": str(image_path),
                "original_shape": image_array.shape,
                "source_format": image_path.suffix.lower(),
            }

            # Create Image object
            return Image(image_array, metadata=metadata)

        except Exception as e:
            self._logger.error(f"Failed to load image {image_path}: {e}")
            return None

    def compute(self, op_input, op_output, context):
        """Process one prompt and emit it."""

        # Check if we have more prompts to process
        if self._current_index >= len(self._prompts_data):
            # No more prompts to process
            self._logger.info("All prompts have been processed")
            self.fragment.stop_execution()
            return

        # Get the current prompt data
        prompt_data = self._prompts_data[self._current_index]

        # Load the associated image
        image = self._load_image(prompt_data["image"])
        if image is None:
            self._logger.error("Skipping prompt due to image load failure")
            self._current_index += 1
            return

        # Generate a unique request ID
        request_id = str(uuid.uuid4())

        # Emit all the data
        op_output.emit(image, "image")
        op_output.emit(prompt_data["prompt"], "prompt")
        op_output.emit(prompt_data["output_type"], "output_type")
        op_output.emit(request_id, "request_id")
        op_output.emit(prompt_data["generation_params"], "generation_params")

        self._logger.info(
            f"Emitted prompt {self._current_index + 1}/{len(self._prompts_data)}: "
            f"'{prompt_data['prompt'][:50]}...' with image {prompt_data['image']}"
        )

        # Move to the next prompt
        self._current_index += 1
