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

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from monai.deploy.core import Fragment, Image, Operator, OperatorSpec
from monai.deploy.utils.importutil import optional_import

PILImage, _ = optional_import("PIL", name="Image")


class VLMResultsWriterOperator(Operator):
    """Write vision-language model results to disk based on output type.

    This operator receives results from the VLM inference operator and writes
    them to the output directory in the appropriate format:

    - json: Writes the result as a JSON file named {request_id}.json
    - image: Writes the image as a PNG file named {request_id}.png
    - image_overlay: Writes the image with overlay as a PNG file named {request_id}_overlay.png

    The operator handles results sequentially and writes each one to disk as it's received.

    Inputs:
        result: The generated result (format depends on output_type)
        output_type: The output type (json, image, or image_overlay)
        request_id: The request ID used for naming output files
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        output_folder: Path,
        **kwargs,
    ) -> None:
        """Initialize the VLMResultsWriterOperator.

        Args:
            fragment: An instance of the Application class
            output_folder: Path to folder where results will be written
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._output_folder = Path(output_folder)

        # Create output directory if it doesn't exist
        self._output_folder.mkdir(parents=True, exist_ok=True)

        super().__init__(fragment, *args, **kwargs)

        # Track number of results written
        self._results_written = 0

    def setup(self, spec: OperatorSpec):
        """Define the operator inputs."""
        spec.input("result")
        spec.input("output_type")
        spec.input("request_id")

    def _write_json_result(self, result: Dict[str, Any], request_id: str):
        """Write JSON result to disk."""
        output_path = self._output_folder / f"{request_id}.json"

        try:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            self._logger.info(f"Wrote JSON result to {output_path}")
        except Exception as e:
            self._logger.error(f"Failed to write JSON result: {e}")

    def _write_image_result(self, image: Image, request_id: str, suffix: str = ""):
        """Write image result to disk."""
        output_filename = f"{request_id}{suffix}.png"
        output_path = self._output_folder / output_filename

        try:
            # Get numpy array from Image object
            image_array = image.asnumpy()

            # Ensure HWC format
            if image_array.ndim == 3 and image_array.shape[0] <= 4:  # Likely CHW
                image_array = np.transpose(image_array, (1, 2, 0))

            # Convert to uint8 if needed
            if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)

            # Save using PIL
            pil_image = PILImage.fromarray(image_array)
            pil_image.save(output_path)

            self._logger.info(f"Wrote image result to {output_path}")

        except Exception as e:
            self._logger.error(f"Failed to write image result: {e}")

    def compute(self, op_input, op_output, context):
        """Write results to disk based on output type."""
        # Receive inputs
        result = op_input.receive("result")
        output_type = op_input.receive("output_type")
        request_id = op_input.receive("request_id")

        self._logger.info(f"Writing result for request {request_id} with output type {output_type!r}")

        try:
            if output_type == "json":
                if isinstance(result, dict):
                    self._write_json_result(result, request_id)
                else:
                    # Convert to dict if needed
                    self._write_json_result({"result": str(result)}, request_id)

            elif output_type == "image":
                if isinstance(result, Image):
                    self._write_image_result(result, request_id)
                else:
                    self._logger.error(f"Expected Image object for image output, got {type(result)}")

            elif output_type == "image_overlay":
                if isinstance(result, Image):
                    self._write_image_result(result, request_id, suffix="_overlay")
                else:
                    self._logger.error(f"Expected Image object for image_overlay output, got {type(result)}")

            else:
                self._logger.warning(f"Unknown output type: {output_type}")
                # Write as JSON fallback
                self._write_json_result({"result": str(result), "output_type": output_type}, request_id)

            self._results_written += 1
            self._logger.info(f"Total results written: {self._results_written}")

        except Exception as e:
            self._logger.error(f"Error writing result for request {request_id}: {e}")

            # Try to write error file
            error_path = self._output_folder / f"{request_id}_error.json"
            try:
                with open(error_path, "w") as f:
                    json.dump(
                        {
                            "request_id": request_id,
                            "error": str(e),
                            "output_type": output_type,
                        },
                        f,
                        indent=2,
                    )
            except Exception:
                pass
