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
from pathlib import Path
from typing import Optional

import numpy as np

from monai.deploy.core import ConditionType, Fragment, Image, Operator, OperatorSpec
from monai.deploy.utils.importutil import optional_import

PILImage, _ = optional_import("PIL", name="Image")


# @md.env(pip_packages=["Pillow >= 8.0.0"])
class ImageFileLoader(Operator):
    """Load a single image file (JPEG, PNG, BMP, TIFF) and convert to Image object.

    This operator loads a single image file specified via input path and outputs an Image object.
    It can be chained with GenericDirectoryScanner for batch processing of multiple images.

    By default it outputs channel-first arrays (CHW) to match many MONAI pipelines. For 2D RGB models
    whose bundle preprocessing includes EnsureChannelFirstd(channel_dim=-1), set ``channel_first=False``
    to emit HWC arrays so the bundle transform handles channel movement.

    Named Inputs:
        file_path: Path to the image file to load (optional, overrides input_path)

    Named Outputs:
        image: Image object loaded from file
        filename: Name of the loaded file (without extension)
    """

    SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

    def __init__(
        self,
        fragment: Fragment,
        *args,
        input_path: Optional[Path] = None,
        channel_first: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the ImageFileLoader.

        Args:
            fragment: An instance of the Application class
            input_path: Default path to image file (can be overridden by input)
            channel_first: If True (default), emit CHW arrays. If False, emit HWC arrays.
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._input_path = Path(input_path) if input_path else None
        self._channel_first = bool(channel_first)

        # Port names
        self._input_name_path = "file_path"
        self._output_name_image = "image"
        self._output_name_filename = "filename"

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Define the operator inputs and outputs."""
        spec.input(self._input_name_path).condition(ConditionType.NONE)
        spec.output(self._output_name_image)
        spec.output(self._output_name_filename).condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        """Load the image file and emit it."""

        # Try to get file path from input port
        input_path = None
        try:
            input_path = op_input.receive(self._input_name_path)
        except Exception:
            pass

        # Validate input path or fall back to object attribute
        if not input_path or not Path(input_path).is_file():
            self._logger.info(f"No or invalid file path from input port: {input_path}")
            # Try to fall back to use the object attribute if it is valid
            if self._input_path and self._input_path.is_file():
                input_path = self._input_path
            else:
                raise ValueError(f"No valid file path from input port or obj attribute: {self._input_path}")

        # Convert to Path object
        image_path = Path(input_path)

        # Validate file extension
        if image_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {image_path.suffix}. "
                f"Supported extensions: {self.SUPPORTED_EXTENSIONS}"
            )

        try:
            # Load and process the image
            image_obj = self._load_image(image_path)

            # Emit the image and filename
            op_output.emit(image_obj, self._output_name_image)
            op_output.emit(image_path.stem, self._output_name_filename)

            self._logger.info(f"Successfully loaded and emitted image: {image_path.name}")

        except Exception as e:
            self._logger.error(f"Failed to load image {image_path}: {e}")
            raise

    def _load_image(self, image_path: Path) -> Image:
        """Load an image file and return as Image object."""
        # Load image using PIL
        pil_image = PILImage.open(image_path)

        # Convert to RGB if necessary
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Convert to numpy array (HWC float32). Intensity scaling (to [0,1]) is typically handled by bundle.
        image_array = np.array(pil_image).astype(np.float32)

        # Convert to channel-first when requested
        if self._channel_first:
            # PIL loads HWC; convert to CHW
            image_array = np.transpose(image_array, (2, 0, 1))

        # Create metadata
        metadata = {
            "filename": str(image_path),
            "original_shape": image_array.shape,
            "source_format": image_path.suffix.lower(),
        }

        # Create Image object
        return Image(image_array, metadata=metadata)


def test():
    """Test the ImageFileLoader operator."""
    import tempfile

    from PIL import Image as PILImageCreate

    # Create a temporary directory with a test image
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test image
        test_image_path = temp_path / "test_image.jpg"
        img = PILImageCreate.new("RGB", (100, 100), color=(128, 64, 192))
        img.save(test_image_path)

        # Test the operator
        fragment = Fragment()
        loader = ImageFileLoader(fragment, input_path=test_image_path)

        # Simulate setup
        from monai.deploy.core import OperatorSpec

        spec = OperatorSpec()
        loader.setup(spec)

        # Simulate compute call
        class MockInput:
            def receive(self, name):
                # Simulate no input from port, will fall back to object attribute
                raise Exception("No input")

        class MockOutput:
            def emit(self, data, name):
                if name == "filename":
                    print(f"Emitted filename: {data}")
                elif name == "image":
                    print(f"Emitted image with shape: {data.asnumpy().shape}")

        mock_input = MockInput()
        mock_output = MockOutput()

        loader.compute(mock_input, mock_output, None)


if __name__ == "__main__":
    test()
