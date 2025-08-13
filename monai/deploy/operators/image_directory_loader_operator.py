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
from typing import List

import numpy as np

from monai.deploy.core import Fragment, Image, Operator, OperatorSpec
from monai.deploy.utils.importutil import optional_import

PILImage, _ = optional_import("PIL", name="Image")


# @md.env(pip_packages=["Pillow >= 8.0.0"])
class ImageDirectoryLoader(Operator):
    """Load common image files (JPEG, PNG, BMP, TIFF) from a directory and convert them to Image objects.

    This operator processes image files one at a time to avoid buffer overflow issues and supports
    batch processing of multiple images in a directory.

    By default it outputs channel-first arrays (CHW) to match many MONAI pipelines. For 2D RGB models
    whose bundle preprocessing includes EnsureChannelFirstd(channel_dim=-1), set ``channel_first=False``
    to emit HWC arrays so the bundle transform handles channel movement.

    Named Outputs:
        image: Image object loaded from file
        filename: Name of the loaded file (without extension)
    """

    SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

    def __init__(
        self,
        fragment: Fragment,
        *args,
        input_folder: Path,
        channel_first: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the ImageDirectoryLoader.

        Args:
            fragment: An instance of the Application class
            input_folder: Path to folder containing image files
            channel_first: If True (default), emit CHW arrays. If False, emit HWC arrays.
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._input_folder = Path(input_folder)
        self._channel_first = bool(channel_first)

        super().__init__(fragment, *args, **kwargs)

    def _find_image_files(self) -> List[Path]:
        """Find all supported image files in the input directory."""
        image_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            image_files.extend(self._input_folder.rglob(f"*{ext}"))
            image_files.extend(self._input_folder.rglob(f"*{ext.upper()}"))

        # Sort files for consistent ordering
        image_files.sort()
        return image_files

    def setup(self, spec: OperatorSpec):
        """Define the operator outputs."""
        spec.output("image")
        spec.output("filename")

        # Pre-initialize the image files list
        self._image_files = self._find_image_files()
        self._current_index = 0

        if not self._image_files:
            self._logger.warning(f"No image files found in {self._input_folder}")
        else:
            self._logger.info(f"Found {len(self._image_files)} image files to process")

    def compute(self, op_input, op_output, context):
        """Load one image and emit it."""

        # Check if we have more images to process
        if self._current_index >= len(self._image_files):
            # No more images to process
            self._logger.info("All images have been processed")
            self.fragment.stop_execution()
            return

        # Get the current image path
        image_path = self._image_files[self._current_index]

        try:
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
            image_obj = Image(image_array, metadata=metadata)

            # Emit the image and filename
            op_output.emit(image_obj, "image")
            op_output.emit(image_path.stem, "filename")

            self._logger.info(
                f"Loaded and emitted image: {image_path.name} ({self._current_index + 1}/{len(self._image_files)})"
            )

        except Exception as e:
            self._logger.error(f"Failed to load image {image_path}: {e}")

        # Move to the next image
        self._current_index += 1


def test():
    """Test the ImageDirectoryLoader operator."""
    import tempfile

    from PIL import Image as PILImageCreate

    # Create a temporary directory with test images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test images
        for i in range(3):
            img = PILImageCreate.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
            img.save(temp_path / f"test_{i}.jpg")

        # Test the operator
        fragment = Fragment()
        loader = ImageDirectoryLoader(fragment, input_folder=temp_path)

        # Simulate setup
        from monai.deploy.core import OperatorSpec

        spec = OperatorSpec()
        loader.setup(spec)

        print(f"Found {len(loader._image_files)} test images")

        # Simulate compute calls
        class MockOutput:
            def emit(self, data, name):
                if name == "filename":
                    print(f"Emitted filename: {data}")
                elif name == "image":
                    print(f"Emitted image with shape: {data.asnumpy().shape}")

        mock_output = MockOutput()

        # Process all images
        while loader._current_index < len(loader._image_files):
            loader.compute(None, mock_output, None)


if __name__ == "__main__":
    test()
