# Copyright 2021-2023 MONAI Consortium
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

import numpy as np

from monai.deploy.core import ConditionType, Fragment, Operator, OperatorSpec
from monai.deploy.utils.importutil import optional_import

SimpleITK, _ = optional_import("SimpleITK")


# @md.env(pip_packages=["SimpleITK>=2.0.2"])
class NiftiDataLoader(Operator):
    """
    This operator reads a nifti image, extracts the numpy array and forwards it to the next operator

    Named input:
        image_path: Path to the image file, optional. Use it to override the input path set on the object.

    Named output:
        image: A Numpy array object. Downstream receiver optional.
    """

    def __init__(self, fragment: Fragment, *args, input_path: Path, **kwargs) -> None:
        """Creates an instance with the file path to load image from.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            input_path (Path): The file Path to read from, overridden by valid named input on compute.
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.input_path = input_path  # Allow to be None, to be overridden when compute is called.
        self.input_name_path = "image_path"
        self.output_name_image = "image"

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_path).condition(ConditionType.NONE)
        spec.output(self.output_name_image).condition(ConditionType.NONE)  # Fine for no or not-ready receiver ports.

    def compute(self, op_input, op_output, context):
        """Performs computation with the provided context."""

        # The named input port is optional, so must check for and validate the data
        input_path = None
        try:
            input_path = op_input.receive(self.input_name_path)
        except Exception:
            pass

        if not input_path or not Path(input_path).is_file:
            self._logger.info(f"No or invalid file path from the optional input port: {input_path}")
            # Try to fall back to use the object attribute if it is valid
            if self.input_path and self.input_path.is_file():
                input_path = self.input_path
            else:
                raise ValueError(f"No valid file path from input port or obj attribute: {self.input_path}")

        image_np = self.convert_and_save(input_path)
        op_output.emit(image_np, self.output_name_image)

    def convert_and_save(self, nii_path):
        """
        reads the nifti image and returns a numpy image array
        """
        image_reader = SimpleITK.ImageFileReader()
        image_reader.SetFileName(str(nii_path))
        image = image_reader.Execute()
        image_np = SimpleITK.GetArrayFromImage(image)

        # Get image metadata to properly distinguish between different image types
        spatial_dims = image.GetDimension()  # Actual spatial dimensions (2D, 3D, etc.)
        num_components = image.GetNumberOfComponentsPerPixel()  # Components/channels per pixel
        
        self._logger.debug(f"Image spatial dimensions: {spatial_dims}, components per pixel: {num_components}, array shape: {image_np.shape}")

        # Handle different dimensionalities properly using SimpleITK metadata
        if spatial_dims == 2:
            if num_components == 1:
                # 2D grayscale: transpose from (y, x) to (x, y)
                image_np = np.transpose(image_np, [1, 0])
            else:
                # 2D with multiple components/channels: transpose from (y, x, c) to (x, y, c)
                # SimpleITK stores multi-component 2D images as (y, x, c)
                image_np = np.transpose(image_np, [1, 0, 2])
        elif spatial_dims == 3:
            if num_components == 1:
                # 3D grayscale volume: transpose from (z, y, x) to (x, y, z)
                image_np = np.transpose(image_np, [2, 1, 0])
            else:
                # 3D volume with multiple components: transpose from (z, y, x, c) to (x, y, z, c)
                # SimpleITK stores multi-component 3D images as (z, y, x, c)
                image_np = np.transpose(image_np, [2, 1, 0, 3])
        else:
            # For other spatial dimensions, log a warning and return as-is
            self._logger.warning(
                f"Unexpected {spatial_dims}D spatial image with {num_components} components per pixel, "
                f"array shape {image_np.shape} from {nii_path}, returning without transpose"
            )

        return image_np


def test():
    # Make sure the file path is correct.
    filepath = Path(__file__).parent.resolve() / "../../../inputs/lung_seg_ct/nii/volume-covid19-A-0001.nii"
    fragment = Fragment()
    nii_operator = NiftiDataLoader(fragment, input_path=filepath)
    _ = nii_operator.convert_and_save(filepath)


def main():
    test()


if __name__ == "__main__":
    main()
