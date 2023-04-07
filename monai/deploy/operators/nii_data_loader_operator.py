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

from pathlib import Path

import numpy as np

import monai.deploy.core as md
from monai.deploy.core import ConditionType, Fragment, Operator, OperatorSpec
from monai.deploy.utils.importutil import optional_import

SimpleITK, _ = optional_import("SimpleITK")


# @md.input("image_path", DataPath, IOType.DISK)
# @md.output("image", np.ndarray, IOType.IN_MEMORY)
# @md.env(pip_packages=["SimpleITK>=2.0.2"])
class NiftiDataLoader(Operator):
    """
    This operator reads a nifti image, extracts the numpy array and forwards it to the next operator

    Named input:
        image_path: Path to the image file, optional, used to override the path set in the object.

    Named output:
        image: A Numpy object in memory. Downstream receiver optional.
    """

    def __init__(self, fragment: Fragment, *args, input_path: Path, **kwargs) -> None:
        """Creates an instance with the file path to load image from.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            input_path (Path): The Path to read the image file from, overriden by the named input.
        """
        self.input_path = input_path  # Allow to be None, to be overridden when compute is called.
        self.input_name_path = "image_path"
        self.output_name_image = "image"

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_path).condition(ConditionType.NONE)
        spec.output(self.output_name_image).condition(ConditionType.NONE)  # Fine for no or not-ready receiver ports.

    def compute(self, op_input, op_output, context):
        input_path = None
        try:
            input_path = op_input.receive(self.input_name_path)
        except Exception:
            if self.input_path and not self.input_path.is_file():
                input_path = self.input_path
            else:
                raise ValueError("No path given to load image from.")

        image_np = self.convert_and_save(input_path)
        op_output.emit(image_np, self.output_name_image)

    def convert_and_save(self, nii_path):
        """
        reads the nifti image and returns a numpy image array
        """
        image_reader = SimpleITK.ImageFileReader()
        image_reader.SetFileName(str(nii_path))
        image = image_reader.Execute()
        image_np = np.transpose(SimpleITK.GetArrayFromImage(image), [2, 1, 0])
        return image_np


def test():
    filepath = "/home/mqin/src/monai-deploy-app-sdk/inputs/lung_seg_ct/nii/volume-covid19-A-0001.nii"  #  "/home/gupta/Documents/mni_icbm152_nlin_sym_09a/mni_icbm152_gm_tal_nlin_sym_09a.nii"
    fragment = Fragment()
    nii_operator = NiftiDataLoader(fragment, input_path=filepath)
    _ = nii_operator.convert_and_save(filepath)


def main():
    test()


if __name__ == "__main__":
    main()
