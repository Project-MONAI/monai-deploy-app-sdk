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
class NiftiDataWriter(Operator):

    def __init__(self, fragment: Fragment, *args, output_file: Path, **kwargs) -> None:
        """Creates an instance with the file path to load image from.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            input_path (Path): The file Path to read from, overridden by valid named input on compute.
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

        self.output_file = output_file
        self.input_name_seg = "seg_image"
        self.input_name_output_file = "output_file"

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_seg)
        spec.input(self.input_name_output_file).condition(ConditionType.NONE)  # Optional input not requiring sender.

    def compute(self, op_input, op_output, context):
        """Performs computation with the provided context."""


        seg_image = op_input.receive(self.input_name_seg)


        # If the optional named input, output_folder, has content, use it instead of the one set on the object.
        # Since this input is optional, must check if data present and if Path or str.
        output_file = None
        try:
            output_file = op_input.receive(self.input_name_output_file)
        except Exception:
            pass

        if not output_file or not isinstance(output_file, (Path, str)):
            output_file = self.output_file

        self.convert_and_save(seg_image, output_file)

    def convert_and_save(self, seg_image, nii_path):
        """
        reads the nifti image and returns a numpy image array
        """
        image_writer = SimpleITK.ImageFileWriter()

        image = SimpleITK.GetImageFromArray(seg_image._data)
        image.SetSpacing(seg_image.metadata()["pixdim"])
        
        if len(seg_image.metadata()["direction"]) == 16:
            direction = []
            direction.extend(seg_image.metadata()["direction"][0:3])
            direction.extend(seg_image.metadata()["direction"][4:7])
            direction.extend(seg_image.metadata()["direction"][8:11])
            image.SetDirection(direction)
        else:
            image.SetDirection(seg_image.metadata()["direction"])
            
        image.SetOrigin(seg_image.metadata()["origin"])
        
        image_writer.SetFileName(nii_path)
        image_writer.Execute(image)


def test():
    ...


def main():
    test()


if __name__ == "__main__":
    main()
