# Copyright 2021-2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext
from monai.deploy.utils.importutil import optional_import

SimpleITK, _ = optional_import("SimpleITK")


@md.input("image_path", DataPath, IOType.DISK)
@md.output("image", np.ndarray, IOType.IN_MEMORY)
class NiftiDataLoader(Operator):
    """
    This operator reads a nifti image, extracts the numpy array and forwards it to the next operator
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        input_path = op_input.get().path
        image_np = self.convert_and_save(input_path)
        op_output.set(image_np)

    def convert_and_save(self, nii_path):
        """
        reads the nifti image and
        """
        image_reader = SimpleITK.ImageFileReader()
        image_reader.SetFileName(str(nii_path))
        image = image_reader.Execute()
        image_np = np.transpose(SimpleITK.GetArrayFromImage(image), [2, 1, 0])
        return image_np


def test():
    filepath = "/home/gupta/Documents/mni_icbm152_nlin_sym_09a/mni_icbm152_gm_tal_nlin_sym_09a.nii"
    nii_operator = NiftiDataLoader()
    _ = nii_operator.convert_and_save(filepath)


def main():
    test()


if __name__ == "__main__":
    main()
