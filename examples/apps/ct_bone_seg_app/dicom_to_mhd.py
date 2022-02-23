# Copyright 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext
import SimpleITK as sitk
import os

@md.input("image", DataPath, IOType.DISK)
@md.output("image", DataPath, IOType.DISK)
@md.env(pip_packages=["SimpleITK==1.2.4"])
class DicomToMhd(Operator):
    """
    If input is DICOM: This operator converts a dicom image to a mhd image
    If input is MHD: This operator sets the output path to input path
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        input_path = op_input.get().path
        if input_path.is_dir():
            current_file_dir = os.path.abspath(os.path.dirname(__file__))
            op_output.set(DataPath(current_file_dir))
            output_path = op_output.get().path
            
            output_file_path = os.path.join(output_path, "intermediate_mhd_data.mhd")

            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(input_path))
            reader.SetFileNames(dicom_names)

            fixed = reader.Execute()
            sitk.WriteImage(fixed, output_file_path)
        else:
            if os.path.isfile(input_path):
                extension = os.path.splitext(input_path)[1]
                if extension == '.mhd':
                    # Input path is a MHD file
                    # Setting output folder as input folder
                    op_output.set(DataPath(input_path))
                else:
                    raise IOError('Unsupported extension')
            else:
                raise IOError('Invalid input path')


        