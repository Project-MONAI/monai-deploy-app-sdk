# Copyright 2021 MONAI Consortium
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
# If `pip_packages` is specified, the definition will be aggregated with the package dependency list of other
# operators and the application in packaging time.
@md.env(pip_packages=["SimpleITK==1.2.4"])
class DicomToMhd(Operator):
    """
    If input is DICOM: This operator converts a dicom image to a mhd image
    If input is MHD: This operator sets the output path to input path
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        input_path = op_input.get().path
        if input_path.is_dir():
            print(input_path)
            _current_dir = os.path.abspath(os.path.dirname(__file__))
            op_output.set(DataPath(_current_dir))
            output_path = op_output.get().path
            if output_path.is_dir():
                print(output_path)
            
            output_file_path = os.path.join(output_path, "intermediate_mhd_data.mhd")

            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(input_path))
            reader.SetFileNames(dicom_names)

            fixed = reader.Execute()
            sitk.WriteImage(fixed, output_file_path)
        else:
            print("Input path is not a directory")
            if os.path.isfile(input_path):
                extension = os.path.splitext(input_path)[1]
                if extension == '.mhd':
                    print("Input path is a MHD file")
                    print("Setting output folder as input folder")
                    op_output.set(DataPath(input_path))
                else:
                    raise IOError('Unsupported extension')
            else:
                raise IOError('Invalid input path')


        