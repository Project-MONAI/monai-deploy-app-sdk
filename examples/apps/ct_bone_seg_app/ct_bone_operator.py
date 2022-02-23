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
from monai.deploy.core import DataPath, ExecutionContext, Image, InputContext, IOType, Operator, OutputContext
import os

@md.input("image", DataPath, IOType.DISK)
@md.output("image", DataPath, IOType.DISK)
@md.env(pip_packages=["patchelf", "clcat"])
class CtBoneOperator(Operator):
    """This Operator implements CT Bone segmentation
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        
        input_path = op_input.get().path
        inputfile = ""
        
        if input_path.is_dir():
            inputfile = os.path.join(input_path, "intermediate_mhd_data.mhd")
        elif input_path.is_file():
            inputfile = str(input_path)
        else:
            raise("Input path invalid from input context")

        output_path = op_output.get().path
        output_path.mkdir(parents=True, exist_ok=True)
        outputfile = os.path.join(output_path, "bone_mask.mhd")
        
        import clcat.ct_algos as cact
        status = cact.ct_bone(inputfile, outputfile)
        
        if input_path.is_dir():
            os.remove(inputfile)
            inputrawfile  = os.path.join(input_path, "intermediate_mhd_data.raw")
            os.remove(inputrawfile)

        if status != 0:
            raise("Bone segmentation failed")
