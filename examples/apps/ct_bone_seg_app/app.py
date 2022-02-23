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

from ct_bone_operator import CtBoneOperator
from dicom_to_mhd import DicomToMhd
from monai.deploy.core import Application, env, resource

@resource(cpu=1, gpu=1)
class App(Application):
    """This is a CT Bone application.
    """

    name = "ct_bone_app"
    description = "CT Bone segmentation app."
    version = "0.1.0"

    def compose(self):
        """This application has two operators.
        """
        
        dcm_to_mhd = DicomToMhd()
        bone_op = CtBoneOperator()
        
        self.add_flow(dcm_to_mhd, bone_op)

if __name__ == "__main__":
    App(do_run=True)
