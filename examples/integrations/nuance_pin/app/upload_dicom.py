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

import glob
import os

import pydicom

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext


@md.input("image", DataPath, IOType.DISK)
class NuancePINUploadDicom(Operator):

    def __init__(self, upload_document, upload_gsps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upload_document = upload_document
        self.upload_gsps = upload_gsps

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        input_path = op_input.get().path
        dcm_files = glob.glob(f"{os.path.sep}".join([f"{input_path}", "**", "*.dcm"]), recursive=True)

        for dcm_file in dcm_files:
            ds = pydicom.dcmread(dcm_file)
            series_uid = ds[0x0020000D].value

            self.upload_gsps(
                os.path.join(self.output_path, dcm_file),
                series_uid=series_uid,
            )
