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

import os.path
from typing import Any
from zipfile import BadZipFile, ZipFile

from monai.deploy.utils.importutil import optional_import

from .model import Model

torch, _ = optional_import("torch")


class TorchScriptModel(Model):
    """Represents TorchScript model.

    TorchScript serialization format (TorchScript model file) is created by torch.jit.save() method and
    the serialized model (which usually has .pt or .pth extension) is a ZIP archive containing many files.
    (https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md)

    We consider that the model is a torchscript model if its unzipped archive contains files named 'data.pkl' and
    'constants.pkl', and folders named 'code' and 'data'.
    """

    model_type: str = "torch_script"

    @property
    def predictor(self) -> "torch.nn.Module":
        if self._predictor is None:
            self._predictor = torch.jit.load(self.path)
        return self._predictor

    @predictor.setter
    def predictor(self, predictor: Any):
        self._predictor = predictor

    @classmethod
    def accept(cls, path: str):
        prefix_code = False
        prefix_data = False
        prefix_constants_pkl = False
        prefix_data_pkl = False

        if not os.path.isfile(path):
            return False, None

        try:
            zip_file = ZipFile(path)
            for data in zip_file.filelist:
                file_name = data.filename
                pivot = file_name.find("/")
                if pivot != -1 and not prefix_code and file_name[pivot:].startswith("/code/"):
                    prefix_code = True
                if pivot != -1 and not prefix_data and file_name[pivot:].startswith("/data/"):
                    prefix_data = True
                if pivot != -1 and not prefix_constants_pkl and file_name[pivot:] == "/constants.pkl":
                    prefix_constants_pkl = True
                if pivot != -1 and not prefix_data_pkl and file_name[pivot:] == "/data.pkl":
                    prefix_data_pkl = True
        except BadZipFile:
            return False, None

        if prefix_code and prefix_data and prefix_constants_pkl and prefix_data_pkl:
            return True, cls.model_type

        return False, None
