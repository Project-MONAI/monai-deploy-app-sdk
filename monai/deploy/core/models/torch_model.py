# Copyright 2021-2025 MONAI Consortium
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
    the serialized model (which usually has .pt or .pth extension) is a ZIP archive.
    (https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md)

    We identify a file as a TorchScript model if its unzipped archive contains a 'code/' directory
    and a 'data.pkl' file. For tensor constants, it may contain either a 'constants.pkl' file (older format)
    or a 'constants/' directory (newer format).

    When predictor property is accessed or the object is called (__call__), the model is loaded in `evaluation mode`
    from the serialized model file (if it is not loaded yet) and the model is ready to be used.
    """

    model_type: str = "torch_script"

    @property
    def predictor(self) -> "torch.nn.Module":  # type: ignore
        """Get the model's predictor (torch.nn.Module)

        If the predictor is not loaded, load it from the model file in evaluation mode.
        (https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html?highlight=eval#torch.jit.ScriptModule.eval)

        Returns:
            torch.nn.Module: the model's predictor
        """
        if self._predictor is None:
            # Use a device to dynamically remap, depending on the GPU availability.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._predictor = torch.jit.load(self.path, map_location=device).eval()
        return self._predictor

    @predictor.setter
    def predictor(self, predictor: Any):
        self._predictor = predictor

    def eval(self) -> "TorchScriptModel":
        """Set the model in evaluation model.

        This is a proxy method for torch.jit.ScriptModule.eval().
        See https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html?highlight=eval#torch.jit.ScriptModule.eval

        Returns:
            self
        """
        self.predictor.eval()
        return self

    def train(self, mode: bool = True) -> "TorchScriptModel":
        """Set the model in training mode.

        This is a proxy method for torch.jit.ScriptModule.train().
        See https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html?highlight=train#torch.jit.ScriptModule.train

        Args:
            mode (bool): whether the model is in training mode

        Returns:
            self
        """
        self.predictor.train(mode)
        return self

    @classmethod
    def accept(cls, path: str):
        # These are the files and directories we expect to find in a TorchScript zip archive.
        # See: https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/docs/serialization.md
        has_code_dir = False
        has_constants_dir = False
        has_constants_pkl = False
        has_data_pkl = False

        if not os.path.isfile(path):
            return False, None

        try:
            with ZipFile(path) as zip_file:
                # Top-level directory name in the zip file (e.g., 'model_name/')
                top_level_dir = ""
                if "/" in zip_file.filelist[0].filename:
                    top_level_dir = zip_file.filelist[0].filename.split("/", 1)[0] + "/"

                filenames = {f.filename for f in zip_file.filelist}

                # Check for required files and directories
                has_data_pkl = (top_level_dir + "data.pkl") in filenames
                has_code_dir = any(f.startswith(top_level_dir + "code/") for f in filenames)

                # Check for either constants.pkl (older format) or constants/ (newer format)
                has_constants_pkl = (top_level_dir + "constants.pkl") in filenames
                has_constants_dir = any(f.startswith(top_level_dir + "constants/") for f in filenames)

        except (BadZipFile, IndexError):
            return False, None

        # A valid TorchScript model must have code/, data.pkl, and either constants.pkl or constants/
        if has_code_dir and has_data_pkl and (has_constants_pkl or has_constants_dir):
            return True, cls.model_type

        return False, None
