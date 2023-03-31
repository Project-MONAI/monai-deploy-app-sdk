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

import os
from pathlib import Path

from monai.deploy.core import Operator, OperatorSpec


# @md.input("image", DataPath, IOType.DISK)
# @md.output("image", Image, IOType.IN_MEMORY)
# # If `pip_packages` is specified, the definition will be aggregated with the package dependency list of other
# # operators and the application in packaging time.
# # @md.env(pip_packages=["scikit-image >= 0.17.2"])
class SobelOperator(Operator):
    """This Operator implements a Sobel edge detector.

    It has a single input and single output.
    """

    DEFAULT_INPUT_FOLDER = Path(os.path.join(os.path.dirname(__file__))) / "input"

    def __init__(self, *args, input_folder: Path, **kwargs):
        # TODO: what is this for? Many examples use this. Related to CountConditions?
        self.index = 0

        # May want to validate the path, but should really be validate when the compute
        # is called as the input path can set for each compute call
        self.input_folder = (
            input_folder if (input_folder and input_folder.is_dir()) else SobelOperator.DEFAULT_INPUT_FOLDER
        )

        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out1")
        # spec.param("input_folder", Path("."))

    def compute(self, op_input, op_output, context):
        from skimage import filters, io

        self.index += 1
        print(f"# of times pperator {__name__} called: {self.index}")

        # TODO: Ideally the op_input or execution context should provide the file path
        # to read data from, for operators that are File input based.
        #
        # Need to use a temporary way to get input path. e.g. value set on init
        input_folder = self.input_folder  # op_input.get().path
        print(f"Input from: {input_folder}")
        if input_folder.is_dir():
            input_file = next(input_folder.glob("*.*"))  # take the first file

        data_in = io.imread(input_file)[:, :, :3]  # discard alpha channel if exists
        data_out = filters.sobel(data_in)

        op_output.emit(data_out, "out1")
