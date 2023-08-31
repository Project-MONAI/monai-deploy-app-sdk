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

from pathlib import Path

from monai.deploy.core import Fragment, Operator, OperatorSpec


# # If `pip_packages` is specified, the definition will be aggregated with the package dependency list of other
# # operators and the application in packaging time.
# # @md.env(pip_packages=["scikit-image >= 0.17.2"])
class SobelOperator(Operator):
    """This Operator implements a Sobel edge detector.

    It has the following input and output:
        single input:
          a image file, first one found in the input folder
        single output:
          array object in memory
    """

    DEFAULT_INPUT_FOLDER = Path.cwd() / "input"

    def __init__(self, fragment: Fragment, *args, input_path: Path, **kwargs):
        """Create an instance to be part of the given application (fragment).

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment
            input_path (Path): The path of the input image file or folder containing the image file
        """
        self.index = 0

        # May want to validate the path, but it should really be validated when the compute function is called, also,
        # when file path as input is supported in the operator or execution context, input_folder needs not an attribute.
        self.input_path = input_path if input_path else SobelOperator.DEFAULT_INPUT_FOLDER

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out1")

    def compute(self, op_input, op_output, context):
        from skimage import filters, io

        self.index += 1
        print(f"Number of times operator {self.name} whose class is defined in {__name__} called: {self.index}")

        # Ideally the op_input or execution context should provide the file path
        # to read data from, for operators that are file input based.
        # For now, use a temporary way to get input path. e.g. value set on init
        input_path = self.input_path
        print(f"Input from: {input_path}, whose absolute path: {input_path.absolute()}")
        if input_path.is_dir():
            input_path = next(input_path.glob("*.*"))  # take the first file

        data_in = io.imread(input_path)[:, :, :3]  # discard alpha channel if exists
        data_out = filters.sobel(data_in)

        op_output.emit(data_out, "out1")
