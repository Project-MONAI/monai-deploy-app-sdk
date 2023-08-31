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

from monai.deploy.core import ConditionType, Fragment, Operator, OperatorSpec


# If `pip_packages` is specified, the definition will be aggregated with the package dependency list of other
# operators and the application in packaging time.
# @md.env(pip_packages=["scikit-image >= 0.17.2"])
class GaussianOperator(Operator):
    """This Operator implements a smoothening based on Gaussian.

    It has the following input and output:
        single input:
          an image array object
        single output:
          an image arrary object, without enforcing a downsteam receiver

    Besides, this operator also saves the image file in the given output folder.
    """

    DEFAULT_OUTPUT_FOLDER = Path.cwd() / "output"

    def __init__(self, fragment: Fragment, *args, output_folder: Path, **kwargs):
        """Create an instance to be part of the given application (fragment).

        Args:
            fragment (Fragment): The instance of Application class which is derived from Fragment
            output_folder (Path): The folder to save the output file.
        """
        self.output_folder = output_folder if output_folder else GaussianOperator.DEFAULT_OUTPUT_FOLDER
        self.index = 0

        # If `self.sigma_default` is set here (e.g., `self.sigma_default = 0.2`), then
        # the default value by `param()` in `setup()` will be ignored.
        # (you can just call `spec.param("sigma_default")` in `setup()` to use the
        # default value)
        self.sigma_default = 0.2
        self.channel_axis = 2

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in1")
        spec.output("out1").condition(ConditionType.NONE)  # Condition is for no or not-ready receiver ports.
        spec.param("sigma_default", 0.2)
        spec.param("channel_axis", 2)

    def compute(self, op_input, op_output, context):
        import numpy as np
        from skimage.filters import gaussian
        from skimage.io import imsave

        self.index += 1
        print(f"Number of times operator {self.name} whose class is defined in {__name__} called: {self.index}")

        data_in = op_input.receive("in1")
        data_out = gaussian(data_in, sigma=self.sigma_default, channel_axis=self.channel_axis)

        # Make sure the data type is what PIL Image can support, as the imsave function calls PIL Image fromarray()
        # Some details can be found at https://stackoverflow.com/questions/55319949/pil-typeerror-cannot-handle-this-data-type
        print(f"Data type of output: {type(data_out)!r}, max = {np.max(data_out)!r}")
        if np.max(data_out) <= 1:
            data_out = (data_out * 255).astype(np.uint8)
        print(f"Data type of output post conversion: {type(data_out)!r}, max = {np.max(data_out)!r}")

        # For now, use attribute of self to find the output path.
        self.output_folder.mkdir(parents=True, exist_ok=True)
        output_path = self.output_folder / "final_output.png"
        imsave(output_path, data_out)

        op_output.emit(data_out, "out1")
