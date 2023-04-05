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

from skimage.filters import gaussian
from skimage.io import imsave

from monai.deploy.core import Operator, OperatorSpec


# If `pip_packages` is specified, the definition will be aggregated with the package dependency list of other
# operators and the application in packaging time.
# @md.env(pip_packages=["scikit-image >= 0.17.2"])
class GaussianOperator(Operator):
    """This Operator implements a smoothening based on Gaussian.

    It ingests a single input, an array, but emits none, instead it saves a file in the output folder.
    """

    DEFAULT_OUTPUT_FOLDER = Path.cwd() / "output"

    def __init__(self, *args, output_folder: Path, **kwargs):
        # If `self.sigma_default` is set here (e.g., `self.sigma_default = 0.2`), then
        # the default value by `param()` in `setup()` will be ignored.
        # (you can just call `spec.param("sigma_default")` in `setup()` to use the
        # default value)
        self.output_folder = output_folder if output_folder else GaussianOperator.DEFAULT_OUTPUT_FOLDER
        self.index = 0

        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in1")
        # spec.output("out1")  # Cannot have a output without having downstream receiver
        spec.param("sigma_default", 0.2)
        spec.param("channel_axis", 2)

    def compute(self, op_input, op_output, context):
        self.index += 1
        print(f"Number of times operator {self.name} whose class is defined in {__name__} called: {self.index}")

        data_in = op_input.receive("in1")
        data_out = gaussian(data_in, sigma=self.sigma_default, channel_axis=self.channel_axis)

        # For now, use attribute of self to find the output path.
        self.output_folder.mkdir(parents=True, exist_ok=True)
        output_path = self.output_folder / "final_output.png"
        imsave(output_path, data_out)

        # op_input.emit(data_out, "out1")  # CANNOT emit a dangling output!!!
