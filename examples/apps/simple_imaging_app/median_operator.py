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

from monai.deploy.core import Fragment, Operator, OperatorSpec


# If `pip_packages` is specified, the definition will be aggregated with the package dependency list of other
# operators and the application in packaging time.
# @md.env(pip_packages=["scikit-image >= 0.17.2"])
class MedianOperator(Operator):
    """This Operator implements a noise reduction.

    The algorithm is based on the median operator.
    It ingests a single input and provides a single output, both are in-memory image arrays
    """

    # Define __init__ method with super().__init__() if you want to override the default behavior.
    def __init__(self, fragment: Fragment, *args, **kwargs):
        """Create an instance to be part of the given application (fragment).

        Args:
            fragment (Fragment): The instance of Application class which is derived from Fragment
        """

        self.index = 0

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in1")
        spec.output("out1")

    def compute(self, op_input, op_output, context):
        from skimage.filters import median

        self.index += 1
        print(f"Number of times operator {self.name} whose class is defined in {__name__} called: {self.index}")
        data_in = op_input.receive("in1")
        data_out = median(data_in)
        op_output.emit(data_out, "out1")
