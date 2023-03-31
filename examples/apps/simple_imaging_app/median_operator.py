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

from skimage.filters import median

from monai.deploy.core import Operator, OperatorSpec


# If `pip_packages` is specified, the definition will be aggregated with the package dependency list of other
# operators and the application in packaging time.
# @md.env(pip_packages=["scikit-image >= 0.17.2"])
class MedianOperator(Operator):
    """This Operator implements a noise reduction.

    The algorithm is based on the median operator.
    It ingests a single input and provides a single output.
    """

    # Define __init__ method with super().__init__() if you want to override the default behavior.
    def __init__(self, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)
        self.index = 0

    def setup(self, spec: OperatorSpec):
        spec.input("in1")
        spec.output("out1")

    def compute(self, op_input, op_output, context):
        self.index += 1
        print(f"# of times pperator {__name__} called: {self.index}")
        data_in = op_input.receive("in1")
        data_out = median(data_in)
        op_output.emit(data_out, "out1")  # TODO: change this to use Image class
