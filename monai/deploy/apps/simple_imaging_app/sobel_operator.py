# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from skimage import filters, io
import pathlib

from monai.deploy.core.base_operator import BaseOperator


class SobelOperator(BaseOperator):
    """This Operator implements a Sobel edge detector.

    It has a single input and single output.
    """

    def __init__(self):
        super().__init__()

    def execute(self, execution_context):
        """Performs execution for this operator.

        The input for this operator is hardcoded.
        In near future this will be changed where the input is provided via
        inversion of control mecchanism.
        """
        super().execute(execution_context)
        data_in = io.imread(pathlib.Path(__file__).parent.resolve() / "brain_mr_input.jpg")
        data_out = filters.sobel(data_in)
        execution_context.set_operator_output(self._uid, 0, data_out)
