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

from skimage.filters import median

from monai.deploy.core.base_operator import BaseOperator


class MedianOperator(BaseOperator):

    """This Operator implements a noise reduction.

    The algorithm is based on the median operator.
    It ingests a single input and provides a single output.
    """

    def __init__(self):
        super().__init__()

    def execute(self, execution_context):
        super().execute(execution_context)
        data_in = execution_context.get_operator_input(self._uid, 0)
        data_out = median(data_in)
        execution_context.set_operator_output(self._uid, 0, data_out)