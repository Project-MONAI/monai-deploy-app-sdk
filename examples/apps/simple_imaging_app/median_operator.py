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

from monai.deploy.core import ExecutionContext, Image, IOType, Operator, input, output


@input("image", Image, IOType.IN_MEMORY)
@output("image", Image, IOType.IN_MEMORY)
class MedianOperator(Operator):
    """This Operator implements a noise reduction.

    The algorithm is based on the median operator.
    It ingests a single input and provides a single output.
    """

    def execute(self, context: ExecutionContext):
        from skimage.filters import median

        data_in = context.get_input("image").asnumpy()
        data_out = median(data_in)
        context.set_output(Image(data_out))
