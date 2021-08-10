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

from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, Operator, OutputContext, input, output


@input("image", Image, IOType.IN_MEMORY)
@output("image", Image, IOType.IN_MEMORY)
# If `pip_packages` is not specified, the operator will use a default package dependency list (requirements.txt)
# in the same directory (or a parent directory up to App folder, if not exists) of the operator.
# If `pip_packages` is specified, the definition will be aggregated with the package dependency list of other
# operators and the application in packaging time.
# @env(pip_packages=["scikit-image >= 0.18.0"])
class MedianOperator(Operator):
    """This Operator implements a noise reduction.

    The algorithm is based on the median operator.
    It ingests a single input and provides a single output.
    """

    def compute(self, input: InputContext, output: OutputContext, context: ExecutionContext):
        from skimage.filters import median

        # model = context.models.get()  # PyTorchModel object that inherits Model class
        # # model.name for accessing the model's name
        # # model.path for accessing the model's path
        # result = model.infer(input.get().asnumpy())

        data_in = input.get().asnumpy()
        data_out = median(data_in)
        output.set(Image(data_out))
