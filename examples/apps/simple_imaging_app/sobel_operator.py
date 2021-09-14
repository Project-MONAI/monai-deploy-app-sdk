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

from monai.deploy.core import (
    DataPath,
    ExecutionContext,
    Image,
    InputContext,
    IOType,
    Operator,
    OutputContext,
    input,
    output,
)


@input("image", DataPath, IOType.DISK)
@output("image", Image, IOType.IN_MEMORY)
# If `pip_packages` is specified, the definition will be aggregated with the package dependency list of other
# operators and the application in packaging time.
# @env(pip_packages=["scikit-image >= 0.17.2"])
class SobelOperator(Operator):
    """This Operator implements a Sobel edge detector.

    It has a single input and single output.
    """

    def compute(self, input: InputContext, output: OutputContext, context: ExecutionContext):
        from skimage import filters, io

        input_path = input.get().path

        data_in = io.imread(input_path)[:, :, :3]  # discard alpha channel if exists
        data_out = filters.sobel(data_in)

        output.set(Image(data_out))
