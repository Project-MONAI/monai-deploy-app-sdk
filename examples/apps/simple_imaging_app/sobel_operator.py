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

from monai.deploy.core import Blob, ExecutionContext, Image, IOType, Operator, input, output


@input("image", Blob, IOType.DISK)
@output("image", Image, IOType.IN_MEMORY)
class SobelOperator(Operator):
    """This Operator implements a Sobel edge detector.

    It has a single input and single output.
    """

    def execute(self, context: ExecutionContext):
        """Performs execution for this operator.

        The input for this operator is hardcoded.
        In near future this will be changed where the input is provided via
        inversion of control mecchanism.
        """
        import pathlib

        from skimage import filters, io

        data_in = io.imread(pathlib.Path(__file__).parent.resolve() / "brain_mr_input.jpg")
        data_out = filters.sobel(data_in)

        context.set_output(Image(data_out), "image")
