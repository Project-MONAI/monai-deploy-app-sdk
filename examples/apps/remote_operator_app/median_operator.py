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
from tempfile import mkdtemp

import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, OutputContext
from monai.deploy.operators.remotes.docker_operator import RemoteDockerOperator


@md.input("image", Image, IOType.IN_MEMORY)
@md.output("image", Image, IOType.IN_MEMORY)
class MedianOperator(RemoteDockerOperator):
    """This operator is a subclass of the base operator to demonstrate the usage of inheritance."""

    def to_input_folder(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        from skimage.io import imsave

        data_in = op_input.get().asnumpy()

        input_folder = Path(mkdtemp(dir=context.workdir))
        imsave(input_folder / "input.png", data_in)

        return input_folder

    def to_output_folder(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        # Return the output folder
        return Path(".")

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        from skimage.io import imread

        super().compute(op_input, op_output, context)

        output_folder = self.to_output_folder(op_input, op_output, context)

        output_path = output_folder
        if output_path.is_dir():
            output_path = next(output_path.glob("*.*"))  # take the first file

        data_out = imread(output_path)
        op_output.set(Image(data_out))
