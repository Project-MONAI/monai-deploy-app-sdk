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

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, Image, InputContext, IOType, Operator, OutputContext


@md.input("image", Image, IOType.IN_MEMORY)
@md.output("image", DataPath, IOType.DISK)
# If `pip_packages` is specified, the definition will be aggregated with the package dependency list of other
# operators and the application in packaging time.
# @md.env(pip_packages=["scikit-image >= 0.17.2"])
class GaussianOperator(Operator):
    """This Operator implements a smoothening based on Gaussian.

    It ingests a single input and provides a single output.
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        import numpy as np
        from skimage.filters import gaussian
        from skimage.io import imsave

        data_in = op_input.get().asnumpy()
        data_out = gaussian(data_in, sigma=0.2, channel_axis=2)  # Add the param introduced in 0.19.

        # Make sure the data type is what PIL Image can support, as the imsave function calls PIL Image fromarray()
        # Some details can be found at https://stackoverflow.com/questions/55319949/pil-typeerror-cannot-handle-this-data-type
        if np.max(data_out) <= 1:
            data_out = (data_out * 255).astype(np.uint8)

        output_folder = op_output.get().path
        output_path = output_folder / "final_output.png"
        imsave(output_path, data_out)
