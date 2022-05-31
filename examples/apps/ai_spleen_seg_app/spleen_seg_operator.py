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

import logging
from os import path

from numpy import uint8

import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, Operator, OutputContext
from monai.deploy.operators.monai_bundle_inference_operator import MonaiBundleInferenceOperator


@md.input("image", Image, IOType.IN_MEMORY)
@md.output("seg_image", Image, IOType.IN_MEMORY)
@md.env(pip_packages=["monai>=0.8.1", "torch>=1.10.2", "numpy>=1.21", "nibabel"])
class SpleenSegOperator(Operator):
    """Performs Spleen segmentation with a 3D image converted from a DICOM CT series.

    This operator makes use of the App SDK MonaiBundleInferenceOperator in a composition approach.
    Parsing of the bundle, transforms and inference are done in the MonaiBundleInferenceOperator.

    Single named input of Image type and single named output of Image type are supported here.
    Mapping of the I/O of the operator to the "keys" in MONAI transforms is not supported yet.
    """

    def __init__(self, model_name:str = "", *args, **kwargs):

        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__(*args, **kwargs)
        self._model_name = model_name

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        input_image = op_input.get("image")
        if not input_image:
            raise ValueError("Input image is not found.")

        # Delegates inference and saving output to the built-in operator.
        infer_operator = MonaiBundleInferenceOperator(self._model_name)

        # Now let the built-in operator handles the work with the I/O spec and execution context.
        infer_operator.compute(op_input, op_output, context)
