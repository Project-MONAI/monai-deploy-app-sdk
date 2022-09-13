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

import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, Operator, OutputContext
from monai.deploy.operators.monai_seg_inference_operator import InMemImageReader, MonaiSegInferenceOperator
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    Compose,
    CropForegroundd,
    Invertd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)


@md.input("image", Image, IOType.IN_MEMORY)
@md.output("seg_image", Image, IOType.IN_MEMORY)
@md.env(pip_packages=["monai>=0.8.1", "torch>=1.5", "numpy>=1.21", "nibabel"])
class LungNoduleSegOperator(Operator):

    def __init__(self):

        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        input_image = op_input.get("image")
        if not input_image:
            raise ValueError("Input image is not found.")

        _reader = InMemImageReader(input_image)
        pre_transforms = self.pre_process(_reader)
        post_transforms = self.post_process(pre_transforms)

        # Delegates inference and saving output to the built-in operator.
        infer_operator = MonaiSegInferenceOperator(
            (224, 224, 32),
            pre_transforms,
            post_transforms,
        )

        # Setting the keys used in the dictironary based transforms may change.
        infer_operator.input_dataset_key = self._input_dataset_key
        infer_operator.pred_dataset_key = self._pred_dataset_key

        # Now let the built-in operator handles the work with the I/O spec and execution context.
        infer_operator.compute(op_input, op_output, context)

    def pre_process(self, img_reader) -> Compose:
        """Composes transforms for preprocessing input before predicting on a model."""

        my_key = self._input_dataset_key
        return Compose(
            [
                LoadImaged(keys=my_key, reader=img_reader),
                AddChanneld(keys=my_key),
                Spacingd(keys=my_key, pixdim=(0.8, 0.8, 5.0), mode=("bilinear"), align_corners=[True]),
                Orientationd(keys=my_key, axcodes="RAS"),
                ScaleIntensityRanged(my_key, a_min=-1000, a_max=500, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(my_key, source_key=my_key),
                ToTensord(my_key),
            ]
        )

    def post_process(self, pre_transforms: Compose) -> Compose:
        """Composes transforms for postprocessing the prediction results."""

        pred_key = self._pred_dataset_key
        return Compose(
            [
                Activationsd(keys=pred_key, softmax=True),
                Invertd(
                    keys=pred_key, transform=pre_transforms, orig_keys=self._input_dataset_key, nearest_interp=False
                ),
                AsDiscreted(keys=pred_key, argmax=True),
            ]
        )
