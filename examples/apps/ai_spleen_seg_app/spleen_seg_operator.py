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

from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, Operator, OutputContext, env, input, output
from monai.deploy.core.domain.monai_types import ComposeInterface
from monai.deploy.operators.monai_seg_inference_operator import InMemImageReader, MonaiSegInferenceOperator
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)


@input("image", Image, IOType.IN_MEMORY)
@output("seg_image", Image, IOType.IN_MEMORY)
@env(pip_packages=["monai==0.6.0", "torch>=1.5", "numpy>=1.17", "nibabel"])
class SpleenSegOperator(Operator):
    """Performs Spleen segmentation with 3D image converted from a DICOM CT series.

    This operator makes use of the App SDK MonaiSegInferenceOperator in a compsition approach.
    It creates the pre-transforms as well as post-transforms with Monai dictionary based transforms.
    Note that the App SDK InMemImageReader, derived from Monai ImageReader, is passed to LoadImaged.
    This derived reader is needed to parse the in memory image object, and return the expected data structure.
    Loading of the model, and predicting using in-proc PyTorch inference is done by MonaiSegInferenceOperator.
    """

    def __init__(self, testing: bool = False):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()
        self.testing = testing
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"

    def compute(self, input: InputContext, output: OutputContext, context: ExecutionContext):

        input_image = input.get("image")
        if not input_image:
            raise ValueError("Input image is not found.")

        if self.testing:
            seg_image = self.infer_and_save(input_image)
            output.set(seg_image, "seg_image")
        else:
            # This operator gets an in-memory Image object, so a specialized ImageReader is needed.
            self._reader = InMemImageReader(input_image)
            pre_transforms = self.pre_process(self._reader)
            post_transforms = self.post_process(pre_transforms)

            # Delegates inference and saving output to the built-in operator.
            infer_operator = MonaiSegInferenceOperator(
                (
                    160,
                    160,
                    160,
                ),
                pre_transforms,
                post_transforms,
            )

            # Setting the keys used in the dictironary based transforms may change.
            infer_operator.input_dataset_key = self._input_dataset_key
            infer_operator.pred_dataset_key = self._pred_dataset_key

            # Now let the built-in operator handles the work with the I/O spec and execution context.
            infer_operator.compute(input, output, context)

    def pre_process(self, img_reader) -> ComposeInterface:
        """Composes transforms for preprocessing input before predicting on a model."""

        my_key = self._input_dataset_key
        return Compose(
            [
                LoadImaged(keys=my_key, reader=img_reader),
                EnsureChannelFirstd(keys=my_key),
                Spacingd(keys=my_key, pixdim=[1.0, 1.0, 1.0], mode=["blinear"], align_corners=True),
                ScaleIntensityRanged(keys=my_key, a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=my_key, source_key=my_key),
                ToTensord(keys=my_key),
            ]
        )

    def post_process(self, pre_transforms: ComposeInterface, out_dir: str = "./infer_output") -> ComposeInterface:
        """Composes transforms for postprocessing the prediction results."""

        pred_key = self._pred_dataset_key
        return Compose(
            [
                Activationsd(keys=pred_key, softmax=True),
                AsDiscreted(keys=pred_key, argmax=True),
                Invertd(
                    keys=pred_key, transform=pre_transforms, orig_keys=self._input_dataset_key, nearest_interp=True
                ),
                SaveImaged(keys=pred_key, output_dir=out_dir, output_postfix="seg", resample=False),
            ]
        )

    def infer_and_save(self, image):
        """Prints out the image obj, and bounce it back, for testing only."""

        image_data = image.asnumpy()
        image_shape = image_data.shape
        print(image_shape)
        print(vars(image))

        # Dummy for now.
        return image
