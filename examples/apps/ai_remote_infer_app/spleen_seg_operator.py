# Copyright 2025 MONAI Consortium
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
from pathlib import Path

from numpy import uint8

from monai.deploy.core import AppContext, ConditionType, Fragment, Operator, OperatorSpec
from monai.deploy.operators.monai_seg_inference_operator import InfererType, InMemImageReader, MonaiSegInferenceOperator
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)


class SpleenSegOperator(Operator):
    """Performs Spleen segmentation with a 3D image converted from a DICOM CT series."""

    DEFAULT_OUTPUT_FOLDER = Path.cwd() / "output/saved_images_folder"

    def __init__(
        self,
        fragment: Fragment,
        *args,
        app_context: AppContext,
        model_path: Path,
        model_name: str,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
        **kwargs,
    ):

        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"

        self.model_path = model_path
        self.model_name = model_name
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.app_context = app_context
        self.input_name_image = "image"
        self.output_name_seg = "seg_image"
        self.output_name_saved_images_folder = "saved_images_folder"

        # The base class has an attribute called fragment to hold the reference to the fragment object
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_image)
        spec.output(self.output_name_seg)
        spec.output(self.output_name_saved_images_folder).condition(
            ConditionType.NONE
        )  # Output not requiring a receiver

    def compute(self, op_input, op_output, context):
        input_image = op_input.receive(self.input_name_image)
        if not input_image:
            raise ValueError("Input image is not found.")

        # This operator gets an in-memory Image object, so a specialized ImageReader is needed.
        _reader = InMemImageReader(input_image)

        pre_transforms = self.pre_process(_reader, str(self.output_folder))
        post_transforms = self.post_process(pre_transforms, str(self.output_folder))

        # Delegates inference and saving output to the built-in operator.
        infer_operator = MonaiSegInferenceOperator(
            self.fragment,
            roi_size=(
                96,
                96,
                96,
            ),
            pre_transforms=pre_transforms,
            post_transforms=post_transforms,
            overlap=0.6,
            app_context=self.app_context,
            model_name=self.model_name,
            inferer=InfererType.SLIDING_WINDOW,
            sw_batch_size=4,
            model_path=self.model_path,
            name="monai_seg_remote_inference_op",
        )

        # Setting the keys used in the dictionary based transforms may change.
        infer_operator.input_dataset_key = self._input_dataset_key
        infer_operator.pred_dataset_key = self._pred_dataset_key

        # Now emit data to the output ports of this operator
        op_output.emit(infer_operator.compute_impl(input_image, context), self.output_name_seg)
        op_output.emit(self.output_folder, self.output_name_saved_images_folder)

    def pre_process(self, img_reader, out_dir: str = "./input_images") -> Compose:
        """Composes transforms for preprocessing input before predicting on a model."""

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        my_key = self._input_dataset_key

        return Compose(
            [
                LoadImaged(keys=my_key, reader=img_reader),
                EnsureChannelFirstd(keys=my_key),
                # The SaveImaged transform can be commented out to save 5 seconds.
                # Uncompress NIfTI file, nii, is used favoring speed over size, but can be changed to nii.gz
                SaveImaged(
                    keys=my_key,
                    output_dir=out_dir,
                    output_postfix="",
                    resample=False,
                    output_ext=".nii",
                ),
                Orientationd(keys=my_key, axcodes="RAS"),
                Spacingd(keys=my_key, pixdim=[1.5, 1.5, 2.9], mode=["bilinear"]),
                ScaleIntensityRanged(keys=my_key, a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
                EnsureTyped(keys=my_key),
            ]
        )

    def post_process(self, pre_transforms: Compose, out_dir: str = "./prediction_output") -> Compose:
        """Composes transforms for postprocessing the prediction results."""

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        pred_key = self._pred_dataset_key

        return Compose(
            [
                Activationsd(keys=pred_key, softmax=True),
                Invertd(
                    keys=pred_key,
                    transform=pre_transforms,
                    orig_keys=self._input_dataset_key,
                    nearest_interp=False,
                    to_tensor=True,
                ),
                AsDiscreted(keys=pred_key, argmax=True),
                # The SaveImaged transform can be commented out to save 5 seconds.
                # Uncompress NIfTI file, nii, is used favoring speed over size, but can be changed to nii.gz
                SaveImaged(
                    keys=pred_key,
                    output_dir=out_dir,
                    output_postfix="seg",
                    output_dtype=uint8,
                    resample=False,
                    output_ext=".nii",
                ),
            ]
        )
