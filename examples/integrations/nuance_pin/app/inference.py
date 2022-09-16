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

from monai.deploy.operators.monai_seg_inference_operator import InMemImageReader
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureTyped,
    Orientationd,
    ScaleIntensityRanged,
    AddChanneld,
    Spacingd,
    EnsureChannelFirstd,
    LoadImaged,
)
from monai.transforms.utility.dictionary import ToDeviced, ToTensord
from monai.deploy.utils.importutil import optional_import
import logging

import torch

import monai.deploy.core as md
from monai.apps.detection.transforms.dictionary import AffineBoxToWorldCoordinated, ClipBoxToImaged, ConvertBoxModed
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, OutputContext, Operator

sliding_window_inference, _ = optional_import("monai.inferers", name="sliding_window_inference")


@md.input("image", Image, IOType.IN_MEMORY)
@md.output("boxes", Image, IOType.IN_MEMORY)
@md.env(pip_packages=["monai>=0.8.1", "torch>=1.5", "numpy>=1.21", "nibabel"])
class CovidDetectionInferenceOperator(Operator):

    def __init__(self, model_path: str = "model/model.ts"):

        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()
        self._input_dataset_key = "image"
        self._pred_box_regression = "box_regression"
        self._pred_classification = "box_classification"
        self._pred_labels = "labels"

        # preload the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading TorchScript model from: {model_path}")
        self.model = torch.jit.load(model_path, map_location=self.device)

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        input_image = op_input.get("image")
        if not input_image:
            raise ValueError("Input image not found.")

        image_reader = InMemImageReader(input_image)

        with torch.no_grad():
            processed_image = self.pre_process(image_reader)(
                (
                    {
                        self._input_dataset_key: input_image.metadata().get("SeriesInstanceUID", "Img_in_context"),
                    },
                    image_reader,
                )
            )
            pred_boxes = self.model(processed_image[self._input_dataset_key])
            # processed_image[self._pred_box_regression] = pred_boxes['box_regression']
            # processed_image[self._pred_classification] = pred_boxes['classification']
            # processed_image[self._pred_labels] = ('box_regression', 'classification')
            # pred_boxes = self.post_process()(
            #     [
            #         {
            #             self._pred_box_regression: pred_boxes['box_regression'][i],
            #             self._pred_classification: pred_boxes['classification'][i],
            #             self._input_dataset_key: processed_image['image'],
            #             self._pred_labels: processed_image[self._pred_labels],
            #         } for i in range(len(pred_boxes))
            #     ]
            # )
            # pred_boxes = self.post_process()(processed_image)

            print(f"Output array shaped: {pred_boxes.shape}")
            op_output.set(pred_boxes, "boxes")

    def pre_process(self, img_reader) -> Compose:
        """Composes transforms for preprocessing input before predicting on a model."""

        image_key = self._input_dataset_key
        return Compose(
            [
                LoadImaged(
                    keys=image_key,
                    reader=img_reader,
                ),
                ToTensord(
                    keys=image_key,
                ),
                ToDeviced(
                    keys=image_key,
                    device='cuda'
                ),
                EnsureChannelFirstd(keys=image_key),
                Orientationd(
                    keys=image_key,
                    axcodes="RAS",
                ),
                AddChanneld(keys=image_key),
                Spacingd(
                    keys=image_key,
                    pixdim=(0.703125, 0.703125, 1.25)
                ),
                ScaleIntensityRanged(
                    image_key,
                    a_min=-1024.0,
                    a_max=300.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                ),
                EnsureTyped(image_key),
            ],
            unpack_items=True,
            map_items=False,
        )

    def post_process(self) -> Compose:
        """Composes transforms for postprocessing the prediction results."""

        image_key = self._input_dataset_key
        pred_key = self._pred_box_regression
        label_key = self._pred_labels
        return Compose(
            [
                ClipBoxToImaged(
                    box_keys=pred_key,
                    box_ref_image_keys=image_key,
                    label_keys=label_key
                ),
                AffineBoxToWorldCoordinated(
                    box_keys=pred_key,
                    box_ref_image_keys=image_key,
                    affine_lps_to_ras=True,
                ),
                ConvertBoxModed(
                    box_keys=pred_key,
                    src_mode="xyzxyz",
                    dst_mode="cccwhd"
                ),
                DeleteItemsd(
                    keys=self._input_dataset_key
                )
            ]
        )
