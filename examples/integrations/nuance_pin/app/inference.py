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

import torch

import monai.deploy.core as md
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.transforms.dictionary import AffineBoxToWorldCoordinated, ClipBoxToImaged, ConvertBoxModed
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, Operator, OutputContext
from monai.deploy.operators.monai_seg_inference_operator import InMemImageReader
from monai.deploy.utils.importutil import optional_import
from monai.transforms import (
    AddChanneld,
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.transforms.utility.dictionary import ToDeviced, ToTensord

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
        self._pred_label = "box_label"
        self._pred_score = "box_score"
        self._pred_labels = "labels"

        # preload the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"Loading TorchScript model from: {model_path}")
        self.model = torch.jit.load(model_path, map_location=self.device)

        self.logger.info("Loading model into RetinaNetDetector")
        self.returned_layers = [1, 2]
        self.base_achor_shapes = [[6, 8, 4], [8, 6, 5], [10, 10, 6]]
        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=[2**l for l in range(len(self.returned_layers) + 1)],
            base_anchor_shapes=self.base_achor_shapes,
        )
        self.detector = RetinaNetDetector(
            network=self.model,
            anchor_generator=anchor_generator,
        )
        self.detector.set_box_selector_parameters(
            score_thresh=0.02,
            topk_candidates_per_level=1000,
            nms_thresh=0.22,
            detections_per_img=100,
        )
        self.detector.set_sliding_window_inferer(
            roi_size=[192, 192, 80],
            overlap=0.25,
            sw_batch_size=1,
            mode="gaussian",
            device="cpu",
        )
        self.detector.eval()

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

            inference_outputs = self.detector(processed_image[self._input_dataset_key], use_inferer=True)

            pred_boxes = []
            processed_image[self._input_dataset_key] = torch.squeeze(processed_image[self._input_dataset_key], dim=0)
            for inference_output in inference_outputs:
                processed_image[self._pred_box_regression] = inference_output[self.detector.target_box_key]
                processed_image[self._pred_labels] = inference_output[self.detector.target_label_key]
                processed_image[self._pred_score] = inference_output[self.detector.pred_score_key]

                pred_boxes.append(self.post_process()(processed_image))

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

        return Compose(
            [
                ClipBoxToImaged(
                    box_keys=self._pred_box_regression,
                    box_ref_image_keys=self._input_dataset_key,
                    label_keys=[self._pred_labels, self._pred_score]
                ),
                AffineBoxToWorldCoordinated(
                    box_keys=self._pred_box_regression,
                    box_ref_image_keys=self._input_dataset_key,
                    affine_lps_to_ras=True,
                ),
                ConvertBoxModed(
                    box_keys=self._pred_box_regression,
                    src_mode="xyzxyz",
                    dst_mode="cccwhd"
                ),
                DeleteItemsd(
                    keys=self._input_dataset_key
                )
            ]
        )
