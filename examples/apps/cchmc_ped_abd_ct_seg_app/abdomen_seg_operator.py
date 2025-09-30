# Copyright 2021-2025 MONAI Consortium
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
from typing import List

import torch
from numpy import float32, int16

# import custom transforms from post_transforms.py
from post_transforms import CalculateVolumeFromMaskd, ExtractVolumeToTextd, LabelToContourd, OverlayImageLabeld

import monai
from monai.deploy.core import AppContext, Fragment, Model, Operator, OperatorSpec
from monai.deploy.operators.monai_seg_inference_operator import InfererType, InMemImageReader, MonaiSegInferenceOperator
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    CastToTyped,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)


# this operator performs inference with the new version of the bundle
class AbdomenSegOperator(Operator):
    """Performs segmentation inference with a custom model architecture."""

    DEFAULT_OUTPUT_FOLDER = Path.cwd() / "output"

    def __init__(
        self,
        fragment: Fragment,
        *args,
        app_context: AppContext,
        model_path: Path,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
        output_labels: List[int],
        **kwargs,
    ):

        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"

        # self.model_path is compatible with TorchScript and PyTorch model workflows (pythonic and MAP)
        self.model_path = self._find_model_file_path(model_path)

        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.output_labels = output_labels
        self.app_context = app_context
        self.input_name_image = "image"
        self.output_name_seg = "seg_image"
        self.output_name_text_dicom_sr = "result_text_dicom_sr"
        self.output_name_text_mongodb = "result_text_mongodb"
        self.output_name_sc_path = "dicom_sc_dir"

        # the base class has an attribute called fragment to hold the reference to the fragment object
        super().__init__(fragment, *args, **kwargs)

    # find model path; supports TorchScript and PyTorch model workflows (pythonic and MAP)
    def _find_model_file_path(self, model_path: Path):
        # when executing pythonically, model_path is a file
        # when executing as MAP, model_path is a directory (/opt/holoscan/models)
        #   torch.load() from PyTorch workflow needs file path; can't load model from directory
        #   returns first found file in directory in this case
        if model_path:
            if model_path.is_file():
                return model_path
            elif model_path.is_dir():
                for file in model_path.rglob("*"):
                    if file.is_file():
                        return file

        raise ValueError(f"Model file not found in the provided path: {model_path}")

    # load a PyTorch model and register it in app_context
    def _load_pytorch_model(self):

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _kernel_size: tuple = (3, 3, 3, 3, 3, 3)
        _strides: tuple = (1, 2, 2, 2, 2, (2, 2, 1))
        _upsample_kernel_size: tuple = (2, 2, 2, 2, (2, 2, 1))

        # create DynUNet model with the specified architecture parameters + move to computational device (GPU or CPU)
        # parameters pulled from inference.yaml file of the MONAI bundle
        model = monai.networks.nets.dynunet.DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=4,
            kernel_size=_kernel_size,
            strides=_strides,
            upsample_kernel_size=_upsample_kernel_size,
            norm_name="INSTANCE",
            deep_supervision=False,
            res_block=True,
        ).to(_device)

        # load model state dictionary (i.e. mapping param names to tensors) via torch.load
        # weights_only=True to avoid arbitrary code execution during unpickling
        state_dict = torch.load(self.model_path, weights_only=True)

        # assign loaded weights to model architecture via load_state_dict
        model.load_state_dict(state_dict)

        # set model in evaluation (inference) mode
        model.eval()

        # create a MONAI Model object to encapsulate the PyTorch model and metadata
        loaded_model = Model(self.model_path, name="ped_abd_ct_seg")

        # assign loaded PyTorch model as the predictor for the Model object
        loaded_model.predictor = model

        # register the loaded Model object in the application context so other operators can access it
        # MonaiSegInferenceOperator uses _get_model method to load models; looks at app_context.models first
        self.app_context.models = loaded_model

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_image)

        # DICOM SEG
        spec.output(self.output_name_seg)

        # DICOM SR
        spec.output(self.output_name_text_dicom_sr)

        # MongoDB
        spec.output(self.output_name_text_mongodb)

        # DICOM SC
        spec.output(self.output_name_sc_path)

    def compute(self, op_input, op_output, context):
        input_image = op_input.receive(self.input_name_image)
        if not input_image:
            raise ValueError("Input image is not found.")

        # this operator gets an in-memory Image object, so a specialized ImageReader is needed.
        _reader = InMemImageReader(input_image)

        # preprocessing and postprocessing
        pre_transforms = self.pre_process(_reader)
        post_transforms = self.post_process(pre_transforms)

        # if PyTorch model
        if self.model_path.suffix.lower() == ".pt":
            # load the PyTorch model
            self._logger.info("PyTorch model detected")
            self._load_pytorch_model()
        # else, we have TorchScript model
        else:
            self._logger.info("TorchScript model detected")

        # delegates inference and saving output to the built-in operator.
        infer_operator = MonaiSegInferenceOperator(
            self.fragment,
            roi_size=(96, 96, 96),
            pre_transforms=pre_transforms,
            post_transforms=post_transforms,
            overlap=0.75,
            app_context=self.app_context,
            model_name="",
            inferer=InfererType.SLIDING_WINDOW,
            sw_batch_size=4,
            model_path=self.model_path,
            name="monai_seg_inference_op",
        )

        # setting the keys used in the dictionary-based transforms
        infer_operator.input_dataset_key = self._input_dataset_key
        infer_operator.pred_dataset_key = self._pred_dataset_key

        seg_image = infer_operator.compute_impl(input_image, context)

        # DICOM SEG
        op_output.emit(seg_image, self.output_name_seg)

        # grab result_text_dicom_sr and result_text_mongodb from ExractVolumeToTextd transform
        result_text_dicom_sr, result_text_mongodb = self.get_result_text_from_transforms(post_transforms)
        if not result_text_dicom_sr or not result_text_mongodb:
            raise ValueError("Result text could not be generated.")

        # only log volumes for target organs so logs reflect MAP behavior
        self._logger.info(f"Calculated Organ Volumes: {result_text_dicom_sr}")

        # DICOM SR
        op_output.emit(result_text_dicom_sr, self.output_name_text_dicom_sr)

        # MongoDB
        op_output.emit(result_text_mongodb, self.output_name_text_mongodb)

        # DICOM SC
        # temporary DICOM SC (w/o source DICOM metadata) saved in output_folder / temp directory
        dicom_sc_dir = self.output_folder / "temp"

        self._logger.info(f"Temporary DICOM SC saved at: {dicom_sc_dir}")

        op_output.emit(dicom_sc_dir, self.output_name_sc_path)

    def pre_process(self, img_reader) -> Compose:
        """Composes transforms for preprocessing the input image before predicting on a model."""

        my_key = self._input_dataset_key

        return Compose(
            [
                # img_reader: specialized InMemImageReader, derived from MONAI ImageReader
                LoadImaged(keys=my_key, reader=img_reader),
                EnsureChannelFirstd(keys=my_key),
                Orientationd(keys=my_key, axcodes="LPS"),
                Spacingd(keys=my_key, pixdim=[1.5, 1.5, 3.0], mode=["bilinear"]),
                ScaleIntensityRanged(keys=my_key, a_min=-250, a_max=400, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=my_key, source_key=my_key, mode="minimum"),
                EnsureTyped(keys=my_key),
                CastToTyped(keys=my_key, dtype=float32),
            ]
        )

    def post_process(self, pre_transforms: Compose) -> Compose:
        """Composes transforms for postprocessing the prediction results."""

        pred_key = self._pred_dataset_key

        labels = {"background": 0, "liver": 1, "spleen": 2, "pancreas": 3}

        return Compose(
            [
                Activationsd(keys=pred_key, softmax=True),
                Invertd(
                    keys=[pred_key, self._input_dataset_key],
                    transform=pre_transforms,
                    orig_keys=[self._input_dataset_key, self._input_dataset_key],
                    meta_key_postfix="meta_dict",
                    nearest_interp=[False, False],
                    to_tensor=True,
                ),
                AsDiscreted(keys=pred_key, argmax=True),
                # custom post-processing steps
                CalculateVolumeFromMaskd(keys=pred_key, label_names=labels),
                # optional code for saving segmentation masks as a NIfTI
                # SaveImaged(
                #     keys=pred_key,
                #     output_ext=".nii.gz",
                #     output_dir=self.output_folder / "NIfTI",
                #     meta_keys="pred_meta_dict",
                #     separate_folder=False,
                #     output_dtype=int16
                # ),
                # volume data stored in dictionary under pred_key + '_volumes' key
                ExtractVolumeToTextd(
                    keys=[pred_key + "_volumes"], label_names=labels, output_labels=self.output_labels
                ),
                # comment out LabelToContourd for seg masks instead of contours; organ filtering will be lost
                LabelToContourd(keys=pred_key, output_labels=self.output_labels),
                OverlayImageLabeld(image_key=self._input_dataset_key, label_key=pred_key, overlay_key="overlay"),
                SaveImaged(
                    keys="overlay",
                    output_ext=".dcm",
                    # save temporary DICOM SC (w/o source DICOM metadata) in output_folder / temp directory
                    output_dir=self.output_folder / "temp",
                    separate_folder=False,
                    output_dtype=int16,
                ),
            ]
        )

    # grab volume data from ExtractVolumeToTextd transform
    def get_result_text_from_transforms(self, post_transforms: Compose):
        """Extracts the result_text variables from post-processing transforms output."""

        # grab the result_text variables from ExractVolumeToTextd transfor
        for transform in post_transforms.transforms:
            if isinstance(transform, ExtractVolumeToTextd):
                return transform.result_text_dicom_sr, transform.result_text_mongodb
        return None
