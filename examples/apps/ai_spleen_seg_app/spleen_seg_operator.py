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

from typing import Any, Dict, Optional, Union
from monai.data import image_reader

from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Spacingd,
    Orientationd,
    Resized,
    SaveImaged,
    ScaleIntensityRanged,
    CropForegroundd,
    ToTensord,
    EnsureTyped,
)

from monai.deploy.core import (
    DataPath,
    ExecutionContext,
    Image,
    InputContext,
    IOType,
    Operator,
    OutputContext,
    input,
    output,
)

from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.monai_seg_inference_operator import MonaiSegInferenceOperator, InMemImageReader

from monai.deploy.core.domain.image import Image

from os import listdir
from os.path import isfile, join

import numpy as np

@input("image", Image, IOType.IN_MEMORY)
@output("seg_image", Image, IOType.IN_MEMORY)
class SpleenSegOperator(Operator):
    """
    This operator writes out a 3D Volumtric Image to disk in a slice by slice manner
    """
    def __init__(self, testing:bool =False):

        super().__init__()
        self.testing = testing
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"

    def compute(self, input: InputContext, output: OutputContext, context: ExecutionContext):

        input_image = input.get("image")
        if not input_image:
            raise ValueError('Input image is not found.')

        if self.testing:
            seg_image = self.infer_and_save(input_image)
            output.set(seg_image, "seg_image")
        else:
            self._reader = InMemImageReader(input_image)
            pre_transforms = self.compose_pre_transforms(self._reader)
            post_transforms = self.compose_post_transforms(pre_transforms)
            # Delegate inference and saving output to the built-in operator
            infer_operator = MonaiSegInferenceOperator([160, 160, 160], pre_transforms, post_transforms)
            infer_operator.compute(input, output, context)

    def compose_pre_transforms(self, img_reader) -> Compose:
        """Compose transforms for preprocessing input before predicting on a model.
        """

        my_key = self._input_dataset_key
        return Compose([
            LoadImaged(keys=my_key, reader=img_reader),
            EnsureChannelFirstd(keys=my_key),
            Spacingd(keys=my_key, pixdim=[1.0, 1.0, 1.0], mode=["blinear"], align_corners=True),
            ScaleIntensityRanged(keys=my_key, a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=my_key, source_key=my_key),
            ToTensord(keys=my_key)
        ])

    def compose_post_transforms(self, pre_transforms: Compose, out_dir:str="./infer_output") -> Compose:
        """Compose transforms for postprocessing the prediction results."""

        pred_key = self._pred_dataset_key
        return Compose([
            Activationsd(keys=pred_key, softmax=True),
            AsDiscreted(keys=pred_key, argmax=True),
            Invertd(
                keys=pred_key,
                transform = pre_transforms,
                orig_keys=self._input_dataset_key,
                nearest_interp=True
            ),
            SaveImaged(keys=pred_key,
                output_dir=out_dir,
                output_postfix="seg",
                resample=False)
        ])

    def infer_and_save(self, image):
        """Prints out the image obj, and bounce it back, for testing only.
        """

        image_data = image.asnumpy()
        image_shape = image_data.shape
        print(image_shape)
        print(vars(image))
        image_data_new = np.expand_dims(image_data, -1)
        print(image_data_new.shape)

        # Dummy for now.
        return image


def test():
    data_path = "input"

    files = []
    loader = DICOMDataLoaderOperator()
    loader._list_files(data_path, files)
    study_list = loader._load_data(files)
    series = study_list[0].get_all_series()[0]

    dcm_to_vol_op = DICOMSeriesToVolumeOperator()
    dcm_to_vol_op.prepare_series(series)
    voxels = dcm_to_vol_op.generate_voxel_data(series)
    metadata = dcm_to_vol_op.create_metadata(series)
    image = dcm_to_vol_op.create_volumetric_image(voxels, metadata)

    seg_op = SpleenSegOperator(testing=True)
    seg_op.infer_and_save(image)

if __name__ == "__main__":
    test()
