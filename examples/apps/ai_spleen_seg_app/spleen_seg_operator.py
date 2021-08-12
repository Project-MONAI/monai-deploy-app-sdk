# Copyright 2020 - 2021 MONAI Consortium
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


from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_study import DICOMStudy
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator

from monai.deploy.core.domain.image import Image

from os import listdir
from os.path import isfile, join

import numpy as np

@input("image", Image, IOType.IN_MEMORY)
@output("seg_image", Image, IOType.IN_MEMORY)
#@output("seg_image", DataPath, IOType.DISK)

class SpleenSegOperator(Operator):
    """
    This operator writes out a 3D Volumtric Image to disk in a slice by slice manner
    """


    def compute(self, input: InputContext, output: OutputContext, context: ExecutionContext):
        input_image = input.get("image")
        # output_dir = output.get().path
        # output_dir.mkdir(parents=True, exist_ok=True)
        seg_image = self.infer_and_save(input_image, output)
        output.set(seg_image, "seg_image")

    def infer_and_save(self, image, output):
        """
        extracts the slices in originally acquired direction (often axial)
        and saves then in PNG format slice by slice in the specified directory
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
    out_path = "output"

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

    seg_op = SpleenSegOperator()
    seg_op.infer_and_save(image, out_path)

if __name__ == "__main__":
    test()
