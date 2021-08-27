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


from os import makedirs
from os.path import join
import numpy as np

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
    env
)
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.utils.importutil import optional_import


@input("image", Image, IOType.IN_MEMORY)
@output("image", DataPath, IOType.DISK)
@env(pip_packages=["numpy>=1.17"])   # Shown as example, as it is actually not required by this class.
class ImageConsumerOperator(Operator):
    """
    This operator demonstrates consuming the Image object of a single slice 3D Image (from DICOM XR)
    """

    def compute(self, input: InputContext, output: OutputContext, context: ExecutionContext):
        input_img = input.get("image")
        output_dir = output.get().path
        output_dir.mkdir(parents=True, exist_ok=True)
        self.parse_image(input_img, output_dir)

        # Cannot run the follow statement
        # output.set(output_dir, "image")

    def parse_image(self, image, output_dir):
        """
        Parses the image object to extract the data arrary as well as metadata
        then in PNG format slice by slice in the specified directory
        """
        image_data = image.asnumpy()
        image_shape = image_data.shape
        img_meta_data = image.metadata()

        print(f'Image data:\n): {image_data}')
        print(f'Image data type: {type(image_data)}')
        print(f'Image shape {type(image_shape)}: {image_shape}')

        if img_meta_data and isinstance(img_meta_data, dict):
            print(f'Image metadata {type(img_meta_data)}: {img_meta_data}')
        else:
            print(f'No Image metadata.')

        output_file_path = join(output_dir, 'success.txt')
        with open(output_file_path,'w') as out_file:
            out_file.writelines('Success.')

def test():
    data_path = "input"
    out_path = "output"
    makedirs(out_path, exist_ok=True)

    print(f'Reading DCM from folder: {input}')
    print(f'Saving some output to folder: {out_path}')

    files = []
    loader = DICOMDataLoaderOperator()
    loader._list_files(
        data_path,
        files,
    )
    study_list = loader._load_data(files)

    series = None
    for study in study_list:
        for series_item in study.get_all_series():
            series = series_item
            break
        if series:
            break

    if not series:
        print('Failed to find any series.')
        exit(1)

    print(f'1st series ({type(series)}): {series}')

    series_to_vol = DICOMSeriesToVolumeOperator()
    series_to_vol.prepare_series(series)
    voxels = series_to_vol.generate_voxel_data(series)
    metadata = series_to_vol.create_metadata(series)
    image = series_to_vol.create_volumetric_image(voxels, metadata)

    image_consumer = ImageConsumerOperator()
    image_consumer.parse_image(image, out_path)

if __name__ == "__main__":
    test()
