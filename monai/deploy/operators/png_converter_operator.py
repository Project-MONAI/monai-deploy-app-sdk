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

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, Image, InputContext, IOType, Operator, OutputContext
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.utils.importutil import optional_import

PILImage, _ = optional_import("PIL", name="Image")


@md.input("image", Image, IOType.IN_MEMORY)
@md.output("image", DataPath, IOType.DISK)
@md.env(pip_packages=["Pillow >= 8.0.0"])
class PNGConverterOperator(Operator):
    """
    This operator writes out a 3D Volumtric Image to disk in a slice by slice manner
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        input_path = op_input.get("image")
        output_dir = op_output.get().path
        output_dir.mkdir(parents=True, exist_ok=True)
        self.convert_and_save(input_path, output_dir)

    def convert_and_save(self, image, path):
        """
        extracts the slices in originally acquired direction (often axial)
        and saves then in PNG format slice by slice in the specified directory
        """
        image_data = image.asnumpy()
        image_shape = image_data.shape

        num_images = image_shape[0]

        for i in range(0, num_images):
            input_data = image_data[:, :, i]
            pil_image = PILImage.fromarray(input_data)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            pil_image.save(join(str(path), join(str(i) + ".png")))


def main():
    op1 = DICOMSeriesToVolumeOperator()
    data_path = "/home/rahul/medical-images/lung-ct-2/"
    out_path = "/home/rahul/monai-output/"
    makedirs(out_path, exist_ok=True)

    files = []
    loader = DICOMDataLoaderOperator()
    loader._list_files(
        data_path,
        files,
    )
    study_list = loader._load_data(files)

    series = study_list[0].get_all_series()[0]
    op1.prepare_series(series)
    voxels = op1.generate_voxel_data(series)
    metadata = op1.create_metadata(series)
    image = op1.create_volumetric_image(voxels, metadata)

    op2 = PNGConverterOperator()
    op2.convert_and_save(image, out_path)

    print(series)
    # print(metadata.keys())


if __name__ == "__main__":
    main()
