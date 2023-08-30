# Copyright 2021-2023 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from monai.deploy.conditions import CountCondition
from monai.deploy.core import AppContext, Application
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.png_converter_operator import PNGConverterOperator


class App(Application):
    """This application loads DICOM files, converts them to 3D image, then to PNG files on disk.

    This showcases the MONAI Deploy application framework
    """

    def compose(self):
        # Use command line options over environment variables to init context.
        app_context: AppContext = Application.init_app_context(self.argv)
        input_dcm_folder = Path(app_context.input_path)
        output_folder = Path(app_context.output_path)
        print(f"input_dcm_folder: {input_dcm_folder}")

        # Set the first operator to run only once by setting the count condition to 1
        study_loader_op = DICOMDataLoaderOperator(
            self, CountCondition(self, 1), input_folder=input_dcm_folder, name="dcm_loader"
        )
        series_selector_op = DICOMSeriesSelectorOperator(self, name="series_selector")
        series_to_vol_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol")
        png_converter_op = PNGConverterOperator(self, output_folder=output_folder, name="png_converter")

        # Create the execution DAG by linking operators' named output to named input.
        self.add_flow(study_loader_op, series_selector_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(
            series_selector_op, series_to_vol_op, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(series_to_vol_op, png_converter_op, {("image", "image")})


if __name__ == "__main__":
    App().run()
