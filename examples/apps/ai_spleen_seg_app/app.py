<<<<<<< HEAD
# Copyright 2020 - 2021 MONAI Consortium
=======
# Copyright 2021 MONAI Consortium
>>>>>>> dd1a51d8fffbc91621333fbffa5a2350571ff50e
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
from monai.deploy.core import Application
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator
from spleen_seg_operator import SpleenSegOperator

class AISpleenSegApp(Application):
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

        super().__init__(do_run=False)
        print(f'Input: {self._context.input_path}, \
            Output: {self._context.output_path}, \
            Models: {self._context.model_path}'
        )

    def __call__(self):
        #Delegate to base class as there is nothing more to add
        super().run()

    def compose(self):

        study_loader_op = DICOMDataLoaderOperator()
        series_selector_op = DICOMSeriesSelectorOperator()
        series_to_vol_op = DICOMSeriesToVolumeOperator()
        spleen_seg_op = SpleenSegOperator(testing=False)
        dicom_seg_writer = DICOMSegmentationWriterOperator()

        self.add_flow(study_loader_op, series_selector_op, {"dicom_study_list": "dicom_study_list"})
        self.add_flow(series_selector_op, series_to_vol_op, {"dicom_series": "dicom_series"})
        self.add_flow(series_to_vol_op, spleen_seg_op, {"image": "image"})
        self.add_flow(series_selector_op, dicom_seg_writer, {"dicom_series": "dicom_series"})
        self.add_flow(spleen_seg_op, dicom_seg_writer, {"seg_image": "seg_image"})

if __name__ == "__main__":
    # Creat the app without immediately running it. The base class needs to change default run=False
    app_instance = AISpleenSegApp()
    app_instance()
