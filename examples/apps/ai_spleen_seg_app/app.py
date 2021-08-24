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
from monai.deploy.core import Application, env, resource
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator
from spleen_seg_operator import SpleenSegOperator

@resource(cpu=1, gpu=1, memory="7Gi")
# pip_packages can be a string that is a path(str) to requirements.txt file or a list of packages.
# The monai pkg is not required by this class, instead by the included operators.
@env(pip_packages=["monai == 0.6.0"])
class AISpleenSegApp(Application):
    def __init__(self, *args, **kwargs):
        """Creates an application instance."""
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

        super().__init__(*args, **kwargs)

        self.logger.debug(f'App Path: {self.path}, \
            Input: {self.context.input_path}, \
            Output: {self.context.output_path},\
            Models: {self.context.model_path}'
        )

    def run(self):
        # This method calls the base class to run. Can be omitted if simply calling through.
        self.logger.debug(f'Begin {self.run.__name__}')
        super().run()
        self.logger.debug(f'End {self.run.__name__}')

    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""

        self.logger.debug(f'Begin {self.compose.__name__}')
        # Creates the custom operator(s) as well as SDK built-in operator(s).
        study_loader_op = DICOMDataLoaderOperator()
        series_selector_op = DICOMSeriesSelectorOperator()
        series_to_vol_op = DICOMSeriesToVolumeOperator()
        spleen_seg_op = SpleenSegOperator()  # This is the app specific operator.
        dicom_seg_writer = DICOMSegmentationWriterOperator()

        # Create the processing pipeline DAG, by specifying the upstream and downstream operators, and
        # ensuring the output from the former matches the input of the latter, in both name and type.
        self.add_flow(study_loader_op, series_selector_op, {"dicom_study_list": "dicom_study_list"})
        self.add_flow(series_selector_op, series_to_vol_op, {"dicom_series": "dicom_series"})
        self.add_flow(series_to_vol_op, spleen_seg_op, {"image": "image"})
        # Note below the dicom_seg_writer requires two inputs, each coming from a upstream operator.
        self.add_flow(series_selector_op, dicom_seg_writer, {"dicom_series": "dicom_series"})
        self.add_flow(spleen_seg_op, dicom_seg_writer, {"seg_image": "seg_image"})

        self.logger.debug(f'End {self.compose.__name__}')

if __name__ == "__main__":
    # Creates the app and test it standalone.
    logging.basicConfig(level=logging.DEBUG)
    app_instance = AISpleenSegApp()  # Optional params' defaults are fine.
    app_instance.run()
