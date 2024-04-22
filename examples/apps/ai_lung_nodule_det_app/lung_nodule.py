# Copyright 2021-2022 MONAI Consortium
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
from typing import Callable, Optional

from inference import LungNoduleInferenceOperator
from post_inference_ops import GenerateGSPSOp

from monai.deploy.conditions import CountCondition
from monai.deploy.core import AppContext, Application
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator


# @resource(cpu=1, gpu=1, memory="7Gi")
# The monai pkg is not required by this class, instead by the included operators.
class LungNoduleDetectionApp(Application):
    def __init__(self, *args, upload_document: Optional[Callable]=None, upload_gsps: Optional[Callable]=None, **kwargs):
        """Creates an application instance."""
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.upload_document = upload_document
        self.upload_gsps = upload_gsps
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        # This method calls the base class to run. Can be omitted if simply calling through.
        self._logger.info(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._logger.info(f"End {self.run.__name__}")

    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""

        logging.info(f"Begin {self.compose.__name__}")

        # Use command line options over environment variables to init context.
        app_context: AppContext = Application.init_app_context(self.argv)
        app_input_path = Path(app_context.input_path)
        app_output_path = Path(app_context.output_path)

        dicom_selection_rules = """
        {
            "selections": [
                {
                    "name": "CT Series",
                    "conditions": {
                        "Modality": "CT"
                    }
                }
            ]
        }
        """

        # Create the custom operator(s) as well as SDK built-in operator(s).
        study_loader_op = DICOMDataLoaderOperator(
             self, CountCondition(self, 1), input_folder=app_input_path, name="dcm_loader_op"
        )
        series_selector_op = DICOMSeriesSelectorOperator(self, rules=dicom_selection_rules, name="series_selector_op")
        series_to_vol_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_op")
        detection_op = LungNoduleInferenceOperator(self, app_context=app_context, model_path=app_context.model_path, name="detection_op")
        gsps_op = GenerateGSPSOp(self, upload_gsps_fn=self.upload_gsps, app_context=app_context, model_path=app_context.model_path, name="gsps_op")  

        self.add_flow(study_loader_op, series_selector_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(
            series_selector_op, series_to_vol_op, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(series_to_vol_op, detection_op, {("image", "image")})

        self.add_flow(detection_op, gsps_op, {("detections", "detection_predictions")})
        self.add_flow(series_selector_op, gsps_op, {("study_selected_series_list", "original_dicom")})

        logging.info(f"End {self.compose.__name__}")


if __name__ == "__main__":
    # Creates the app and test it standalone. When running is this mode, please note the following:
    #     -m <model file>, for model file path
    #     -i <DICOM folder>, for input DICOM CT series folder
    #     -o <output folder>, for the output folder, default $PWD/output
    # e.g.
    #     monai-deploy exec app.py -i input -m model/model.ts
    #
    app_instance = LungNoduleDetectionApp().run()
