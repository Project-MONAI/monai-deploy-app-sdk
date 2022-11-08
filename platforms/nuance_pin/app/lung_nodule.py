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
from typing import Optional

from ai_service import AiJobProcessor
from app.inference import LungNoduleInferenceOperator
from app.post_inference_ops import CreatePINDiagnosticsReportOp, GenerateGSPSOp

from monai.deploy.core import Application, resource
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator


@resource(cpu=1, gpu=1, memory="7Gi")
# The monai pkg is not required by this class, instead by the included operators.
class LungNoduleDetectionApp(Application):
    def __init__(self, pin_processor: Optional[AiJobProcessor] = None, *args, **kwargs):
        """Creates an application instance."""
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.pin_processor = pin_processor
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        # This method calls the base class to run. Can be omitted if simply calling through.
        self._logger.info(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._logger.info(f"End {self.run.__name__}")

    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""

        logging.info(f"Begin {self.compose.__name__}")

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
        study_loader_op = DICOMDataLoaderOperator()
        series_selector_op = DICOMSeriesSelectorOperator(dicom_selection_rules)
        series_to_vol_op = DICOMSeriesToVolumeOperator()
        detection_op = LungNoduleInferenceOperator()
        gsps_op = GenerateGSPSOp(pin_processor=self.pin_processor)
        pin_report_op = CreatePINDiagnosticsReportOp(pin_processor=self.pin_processor)

        self.add_flow(study_loader_op, series_selector_op, {"dicom_study_list": "dicom_study_list"})
        self.add_flow(
            series_selector_op, series_to_vol_op, {"study_selected_series_list": "study_selected_series_list"}
        )
        self.add_flow(series_to_vol_op, detection_op, {"image": "image"})

        self.add_flow(detection_op, pin_report_op, {"detections": "detection_predictions"})
        self.add_flow(series_selector_op, pin_report_op, {"study_selected_series_list": "original_dicom"})

        self.add_flow(detection_op, gsps_op, {"detections": "detection_predictions"})
        self.add_flow(series_selector_op, gsps_op, {"study_selected_series_list": "original_dicom"})

        logging.info(f"End {self.compose.__name__}")


if __name__ == "__main__":
    # Creates the app and test it standalone. When running is this mode, please note the following:
    #     -m <model file>, for model file path
    #     -i <DICOM folder>, for input DICOM CT series folder
    #     -o <output folder>, for the output folder, default $PWD/output
    # e.g.
    #     monai-deploy exec app.py -i input -m model/model.ts
    #
    logging.basicConfig(level=logging.DEBUG)
    app_instance = LungNoduleDetectionApp(do_run=True)
