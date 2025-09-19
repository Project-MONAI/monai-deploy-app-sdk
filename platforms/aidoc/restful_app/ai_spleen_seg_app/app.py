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

import json
import logging
import os
from pathlib import Path
from typing import List, Union

# Required for setting SegmentDescription attributes. Direct import as this is not part of App SDK package.
from pydicom.sr.codedict import codes

from monai.deploy.conditions import CountCondition
from monai.deploy.core import AppContext, Application
from monai.deploy.core.domain import Image
from monai.deploy.core.io_type import IOType
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator, SegmentDescription
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator

# Use a local fixed version. from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.monai_bundle_inference_operator import (
    BundleConfigNames,
    IOMapping,
    MonaiBundleInferenceOperator,
)
from monai.deploy.operators.stl_conversion_operator import STLConversionOperator

from .results_message import (
    AggregatedResults,
    AlgorithmClass,
    DetailedResult,
    MeasurementResult,
    Results,
)


class AISpleenSegApp(Application):
    """Demonstrates inference with built-in MONAI Bundle inference operator with DICOM files as input/output

    This application loads a set of DICOM instances, select the appropriate series, converts the series to
    3D volume image, performs inference with the built-in MONAI Bundle inference operator, including pre-processing
    and post-processing, save the segmentation image in a DICOM Seg OID in an instance file, and optionally the
    surface mesh in STL format.

    Pertinent MONAI Bundle:
      https://github.com/Project-MONAI/model-zoo/tree/dev/models/spleen_ct_segmentation

    Execution Time Estimate:
      With a Nvidia GV100 32GB GPU, for an input DICOM Series of 515 instances, the execution time is around
      25 seconds with saving both DICOM Seg and surface mesh STL file, and 15 seconds with DICOM Seg only.
    """

    def __init__(self, *args, status_callback=None, **kwargs):
        """Creates an application instance."""
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._status_callback = status_callback
        self._app_input_path = None  # to be set in compose
        self._app_output_path = None  # to be set in compose
        super().__init__(*args, **kwargs)

    def _get_files_in_folder(self, folder_path: Union[str, Path]) -> List[str]:
        """Traverses a folder and returns a list of full paths of all files.

        Args:
            folder_path (Union[str, Path]): The path to the folder to traverse.

        Returns:
            List[str]: A list of absolute paths to the files in the folder.
        """
        if not os.path.isdir(folder_path):
            self._logger.warning(f"Output folder '{folder_path}' not found, returning empty file list.")
            return []

        file_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(full_path)
        return file_paths

    def run(self, *args, **kwargs):
        # This method calls the base class to run. Can be omitted if simply calling through.
        self._logger.info(f"Begin {self.run.__name__}")
        try:
            super().run(*args, **kwargs)

            if self._status_callback:
                # Create the results object using the Pydantic models
                ai_results = Results(
                    aggregated_results=AggregatedResults(
                        name="Spleen Segmentation",
                        algorithm_class={AlgorithmClass.MEASUREMENT},
                    ),
                    detailed_results={
                        "Spleen Segmentation": DetailedResult(
                            measurement=MeasurementResult(
                                measurements_text="Spleen segmentation completed successfully.",
                            )
                        )
                    },
                )

                output_files = self._get_files_in_folder(self._app_output_path)

                callback_msg_dict = {
                    "run_success": True,
                    "output_files": output_files,
                    "error_message": None,
                    "error_code": None,
                    "result": ai_results.model_dump_json(),
                }
                self._status_callback(json.dumps(callback_msg_dict))

        except Exception as e:
            self._logger.error(f"Error in {self.run.__name__}: {e}")
            # Let the caller to handle and report the error
            raise e

        self._logger.info(f"End {self.run.__name__}")

    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""

        logging.info(f"Begin {self.compose.__name__}")

        # Use Commandline options over environment variables to init context.
        app_context: AppContext = Application.init_app_context(self.argv)
        self._app_input_path = Path(app_context.input_path)
        self._app_output_path = Path(app_context.output_path)

        # Create the custom operator(s) as well as SDK built-in operator(s).
        study_loader_op = DICOMDataLoaderOperator(
            self, CountCondition(self, 1), input_folder=self._app_input_path, name="study_loader_op"
        )
        series_selector_op = DICOMSeriesSelectorOperator(self, rules=Sample_Rules_Text, name="series_selector_op")
        series_to_vol_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_op")

        # Create the inference operator that supports MONAI Bundle and automates the inference.
        # The IOMapping labels match the input and prediction keys in the pre and post processing.
        # The model_name is optional when the app has only one model.
        # The bundle_path argument optionally can be set to an accessible bundle file path in the dev
        # environment, so when the app is packaged into a MAP, the operator can complete the bundle parsing
        # during init.

        config_names = BundleConfigNames(config_names=["inference"])  # Same as the default

        bundle_spleen_seg_op = MonaiBundleInferenceOperator(
            self,
            input_mapping=[IOMapping("image", Image, IOType.IN_MEMORY)],
            output_mapping=[IOMapping("pred", Image, IOType.IN_MEMORY)],
            app_context=app_context,
            bundle_config_names=config_names,
            name="bundle_spleen_seg_op",
        )

        # Create DICOM Seg writer providing the required segment description for each segment with
        # the actual algorithm and the pertinent organ/tissue. The segment_label, algorithm_name,
        # and algorithm_version are of DICOM VR LO type, limited to 64 chars.
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html
        segment_descriptions = [
            SegmentDescription(
                segment_label="Spleen",
                segmented_property_category=codes.SCT.Organ,
                segmented_property_type=codes.SCT.Spleen,
                algorithm_name="volumetric (3D) segmentation of the spleen from CT image",
                algorithm_family=codes.DCM.ArtificialIntelligence,
                algorithm_version="0.3.2",
            )
        ]

        custom_tags = {"SeriesDescription": "AI generated Seg, not for clinical use."}

        dicom_seg_writer = DICOMSegmentationWriterOperator(
            self,
            segment_descriptions=segment_descriptions,
            custom_tags=custom_tags,
            output_folder=self._app_output_path,
            name="dicom_seg_writer",
        )

        # reporter_op = ExecutionStatusReporterOperator(self, status_callback=self._status_callback)

        # Create the processing pipeline, by specifying the source and destination operators, and
        # ensuring the output from the former matches the input of the latter, in both name and type.
        self.add_flow(study_loader_op, series_selector_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(
            series_selector_op, series_to_vol_op, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(series_to_vol_op, bundle_spleen_seg_op, {("image", "image")})
        # Note below the dicom_seg_writer requires two inputs, each coming from a source operator.
        self.add_flow(
            series_selector_op, dicom_seg_writer, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(bundle_spleen_seg_op, dicom_seg_writer, {("pred", "seg_image")})
        # Create the surface mesh STL conversion operator and add it to the app execution flow, if needed, by
        # uncommenting the following couple lines.
        stl_conversion_op = STLConversionOperator(
            self, output_file=self._app_output_path.joinpath("stl/spleen.stl"), name="stl_conversion_op"
        )
        self.add_flow(bundle_spleen_seg_op, stl_conversion_op, {("pred", "image")})

        # Connect the reporter operator to the end of the pipeline.
        # It will be triggered after the DICOM SEG file is written.
        # self.add_flow(stl_conversion_op, reporter_op, {("stl_bytes", "data")})

        logging.info(f"End {self.compose.__name__}")


# This is a sample series selection rule in JSON, simply selecting CT series.
# If the study has more than 1 CT series, then all of them will be selected.
# Please see more detail in DICOMSeriesSelectorOperator.
Sample_Rules_Text = """
{
    "selections": [
        {
            "name": "CT Series",
            "conditions": {
                "StudyDescription": "(.*?)",
                "Modality": "(?i)CT",
                "SeriesDescription": "(.*?)"
            }
        }
    ]
}
"""

if __name__ == "__main__":
    # Creates the app and test it standalone. When running is this mode, please note the following:
    #     -m <model file>, for model file path
    #     -i <DICOM folder>, for input DICOM CT series folder
    #     -o <output folder>, for the output folder, default $PWD/output
    # e.g.
    #     monai-deploy exec app.py -i input -m model/model.ts
    #
    logging.info(f"Begin {__name__}")
    AISpleenSegApp().run()
    logging.info(f"End {__name__}")
