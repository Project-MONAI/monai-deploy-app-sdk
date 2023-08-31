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

import logging
from pathlib import Path

from livertumor_seg_operator import LiverTumorSegOperator
from pydicom.sr.codedict import codes  # Required for setting SegmentDescription attributes.

from monai.deploy.conditions import CountCondition
from monai.deploy.core import AppContext, Application
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator, SegmentDescription
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.stl_conversion_operator import STLConversionOperator

# This sample example completes the processing of a DICOM series with around 600 instances within 45 seconds,
# and time reduces to about 23 seconds if the STL generation is disabled,
# on a desktop with Ubuntu 20.04, 32GB of RAM, and a Nvidia GPU GV100 with 32GB of memory.


# @resource(cpu=1, gpu=1, memory="7Gi")
class AILiverTumorApp(Application):
    def __init__(self, *args, **kwargs):
        """Creates an application instance."""

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        # This method calls the base class to run. Can be omitted if simply calling through.
        self._logger.info(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._logger.info(f"End {self.run.__name__}")

    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""

        self._logger.info(f"Begin {self.compose.__name__}")
        # Use command line options over environment variables to init context.
        app_context: AppContext = Application.init_app_context(self.argv)
        app_input_path = Path(app_context.input_path)
        app_output_path = Path(app_context.output_path)
        model_path = Path(app_context.model_path)

        self._logger.info(f"App input and output path: {app_input_path}, {app_output_path}")

        # Creates the custom operator(s) as well as SDK built-in operator(s).
        study_loader_op = DICOMDataLoaderOperator(
            self, CountCondition(self, 1), input_folder=app_input_path, name="dcm_loader_op"
        )
        series_selector_op = DICOMSeriesSelectorOperator(self, rules=Sample_Rules_Text, name="series_selector_op")
        series_to_vol_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_op")
        # Model specific inference operator, supporting MONAI transforms.
        liver_tumor_seg_op = LiverTumorSegOperator(self, app_context=app_context, model_path=model_path, name="seg_op")
        #     self, model_path=model_path, output_folder=app_output_path, name="seg_op"
        # )

        # Create the surface mesh STL conversion operator
        stl_op = STLConversionOperator(self, output_file=app_output_path.joinpath("stl/mesh.stl"), name="stl_op")

        # Create DICOM Seg writer providing the required segment description for each segment with
        # the actual algorithm and the pertinent organ/tissue.
        # The segment_label, algorithm_name, and algorithm_version are limited to 64 chars.
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html
        # User can Look up SNOMED CT codes at, e.g.
        # https://bioportal.bioontology.org/ontologies/SNOMEDCT

        _algorithm_name = "3D segmentation of the liver and tumor from CT image"
        _algorithm_family = codes.DCM.ArtificialIntelligence
        _algorithm_version = "0.1.0"

        segment_descriptions = [
            SegmentDescription(
                segment_label="Liver",
                segmented_property_category=codes.SCT.Organ,
                segmented_property_type=codes.SCT.Liver,
                algorithm_name=_algorithm_name,
                algorithm_family=_algorithm_family,
                algorithm_version=_algorithm_version,
            ),
            SegmentDescription(
                segment_label="Tumor",
                segmented_property_category=codes.SCT.Tumor,
                segmented_property_type=codes.SCT.Tumor,
                algorithm_name=_algorithm_name,
                algorithm_family=_algorithm_family,
                algorithm_version=_algorithm_version,
            ),
        ]

        dicom_seg_writer = DICOMSegmentationWriterOperator(
            self, segment_descriptions=segment_descriptions, output_folder=app_output_path, name="dcm_seg_writer_op"
        )
        # Create the processing pipeline, by specifying the source and destination operators, and
        # ensuring the output from the former matches the input of the latter, in both name and type.
        self.add_flow(study_loader_op, series_selector_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(
            series_selector_op, series_to_vol_op, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(series_to_vol_op, liver_tumor_seg_op, {("image", "image")})

        # Note below the dicom_seg_writer requires two inputs, each coming from a source operator.
        self.add_flow(
            series_selector_op, dicom_seg_writer, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(liver_tumor_seg_op, dicom_seg_writer, {("seg_image", "seg_image")})

        # Add the stl mesh operator to save the mesh in stl format.
        self.add_flow(liver_tumor_seg_op, stl_op, {("seg_image", "image")})

        self._logger.info(f"End {self.compose.__name__}")


# This is a sample series selection rule in JSON, simply selecting CT series.
# If the study has more than 1 CT series, then all of them will be selected.
# Please see more detail in DICOMSeriesSelectorOperator.
# For list of string values, e.g. "ImageType": ["PRIMARY", "ORIGINAL"], it is a match if all elements
# are all in the multi-value attribute of the DICOM series.

Sample_Rules_Text = """
{
    "selections": [
        {
            "name": "CT Series",
            "conditions": {
                "Modality": "(?i)CT",
                "ImageType": ["PRIMARY", "ORIGINAL"],
                "PhotometricInterpretation": "MONOCHROME2"
            }
        }
    ]
}
"""

if __name__ == "__main__":
    # Creates the app and test it standalone.
    logging.info(f"Begin {__name__}")
    AILiverTumorApp().run()
    logging.info(f"End {__name__}")
