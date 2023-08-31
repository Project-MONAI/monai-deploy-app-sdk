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

from pydicom.sr.codedict import codes  # Required for setting SegmentDescription attributes.
from unetr_seg_operator import UnetrSegOperator

from monai.deploy.conditions import CountCondition
from monai.deploy.core import AppContext, Application
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator, SegmentDescription
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.stl_conversion_operator import STLConversionOperator

# This sample example completes the processing of a DICOM series with around 300 instances within 43 seconds,
# 39 if not saving intermediate image file, and 28 seconds if the STL generation is also disabled,
# on a desktop with Ubuntu 20.04, 32GB of RAM, and a Nvidia GPU GV100 with 32GB of memory.


# @resource(cpu=1, gpu=1, memory="7Gi")
class AIUnetrSegApp(Application):
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
        seg_op = UnetrSegOperator(self, app_context=app_context, model_path=model_path, name="seg_op")

        # Create the surface mesh STL conversion operator, for all segments
        stl_conversion_op = STLConversionOperator(
            self,
            output_file=app_output_path.joinpath("stl/mesh.stl"),
            keep_largest_connected_component=False,
            name="stl_op",
        )

        # Create DICOM Seg writer providing the required segment description for each segment with
        # the actual algorithm and the pertinent organ/tissue.
        # The segment_label, algorithm_name, and algorithm_version are limited to 64 chars.
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html

        _algorithm_name = "3D multi-organ segmentation from CT image"
        _algorithm_family = codes.DCM.ArtificialIntelligence
        _algorithm_version = "0.1.0"

        # List of (Segment name, [Code menaing str]), not including background which is value of 0.
        # User must provide correct codes, which can be looked at, e.g.
        # https://bioportal.bioontology.org/ontologies/SNOMEDCT
        # Alternatively, consult the concept and code dictionaries in PyDicom

        organs = [
            ("Spleen",),
            ("Right Kidney", "Kidney"),
            ("Left Kideny", "Kidney"),
            ("Gallbladder",),
            ("Esophagus",),
            ("Liver",),
            ("Stomach",),
            ("Aorta",),
            ("Inferior vena cava", "InferiorVenaCava"),
            ("Portal and Splenic Veins", "SplenicVein"),
            ("Pancreas",),
            ("Right adrenal gland", "AdrenalGland"),
            ("Left adrenal gland", "AdrenalGland"),
        ]

        segment_descriptions = [
            SegmentDescription(
                segment_label=organ[0],
                segmented_property_category=codes.SCT.Organ,
                segmented_property_type=codes.SCT.__getattr__(organ[1] if len(organ) > 1 else organ[0]),
                algorithm_name=_algorithm_name,
                algorithm_family=_algorithm_family,
                algorithm_version=_algorithm_version,
            )
            for organ in organs
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
        self.add_flow(series_to_vol_op, seg_op, {("image", "image")})
        self.add_flow(seg_op, stl_conversion_op, {("seg_image", "image")})

        # Note below the dicom_seg_writer requires two inputs, each coming from a source operator.
        self.add_flow(
            series_selector_op, dicom_seg_writer, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(seg_op, dicom_seg_writer, {("seg_image", "seg_image")})

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
    # Creates the app and test it standalone. When running is this mode, please note the following:
    #     -m <model file>, for model file path
    #     -i <DICOM folder>, for input DICOM CT series folder
    #     -o <output folder>, for the output folder, default $PWD/output
    # e.g.
    #     python3 app.py -i input -m model/model.ts
    #

    logging.info(f"Begin {__name__}")
    AIUnetrSegApp().run()
    logging.info(f"End {__name__}")
