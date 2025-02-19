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

import logging
from pathlib import Path

# custom inference operator
from abdomen_seg_operator import AbdomenSegOperator

# custom DICOM Secondary Capture (SC) writer operator
from dicom_sc_writer_operator import DICOMSCWriterOperator

# custom MongoDB operators
from mongodb_entry_creator_operator import MongoDBEntryCreatorOperator
from mongodb_writer_operator import MongoDBWriterOperator

# required for setting SegmentDescription attributes
# direct import as this is not part of App SDK package
from pydicom.sr.codedict import codes

from monai.deploy.conditions import CountCondition
from monai.deploy.core import Application
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator, SegmentDescription
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.dicom_text_sr_writer_operator import DICOMTextSRWriterOperator, EquipmentInfo, ModelInfo


# inherit new Application class instance, AIAbdomenSegApp, from MONAI Application base class
# base class provides support for chaining up operators and executing application
class AIAbdomenSegApp(Application):
    """Demonstrates inference with customized CCHMC pediatric abdominal segmentation bundle inference operator, with
    DICOM files as input/output

    This application loads a set of DICOM instances, selects the appropriate series, converts the series to
    3D volume image, performs inference with a custom inference operator, including pre-processing
    and post-processing, saves a DICOM SEG (organ contours), a DICOM Secondary Capture (organ contours overlay),
    and a DICOM SR (organ volumes), and writes organ volumes and relevant DICOM tags to the MONAI Deploy Express
    MongoDB database (optional).

    Pertinent MONAI Bundle:
      https://github.com/cchmc-dll/pediatric_abdominal_segmentation_bundle/tree/original

    Execution Time Estimate:
      With a NVIDIA GeForce RTX 3090 24GB GPU, for an input DICOM Series of 204 instances, the execution time is around
      25 seconds for DICOM SEG, DICOM SC, and DICOM SR outputs, as well as the MDE MongoDB database write.
    """

    def __init__(self, *args, **kwargs):
        """Creates an application instance."""
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        # this method calls the base class to run; can be omitted if simply calling through
        self._logger.info(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._logger.info(f"End {self.run.__name__}")

    # use compose method to instantiate operators and connect them to form a Directed Acyclic Graph (DAG)
    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""

        logging.info(f"Begin {self.compose.__name__}")

        # use Commandline options over environment variables to init context
        app_context = Application.init_app_context(self.argv)
        app_input_path = Path(app_context.input_path)
        app_output_path = Path(app_context.output_path)
        model_path = Path(app_context.model_path)

        # create the custom operator(s) as well as SDK built-in operator(s)
        # DICOM Data Loader op
        study_loader_op = DICOMDataLoaderOperator(
            self, CountCondition(self, 1), input_folder=app_input_path, name="study_loader_op"
        )

        # custom DICOM Series Selector op
        # all_matched and sort_by_sop_instance_count = True; want all series that meet the selection criteria
        # to be matched, and SOP sorting
        series_selector_op = DICOMSeriesSelectorOperator(
            self, rules=Sample_Rules_Text, all_matched=True, sort_by_sop_instance_count=True, name="series_selector_op"
        )

        # DICOM Series to Volume op
        series_to_vol_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_op")

        # custom inference op
        # output_labels specifies which of the organ segmentations are desired in the DICOM SEG, DICOM SC, and DICOM SR outputs
        # 1 = Liver, 2 = Spleen, 3 = Pancreas; all segmentations performed, but visibility in outputs (SEG, SC, SR) controlled here
        # all organ volumes will be written to MongoDB
        output_labels = [1, 2, 3]
        abd_seg_op = AbdomenSegOperator(
            self, app_context=app_context, model_path=model_path, output_labels=output_labels, name="abd_seg_op"
        )

        # create DICOM Seg writer providing the required segment description for each segment with
        # the actual algorithm and the pertinent organ/tissue; the segment_label, algorithm_name,
        # and algorithm_version are of DICOM VR LO type, limited to 64 chars
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html

        # general algorithm information
        _algorithm_name = "CCHMC Pediatric CT Abdominal Segmentation"
        _algorithm_family = codes.DCM.ArtificialIntelligence
        _algorithm_version = "0.4.3"

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
                segment_label="Spleen",
                segmented_property_category=codes.SCT.Organ,
                segmented_property_type=codes.SCT.Spleen,
                algorithm_name=_algorithm_name,
                algorithm_family=_algorithm_family,
                algorithm_version=_algorithm_version,
            ),
            SegmentDescription(
                segment_label="Pancreas",
                segmented_property_category=codes.SCT.Organ,
                segmented_property_type=codes.SCT.Pancreas,
                algorithm_name=_algorithm_name,
                algorithm_family=_algorithm_family,
                algorithm_version=_algorithm_version,
            ),
        ]

        # custom tags - add Device UID to DICOM SEG to match SR and SC tags
        custom_tags_seg = {"SeriesDescription": "AI Generated DICOM SEG; Not for Clinical Use.", "DeviceUID": "0.0.1"}
        custom_tags_sr = {"SeriesDescription": "AI Generated DICOM SR; Not for Clinical Use."}
        custom_tags_sc = {"SeriesDescription": "AI Generated DICOM Secondary Capture; Not for Clinical Use."}

        # DICOM SEG Writer op writes content from segment_descriptions to output DICOM images as DICOM tags
        dicom_seg_writer = DICOMSegmentationWriterOperator(
            self,
            segment_descriptions=segment_descriptions,
            custom_tags=custom_tags_seg,
            # store DICOM SEG in SEG subdirectory; necessary for routing in CCHMC MDE workflow definition
            output_folder=app_output_path / "SEG",
            # omit_empty_frames is a default parameteter (type bool) of DICOMSegmentationWriterOperator
            # dictates whether or not to omit frames that contain no segmented pixels from the output segmentation
            # default value is True; changed to False to ensure input and output DICOM series #'s match
            omit_empty_frames=False,
            name="dicom_seg_writer",
        )

        # model and equipment info
        my_model_info = ModelInfo("CCHMC CAIIR", "CCHMC Pediatric CT Abdominal Segmentation", "0.4.3", "0.0.1")
        my_equipment = EquipmentInfo(manufacturer="The MONAI Consortium", manufacturer_model="MONAI Deploy App SDK")

        # DICOM SR Writer op
        dicom_sr_writer = DICOMTextSRWriterOperator(
            self,
            # copy_tags is a default parameteter (type bool) of DICOMTextSRWriterOperator; default value is True
            # dictates whether or not to copy DICOM attributes from the selected DICOM series
            # changed to True to copy DICOM attributes so DICOM SR has same Study UID
            copy_tags=True,
            model_info=my_model_info,
            equipment_info=my_equipment,
            custom_tags=custom_tags_sr,
            # store DICOM SR in SR subdirectory; necessary for routing in CCHMC MDE workflow definition
            output_folder=app_output_path / "SR",
        )

        # custom DICOM SC Writer op
        dicom_sc_writer = DICOMSCWriterOperator(
            self,
            model_info=my_model_info,
            equipment_info=my_equipment,
            custom_tags=custom_tags_sc,
            # store DICOM SC in SC subdirectory; necessary for routing in CCHMC MDE workflow definition
            output_folder=app_output_path / "SC",
        )

        # MongoDB database, collection, and MAP version info
        database_name = "CTLiverSpleenSegPredictions"
        collection_name = "OrganVolumes"
        map_version = "0.0.1"

        # custom MongoDB Entry Creator op
        mongodb_entry_creator = MongoDBEntryCreatorOperator(self, map_version=map_version)

        # custom MongoDB Writer op
        mongodb_writer = MongoDBWriterOperator(self, database_name=database_name, collection_name=collection_name)

        # create the processing pipeline, by specifying the source and destination operators, and
        # ensuring the output from the former matches the input of the latter, in both name and type
        # instantiate and connect operators using self.add_flow(); specify current operator, next operator, and tuple to match I/O
        self.add_flow(study_loader_op, series_selector_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(
            series_selector_op, series_to_vol_op, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(series_to_vol_op, abd_seg_op, {("image", "image")})

        # note below the dicom_seg_writer, dicom_sr_writer, dicom_sc_writer, and mongodb_entry_creator each require
        # two inputs, each coming from a source operator

        # DICOM SEG
        self.add_flow(
            series_selector_op, dicom_seg_writer, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(abd_seg_op, dicom_seg_writer, {("seg_image", "seg_image")})

        # DICOM SR
        self.add_flow(
            series_selector_op, dicom_sr_writer, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(abd_seg_op, dicom_sr_writer, {("result_text_dicom_sr", "text")})

        # DICOM SC
        self.add_flow(
            series_selector_op, dicom_sc_writer, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(abd_seg_op, dicom_sc_writer, {("dicom_sc_dir", "dicom_sc_dir")})

        # MongoDB
        self.add_flow(
            series_selector_op, mongodb_entry_creator, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(abd_seg_op, mongodb_entry_creator, {("result_text_mongodb", "text")})
        self.add_flow(mongodb_entry_creator, mongodb_writer, {("mongodb_database_entry", "mongodb_database_entry")})

        logging.info(f"End {self.compose.__name__}")


# series selection rule in JSON, which selects for axial CT series; flexible ST choices:
# StudyDescription: matches any value
# Modality: matches "CT" value (case-insensitive); filters out non-CT modalities
# ImageType: matches value that contains "PRIMARY", "ORIGINAL", and "AXIAL"; filters out most cor and sag views
# SeriesDescription: matches any values that do not contain "cor" or "sag" (case-insensitive); filters out cor and sag views
# SliceThickness: supports list, string, and numerical matching:
# [3, 5]: matches ST values between 3 and 5
# "^(5(\\\\.0+)?|5)$": RegEx; matches ST values of 5, 5.0, 5.00, etc.
# 5: matches ST values of 5, 5.0, 5.00, etc.
# all valid series will be selected; downstream operators only perform inference and write outputs for 1st selected series
# please see more detail in DICOMSeriesSelectorOperator

Sample_Rules_Text = """
{
    "selections": [
        {
            "name": "Axial CT Series",
            "conditions": {
                "StudyDescription": "(.*?)",
                "Modality": "(?i)CT",
                "ImageType": ["PRIMARY", "ORIGINAL", "AXIAL"],
                "SeriesDescription": "(?i)^(?!.*(cor|sag)).*$",
                "SliceThickness": [3, 5]
            }
        }
    ]
}
"""

# if executing application code using python interpreter:
if __name__ == "__main__":
    # creates the app and test it standalone; when running is this mode, please note the following:
    #     -m <model file>, for model file path
    #     -i <DICOM folder>, for input DICOM CT series folder
    #     -o <output folder>, for the output folder, default $PWD/output
    # e.g.
    #     monai-deploy exec app.py -i input -m model/dynunet_FT.ts
    #
    logging.info(f"Begin {__name__}")
    AIAbdomenSegApp().run()
    logging.info(f"End {__name__}")
