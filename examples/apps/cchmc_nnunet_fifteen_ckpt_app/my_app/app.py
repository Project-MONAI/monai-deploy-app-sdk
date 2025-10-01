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

# custom DICOMSCWriterOperator (Secondary Capture)
from dicom_sc_writer_operator import DICOMSCWriterOperator

# custom DICOMSeriesSelectorOperator
from dicom_series_selector_operator import DICOMSeriesSelectorOperator

# custom inference operator
from nnunet_seg_operator import NNUnetSegOperator

# required for setting SegmentDescription attributes
# direct import as this is not part of App SDK package
from pydicom.sr.codedict import codes

from monai.deploy.conditions import CountCondition
from monai.deploy.core import Application
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator, SegmentDescription
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.dicom_text_sr_writer_operator import DICOMTextSRWriterOperator, EquipmentInfo, ModelInfo


# inherit new Application class instance, AIAbdomenSegApp, from MONAI Application base class
# base class provides support for chaining up operators and executing application
class UTEAirwayNNUnetApp(Application):
    """Demonstrates inference with nnU-Net ensemble models for airway segmentation.

    This application loads a set of DICOM instances, selects the appropriate series, converts the series to
    3D volume image, performs inference with the NNUnetSegOperator, including pre-processing
    and post-processing, saves a DICOM SEG (airway contour), a DICOM Secondary Capture (airway contour overlay),
    and a DICOM SR (airway volume).

    Pertinent MONAI Bundle:
      This MAP is designed to work with a MONAI bundle compatible with nnU-Net.
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

        # Temporary bug fix for MAP execution where model path copy is messed up - need fix to app-sdk package function
        # Check if the model_path has a subfolder named 'models' and set model_path to that subfolder if it exists
        models_subfolder = model_path / "models"
        if models_subfolder.exists() and models_subfolder.is_dir():
            self._logger.info(f"Found 'models' subfolder in {model_path}. Setting model_path to {models_subfolder}")
            model_path = models_subfolder

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
        # 1 = airway
        output_labels = [1]
        nnunet_seg_op = NNUnetSegOperator(
            self,
            app_context=app_context,
            model_path=model_path,
            output_folder=app_output_path,
            output_labels=output_labels,
            name="nnunet_seg_op",
        )

        # create DICOM Seg writer providing the required segment description for each segment with
        # the actual algorithm and the pertinent organ/tissue; the segment_label, algorithm_name,
        # and algorithm_version are of DICOM VR LO type, limited to 64 chars
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html

        # general algorithm information
        _algorithm_name = "UTE_nnunet_airway"
        _algorithm_family = codes.DCM.ArtificialIntelligence
        _algorithm_version = "1.0.0"

        segment_descriptions = [
            SegmentDescription(
                segment_label="Airway",
                segmented_property_category=codes.SCT.BodyStructure,
                segmented_property_type=codes.SCT.TracheaAndBronchus,
                algorithm_name=_algorithm_name,
                algorithm_family=_algorithm_family,
                algorithm_version=_algorithm_version,
            ),
        ]

        # model info is algorithm information
        my_model_info = ModelInfo(
            creator="UTE",  # institution name
            name=_algorithm_name,  # algorithm name
            version=_algorithm_version,  # algorithm version
            uid="1.0.0",  # MAP version
        )

        # equipment info is MONAI Deploy App SDK information
        my_equipment = EquipmentInfo(
            manufacturer="The MONAI Consortium",
            manufacturer_model="MONAI Deploy App SDK",
            software_version_number="3.0.0",  # MONAI Deploy App SDK version
        )

        # custom tags - add AlgorithmName for monitoring purposes
        custom_tags_seg = {
            "SeriesDescription": "AI Generated DICOM SEG; Not for Clinical Use.",
            "AlgorithmName": f"{my_model_info.name}:{my_model_info.version}:{my_model_info.uid}",
        }
        custom_tags_sr = {
            "SeriesDescription": "AI Generated DICOM SR; Not for Clinical Use.",
            "AlgorithmName": f"{my_model_info.name}:{my_model_info.version}:{my_model_info.uid}",
        }
        custom_tags_sc = {
            "SeriesDescription": "AI Generated DICOM Secondary Capture; Not for Clinical Use.",
            "AlgorithmName": f"{my_model_info.name}:{my_model_info.version}:{my_model_info.uid}",
        }

        # DICOM SEG Writer op writes content from segment_descriptions to output DICOM images as DICOM tags
        dicom_seg_writer = DICOMSegmentationWriterOperator(
            self,
            segment_descriptions=segment_descriptions,
            model_info=my_model_info,
            custom_tags=custom_tags_seg,
            # store DICOM SEG in SEG subdirectory; necessary for routing in CCHMC MDE workflow definition
            output_folder=app_output_path / "SEG",
            # omit_empty_frames is a default parameteter (type bool) of DICOMSegmentationWriterOperator
            # dictates whether or not to omit frames that contain no segmented pixels from the output segmentation
            # default value is True; changed to False to ensure input and output DICOM series #'s match
            omit_empty_frames=False,
            name="dicom_seg_writer",
        )

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

        # create the processing pipeline, by specifying the source and destination operators, and
        # ensuring the output from the former matches the input of the latter, in both name and type
        # instantiate and connect operators using self.add_flow(); specify current operator, next operator, and tuple to match I/O
        self.add_flow(study_loader_op, series_selector_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(
            series_selector_op, series_to_vol_op, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(series_to_vol_op, nnunet_seg_op, {("image", "image")})

        # note below the dicom_seg_writer, dicom_sr_writer, and dicom_sc_writer each require two inputs,
        # each coming from a source operator

        # DICOM SEG
        self.add_flow(
            series_selector_op, dicom_seg_writer, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(nnunet_seg_op, dicom_seg_writer, {("seg_image", "seg_image")})

        # DICOM SR
        self.add_flow(
            series_selector_op, dicom_sr_writer, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(nnunet_seg_op, dicom_sr_writer, {("result_text", "text")})

        # DICOM SC
        self.add_flow(
            series_selector_op, dicom_sc_writer, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(nnunet_seg_op, dicom_sc_writer, {("dicom_sc_dir", "dicom_sc_dir")})

        logging.info(f"End {self.compose.__name__}")


# series selection rule in JSON, which selects for Axial T2 MR series:
# StudyDescription (Type 3): matches any value
# Modality (Type 1): matches "MR" value (case-insensitive); filters out non-MR modalities
# ImageOrientationPatient (Type 1): matches Axial orientations; filters out Sagittal and Coronal orientations
# MRAcquisitionType (Type 2): matches "2D" value (case-insensitive); filters out 3D acquisitions
# RepetitionTime (Type 2C): matches values greater than 1200; filters for T2 acquisitions
# EchoTime (Type 2): matches values bewtween 75 and 100 (inclusive); filters out SSH series
# EchoTrainLength (Type 2): matches values less than 50; filters out SSH series
# FlipAngle (Type 3): matches values greater than 75; filters for T2 acquisitions
# all valid series will be selected; downstream operators only perform inference and write outputs for 1st selected series
# please see more detail in DICOMSeriesSelectorOperator

Sample_Rules_Text = """
"""

# if executing application code using python interpreter:
if __name__ == "__main__":
    # creates the app and test it standalone; when running is this mode, please note the following:
    #     -m <model file>, for model file path
    #     -i <DICOM folder>, for input DICOM MR series folder
    #     -o <output folder>, for the output folder, default $PWD/output
    # e.g.
    #     monai-deploy exec app.py -i input -m model/ls_swinunetr_FT.pt
    #
    logging.info(f"Begin {__name__}")
    UTEAirwayNNUnetApp().run()
    logging.info(f"End {__name__}")
