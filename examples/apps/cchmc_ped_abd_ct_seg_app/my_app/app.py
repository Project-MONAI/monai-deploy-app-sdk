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

# custom DICOMSeriesSelectorOperator
from dicom_series_selector_operator import DICOMSeriesSelectorOperator

# custom DICOMTextSRWriterOperator
from dicom_text_sr_writer_operator import DICOMTextSRWriterOperator, EquipmentInfo, ModelInfo

# required for setting SegmentDescription attributes
# direct import as this is not part of App SDK package
from pydicom.sr.codedict import codes

from monai.deploy.conditions import CountCondition
from monai.deploy.core import Application
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_encapsulated_pdf_writer_operator import DICOMEncapsulatedPDFWriterOperator
from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator, SegmentDescription
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.segmentation_metrics_operator import SegmentationMetricsOperator
from monai.deploy.operators.segmentation_zscore_operator import SegmentationZScoreOperator
from monai.deploy.operators.segmentation_contour_operator import SegmentationContourOperator
from monai.deploy.operators.segmentation_overlay_operator import SegmentationOverlayOperator



# inherit new Application class instance, AIAbdomenSegApp, from MONAI Application base class
# base class provides support for chaining up operators and executing application
class AIAbdomenSegApp(Application):
    """Demonstrates inference with customized CCHMC pediatric abdominal segmentation bundle inference operator, with
    DICOM files as input/output

    This application loads a set of DICOM instances, selects the appropriate series, converts the series to
    3D volume image, performs inference with a custom inference operator, including pre-processing
    and post-processing, saves a DICOM SEG (organ contours), a DICOM Secondary Capture (organ contours overlay),
    and a DICOM SR (organ volumes).

    Pertinent MONAI Bundle:
      https://github.com/cchmc-dll/pediatric_abdominal_segmentation_bundle/tree/original

    Execution Time Estimate:
      With a NVIDIA GeForce RTX 3090 24GB GPU, for an input DICOM Series of 204 instances, the execution time is around
      30 seconds for DICOM SEG, DICOM SC, and DICOM SR outputs.
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
        # specify output labes for largest connected component application, background or 0 will be ignored
        output_labels = {'background': 0, 'liver': 1, 'spleen': 2, 'pancreas': 3}
        abd_seg_op = AbdomenSegOperator(
            self, app_context=app_context, model_path=model_path, output_folder=app_output_path, output_labels=output_labels, name="abd_seg_op"
        )

        # create DICOM Seg writer providing the required segment description for each segment with
        # the actual algorithm and the pertinent organ/tissue; the segment_label, algorithm_name,
        # and algorithm_version are of DICOM VR LO type, limited to 64 chars
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html

        # general algorithm information
        _algorithm_name = "CCHMC Pediatric CT Liver-Spleen Segmentation"
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

        # model info is algorithm information
        my_model_info = ModelInfo(
            creator="CCHMC CAIIR", # institution name
            name=_algorithm_name, # algorithm name
            version=_algorithm_version, # algorithm version
            uid="0.0.6", # MAP version
        )

        # equipment info is MONAI Deploy App SDK information
        my_equipment_info = EquipmentInfo(
            manufacturer="The MONAI Consortium",
            manufacturer_model="MONAI Deploy App SDK",
            software_version_number="3.1.0", # MONAI Deploy App SDK version
        )

        # custom tags - add AlgorithmName for monitoring purposes
        custom_tags_seg = {
            "SeriesDescription": "AI Generated DICOM SEG; Not for Clinical Use.",
            "AlgorithmName": f"{my_model_info.name}:{my_model_info.version}:{my_model_info.uid}"
        }
        custom_tags_sr = {
            "SeriesDescription": "AI Generated DICOM SR; Not for Clinical Use.",
            "AlgorithmName": f"{my_model_info.name}:{my_model_info.version}:{my_model_info.uid}"
        }
        custom_tags_sc = {
            "SeriesDescription": "AI Generated DICOM Secondary Capture; Not for Clinical Use.",
            "AlgorithmName": f"{my_model_info.name}:{my_model_info.version}:{my_model_info.uid}"
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
        
        # Segmentation Metrics Operator - computes volume, slices, intensity stats
        # The label_dict will be provided by the inference operator based on organs to analyze
        seg_metrics_op = SegmentationMetricsOperator(self, name="seg_metrics_op", labels_dict = {"liver": 1, "spleen": 2}, use_gpu=True)

        # Segmentation Z-Score Operator - computes z-scores and percentiles, generates PDF report
        assets_path = model_path.parent / "assets"  # Assumes assets folder is in model directory
        seg_zscore_op = SegmentationZScoreOperator(
            self,
            assets_path=str(assets_path),
            generate_plots=True,
            name="seg_zscore_op",
            additional_metrics_map={'liver_hu': {'organ':'liver', 'metric':'mean_intensity'},}
        )
        
        # DICOM Encapsulated PDF Writer - creates DICOM PDF from zscore report
        custom_tags_pdf = {"SeriesDescription": "AI Generated Z-Score Report; Not for Clinical Use."}
        dicom_pdf_writer = DICOMEncapsulatedPDFWriterOperator(
            self,
            output_folder=app_output_path / "PDF",
            model_info=my_model_info,
            equipment_info=my_equipment_info,
            custom_tags=custom_tags_pdf,
            name="dicom_pdf_writer",
        )

        # DICOM Contour op - Creates Segmentation Contour DICOMs from segmentation
        dicom_contour_creator_op = SegmentationContourOperator(
            self,
            labels_dict = {"liver": 1, "spleen": 2},
        )
        
        # DICOM Overlay op - Creates Segmentation Overlay DICOMs from segmentation
        dicom_overlay_creator_op = SegmentationOverlayOperator(
            self,
            use_gpu=True,
        )
                
        # DICOM SR Writer op
        dicom_sr_writer = DICOMTextSRWriterOperator(
            self,
            # copy_tags is a default parameteter (type bool) of DICOMTextSRWriterOperator; default value is True
            # dictates whether or not to copy DICOM attributes from the selected DICOM series
            # changed to True to copy DICOM attributes so DICOM SR has same Study UID
            copy_tags=True,
            model_info=my_model_info,
            equipment_info=my_equipment_info,
            custom_tags=custom_tags_sr,
            # store DICOM SR in SR subdirectory; necessary for routing in CCHMC MDE workflow definition
            output_folder=app_output_path / "SR",
        )

        # # custom DICOM SC Writer op
        dicom_sc_writer = DICOMSCWriterOperator(
            self,
            model_info=my_model_info,
            equipment_info=my_equipment_info,
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
        self.add_flow(series_to_vol_op, abd_seg_op, {("image", "image")})
        
        # Segmentation Metrics Pipeline - compute metrics from segmentation and input scan
        #self.add_flow(abd_seg_op, seg_metrics_op, {("seg_image", "segmentation_mask")})
        self.add_flow(abd_seg_op, seg_metrics_op, {("seg_metatensor", "segmentation_mask")})
        self.add_flow(series_to_vol_op, seg_metrics_op, {("image", "input_scan")})
        
        # Z-Score Pipeline - compute z-scores from metrics and patient demographics
        self.add_flow(seg_metrics_op, seg_zscore_op, {("metrics_dict", "metrics_dict")})
        ## pass dicom study list to extract patient demographics
        self.add_flow(series_selector_op, seg_zscore_op, {("study_selected_series_list", "study_selected_series_list")})
        
        # DICOM SEG
        self.add_flow(
            series_selector_op, dicom_seg_writer, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(abd_seg_op, dicom_seg_writer, {("seg_image", "seg_image")})
        
        # DICOM PDF - encapsulated PDF with zscore plots
        self.add_flow(
            series_selector_op, dicom_pdf_writer, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(seg_zscore_op, dicom_pdf_writer, {("pdf_bytes", "pdf_bytes")})
        
        # DICOM SR - now uses zscore_dict output instead of text from transforms
        self.add_flow(
            series_selector_op, dicom_sr_writer, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(seg_zscore_op, dicom_sr_writer, {("zscore_dict", "dict")})
        
        # DICOM Contour Creation
        self.add_flow(
            abd_seg_op, dicom_contour_creator_op, {("seg_image", "segmentation_mask")}
        )
        
        # DICOM Overlay Creation
        self.add_flow(
            dicom_contour_creator_op, dicom_overlay_creator_op, {("contour", "segmentation_mask")}
        )
        self.add_flow(
            series_to_vol_op, dicom_overlay_creator_op, {("image", "input_scan")}
        )

        # note below the dicom_seg_writer, dicom_sr_writer, and dicom_sc_writer each require two inputs,
        # each coming from a source operator
        
        # DICOM SC
        self.add_flow(
            series_selector_op, dicom_sc_writer, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(dicom_overlay_creator_op, dicom_sc_writer, {("overlay", "input_overlay_image")})



# series selection rule in JSON, which selects for standard axial CT series:
# StudyDescription (Type 3): matches any value
# Modality (Type 1): matches "CT" value (case-insensitive); filters out non-CT modalities
# ImageOrientationPatient (Type 1): matches Axial orientation; filters out Cor and Sag orientations
# ImageType (Type 1): matches value that contains "PRIMARY"; filters out secondary and reformatted series
# SliceThickness (Type 2): matches ST values between 2 and 5, inclusive; filters out thin slices
# SeriesDescription (Type 3): matches any values that do not contain "cor" or "sag" (case-insensitive); filters out Cor and Sag views
# all valid series will be selected; downstream operators only perform inference and write outputs for 1st selected series
# please see more detail in DICOMSeriesSelectorOperator

Sample_Rules_Text = """
{
    "selections": [
        {
            "name": "Standard Axial CT Series",
            "conditions": {
                "StudyDescription": "(.*?)",
                "Modality": "(?i)CT",
                "ImageOrientationPatient": "Axial",
                "ImageType": ["PRIMARY"],
                "SliceThickness": [2, 5],
                "SeriesDescription": "(?i)^(?!.*(cor|sag)).*$"
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
    #     monai-deploy exec app.py -i input -m model/new_bundle.ts
    #
    logging.info(f"Begin {__name__}")
    AIAbdomenSegApp().run()
    logging.info(f"End {__name__}")
