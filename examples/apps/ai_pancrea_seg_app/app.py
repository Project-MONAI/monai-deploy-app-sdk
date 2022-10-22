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

# Required for setting SegmentDescription attributes. Direct import as this is not part of App SDK package.
from pydicom.sr.codedict import codes

import monai.deploy.core as md
from monai.deploy.core import Application, resource
from monai.deploy.core.domain import Image
from monai.deploy.core.io_type import IOType
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator, SegmentDescription
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.monai_bundle_inference_operator import (
    BundleConfigNames,
    IOMapping,
    MonaiBundleInferenceOperator,
)

# from monai.deploy.operators.stl_conversion_operator import STLConversionOperator  # import as needed.


@resource(cpu=1, gpu=1, memory="7Gi")
@md.env(pip_packages=["torch>=1.12.0"])
# pip_packages can be a string that is a path(str) to requirements.txt file or a list of packages.
# The monai pkg is not required by this class, instead by the included operators.
class AIPancreasSegApp(Application):
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

        logging.info(f"Begin {self.compose.__name__}")

        # Create the custom operator(s) as well as SDK built-in operator(s).
        study_loader_op = DICOMDataLoaderOperator()
        series_selector_op = DICOMSeriesSelectorOperator(Sample_Rules_Text)
        series_to_vol_op = DICOMSeriesToVolumeOperator()

        # Create the inference operator that supports MONAI Bundle and automates the inference.
        # The IOMapping labels match the input and prediction keys in the pre and post processing.
        # The model_name is optional when the app has only one model.
        # The bundle_path argument optionally can be set to an accessible bundle file path in the dev
        # environment, so when the app is packaged into a MAP, the operator can complete the bundle parsing
        # during init to provide the optional packages info, parsed from the bundle, to the packager
        # for it to install the packages in the MAP docker image.
        # Setting output IOType to DISK only works only for leaf operators, not the case in this example.
        #
        # Pertinent MONAI Bundle:
        #   https://github.com/Project-MONAI/model-zoo/tree/dev/models/spleen_ct_segmentation

        bundle_spleen_seg_op = MonaiBundleInferenceOperator(
            input_mapping=[IOMapping("image", Image, IOType.IN_MEMORY)],
            output_mapping=[IOMapping("pred", Image, IOType.IN_MEMORY)],
            bundle_config_names=BundleConfigNames(config_names=["inference"]),  # Same as the default
        )

        # Create DICOM Seg writer providing the required segment description for each segment with
        # the actual algorithm and the pertinent organ/tissue. The segment_label, algorithm_name,
        # and algorithm_version are of DICOM VR LO type, limited to 64 chars.
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html
        segment_descriptions = [
            SegmentDescription(
                segment_label="Pancreas",
                segmented_property_category=codes.SCT.Organ,
                segmented_property_type=codes.SCT.Pancreas,
                algorithm_name="volumetric (3D) segmentation of the pancreas from CT image",
                algorithm_family=codes.DCM.ArtificialIntelligence,
                algorithm_version="0.3.0",
            )
        ]

        custom_tags = {"SeriesDescription": "AI generated Seg for research use only. Not for clinical use."}

        dicom_seg_writer = DICOMSegmentationWriterOperator(
            segment_descriptions=segment_descriptions, custom_tags=custom_tags
        )

        # Create the processing pipeline, by specifying the source and destination operators, and
        # ensuring the output from the former matches the input of the latter, in both name and type.
        self.add_flow(study_loader_op, series_selector_op, {"dicom_study_list": "dicom_study_list"})
        self.add_flow(
            series_selector_op, series_to_vol_op, {"study_selected_series_list": "study_selected_series_list"}
        )
        self.add_flow(series_to_vol_op, bundle_spleen_seg_op, {"image": "image"})
        # Note below the dicom_seg_writer requires two inputs, each coming from a source operator.
        self.add_flow(
            series_selector_op, dicom_seg_writer, {"study_selected_series_list": "study_selected_series_list"}
        )
        self.add_flow(bundle_spleen_seg_op, dicom_seg_writer, {"pred": "seg_image"})
        # Create the surface mesh STL conversion operator and add it to the app execution flow, if needed, by
        # uncommenting the following couple lines.
        # stl_conversion_op = STLConversionOperator(output_file="stl/spleen.stl")
        # self.add_flow(bundle_spleen_seg_op, stl_conversion_op, {"pred": "image"})

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
    logging.basicConfig(level=logging.DEBUG)
    app_instance = AIPancreasSegApp(do_run=True)
