# # Copyright 2021-2022 MONAI Consortium
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #     http://www.apache.org/licenses/LICENSE-2.0
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import logging

# # Required for setting SegmentDescription attributes. Direct import as this is not part of App SDK package.
# from pydicom.sr.codedict import codes

# import monai.deploy.core as md
# from monai.deploy.core import Application, resource
# from monai.deploy.core.domain import Image
# from monai.deploy.core.io_type import IOType
# from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
# from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator, SegmentDescription
# from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
# from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
# from monai.deploy.operators.monai_bundle_inference_operator import (
#     BundleConfigNames,
#     IOMapping,
#     MonaiBundleInferenceOperator,
# )


# @resource(cpu=1, gpu=1, memory="7Gi")
# # Enforcing torch>=1.12.0 because one of the Bundles/TorchScripts, Pancreas CT Seg, was created
# # with this version, and would fail to jit.load with lower version of torch.
# # The Bundle Inference Operator as of now only requires torch>=1.10.2, and does not yet dynamically
# # parse the MONAI bundle to get the required pip package or ver on initialization, hence it does not set
# # its own @env decorator accordingly when the app is being packaged into a MONAI Package.
# @md.env(pip_packages=["torch>=1.12.0"])
# # pip_packages can be a string that is a path(str) to requirements.txt file or a list of packages.
# # The monai pkg is not required by this class, instead by the included operators.
# class App(Application):
#     """This example demonstrates how to create a multi-model/multi-AI application.

#     The important steps are:
#         1. Place the model TorchScripts in a defined folder structure, see below for details
#         2. Pass the model name to the inference operator instance in the app
#         3. Connect the input to and output from the inference operators, as required by the app

#     Required Model Folder Structure:
#         1. The model TorchScripts, be it MONAI Bundle compliant or not, must be placed in
#            a parent folder, whose path is used as the path to the model(s) on app execution
#         2. Each TorchScript file needs to be in a sub-folder, whose name is the model name

#     An example is shown below, where the `parent_foler` name can be the app's own choosing, and
#     the sub-folder names become model names, `pancreas_ct_dints` and `spleen_model`, respectively.

#         <parent_fodler>
#         ├── pancreas_ct_dints
#         │   └── model.ts
#         └── spleen_ct
#             └── model.ts

#     Note:
#     1. The TorchScript files of MONAI Bundles can be downloaded from MONAI Model Zoo, at
#        https://github.com/Project-MONAI/model-zoo/tree/dev/models
#     2. The input DICOM instances are from a DICOM Series of CT Abdomen, similar to the ones
#        used in the Spleen Segmentation example
#     3. This example is purely for technical demonstration, not for clinical use
#     """

#     def __init__(self, *args, **kwargs):
#         """Creates an application instance."""
#         self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
#         super().__init__(*args, **kwargs)

#     def run(self, *args, **kwargs):
#         # This method calls the base class to run. Can be omitted if simply calling through.
#         self._logger.info(f"Begin {self.run.__name__}")
#         super().run(*args, **kwargs)
#         self._logger.info(f"End {self.run.__name__}")

#     def compose(self):
#         """Creates the app specific operators and chain them up in the processing DAG."""

#         logging.info(f"Begin {self.compose.__name__}")

#         # Create the custom operator(s) as well as SDK built-in operator(s).
#         study_loader_op = DICOMDataLoaderOperator()
#         series_selector_op = DICOMSeriesSelectorOperator(Sample_Rules_Text)
#         series_to_vol_op = DICOMSeriesToVolumeOperator()

#         # Create the inference operator that supports MONAI Bundle and automates the inference.
#         # The IOMapping labels match the input and prediction keys in the pre and post processing.
#         # The model_name is optional when the app has only one model.
#         # The bundle_path argument optionally can be set to an accessible bundle file path in the dev
#         # environment, so when the app is packaged into a MAP, the operator can complete the bundle parsing
#         # during init to provide the optional packages info, parsed from the bundle, to the packager
#         # for it to install the packages in the MAP docker image.
#         # Setting output IOType to DISK only works only for leaf operators, not the case in this example.
#         # When multiple models/bundles are supported, create an inference operator for each.
#         #
#         # Pertinent MONAI Bundle:
#         #   https://github.com/Project-MONAI/model-zoo/tree/dev/models/spleen_ct_segmentation, v0.3.2
#         #   https://github.com/Project-MONAI/model-zoo/tree/dev/models/pancreas_ct_dints_segmentation, v0.3

#         config_names = BundleConfigNames(config_names=["inference"])  # Same as the default

#         # This is the inference operator for the spleen_model bundle. Note the model name.
#         bundle_spleen_seg_op = MonaiBundleInferenceOperator(
#             input_mapping=[IOMapping("image", Image, IOType.IN_MEMORY)],
#             output_mapping=[IOMapping("pred", Image, IOType.IN_MEMORY)],
#             bundle_config_names=config_names,
#             model_name="spleen_ct",
#         )

#         # This is the inference operator for the pancreas_ct_dints bundle. Note the model name.
#         bundle_pancreas_seg_op = MonaiBundleInferenceOperator(
#             input_mapping=[IOMapping("image", Image, IOType.IN_MEMORY)],
#             output_mapping=[IOMapping("pred", Image, IOType.IN_MEMORY)],
#             model_name="pancreas_ct_dints",
#         )

#         # Create DICOM Seg writer providing the required segment description for each segment with
#         # the actual algorithm and the pertinent organ/tissue. The segment_label, algorithm_name,
#         # and algorithm_version are of DICOM VR LO type, limited to 64 chars.
#         # https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html
#         #
#         # NOTE: Each generated DICOM Seg will be a dcm file with the name based on the SOP instance UID.

#         # Description for the Spleen seg, and the seg writer obj
#         seg_descriptions_spleen = [
#             SegmentDescription(
#                 segment_label="Spleen",
#                 segmented_property_category=codes.SCT.Organ,
#                 segmented_property_type=codes.SCT.Spleen,
#                 algorithm_name="volumetric (3D) segmentation of the spleen from CT image",
#                 algorithm_family=codes.DCM.ArtificialIntelligence,
#                 algorithm_version="0.3.2",
#             )
#         ]

#         custom_tags_spleen = {"SeriesDescription": "AI Spleen Seg for research use only. Not for clinical use."}
#         dicom_seg_writer_spleen = DICOMSegmentationWriterOperator(
#             segment_descriptions=seg_descriptions_spleen, custom_tags=custom_tags_spleen
#         )

#         # Description for the Pancreas seg, and the seg writer obj
#         seg_descriptions_pancreas = [
#             SegmentDescription(
#                 segment_label="Pancreas",
#                 segmented_property_category=codes.SCT.Organ,
#                 segmented_property_type=codes.SCT.Pancreas,
#                 algorithm_name="volumetric (3D) segmentation of the pancreas from CT image",
#                 algorithm_family=codes.DCM.ArtificialIntelligence,
#                 algorithm_version="0.3.0",
#             )
#         ]
#         custom_tags_pancreas = {"SeriesDescription": "AI Pancreas Seg for research use only. Not for clinical use."}

#         dicom_seg_writer_pancreas = DICOMSegmentationWriterOperator(
#             segment_descriptions=seg_descriptions_pancreas, custom_tags=custom_tags_pancreas
#         )

#         # NOTE: Sharp eyed readers can already see that the above instantiation of object can be simply parameterized.
#         #       Very true, but leaving them as if for easy reading. In fact the whole app can be parameterized for general use.

#         # Create the processing pipeline, by specifying the upstream and downstream operators, and
#         # ensuring the output from the former matches the input of the latter, in both name and type.
#         self.add_flow(study_loader_op, series_selector_op, {"dicom_study_list": "dicom_study_list"})
#         self.add_flow(
#             series_selector_op, series_to_vol_op, {"study_selected_series_list": "study_selected_series_list"}
#         )

#         # Feed the input image to all inference operators
#         self.add_flow(series_to_vol_op, bundle_spleen_seg_op, {"image": "image"})
#         # The Pancreas CT Seg bundle requires PyTorch 1.12.0 to avoid failure to load.
#         self.add_flow(series_to_vol_op, bundle_pancreas_seg_op, {"image": "image"})

#         # Create DICOM Seg for one of the inference output
#         # Note below the dicom_seg_writer requires two inputs, each coming from a upstream operator.
#         self.add_flow(
#             series_selector_op, dicom_seg_writer_spleen, {"study_selected_series_list": "study_selected_series_list"}
#         )
#         self.add_flow(bundle_spleen_seg_op, dicom_seg_writer_spleen, {"pred": "seg_image"})

#         # Create DICOM Seg for one of the inference output
#         # Note below the dicom_seg_writer requires two inputs, each coming from a upstream operator.
#         self.add_flow(
#             series_selector_op, dicom_seg_writer_pancreas, {"study_selected_series_list": "study_selected_series_list"}
#         )
#         self.add_flow(bundle_pancreas_seg_op, dicom_seg_writer_pancreas, {"pred": "seg_image"})

#         logging.info(f"End {self.compose.__name__}")


# # This is a sample series selection rule in JSON, simply selecting CT series.
# # If the study has more than 1 CT series, then all of them will be selected.
# # Please see more detail in DICOMSeriesSelectorOperator.
# Sample_Rules_Text = """
# {
#     "selections": [
#         {
#             "name": "CT Series",
#             "conditions": {
#                 "StudyDescription": "(.*?)",
#                 "Modality": "(?i)CT",
#                 "SeriesDescription": "(.*?)"
#             }
#         }
#     ]
# }
# """

# if __name__ == "__main__":
#     # Creates the app and test it standalone. When running is this mode, please note the following:
#     #     -m <model file>, for model file path
#     #     -i <DICOM folder>, for input DICOM CT series folder
#     #     -o <output folder>, for the output folder, default $PWD/output
#     # e.g.
#     #     monai-deploy exec app.py -i input -m model/model.ts
#     #
#     logging.basicConfig(level=logging.DEBUG)
#     app_instance = App(do_run=True)
