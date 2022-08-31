# Copyright 2021 MONAI Consortium
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
from unetr_seg_operator import UnetrSegOperator

from monai.deploy.core import Application, resource
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator, SegmentDescription
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.publisher_operator import PublisherOperator
from monai.deploy.operators.stl_conversion_operator import STLConversionOperator


@resource(cpu=1, gpu=1, memory="7Gi")
# pip_packages can be a string that is a path(str) to requirements.txt file or a list of packages.
# The MONAI pkg is not required by this class, instead by the included operators.
class AIUnetrSegApp(Application):
    def __init__(self, *args, **kwargs):
        """Creates an application instance."""

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        # This method calls the base class to run. Can be omitted if simply calling through.
        self._logger.debug(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._logger.debug(f"End {self.run.__name__}")

    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""

        self._logger.debug(f"Begin {self.compose.__name__}")
        # Creates the custom operator(s) as well as SDK built-in operator(s).
        study_loader_op = DICOMDataLoaderOperator()
        series_selector_op = DICOMSeriesSelectorOperator()
        series_to_vol_op = DICOMSeriesToVolumeOperator()
        # Model specific inference operator, supporting MONAI transforms.
        unetr_seg_op = UnetrSegOperator()
        # Create the publisher operator
        publisher_op = PublisherOperator()
        # Create the surface mesh STL conversion operator, for all segments
        stl_conversion_op = STLConversionOperator(
            output_file="stl/multi-organs.stl", keep_largest_connected_component=False
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

        dicom_seg_writer = DICOMSegmentationWriterOperator(segment_descriptions)

        # Create the processing pipeline, by specifying the source and destination operators, and
        # ensuring the output from the former matches the input of the latter, in both name and type.
        self.add_flow(study_loader_op, series_selector_op, {"dicom_study_list": "dicom_study_list"})
        self.add_flow(
            series_selector_op, series_to_vol_op, {"study_selected_series_list": "study_selected_series_list"}
        )
        self.add_flow(series_to_vol_op, unetr_seg_op, {"image": "image"})
        self.add_flow(unetr_seg_op, stl_conversion_op, {"seg_image": "image"})

        # Add the publishing operator to save the input and seg images for Render Server.
        # Note the PublisherOperator has temp impl till a proper rendering module is created.
        self.add_flow(unetr_seg_op, publisher_op, {"saved_images_folder": "saved_images_folder"})

        # Note below the dicom_seg_writer requires two inputs, each coming from a source operator.
        self.add_flow(
            series_selector_op, dicom_seg_writer, {"study_selected_series_list": "study_selected_series_list"}
        )
        self.add_flow(unetr_seg_op, dicom_seg_writer, {"seg_image": "seg_image"})

        self._logger.debug(f"End {self.compose.__name__}")


if __name__ == "__main__":
    # Creates the app and test it standalone. When running is this mode, please note the following:
    #     -m <model file>, for model file path
    #     -i <DICOM folder>, for input DICOM CT series folder
    #     -o <output folder>, for the output folder, default $PWD/output
    # e.g.
    #     python3 app.py -i input -m model/model.ts
    #
    import shutil
    import traceback
    from pathlib import Path

    logging.basicConfig(level=logging.DEBUG)
    # This main function is an example to show how a batch of input can be processed.
    # It assumes that in the app input folder there are a number of subfolders, each
    # containing a discrete input to be processed. Each discrete payload can have
    # multiple DICOM instances file, optionally organized in its own folder structure.
    # The application object is first created, and on its init the model network is
    # loaded as well as pre and post processing transforms. This app object is then
    # run multiple times, each time with a single discrete payload.

    app = AIUnetrSegApp(do_run=False)

    # Preserve the application top level input and output folder path, as the path
    # in the context may change on each run if the I/O arguments are passed in.
    app_input_path = Path(app.context.input_path)
    app_output_path = Path(app.context.output_path)

    # Get subfolders in the input path, assume each one contains a discrete payload
    input_dirs = [path for path in app_input_path.iterdir() if path.is_dir()]

    # Set the output path for each run under the app's output path, and do run
    work_dirs = []
    for idx, dir in enumerate(input_dirs):
        try:
            output_path = app_output_path / f"{dir.name}_output"
            # Note: the work_dir should be mapped to the host drive when used in
            #       a container for better performance.
            work_dir = f".unetr_app_workdir{idx}"
            work_dirs.extend(work_dir)

            logging.info(f"Start processing input in: {dir} with results in: {output_path}")

            # Run app with specific input and output path.
            # Passing in the input and output do have the side effect of changing
            # app context. This side effect will likely be eliminated in later releases.
            app.run(input=dir, output=output_path, workdir=work_dir)

            logging.info(f"Completed processing input in: {dir} with results in: {output_path}")
        except Exception as ex:
            logging.error(f"Failed processing input in {dir}, due to: {ex}\n")
            traceback.print_exc()
        finally:
            # Remove the workdir; alternatively do this later, if storage space is not a concern.
            shutil.rmtree(work_dir, ignore_errors=True)

    # Alternative. Explicitly remove the working dirs at the end of main.
    # [shutil.rmtree(work_dir, ignore_errors=True) for work_dir in work_dirs]
