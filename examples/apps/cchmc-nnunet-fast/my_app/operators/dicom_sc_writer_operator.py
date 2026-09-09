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
import os
from pathlib import Path
from typing import Dict, Optional, Union

import pydicom

from monai.deploy.core import Fragment, Operator, OperatorSpec
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries
from monai.deploy.operators.dicom_utils import EquipmentInfo, ModelInfo, write_common_modules
from monai.deploy.utils.importutil import optional_import
from monai.deploy.utils.version import get_sdk_semver

dcmread, _ = optional_import("pydicom", name="dcmread")
dcmwrite, _ = optional_import("pydicom.filewriter", name="dcmwrite")
generate_uid, _ = optional_import("pydicom.uid", name="generate_uid")
ImplicitVRLittleEndian, _ = optional_import("pydicom.uid", name="ImplicitVRLittleEndian")
Dataset, _ = optional_import("pydicom.dataset", name="Dataset")
FileDataset, _ = optional_import("pydicom.dataset", name="FileDataset")
Sequence, _ = optional_import("pydicom.sequence", name="Sequence")


class DICOMSCWriterOperator(Operator):
    """Class to write a new DICOM Secondary Capture (DICOM SC) instance with source DICOM Series metadata included.

    Named inputs:
        dicom_sc_dir: file path of temporary DICOM SC (w/o source DICOM Series metadata).
        study_selected_series_list: DICOM Series for copying metadata from.

    Named output:
        None.

    File output:
        New, updated DICOM SC file (with source DICOM Series metadata) in the provided output folder.
    """

    # file extension for the generated DICOM Part 10 file
    DCM_EXTENSION = ".dcm"
    # the default output folder for saving the generated DICOM instance file
    # DEFAULT_OUTPUT_FOLDER = Path(os.path.join(os.path.dirname(__file__))) / "output"
    DEFAULT_OUTPUT_FOLDER = Path.cwd() / "output"

    def __init__(
        self,
        fragment: Fragment,
        *args,
        output_folder: Union[str, Path],
        model_info: ModelInfo,
        equipment_info: Optional[EquipmentInfo] = None,
        custom_tags: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Class to write a new DICOM Secondary Capture (DICOM SC) instance with source DICOM Series metadata.

        Args:
            output_folder (str or Path): The folder for saving the generated DICOM SC instance file.
            model_info (ModelInfo): Object encapsulating model creator, name, version and UID.
            equipment_info (EquipmentInfo, optional): Object encapsulating info for DICOM Equipment Module.
                                                      Defaults to None.
            custom_tags (Dict[str, str], optional): Dictionary for setting custom DICOM tags using Keywords and str values only.
                                                    Defaults to None.

        Raises:
            ValueError: If result cannot be found either in memory or from file.
        """

        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")

        # need to init the output folder until the execution context supports dynamic FS path
        # not trying to create the folder to avoid exception on init
        self.output_folder = Path(output_folder) if output_folder else DICOMSCWriterOperator.DEFAULT_OUTPUT_FOLDER
        self.input_name_sc_dir = "dicom_sc_dir"
        self.input_name_study_series = "study_selected_series_list"

        # for copying DICOM attributes from a provided DICOMSeries
        # required input for write_common_modules; will always be True for this implementation
        self.copy_tags = True

        self.model_info = model_info if model_info else ModelInfo()
        self.equipment_info = equipment_info if equipment_info else EquipmentInfo()
        self.custom_tags = custom_tags

        # set own Modality and SOP Class UID
        # Standard SOP Classes: https://dicom.nema.org/dicom/2013/output/chtml/part04/sect_B.5.html
        # Modality, e.g.,
        #   "OT" for PDF
        #   "SR" for Structured Report.
        # Media Storage SOP Class UID, e.g.,
        #   "1.2.840.10008.5.1.4.1.1.88.11" for Basic Text SR Storage
        #   "1.2.840.10008.5.1.4.1.1.104.1" for Encapsulated PDF Storage,
        #   "1.2.840.10008.5.1.4.1.1.88.34" for Comprehensive 3D SR IOD
        #   "1.2.840.10008.5.1.4.1.1.66.4" for Segmentation Storage
        self.modality_type = "OT"  # OT Modality for Secondary Capture
        self.sop_class_uid = (
            "1.2.840.10008.5.1.4.1.1.7.4"  # SOP Class UID for Multi-frame True Color Secondary Capture Image Storage
        )
        # custom OverlayImageLabeld post-processing transform creates an RBG overlay

        # equipment version may be different from contributing equipment version
        try:
            self.software_version_number = get_sdk_semver()  # SDK Version
        except Exception:
            self.software_version_number = ""
        self.operators_name = f"AI Algorithm {self.model_info.name}"

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Set up the named input(s), and output(s) if applicable.

        This operator does not have an output for the next operator, rather file output only.

        Args:
            spec (OperatorSpec): The Operator specification for inputs and outputs etc.
        """

        spec.input(self.input_name_sc_dir)
        spec.input(self.input_name_study_series)

    def compute(self, op_input, op_output, context):
        """Performs computation for this operator and handles I/O.

        For now, only a single result content is supported, which could be in memory or an accessible file.
        The DICOM Series used during inference is required (and copy_tags is hardcoded to True).

        When there are multiple selected series in the input, the first series' containing study will
        be used for retrieving DICOM Study module attributes, e.g. StudyInstanceUID.

        Raises:
            NotADirectoryError: When temporary DICOM SC path is not a directory.
            FileNotFoundError: When result object not in the input, and result file not found either.
            ValueError: Content object and file path not in the inputs, or no DICOM series provided.
            IOError: If the input content is blank.
        """

        # receive the temporary DICOM SC file path and study selected series list
        dicom_sc_dir = Path(op_input.receive(self.input_name_sc_dir))
        if not dicom_sc_dir:
            raise IOError("Temporary DICOM SC path is read but blank.")
        if not dicom_sc_dir.is_dir():
            raise NotADirectoryError(f"Provided temporary DICOM SC path is not a directory: {dicom_sc_dir}")
        self._logger.info(f"Received temporary DICOM SC path: {dicom_sc_dir}")

        study_selected_series_list = op_input.receive(self.input_name_study_series)
        if not study_selected_series_list or len(study_selected_series_list) < 1:
            raise ValueError("Missing input, list of 'StudySelectedSeries'.")

        # retrieve the DICOM Series used during inference in order to grab appropriate study/series level tags
        # this will be the 1st Series in study_selected_series_list
        dicom_series = None
        for study_selected_series in study_selected_series_list:
            if not isinstance(study_selected_series, StudySelectedSeries):
                raise ValueError(f"Element in input is not expected type, {StudySelectedSeries}.")
            selected_series = study_selected_series.selected_series[0]
            dicom_series = selected_series.series
            break

        # log basic DICOM metadata for the retrieved DICOM Series
        self._logger.debug(f"Dicom Series: {dicom_series}")

        # the output folder should come from the execution context when it is supported
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # write the new DICOM SC instance
        self.write(dicom_sc_dir, dicom_series, self.output_folder)

    def write(self, dicom_sc_dir, dicom_series: DICOMSeries, output_dir: Path):
        """Writes a new, updated DICOM SC instance and deletes the temporary DICOM SC instance.
        The new, updated DICOM SC instance is the temporary DICOM SC instance with source
        DICOM Series metadata copied.

        Args:
            dicom_sc_dir: temporary DICOM SC file path.
            dicom_series (DICOMSeries): DICOMSeries object encapsulating the original series.

        Returns:
            None

        File output:
           New, updated DICOM SC file (with source DICOM Series metadata) in the provided output folder.
        """

        if not isinstance(output_dir, Path):
            raise ValueError("output_dir is not a valid Path.")

        output_dir.mkdir(parents=True, exist_ok=True)  # just in case

        # find the temporary DICOM SC file in the directory; there should only be one .dcm file present
        dicom_files = list(dicom_sc_dir.glob("*.dcm"))
        dicom_sc_file = dicom_files[0]

        # load the temporary DICOM SC file using pydicom
        dicom_sc_dataset = pydicom.dcmread(dicom_sc_file)
        self._logger.info(f"Loaded temporary DICOM SC file: {dicom_sc_file}")

        # use write_common_modules to copy metadata from dicom_series
        # this will copy metadata and return an updated Dataset
        ds = write_common_modules(
            dicom_series,
            self.copy_tags,  # always True for this implementation
            self.modality_type,
            self.sop_class_uid,
            self.model_info,
            self.equipment_info,
        )

        # Secondary Capture specific tags
        ds.ImageType = ["DERIVED", "SECONDARY"]

        # for now, only allow str Keywords and str value
        if self.custom_tags:
            for k, v in self.custom_tags.items():
                if isinstance(k, str) and isinstance(v, str):
                    try:
                        ds.update({k: v})
                    except Exception as ex:
                        # best effort for now
                        logging.warning(f"Tag {k} was not written, due to {ex}")

        # merge the copied metadata into the loaded temporary DICOM SC file (dicom_sc_dataset)
        for tag, value in ds.items():
            dicom_sc_dataset[tag] = value

        # save the updated DICOM SC file to the output folder
        # instance file name is the same as the new SOP instance UID
        output_file_path = self.output_folder.joinpath(
            f"{dicom_sc_dataset.SOPInstanceUID}{DICOMSCWriterOperator.DCM_EXTENSION}"
        )
        dicom_sc_dataset.save_as(output_file_path)
        self._logger.info(f"Saved updated DICOM SC file at: {output_file_path}")

        # remove the temporary DICOM SC file
        os.remove(dicom_sc_file)
        self._logger.info(f"Removed temporary DICOM SC file: {dicom_sc_file}")

        # check if the temp directory is empty, then delete it
        if not any(dicom_sc_dir.iterdir()):
            os.rmdir(dicom_sc_dir)
            self._logger.info(f"Removed temporary directory: {dicom_sc_dir}")
        else:
            self._logger.warning(f"Temporary directory {dicom_sc_dir} is not empty, skipping removal.")
