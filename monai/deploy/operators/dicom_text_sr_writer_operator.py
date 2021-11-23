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

import datetime
import logging
from pathlib import Path
from random import randint
from typing import Dict, List, Text, Union

from monai.deploy.utils.importutil import optional_import
from monai.deploy.utils.version import get_sdk_semver

dcmread, _ = optional_import("pydicom", name="dcmread")
dcmwrite, _ = optional_import("pydicom.filewriter", name="dcmwrite")
generate_uid, _ = optional_import("pydicom.uid", name="generate_uid")
ImplicitVRLittleEndian, _ = optional_import("pydicom.uid", name="ImplicitVRLittleEndian")
Dataset, _ = optional_import("pydicom.dataset", name="Dataset")
FileDataset, _ = optional_import("pydicom.dataset", name="FileDataset")
Sequence, _ = optional_import("pydicom.sequence", name="Sequence")

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries
from monai.deploy.exceptions import ItemNotExistsError


# Utility classes considered to be moved into Domain module
class ModelInfo(object):
    """Class encapsulating AI model information, according to IHE AI Results (AIR) Rev 1.1.

    The attributes of the class will be used to populate the Contributing Equipment Sequence in the DICOM IOD
    per IHE AIR Rev 1.1, Section 6.5.3.1 General Result Encoding Requirements, as the following,

    The Creator shall describe each algorithm that was used to generate the results in the
    Contributing Equipment Sequence (0018,A001). Multiple items may be included. The Creator
    shall encode the following details in the Contributing Equipment Sequence,
        - Purpose of Reference Code Sequence (0040,A170) shall be (Newcode1, 99IHE, 1630 "Processing Algorithm")
        - Manufacturer (0008,0070)
        - Manufacturer's Model Name (0008,1090)
        - Software Versions (0018,1020)
        - Device UID (0018,1002)

    Each time an AI Model is modified, for example by training, it would be appropriate to update
    the Device UID.
    """

    def __init__(self, creator: str = "", name: str = "", version: str = "", uid: str = ""):

        self.creator = creator if isinstance(creator, str) else ""
        self.name = name if isinstance(name, str) else ""
        self.version = version if isinstance(version, str) else ""
        self.uid = uid if isinstance(uid, str) else ""


class EquipmentInfo(object):
    """Class encapsulating attributes required for DICOM Equipment Module."""

    def __init__(
        self,
        manufacturer: str = "MONAI Deploy",
        manufacturer_model: str = "MONAI Deploy App SDK",
        series_number: str = "0000",
        software_version_number: str = "0.2",
    ):

        self.manufacturer = manufacturer if isinstance(manufacturer, str) else ""
        self.manufacturer_model = manufacturer_model if isinstance(manufacturer_model, str) else ""
        self.series_number = series_number if isinstance(series_number, str) else ""
        try:
            version_str = get_sdk_semver()  # SDK Version
        except Exception:
            version_str = "0.2"  # Fall back to the initial version
        self.software_version_number = (
            software_version_number if isinstance(software_version_number, str) else version_str
        )


# The SR writer operator class
@md.input("classification_result", Text, IOType.IN_MEMORY)
@md.input("classification_result_file", DataPath, IOType.DISK)
@md.input("study_selected_series_list", List[StudySelectedSeries], IOType.IN_MEMORY)
@md.output("dicom_instance", DataPath, IOType.DISK)
@md.env(pip_packages=["pydicom >= 1.4.2"])
class DICOMTextSRWriterOperator(Operator):

    DCM_EXTENSION = ".dcm"

    def __init__(
        self,
        copy_tags: bool,
        model_info: ModelInfo,
        equipment_info: Union[EquipmentInfo, None] = None,
        custom_tags: Union[Dict[str, str], None] = None,
        *args,
        **kwargs,
    ):
        """Class to write DICOM SR SOP Instance for AI textual result in memeory or in a file.

        Args:
            copy_tags (bool): True for copying DICOM attributes from a provided DICOMSeries.
            model_info (ModelInfo): Object encapsulating model creator, name, version and UID.
            equipment_info (EquipmentInfo, optional): Object encapsulating info for DICOM Equipment Module.
                                                      Defaults to None.
            custom_tags (Dict[str, str], optional): Dictionary for setting custom DICOM tags using Keywords and str values only.
                                                    Defaults to None.

        Raises:
            ValueError: If copy_tags is true and no DICOMSeries object provided, or
                        if result cannot be found either in memory or from file.
        """
        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.copy_tags = copy_tags
        self.model_info = model_info if model_info else ModelInfo()
        self.equipment_info = equipment_info if equipment_info else EquipmentInfo()
        self.custom_tags = custom_tags

        # Set own Modality and SOP Class UID
        # Modality, e.g.,
        #   "OT" for PDF
        #   "SR" for Structured Report.
        # Media Storage SOP Class UID, e.g.,
        #   "1.2.840.10008.5.1.4.1.1.104.1" for Encapsulated PDF Storage,
        #   "1.2.840.10008.5.1.4.1.1.88.34" for Comprehensive 3D SR IOD
        #   "1.2.840.10008.5.1.4.1.1.66.4" for Segmentation Storage
        self.modality_type = "SR"
        self.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.34"
        self.implementation_version_name = "MONAI Deploy App SDK 0.2"
        self.operators_name = f"AI Algorithm {self.model_info.name}"

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        """Performs computation for this operator and handles I/O.

        For now, only a single image object or result content is supported and the selected DICOM
        series for inference is required, because the generated IOD needs to refer to original instance.
        When there are multiple selected series in the input, the first series' containing study will
        be used for retrieving DICOM Study module attributes, e.g. StudyInstanceUID.

        Raises:
            FileNotFoundError: When result object not in the input, and result file not found either.
            ValueError: Neither image object nor image file's folder is in the input, or no selected series.
        """

        # Gets the input, prepares the output folder, and then delegates the processing.
        result_text = ""
        try:
            result_text = str(op_input.get("classification_result")).strip()
        except ItemNotExistsError:
            try:
                file_path = op_input.get("classification_result_file")
            except ItemNotExistsError:
                raise ValueError("None of the named inputs for result can be found.") from None
            # Read file, and if exception, let it bubble up
            with open(file_path.path, "r") as f:
                result_text = f.read().strip()

        if not result_text:
            raise IOError("Input is read but blank.")

        try:
            study_selected_series_list = op_input.get("study_selected_series_list")
        except ItemNotExistsError:
            study_selected_series_list = None

        dicom_series = None  # It can be None of copy_tags is false.
        if self.copy_tags:
            # Get the first DICOM Series, as for now, only expecting this.
            if not study_selected_series_list or len(study_selected_series_list) < 1:
                raise ValueError("Missing input, list of 'StudySelectedSeries'.")
            for study_selected_series in study_selected_series_list:
                if not isinstance(study_selected_series, StudySelectedSeries):
                    raise ValueError("Element in input is not expected type, 'StudySelectedSeries'.")
                for selected_series in study_selected_series.selected_series:
                    dicom_series = selected_series.series

        output_dir = op_output.get().path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Now ready to starting writing the DICOM instance
        self.write(result_text, dicom_series, output_dir)

    def write(self, content_text, dicom_series: Union[DICOMSeries, None], output_dir: Path):
        """Writes DICOM object

        Args:
            content_file (str): file containing the contents
            dicom_series (DicomSeries): DicomSeries object encapsulating the original series.
            model_info (MoelInfo): Object encapsulating model creator, name, version and UID.

        Returns:
            PyDicom Dataset
        """
        self._logger.debug("Writing DICOM object...\n{}")
        output_dir.mkdir(parents=True, exist_ok=True)  # Just in case

        ds = DICOMTextSRWriterOperator.write_common_modules(
            dicom_series, self.copy_tags, self.modality_type, self.sop_class_uid, self.model_info, self.equipment_info
        )

        # SR specific
        ds.VerificationFlag = "UNVERIFIED"  # Not attested by a legally accountable person.

        # Per recommendation of IHE Radiology Technical Framework Supplement
        # AI Results (AIR) Rev1.1 - Trial Implementation
        # Specifically for Qualitative Findings,
        # Qualitative findings shall be encoded in an instance of the DICOM Comprehensive 3D SR SOP
        # Class using TID 1500 (Measurement Report) as the root template.
        # DICOM PS3.16: TID 1500 Measurement Report
        # http://dicom.nema.org/medical/dicom/current/output/chtml/part16/chapter_A.html#sect_TID_1500
        # The value for Procedure Reported (121058, DCM, "Procedure reported") shall describe the
        # imaging procedure analyzed, not the algorithm used.

        # Use text value for example
        ds.ValueType = "TEXT"

        # ConceptNameCode Sequence
        seq_concept_name_code = Sequence()
        ds_concept_name_code = Dataset()
        ds_concept_name_code.CodeValue = "18748-4"
        ds_concept_name_code.CodingSchemeDesignator = "LN"
        ds_concept_name_code.CodeMeaning = "Diagnostic Imaging Report"
        seq_concept_name_code.append(ds_concept_name_code)
        ds.ConceptNameCodeSequence = seq_concept_name_code
        ds.TextValue = content_text  # self._content_to_string(content_file)

        # For now, only allow str Keywords and str value
        if self.custom_tags:
            for k, v in self.custom_tags.items():
                if isinstance(k, str) and isinstance(v, str):
                    try:
                        ds.update({k: v})
                    except Exception as ex:
                        # Best effort for now.
                        logging.warning(f"Tag {k} was not written, due to {ex}")

        # Create the dcm file name, based on series instance UID, then save it.
        file_name = f"{ds.SeriesInstanceUID}_{ds.Modality}{DICOMTextSRWriterOperator.DCM_EXTENSION}"
        file_path = output_dir.joinpath(file_name)
        self.save_dcm_file(ds, file_path)

    @staticmethod
    def save_dcm_file(data_set, file_path: Path, validate_readable: bool = True):
        logging.debug(f"DICOM dataset to be written:{data_set}")

        # Write out the DCM file
        if file_path:
            dcmwrite(str(file_path), data_set, write_like_original=False)
            logging.info(f"Finished writing DICOM instance to file {file_path}")
        if validate_readable:
            # Test reading back
            _ = dcmread(str(file_path))

    # TODO: The following function can be moved into Domain module as it's common.
    @staticmethod
    def write_common_modules(
        dicom_series: Union[DICOMSeries, None],
        copy_tags: bool,
        modality_type: str,
        sop_class_uid: str,
        model_info: Union[ModelInfo, None] = None,
        equipment_info: Union[EquipmentInfo, None] = None,
    ):
        """Writes DICOM object common modules with or without a reference DCIOM Series

        Common modules include Study, Patient, Equipment, Series, and SOP common.

        Args:
            dicom_series (DicomSeries): DicomSeries object encapsulating the original series.
            copy_tags (bool): If true, dicom_series must be provided for copying tags.
            modality_type (str): DICOM Modality Type, e.g. SR.
            sop_class_uid (str): Media Storage SOP Class UID, e.g. "1.2.840.10008.5.1.4.1.1.88.34" for Comprehensive 3D SR IOD.
            model_info (MoelInfo): Object encapsulating model creator, name, version and UID.
            equipment_info(EquipmentInfo): Object encapsulating attributes for DICOM Equipment Module

        Returns:
            pydicom Dataset

        Raises:
            ValueError: When dicom_series is not a DICOMSeries object, and new_study is not True.
        """

        if copy_tags:
            if not isinstance(dicom_series, DICOMSeries):
                raise ValueError("A DICOMSeries object is required for coping tags.")

            if len(dicom_series.get_sop_instances()) < 1:
                raise ValueError("DICOMSeries must have at least one SOP instance.")

            # Get one of the SOP instance's native sop instance dataset
            orig_ds = dicom_series.get_sop_instances()[0].get_native_sop_instance()

        logging.debug("Writing DICOM common modules...")

        # Get and format date and time per DICOM standards.
        dt_now = datetime.datetime.now()
        date_now_dcm = dt_now.strftime("%Y%m%d")
        time_now_dcm = dt_now.strftime("%H%M%S")

        # Generate UIDs and descriptions
        my_sop_instance_uid = generate_uid()
        my_series_instance_uid = generate_uid()
        my_series_description = "CAUTION: Not for Diagnostic Use, for research use only."
        my_series_number = str(DICOMTextSRWriterOperator.random_with_n_digits(4))  # 4 digit number to avoid conflict
        my_study_instance_uid = orig_ds.StudyInstanceUID if copy_tags else generate_uid()

        # File meta info data set
        file_meta = Dataset()
        file_meta.FileMetaInformationGroupLength = 198
        file_meta.FileMetaInformationVersion = bytes("01", "utf-8")  # '\x00\x01'

        file_meta.MediaStorageSOPClassUID = sop_class_uid
        file_meta.MediaStorageSOPInstanceUID = my_sop_instance_uid
        file_meta.TransferSyntaxUID = ImplicitVRLittleEndian  # 1.2.840.10008.1.2, Little Endian Implicit VR
        file_meta.ImplementationClassUID = "1.2.40.0.13.1.1.1"  # Made up. Not registered.
        file_meta.ImplementationVersionName = "MONAI Deploy App SDK 0.2"

        # Write modules to data set
        ds = Dataset()
        ds.file_meta = file_meta
        ds.is_implicit_VR = True
        ds.is_little_endian = True

        # Content Date (0008,0023) and Content Time (0008,0033) are defined to be the date and time that
        # the document content creation started. In the context of analysis results, these may be considered
        # to be the date and time that the analysis that generated the result(s) started executing.
        # Use current time for now, but could potentially use the actual inference start time.
        ds.ContentDate = date_now_dcm
        ds.ContentTime = time_now_dcm

        # The date and time that the original generation of the data in the document started.
        ds.AcquisitionDateTime = date_now_dcm + time_now_dcm  # Result has just been created.

        # Patient Module, mandatory.
        # Copy over from the original DICOM metadata.
        ds.PatientName = orig_ds.get("PatientName", "") if copy_tags else ""
        ds.PatientID = orig_ds.get("PatientID", "") if copy_tags else ""
        ds.IssuerOfPatientID = orig_ds.get("IssuerOfPatientID", "") if copy_tags else ""
        ds.PatientBirthDate = orig_ds.get("PatientBirthDate", "") if copy_tags else ""
        ds.PatientSex = orig_ds.get("PatientSex", "") if copy_tags else ""

        # Study Module, mandatory
        # Copy over from the original DICOM metadata.
        ds.StudyDate = orig_ds.get("StudyDate", "") if copy_tags else date_now_dcm
        ds.StudyTime = orig_ds.get("StudyTime", "") if copy_tags else time_now_dcm
        ds.AccessionNumber = orig_ds.get("AccessionNumber", "") if copy_tags else ""
        ds.StudyDescription = orig_ds.get("StudyDescription", "") if copy_tags else "AI results."
        ds.StudyInstanceUID = my_study_instance_uid
        ds.StudyID = orig_ds.get("StudyID", "") if copy_tags else "1"
        ds.ReferringPhysicianName = orig_ds.get("ReferringPhysicianName", "") if copy_tags else ""

        # Equipment Module, mandatory
        if equipment_info:
            ds.Manufacturer = equipment_info.manufacturer
            ds.ManufacturerModel = equipment_info.manufacturer_model
            ds.SeriesNumber = equipment_info.series_number
            ds.SoftwareVersionNumber = equipment_info.software_version_number

        # SOP Common Module, mandatory
        ds.InstanceCreationDate = date_now_dcm
        ds.InstanceCreationTime = time_now_dcm
        ds.SOPClassUID = sop_class_uid
        ds.SOPInstanceUID = my_sop_instance_uid
        ds.InstanceNumber = "1"
        ds.SpecificCharacterSet = "ISO_IR 100"

        # Series Module, mandatory
        ds.Modality = modality_type
        ds.SeriesInstanceUID = my_series_instance_uid
        ds.SeriesNumber = my_series_number
        ds.SeriesDescription = my_series_description
        ds.SeriesDate = date_now_dcm
        ds.SeriesTime = time_now_dcm
        # Body part copied over, although not mandatory depending on modality
        ds.BodyPartExamined = orig_ds.get("BodyPartExamined", "") if copy_tags else ""
        ds.RequestedProcedureID = orig_ds.get("RequestedProcedureID", "") if copy_tags else ""

        # Contributing Equipment Sequence
        # The Creator shall describe each algorithm that was used to generate the results in the
        # Contributing Equipment Sequence (0018,A001). Multiple items may be included. The Creator
        # shall encode the following details in the Contributing Equipment Sequence:
        #  • Purpose of Reference Code Sequence (0040,A170) shall be (Newcode1, 99IHE, 1630 "Processing Algorithm")
        #  • Manufacturer (0008,0070)
        #  • Manufacturer’s Model Name (0008,1090)
        #  • Software Versions (0018,1020)
        #  • Device UID (0018,1002)

        if model_info:
            # First create the Purpose of Reference Code Sequence
            seq_purpose_of_reference_code = Sequence()
            ds_purpose_of_reference_code = Dataset()
            ds_purpose_of_reference_code.CodeValue = "Newcode1"
            ds_purpose_of_reference_code.CodingSchemeDesignator = "99IHE"
            ds_purpose_of_reference_code.CodeMeaning = '"Processing Algorithm'
            seq_purpose_of_reference_code.append(ds_purpose_of_reference_code)

            seq_contributing_equipment = Sequence()
            ds_contributing_equipment = Dataset()
            ds_contributing_equipment.PurposeOfReferenceCodeSequence = seq_purpose_of_reference_code
            # '(121014, DCM, “Device Observer Manufacturer")'
            ds_contributing_equipment.Manufacturer = model_info.creator
            # u'(121015, DCM, “Device Observer Model Name")'
            ds_contributing_equipment.ManufacturerModel = model_info.name
            # u'(111003, DCM, “Algorithm Version")'
            ds_contributing_equipment.SoftwareVersionNumber = model_info.version
            ds_contributing_equipment.DeviceUID = model_info.uid  # u'(121012, DCM, “Device Observer UID")'
            seq_contributing_equipment.append(ds_contributing_equipment)
            ds.ContributingequipmentSequence = seq_contributing_equipment

        logging.debug("DICOM common modules written:\n{}".format(ds))

        return ds

    @staticmethod
    def random_with_n_digits(n):
        """Random number generator to generate n digit int, where 1 <= n <= 32."""

        assert isinstance(n, int) and n <= 32, "Argument n must be an int, n <= 32."
        n = n if n >= 1 else 1
        range_start = 10 ** (n - 1)
        range_end = (10 ** n) - 1
        return randint(range_start, range_end)


def test():
    from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
    from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator

    current_file_dir = Path(__file__).parent.resolve()
    data_path = current_file_dir.joinpath("../../../examples/ai_spleen_seg_data/dcm")
    out_path = current_file_dir.joinpath("../../../examples/output_sr_op")
    test_report_text = "Dummy AI classification resutls."
    test_copy_tags = True

    loader = DICOMDataLoaderOperator()
    series_selector = DICOMSeriesSelectorOperator()
    sr_writer = DICOMTextSRWriterOperator(
        copy_tags=test_copy_tags, model_info=None, custom_tags={"SeriesDescription": "New AI Series"}
    )

    # Testing with the main entry functions
    dicom_series = None
    if test_copy_tags:
        study_list = loader.load_data_to_studies(Path(data_path).absolute())
        study_selected_series_list = series_selector.filter(None, study_list)
        # Get the first DICOM Series, as for now, only expecting this.
        if not study_selected_series_list or len(study_selected_series_list) < 1:
            raise ValueError("Missing input, list of 'StudySelectedSeries'.")
        for study_selected_series in study_selected_series_list:
            if not isinstance(study_selected_series, StudySelectedSeries):
                raise ValueError("Element in input is not expected type, 'StudySelectedSeries'.")
            for selected_series in study_selected_series.selected_series:
                print(type(selected_series))
                dicom_series = selected_series.series
                print(type(dicom_series))

    sr_writer.write(test_report_text, dicom_series, Path(out_path).absolute())


if __name__ == "__main__":
    test()
