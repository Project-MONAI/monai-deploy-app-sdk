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
from pathlib import Path
from typing import Dict, List, Optional, Text

from monai.deploy.utils.importutil import optional_import

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
from monai.deploy.operators.dicom_utils import EquipmentInfo, ModelInfo, save_dcm_file, write_common_modules
from monai.deploy.utils.version import get_sdk_semver


# The SR writer operator class
@md.input("classification_result", Text, IOType.IN_MEMORY)
@md.input("classification_result_file", DataPath, IOType.DISK)
@md.input("study_selected_series_list", List[StudySelectedSeries], IOType.IN_MEMORY)
@md.output("dicom_instance", DataPath, IOType.DISK)
@md.env(pip_packages=["pydicom >= 1.4.2"])
class DICOMTextSRWriterOperator(Operator):

    # File extension for the generated DICOM Part 10 file.
    DCM_EXTENSION = ".dcm"

    def __init__(
        self,
        copy_tags: bool,
        model_info: ModelInfo,
        equipment_info: Optional[EquipmentInfo] = None,
        custom_tags: Optional[Dict[str, str]] = None,
        *args,
        **kwargs,
    ):
        """Class to write DICOM SR SOP Instance for AI textual result in memory or in a file.

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
        #   "1.2.840.10008.5.1.4.1.1.88.11" for Basic Text SR Storage
        #   "1.2.840.10008.5.1.4.1.1.104.1" for Encapsulated PDF Storage,
        #   "1.2.840.10008.5.1.4.1.1.88.34" for Comprehensive 3D SR IOD
        #   "1.2.840.10008.5.1.4.1.1.66.4" for Segmentation Storage
        self.modality_type = "SR"
        self.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.11"
        # Equipment version may be different from contributing equipment version
        try:
            self.software_version_number = get_sdk_semver()  # SDK Version
        except Exception:
            self.software_version_number = ""
        self.operators_name = f"AI Algorithm {self.model_info.name}"

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        """Performs computation for this operator and handles I/O.

        For now, only a single result content is supported, which could be in memory or an accessible file.
        The DICOM series used during inference is optional, but is required if the
        `copy_tags` is true indicating the generated DICOM object needs to copy study level metadata.

        When there are multiple selected series in the input, the first series' containing study will
        be used for retrieving DICOM Study module attributes, e.g. StudyInstanceUID.

        Raises:
            FileNotFoundError: When result object not in the input, and result file not found either.
            ValueError: Content object and file path not in the inputs, or no DICOM series when required.
            IOError: If the input content is blank.
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

        dicom_series = None  # It can be None if not to copy_tags.
        if self.copy_tags:
            # Get the first DICOM Series to retrieve study level tags.
            if not study_selected_series_list or len(study_selected_series_list) < 1:
                raise ValueError("Missing input, list of 'StudySelectedSeries'.")
            for study_selected_series in study_selected_series_list:
                if not isinstance(study_selected_series, StudySelectedSeries):
                    raise ValueError("Element in input is not expected type, 'StudySelectedSeries'.")
                for selected_series in study_selected_series.selected_series:
                    dicom_series = selected_series.series
                    break

        output_dir = op_output.get().path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Now ready to starting writing the DICOM instance
        self.write(result_text, dicom_series, output_dir)

    def write(self, content_text, dicom_series: Optional[DICOMSeries], output_dir: Path):
        """Writes DICOM object

        Args:
            content_file (str): file containing the contents
            dicom_series (DicomSeries): DicomSeries object encapsulating the original series.
            model_info (MoelInfo): Object encapsulating model creator, name, version and UID.

        Returns:
            PyDicom Dataset
        """
        self._logger.debug("Writing DICOM object...\n")

        if not content_text or not len(content_text.strip()):
            raise ValueError("Content is empty.")
        if not isinstance(output_dir, Path):
            raise ValueError("output_dir is not a valid Path.")

        output_dir.mkdir(parents=True, exist_ok=True)  # Just in case

        ds = write_common_modules(
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
        ds.ValueType = "CONTAINER"

        # ConceptNameCode Sequence
        seq_concept_name_code = Sequence()
        ds.ConceptNameCodeSequence = seq_concept_name_code

        # Concept Name Code Sequence: Concept Name Code
        ds_concept_name_code = Dataset()
        ds_concept_name_code.CodeValue = "18748-4"
        ds_concept_name_code.CodingSchemeDesignator = "LN"
        ds_concept_name_code.CodeMeaning = "Diagnostic Imaging Report"
        seq_concept_name_code.append(ds_concept_name_code)

        ds.ContinuityOfContent = "SEPARATE"

        # Content Sequence
        content_sequence = Sequence()
        ds.ContentSequence = content_sequence

        # Content Sequence: Content 1
        content1 = Dataset()
        content1.RelationshipType = "CONTAINS"
        content1.ValueType = "TEXT"

        # Concept Name Code Sequence
        concept_name_code_sequence = Sequence()
        content1.ConceptNameCodeSequence = concept_name_code_sequence

        # Concept Name Code Sequence: Concept Name Code 1
        concept_name_code1 = Dataset()
        concept_name_code1.CodeValue = "111412"  # or 111413 "Overall Assessment"
        concept_name_code1.CodingSchemeDesignator = "DCM"
        concept_name_code1.CodeMeaning = "Narrative Summary"  # or 111413 'Overall Assessment'
        concept_name_code_sequence.append(concept_name_code1)

        content1.TextValue = content_text  # The actual report content text
        content_sequence.append(content1)

        # For now, only allow str Keywords and str value
        if self.custom_tags:
            for k, v in self.custom_tags.items():
                if isinstance(k, str) and isinstance(v, str):
                    try:
                        ds.update({k: v})
                    except Exception as ex:
                        # Best effort for now.
                        logging.warning(f"Tag {k} was not written, due to {ex}")

        # Instance file name is the same as the new SOP instance UID
        file_path = output_dir.joinpath(f"{ds.SOPInstanceUID}{DICOMTextSRWriterOperator.DCM_EXTENSION}")
        save_dcm_file(ds, file_path)
        self._logger.info(f"DICOM SOP instance saved in {file_path}")


def test():
    from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
    from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator

    current_file_dir = Path(__file__).parent.resolve()
    data_path = current_file_dir.joinpath("../../../inputs/livertumor_ct/dcm/1-CT_series_liver_tumor_from_nii014")
    out_path = "output_sr_op"
    test_report_text = "Tumors detected in Liver using MONAI Liver Tumor Seg model."
    test_copy_tags = True

    loader = DICOMDataLoaderOperator()
    series_selector = DICOMSeriesSelectorOperator()
    sr_writer = DICOMTextSRWriterOperator(
        copy_tags=test_copy_tags,
        model_info=None,
        equipment_info=EquipmentInfo(),
        custom_tags={"SeriesDescription": "Textual report from AI algorithm. Not for clinical use."},
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
