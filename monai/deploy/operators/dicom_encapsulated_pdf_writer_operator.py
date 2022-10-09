# Copyright 2022 MONAI Consortium
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
from ast import Bytes
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

from PyPDF2 import PdfReader

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
from monai.deploy.utils.version import get_sdk_semver

from .dicom_utils import EquipmentInfo, ModelInfo, save_dcm_file, write_common_modules


# The SR writer operator class
@md.input("pdf_bytes", Bytes, IOType.IN_MEMORY)
@md.input("pdf_file", DataPath, IOType.DISK)
@md.input("study_selected_series_list", List[StudySelectedSeries], IOType.IN_MEMORY)
@md.output("dicom_instance", DataPath, IOType.DISK)
@md.env(pip_packages=["pydicom >= 1.4.2"])
class DICOMEncapsulatedPDFWriterOperator(Operator):

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
        """Class to write DICOM Encapsulated PDF Instance with PDF bytes in memory or in a file.

        Args:
            copy_tags (bool): True for copying DICOM attributes from a provided DICOMSeries.
            model_info (ModelInfo): Object encapsulating model creator, name, version and UID.
            equipment_info (EquipmentInfo, optional): Object encapsulating info for DICOM Equipment Module.
                                                      Defaults to None.
            custom_tags (Dict[str, str], optional): Dictionary for setting custom DICOM tags using Keywords
                                                    and str values only. Defaults to None.

        Raises:
            ValueError: If copy_tags is true and no DICOMSeries object provided, or
                        if PDF bytes cannot be found in memory or loaded from the file.
        """
        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.copy_tags = copy_tags
        self.model_info = model_info if model_info else ModelInfo()
        self.equipment_info = equipment_info if equipment_info else EquipmentInfo()
        self.custom_tags = custom_tags

        # Set own Modality and SOP Class UID
        # Modality, e.g.,
        #   "OT" for PDF, "DOC" would do too.
        #   "SR" for Structured Report.
        # Media Storage SOP Class UID, e.g.,
        #   "1.2.840.10008.5.1.4.1.1.88.11" for Basic Text SR Storage
        #   "1.2.840.10008.5.1.4.1.1.104.1" for Encapsulated PDF Storage,
        #   "1.2.840.10008.5.1.4.1.1.88.34" for Comprehensive 3D SR IOD
        #   "1.2.840.10008.5.1.4.1.1.66.4" for Segmentation Storage
        #   '1.2.840.10008.5.1.4.1.1.104.1' for Encapsulated PDF Storage
        self.modality_type = "OT"
        self.sop_class_uid = "1.2.840.10008.5.1.4.1.1.104.1"

        # Equipment version may be different from contributing equipment version
        try:
            self.software_version_number = get_sdk_semver()  # SDK Version
        except Exception:
            self.software_version_number = ""
        self.operators_name = f"AI Algorithm {self.model_info.name}"

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        """Performs computation for this operator and handles I/O.

        For now, only a single result content is supported, which could be in bytes or a path
        to the PDF file. The DICOM series used during inference is optional, but is required if the
        `copy_tags` is true indicating the generated DICOM object needs to copy study level metadata.

        When there are multiple selected series in the input, the first series' containing study will
        be used for retrieving DICOM Study module attributes, e.g. StudyInstanceUID.

        Raises:
            FileNotFoundError: When bytes are not in the input, and the file is not given or found.
            ValueError: Input bytes and PDF file not in the input, or no DICOM series when required.
            IOError: If fails to get the bytes of the PDF
        """

        # Gets the input, prepares the output folder, and then delegates the processing.
        pdf_bytes = ""
        try:
            pdf_bytes = bytes(op_input.get("pdf_bytes")).strip()
        except ItemNotExistsError:
            try:
                file_path = op_input.get("pdf_file")
            except ItemNotExistsError:
                raise ValueError("None of the named inputs can be found.") from None
            # Read file, and if exception, let it bubble up
            with open(file_path.path, "rb") as f:
                pdf_bytes = f.read().strip()

        if not pdf_bytes or not len(pdf_bytes):
            raise IOError("Input is read but blank.")

        try:
            study_selected_series_list = op_input.get("study_selected_series_list")
        except ItemNotExistsError:
            study_selected_series_list = None

        dicom_series = None  # It can be None if not to copy_tags.
        if self.copy_tags:
            # Get the first DICOM Series for retrieving study level tags.
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
        self.write(pdf_bytes, dicom_series, output_dir)

    def write(self, content_bytes, dicom_series: Optional[DICOMSeries], output_dir: Path):
        """Writes DICOM object

        Args:
            content_bytes (bytes): Content bytes of PDF
            dicom_series (DicomSeries): DicomSeries object encapsulating the original series.
            output_dir (Path): Folder path for saving the generated file.
        """
        self._logger.debug("Writing DICOM object...\n")

        if not isinstance(content_bytes, bytes):
            raise ValueError("Input must be bytes.")
        elif not len(content_bytes.strip()):
            raise ValueError("Content is empty.")
        elif not self._is_pdf_bytes(content_bytes):
            raise ValueError("Not PDF bytes.")

        if not isinstance(output_dir, Path):
            raise ValueError("output_dir is not a valid Path.")

        output_dir.mkdir(parents=True, exist_ok=True)  # Just in case

        ds = write_common_modules(
            dicom_series, self.copy_tags, self.modality_type, self.sop_class_uid, self.model_info, self.equipment_info
        )

        # Encapsulated PDF specific
        # SC Equipment Module
        ds.Modality = self.modality_type
        ds.ConversionType = "SD"  # Describes the kind of image conversion, Scanned Doc

        # Encapsulated Document Module
        ds.VerificationFlag = "UNVERIFIED"  # Not attested by a legally accountable person.

        ds.BurnedInAnnotation = "YES"
        ds.DocumentTitle = "Generated Observations"
        ds.MIMETypeOfEncapsulatedDocument = "application/pdf"
        ds.EncapsulatedDocument = content_bytes

        ## ConceptNameCode Sequence
        seq_concept_name_code = Sequence()
        ds_concept_name_code = Dataset()
        ds_concept_name_code.CodeValue = "18748-4"
        ds_concept_name_code.CodingSchemeDesignator = "LN"
        ds_concept_name_code.CodeMeaning = "Diagnostic Imaging Report"
        seq_concept_name_code.append(ds_concept_name_code)
        ds.ConceptNameCodeSequence = seq_concept_name_code

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
        file_path = output_dir.joinpath(f"{ds.SOPInstanceUID}{DICOMEncapsulatedPDFWriterOperator.DCM_EXTENSION}")
        save_dcm_file(ds, file_path)

    def _is_pdf_bytes(self, content: bytes):
        try:
            bytes_stream = BytesIO(content)
            reader = PdfReader(bytes_stream)
            self._logger.debug(f"The PDF has {reader.pages} page(s).")
        except Exception as ex:
            self._logger.exception(f"Cannot read as PDF: {ex}")
            return False
        return True


def test():
    from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
    from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator

    current_file_dir = Path(__file__).parent.resolve()
    dcm_folder = current_file_dir.joinpath("../../../inputs/livertumor_ct/dcm/1-CT_series_liver_tumor_from_nii014")
    pdf_file = current_file_dir.joinpath("../../../inputs/pdf/TestPDF.pdf")
    out_path = "output_pdf_op"
    pdf_bytes = b"Not PDF bytes."
    test_copy_tags = False

    loader = DICOMDataLoaderOperator()
    series_selector = DICOMSeriesSelectorOperator()
    sr_writer = DICOMEncapsulatedPDFWriterOperator(
        copy_tags=test_copy_tags,
        model_info=None,
        equipment_info=EquipmentInfo(),
        custom_tags={"SeriesDescription": "Report from AI algorithm. Not for clinical use."},
    )

    # Testing with the main entry functions
    dicom_series = None
    if test_copy_tags:
        study_list = loader.load_data_to_studies(Path(dcm_folder).absolute())
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

    with open(pdf_file, "rb") as f:
        pdf_bytes = f.read()

    sr_writer.write(pdf_bytes, dicom_series, Path(out_path).absolute())


if __name__ == "__main__":
    test()
