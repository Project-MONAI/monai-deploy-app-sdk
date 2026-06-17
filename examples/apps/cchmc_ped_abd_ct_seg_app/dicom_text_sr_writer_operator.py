# Copyright 2021-2026 MONAI Consortium
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
from typing import Dict, List, Optional, Set, Union

from monai.deploy.utils.importutil import optional_import

dcmread, _ = optional_import("pydicom", name="dcmread")
dcmwrite, _ = optional_import("pydicom.filewriter", name="dcmwrite")
_PYDICOM_UID = "pydicom.uid"
_PYDICOM_DATASET = "pydicom.dataset"
generate_uid, _ = optional_import(_PYDICOM_UID, name="generate_uid")
ImplicitVRLittleEndian, _ = optional_import(_PYDICOM_UID, name="ImplicitVRLittleEndian")
ExplicitVRLittleEndian, _ = optional_import(_PYDICOM_UID, name="ExplicitVRLittleEndian")
Dataset, _ = optional_import(_PYDICOM_DATASET, name="Dataset")
FileDataset, _ = optional_import(_PYDICOM_DATASET, name="FileDataset")
Sequence, _ = optional_import("pydicom.sequence", name="Sequence")

_CODE_MEANING_AREA = "square centimeters"

from monai.deploy.core import ConditionType, Fragment, Operator, OperatorSpec
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries
from monai.deploy.operators.dicom_utils import EquipmentInfo, ModelInfo, save_dcm_file, write_common_modules
from monai.deploy.utils.version import get_sdk_semver


# @md.env(pip_packages=["pydicom >= 1.4.2", "monai"])
class DICOMTextSRWriterOperator(Operator):
    """Class to write DICOM Enhanced SR Instance with provided text input as a Content Sequence.

    Named inputs:
        dict: dictionary content to be encapsulated as a Content Sequence in a DICOM instance file.
        study_selected_series_list: Optional, DICOM series for copying metadata from.

    Named output:
        None

    File output:
        Generated DICOM instance file in the provided output folder.
    """

    # File extension for the generated DICOM Part 10 file.
    DCM_EXTENSION = ".dcm"
    # The default output folder for saving the generated DICOM instance file.
    # DEFAULT_OUTPUT_FOLDER = Path(os.path.join(os.path.dirname(__file__))) / "output"
    DEFAULT_OUTPUT_FOLDER = Path.cwd() / "output"

    def __init__(
        self,
        fragment: Fragment,
        *args,
        output_folder: Union[str, Path],
        model_info: Optional[ModelInfo] = None,
        copy_tags: bool = True,
        equipment_info: Optional[EquipmentInfo] = None,
        custom_tags: Optional[Dict[str, str]] = None,
        included_fields: Optional[List[str]] = None,
        report_code_value: str,
        report_coding_scheme_designator: str,
        report_code_meaning: str,
        **kwargs,
    ):
        """Class to write DICOM Enhanced SR SOP Instance for AI textual result in memory or in a file.

        Args:
            output_folder (str or Path): The folder for saving the generated DICOM instance file.
            copy_tags (bool): True, default, for copying DICOM attributes from a provided DICOMSeries.
                              If True and no DICOMSeries obj provided, runtime exception is thrown.
            model_info (ModelInfo): Object encapsulating model creator, name, version and UID.
            equipment_info (EquipmentInfo, optional): Object encapsulating info for DICOM Equipment Module.
                                                      Defaults to None.
            custom_tags (Dict[str, str], optional): Dictionary for setting custom DICOM tags using Keywords and str values only.
                                                    Defaults to None.
            included_fields (List[str], optional): SR measurement names to include in the output.
                                                   Supports raw metric names such as "volume"
                                                   and fully-qualified output names such as
                                                   "liver.volume". Defaults to None, which
                                                   includes all supported fields.
            report_code_value (str): DICOM Code Value for the report type.
            report_coding_scheme_designator (str): DICOM Coding Scheme Designator for the report.
            report_code_meaning (str): Human-readable meaning of the report code.

        Raises:
            ValueError: If copy_tags is true and no DICOMSeries object provided, or
                        if result cannot be found either in memory or from file.
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

        # Need to init the output folder until the execution context supports dynamic FS path
        # Not trying to create the folder to avoid exception on init
        self.output_folder = Path(output_folder) if output_folder else DICOMTextSRWriterOperator.DEFAULT_OUTPUT_FOLDER
        self.copy_tags = copy_tags
        self.model_info = model_info if model_info else ModelInfo()
        self.equipment_info = equipment_info if equipment_info else EquipmentInfo()
        self.custom_tags = custom_tags
        self.included_fields = self._normalize_included_fields(included_fields)
        self.report_code_value = report_code_value
        self.report_coding_scheme_designator = report_coding_scheme_designator
        self.report_code_meaning = report_code_meaning
        self.input_name_dict = "dict"
        self.input_name_dcm_series = "study_selected_series_list"

        # Set own Modality and SOP Class UID
        # Modality, e.g.,
        #   "OT" for PDF
        #   "SR" for Structured Report.
        # Media Storage SOP Class UID, e.g.,
        #   "1.2.840.10008.5.1.4.1.1.88.11" for Basic Text SR Storage
        #   "1.2.840.10008.5.1.4.1.1.88.22" for Enhanced SR
        #   "1.2.840.10008.5.1.4.1.1.104.1" for Encapsulated PDF Storage
        #   "1.2.840.10008.5.1.4.1.1.88.34" for Comprehensive 3D SR IOD
        #   "1.2.840.10008.5.1.4.1.1.66.4" for Segmentation Storage
        # full list: https://dicom.nema.org/dicom/2013/output/chtml/part04/sect_i.4.html
        self.modality_type = "SR"
        self.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.22"  # Enhanced SR

        # Equipment version may be different from contributing equipment version
        try:
            self.software_version_number = get_sdk_semver()  # SDK Version
        except Exception:
            self.software_version_number = ""
        self.operators_name = f"AI Algorithm {self.model_info.name}"

        super().__init__(fragment, *args, **kwargs)

    def _normalize_included_fields(self, included_fields: Optional[List[str]]) -> Optional[Set[str]]:
        if not included_fields:
            return None

        normalized_fields = {str(field).strip().lower() for field in included_fields if str(field).strip()}
        return normalized_fields or None

    def _should_include_metric(self, biomarker_name: str, metric_name: Optional[str] = None) -> bool:
        if not self.included_fields:
            return True

        normalized_biomarker_name = str(biomarker_name).strip().lower()
        candidate_names = {normalized_biomarker_name}
        if metric_name:
            normalized_metric_name = str(metric_name).strip().lower()
            candidate_names.add(normalized_metric_name)
            candidate_names.add(f"{normalized_biomarker_name}.{normalized_metric_name}")

        return any(candidate in self.included_fields for candidate in candidate_names)

    def _get_formatted_value(self, value) -> str:
        """
        Formats the numeric value based on dynamic rounding rules:
        > 1000: 0 decimals
        > 10: 1 decimal
        < 0.1: 3 decimals
        < 1: 2 decimals
        Else (1 <= value <= 10): 2 decimals (inferred default)
        """
        if value is None:
            return "0"

        try:
            val_float = float(value)
            abs_val = abs(val_float)

            if abs_val > 1000:
                return f"{val_float:.0f}"
            elif abs_val > 10:
                return f"{val_float:.1f}"
            elif abs_val < 0.1:
                return f"{val_float:.3f}"
            elif abs_val < 1:
                return f"{val_float:.2f}"
            else:
                return f"{val_float:.2f}"
        except ValueError:
            return str(value)

    def _build_measurement_item(
        self, biomarker_name: str, value, unit: Optional[str], code_meaning: Optional[str] = None
    ):
        concept_name_code = Dataset()
        concept_name_code.update(
            {"CodeValue": biomarker_name, "CodeMeaning": biomarker_name, "CodingSchemeDesignator": "99_BH"}
        )

        measured_value = Dataset()
        measured_value.NumericValue = self._get_formatted_value(value)

        if unit:
            measurement_units_code = Dataset()
            measurement_units_code.update(
                {"CodeValue": unit, "CodingSchemeDesignator": "UCUM", "CodeMeaning": code_meaning or unit}
            )
            measured_value.MeasurementUnitsCodeSequence = Sequence([measurement_units_code])
        else:
            measured_value.MeasurementUnitsCodeSequence = Sequence([])

        data = Dataset()
        data.update(
            {
                "ValueType": "NUM",
                "RelationshipType": "CONTAINS",
                "ConceptNameCodeSequence": Sequence([concept_name_code]),
                "MeasuredValueSequence": Sequence([measured_value]),
            }
        )
        return data

    def _get_source_modality(self, dicom_series: Optional[DICOMSeries]) -> Optional[str]:
        if dicom_series is None:
            return None

        for attr_name in ("Modality", "modality", "_modality"):
            modality = getattr(dicom_series, attr_name, None)
            if modality:
                return str(modality).upper()

        try:
            sop_instances = dicom_series.get_sop_instances()
            if sop_instances:
                source_dataset = sop_instances[0].get_native_sop_instance()
                modality = getattr(source_dataset, "Modality", None)
                if modality:
                    return str(modality).upper()
        except Exception:
            pass

        return None

    def _normalize_metric_entries(
        self, result_text: Dict, source_modality: Optional[str] = None
    ) -> Dict[str, Dict[str, object]]:
        normalized_entries = {}
        is_ct_source = source_modality == "CT"

        for biomarker_name, biomarker_dict in result_text.items():
            if not isinstance(biomarker_dict, dict):
                continue

            if "biomarker_value" in biomarker_dict and "unit" in biomarker_dict:
                if not self._should_include_metric(biomarker_name):
                    continue
                if biomarker_dict.get("biomarker_value") is None:
                    continue  # Skip absent measurements (e.g. organ not segmented)
                normalized_entries[biomarker_name] = biomarker_dict
                continue

            for metric_name, metric_value in biomarker_dict.items():
                if metric_value is None or isinstance(metric_value, (dict, list, tuple, set)):
                    continue

                if not self._should_include_metric(biomarker_name, metric_name):
                    continue

                metric_name_lower = metric_name.lower()
                output_name = f"{biomarker_name}.{metric_name}"
                unit = None
                code_meaning = None

                if metric_name_lower == "volume":
                    unit = "mL"
                    code_meaning = "milliliter"
                elif metric_name_lower == "area":
                    unit = "cm2"
                    code_meaning = _CODE_MEANING_AREA
                elif "intensity" in metric_name_lower and is_ct_source:
                    unit = "HU"
                    code_meaning = "Hounsfield Unit"
                elif metric_name_lower in {"num.slices", "pixel.count", "num.connected.components"}:
                    unit = None
                    code_meaning = None
                elif metric_name_lower == "error":
                    continue
                elif not isinstance(metric_value, (int, float)):
                    continue

                normalized_entries[output_name] = {
                    "biomarker_value": metric_value,
                    "unit": unit,
                    "code_meaning": code_meaning,
                }

        return normalized_entries

    def _create_content_sequence(self, result_text: Dict, source_modality: Optional[str] = None) -> List[object]:
        """
        Internal helper to parse the dictionary and create the DICOM Content Sequence elements.
        Separated to allow easier testing of logic without full operator execution.
        """
        content_sequence_elements = []
        normalized_result_text = self._normalize_metric_entries(result_text, source_modality)

        for biomarker_name, biomarker_dict in normalized_result_text.items():

            # Parse result_text for Measured Value Sequence writing
            value, unit = biomarker_dict.get("biomarker_value"), biomarker_dict.get("unit")
            if value is None:
                raise ValueError(f"Missing value for biomarker: {biomarker_name}")

            if unit is not None and not isinstance(unit, str):
                raise ValueError(f"Unit must be a string for biomarker: {biomarker_name}")

            # Extract CodeMeaning based on unit
            if unit == "HU":
                code_meaning = "Hounsfield Unit"
            elif unit and unit.lower() in ["ml", "milliliter", "milliliters"]:
                code_meaning = "milliliter"
            elif unit and unit.lower() in ["cm2", "cm^2", "square centimeters"]:
                code_meaning = _CODE_MEANING_AREA
            else:
                code_meaning = unit or ""

            # Apply dynamic rounding
            formatted_value = self._get_formatted_value(value)
            code_meaning_override = biomarker_dict.get("code_meaning")
            if not isinstance(code_meaning_override, str):
                code_meaning_override = None

            self._logger.info(
                f"Preparing Content Sequence for biomarker: {biomarker_name}, value: {value} -> {formatted_value}, unit: {unit}"
            )
            content_sequence_elements.append(
                self._build_measurement_item(
                    biomarker_name,
                    value,
                    unit,
                    code_meaning_override or code_meaning,
                )
            )

            # Add Z-score
            z_score = biomarker_dict.get("z_score")
            if z_score is not None:
                content_sequence_elements.append(
                    self._build_measurement_item(
                        f"{biomarker_name}.z",
                        z_score,
                        None,
                    )
                )

            # Add Percentile if available
            percentile = biomarker_dict.get("percentile_pct")
            if percentile is not None:
                content_sequence_elements.append(
                    self._build_measurement_item(
                        f"{biomarker_name}.p",
                        percentile,
                        "%",
                        "percentile",
                    )
                )

        return content_sequence_elements

    def setup(self, spec: OperatorSpec):
        """Set up the named input(s), and output(s) if applicable.

        This operator does not have an output for the next operator, rather file output only.

        Args:
            spec (OperatorSpec): The Operator specification for inputs and outputs etc.
        """

        spec.input(self.input_name_dict)
        spec.input(self.input_name_dcm_series).condition(ConditionType.NONE)  # Optional input

    def compute(self, op_input, op_output, context):
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
        result_text = op_input.receive(self.input_name_dict)
        if not result_text:
            raise IOError("Input is read but blank.")

        study_selected_series_list = None
        try:
            study_selected_series_list = op_input.receive(self.input_name_dcm_series)
        except Exception:
            pass

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
                if dicom_series is not None:
                    break

        source_modality = self._get_source_modality(dicom_series)
        self._logger.info(f"Detected source modality for DICOM SR content: {source_modality}")

        # Prepare content sequence elements after source_modality is known so that
        # modality-aware normalization (e.g. HU units for CT intensity metrics) applies.
        content_sequence_elements = self._create_content_sequence(result_text, source_modality)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Now ready to starting writing the DICOM instance
        self.write(content_sequence_elements, dicom_series, self.output_folder)

    def write(self, content_text, dicom_series: Optional[DICOMSeries], output_dir: Path):
        """Writes DICOM object

        Args:
            content_text (list): list containing the contents for Content Sequence writing
            dicom_series (DicomSeries): DicomSeries object encapsulating the original series.
            model_info (MoelInfo): Object encapsulating model creator, name, version and UID.

        Returns:
            PyDicom Dataset
        """
        self._logger.debug("Writing DICOM object...\n")

        if not content_text:
            raise ValueError("Content is empty.")
        if not isinstance(output_dir, Path):
            raise ValueError("output_dir is not a valid Path.")

        output_dir.mkdir(parents=True, exist_ok=True)  # Just in case

        ds = write_common_modules(
            dicom_series, self.copy_tags, self.modality_type, self.sop_class_uid, self.model_info, self.equipment_info
        )

        # SR specific
        ds.CompletionFlag = "COMPLETE"  # Estimated degree of completeness
        ds.VerificationFlag = "UNVERIFIED"  # Not attested by a legally accountable person.

        # Required by SR Document Series (Type 2 - Mandatory)
        ds.ReferencedPerformedProcedureStepSequence = Sequence(
            []
        )  # Type 3 for CT/MR Image CIOD; not copied by write_common_modules
        # Required by SR Document General (Type 2 - Mandatory)
        ds.PerformedProcedureCodeSequence = Sequence([])  # Not present for CT/MR Image CIOD

        # Per recommendation of IHE Radiology Technical Framework Supplement
        # AI Results (AIR) Rev1.1 - Trial Implementation
        # Specifically for Qualitative Findings,
        # Qualitative findings shall be encoded in an instance of the DICOM Comprehensive 3D SR SOP
        # Class using TID 1500 (Measurement Report) as the root template.
        # DICOM PS3.16: TID 1500 Measurement Report
        # http://dicom.nema.org/medical/dicom/current/output/chtml/part16/chapter_A.html#sect_TID_1500
        # The value for Procedure Reported (121058, DCM, "Procedure reported") shall describe the
        # imaging procedure analyzed, not the algorithm used.

        # The Comprehensive SR IOD and the Enhanced SR IOD are subsets of the Comprehensive 3D SR IOD, so an Image
        # Display that has implemented support for the Comprehensive 3D SR IOD will have implemented all the
        # capabilities to support the Comprehensive SR IOD and the Enhanced SR IOD

        # Use text value for example
        ds.ValueType = "CONTAINER"

        # ConceptNameCode Sequence
        seq_concept_name_code = Sequence()
        ds.ConceptNameCodeSequence = seq_concept_name_code

        # Concept Name Code Sequence: Concept Name Code
        # Determined via PS3.16 - https://dicom.nema.org/medical/dicom/current/output/html/part16.html#PS3.16
        ds_concept_name_code = Dataset()
        ds_concept_name_code.CodeValue = self.report_code_value
        ds_concept_name_code.CodingSchemeDesignator = self.report_coding_scheme_designator
        ds_concept_name_code.CodeMeaning = self.report_code_meaning
        seq_concept_name_code.append(ds_concept_name_code)

        ds.ContinuityOfContent = "SEPARATE"

        # Content Sequence
        content_sequence = Sequence()
        ds.ContentSequence = content_sequence

        # The actual report content text
        for content_element in content_text:
            content_sequence.append(content_element)

        # For now, only allow str Keywords and str value
        if self.custom_tags:
            for k, v in self.custom_tags.items():
                if isinstance(k, str) and isinstance(v, str):
                    try:
                        ds.update({k: v})
                    except Exception as ex:
                        # Best effort for now
                        logging.warning(f"Tag {k} was not written, due to {ex}")

        # Instance file name is the same as the new SOP instance UID
        file_path = output_dir.joinpath(f"{ds.SOPInstanceUID}{self.DCM_EXTENSION}")

        # write with Explicit VR Little Endian Transfer Syntax to render private tags correctly
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.FileMetaInformationVersion = b"\x00\x01"  # fixes '\x30\x31' writing from write_common_modules
        ds.is_implicit_VR = False
        ds.is_little_endian = True
        save_dcm_file(ds, file_path)
        self._logger.info(f"DICOM SOP instance saved in {file_path}")


def test(test_copy_tags: bool = True):
    from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
    from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator

    current_file_dir = Path(__file__).parent.resolve()
    # Update these paths to match your actual environment or test data location
    data_path = current_file_dir.joinpath("../../../inputs/livertumor_ct/dcm/1-CT_series_liver_tumor_from_nii014")
    out_path = Path("output_sr_op").absolute()

    # UPDATED: Dictionary input with values testing the dynamic rounding logic
    test_data_dict = {
        "Liver_Volume": {"biomarker_value": 1500.12345, "unit": "ml", "z_score": 1.2, "percentile_pct": 95},
        "Tumor_Density_HU": {"biomarker_value": 45.6789, "unit": "HU", "z_score": 2.345, "percentile_pct": 88.1},
        "Small_Nodule_Area": {"biomarker_value": 0.856, "unit": "cm^2", "z_score": 0.5},
        "Tiny_Calcification": {"biomarker_value": 0.0456, "unit": "ml"},
    }

    fragment = Fragment()
    loader = DICOMDataLoaderOperator(fragment, name="loader_op")
    series_selector = DICOMSeriesSelectorOperator(fragment, name="selector_op")
    sr_writer = DICOMTextSRWriterOperator(
        fragment,
        output_folder=out_path,
        copy_tags=test_copy_tags,
        model_info=None,
        equipment_info=EquipmentInfo(),
        custom_tags={"SeriesDescription": "Textual report from AI algorithm. Not for clinical use."},
        report_code_value="126000",
        report_coding_scheme_designator="DCM",
        report_code_meaning="Imaging Measurement Report",
        name="sr_writer",
    )

    dicom_series = None
    if test_copy_tags:
        # Note: This block relies on actual DICOM files existing at data_path
        try:
            study_list = loader.load_data_to_studies(Path(data_path).absolute())
            study_selected_series_list = series_selector.filter(None, study_list)

            if not study_selected_series_list or len(study_selected_series_list) < 1:
                print("Warning: No DICOM series found for test. Running without Series metadata copy.")
                dicom_series = None
            else:
                for study_selected_series in study_selected_series_list:
                    for selected_series in study_selected_series.selected_series:
                        dicom_series = selected_series.series
                        break
        except Exception as e:
            print(f"Skipping DICOM loading due to environment error: {e}")
            dicom_series = None

    # UPDATED TEST LOGIC:
    # 1. Manually trigger the creation of content sequence (test logic)
    print("Testing Content Sequence Creation & Rounding...")
    content_sequence = sr_writer._create_content_sequence(test_data_dict)

    # 2. Write the file
    print(f"Writing SR to {out_path}...")
    try:
        sr_writer.write(content_sequence, dicom_series, out_path)
        print("Test Success: DICOM SR written.")
    except Exception as e:
        print(f"Test Failed during write: {e}")


if __name__ == "__main__":
    # Ensure pydicom and monai are installed before running
    try:
        test(True)
    except Exception as e:
        print(f"Test execution failed: {e}")
