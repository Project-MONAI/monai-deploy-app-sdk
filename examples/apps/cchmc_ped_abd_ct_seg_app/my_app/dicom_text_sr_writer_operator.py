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
from typing import Dict, List, Optional, Union

from monai.deploy.utils.importutil import optional_import

dcmread, _ = optional_import("pydicom", name="dcmread")
dcmwrite, _ = optional_import("pydicom.filewriter", name="dcmwrite")
generate_uid, _ = optional_import("pydicom.uid", name="generate_uid")
ImplicitVRLittleEndian, _ = optional_import("pydicom.uid", name="ImplicitVRLittleEndian")
ExplicitVRLittleEndian, _ = optional_import("pydicom.uid", name="ExplicitVRLittleEndian")
Dataset, _ = optional_import("pydicom.dataset", name="Dataset")
FileDataset, _ = optional_import("pydicom.dataset", name="FileDataset")
Sequence, _ = optional_import("pydicom.sequence", name="Sequence")

from monai.deploy.core import ConditionType, Fragment, Operator, OperatorSpec
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries
from monai.deploy.operators.dicom_utils import EquipmentInfo, ModelInfo, save_dcm_file, write_common_modules
from monai.deploy.utils.version import get_sdk_semver


# @md.env(pip_packages=["pydicom >= 1.4.2", "monai"])
class DICOMTextSRWriterOperator(Operator):
    """Class to write DICOM Text SR Instance with provided text input as a Content Sequence.
    Customized for CT Liver-Spleen Segmentation model.

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
        model_info: ModelInfo,
        copy_tags: bool = True,
        equipment_info: Optional[EquipmentInfo] = None,
        custom_tags: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Class to write DICOM SR SOP Instance for AI textual result in memory or in a file.

        Args:
            output_folder (str or Path): The folder for saving the generated DICOM instance file.
            copy_tags (bool): True, default, for copying DICOM attributes from a provided DICOMSeries.
                              If True and no DICOMSeries obj provided, runtime exception is thrown.
            model_info (ModelInfo): Object encapsulating model creator, name, version and UID.
            equipment_info (EquipmentInfo, optional): Object encapsulating info for DICOM Equipment Module.
                                                      Defaults to None.
            custom_tags (Dict[str, str], optional): Dictionary for setting custom DICOM tags using Keywords and str values only.
                                                    Defaults to None.

        Raises:
            ValueError: If copy_tags is true and no DICOMSeries object provided, or
                        if result cannot be found either in memory or from file.
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

        # Need to init the output folder until the execution context supports dynamic FS path
        # Not trying to create the folder to avoid exception on init
        self.output_folder = Path(output_folder) if output_folder else self.DEFAULT_OUTPUT_FOLDER
        self.copy_tags = copy_tags
        self.model_info = model_info if model_info else ModelInfo()
        self.equipment_info = equipment_info if equipment_info else EquipmentInfo()
        self.custom_tags = custom_tags
        
        self.input_name_dict = "dict"
        self.input_name_dcm_series = "study_selected_series_list"

        # Set own Modality and SOP Class UID
        self.modality_type = "SR"
        self.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.11"
        
        # Equipment version may be different from contributing equipment version
        try:
            self.software_version_number = get_sdk_semver()  # SDK Version
        except Exception:
            self.software_version_number = ""
        self.operators_name = f"AI Algorithm {self.model_info.name}"

        super().__init__(fragment, *args, **kwargs)

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

    def _create_content_sequence(self, result_text: Dict) -> List[Dataset]:
        """
        Internal helper to parse the dictionary and create the DICOM Content Sequence elements.
        Separated to allow easier testing of logic without full operator execution.
        """
        content_sequence_elements = []

        for biomarker_name, biomarker_dict in result_text.items():
            
            # Concept Name Code Sequence
            concept_name_code = Dataset()
            concept_name_code.update({
                "CodeValue": biomarker_name,
                "CodeMeaning": biomarker_name,
                "CodingSchemeDesignator": "99_BH"
            })

            # Parse result_text for Measured Value Sequence writing
            value, unit = biomarker_dict.get("biomarker_value"), biomarker_dict.get("unit")
            if value is None or unit is None:
                raise ValueError(f"Missing value or unit for biomarker: {biomarker_name}")

            # Extract CodeMeaning based on unit
            if biomarker_name.split("_")[-1].lower() == "hu":
                code_meaning = "Hounsfield Unit"
            elif unit.lower() in ["ml", "milliliter", "milliliters"]:
                code_meaning = "milliliter"
            elif unit.lower() in ["cm2", "cm^2", "square centimeters"]:
                code_meaning = "square centimeters"
            else:
                code_meaning = unit  # Fallback

            # Apply dynamic rounding
            formatted_value = self._get_formatted_value(value)
            
            self._logger.info(f"Preparing Content Sequence for biomarker: {biomarker_name}, value: {value} -> {formatted_value}, unit: {unit}")
            
            # Measurement Units Code Sequence
            measurement_units_code = Dataset()
            measurement_units_code.update({
                "CodeValue": unit, 
                "CodingSchemeDesignator": "UCUM",
                "CodeMeaning": code_meaning
            })
                                            
            # Measured Value Sequence
            measured_value = Dataset()
            measured_value.NumericValue = formatted_value
            measured_value.MeasurementUnitsCodeSequence = Sequence([measurement_units_code])

            # NUM Content Item
            data = Dataset()
            data.update({
                "ValueType": "NUM",
                "RelationshipType": "CONTAINS",
                "ConceptNameCodeSequence": Sequence([concept_name_code]),
                "MeasuredValueSequence": Sequence([measured_value])
            })
            content_sequence_elements.append(data)
                        
            ### Add Z-score 
            z_score = biomarker_dict.get("z_score")
            if z_score is not None:
                concept_name_code = Dataset()
                concept_name_code.update({
                    "CodeValue": f"{biomarker_name}_Z",
                    "CodeMeaning": f"{biomarker_name} Z-Score",
                    "CodingSchemeDesignator": "99_BH"
                })
                
                measured_value = Dataset()
                measured_value.NumericValue = self._get_formatted_value(z_score)
                measured_value.MeasurementUnitsCodeSequence = Sequence([]) 

                data = Dataset()
                data.update({
                    "ValueType": "NUM",
                    "RelationshipType": "CONTAINS",
                    "ConceptNameCodeSequence": Sequence([concept_name_code]),
                    "MeasuredValueSequence": Sequence([measured_value])
                })
                content_sequence_elements.append(data)
                
            # Add Percentile if available
            percentile = biomarker_dict.get("percentile_pct")
            if percentile is not None:
                concept_name_code = Dataset()
                concept_name_code.update({
                    "CodeValue": f"{biomarker_name}_P",
                    "CodeMeaning": f"{biomarker_name} Percentile",
                    "CodingSchemeDesignator": "99_BH"
                })
                
                measurement_units_code = Dataset()
                measurement_units_code.update({
                    "CodeValue": "%",
                    "CodingSchemeDesignator": "UCUM",
                    "CodeMeaning": "percentile"
                })
                    
                measured_value = Dataset()
                measured_value.NumericValue = self._get_formatted_value(percentile)
                measured_value.MeasurementUnitsCodeSequence = Sequence([measurement_units_code])

                data = Dataset()
                data.update({
                    "ValueType": "NUM",
                    "RelationshipType": "CONTAINS",
                    "ConceptNameCodeSequence": Sequence([concept_name_code]),
                    "MeasuredValueSequence": Sequence([measured_value])
                })
                content_sequence_elements.append(data)
        
        return content_sequence_elements

    def setup(self, spec: OperatorSpec):
        """Set up the named input(s), and output(s) if applicable."""
        spec.input(self.input_name_dict)
        spec.input(self.input_name_dcm_series).condition(ConditionType.NONE)


    def compute(self, op_input, op_output, context):
        """Performs computation for this operator and handles I/O."""

        # Gets the input
        result_text = op_input.receive(self.input_name_dict)
        if not result_text:
            raise IOError("Input is read but blank.")

        # Prepare content sequences (delegated to helper)
        content_sequence_elements = self._create_content_sequence(result_text)

        study_selected_series_list = None
        try:
            study_selected_series_list = op_input.receive(self.input_name_dcm_series)
        except Exception:
            pass

        dicom_series = None
        if self.copy_tags:
            if not study_selected_series_list or len(study_selected_series_list) < 1:
                raise ValueError("Missing input, list of 'StudySelectedSeries'.")
            for study_selected_series in study_selected_series_list:
                if not isinstance(study_selected_series, StudySelectedSeries):
                    raise ValueError("Element in input is not expected type, 'StudySelectedSeries'.")
                for selected_series in study_selected_series.selected_series:
                    dicom_series = selected_series.series
                    break

        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.write(content_sequence_elements, dicom_series, self.output_folder)

    def write(self, content_text, dicom_series: Optional[DICOMSeries], output_dir: Path):
        """Writes DICOM object"""
        self._logger.debug("Writing DICOM object...\n")

        if not content_text:
            raise ValueError("Content is empty.")
        if not isinstance(output_dir, Path):
            raise ValueError("output_dir is not a valid Path.")

        output_dir.mkdir(parents=True, exist_ok=True) 

        ds = write_common_modules(
            dicom_series, self.copy_tags, self.modality_type, self.sop_class_uid, self.model_info, self.equipment_info
        )

        block = ds.private_block(0x0019, "CCHMC Private", create=True)
        block.add_new(0x01, "UI", f"{dicom_series._series_instance_uid}") 

        ds.VerificationFlag = "UNVERIFIED"
        ds.ValueType = "CONTAINER"

        seq_concept_name_code = Sequence()
        ds.ConceptNameCodeSequence = seq_concept_name_code

        ds_concept_name_code = Dataset()
        ds_concept_name_code.CodeValue = "41806-1"
        ds_concept_name_code.CodingSchemeDesignator = "LN"
        ds_concept_name_code.CodeMeaning = "CT Abdomen Report"
        seq_concept_name_code.append(ds_concept_name_code)

        ds.ContinuityOfContent = "SEPARATE"

        content_sequence = Sequence()
        ds.ContentSequence = content_sequence

        for content_element in content_text:
            content_sequence.append(content_element)

        if self.custom_tags:
            for k, v in self.custom_tags.items():
                if isinstance(k, str) and isinstance(v, str):
                    try:
                        ds.update({k: v})
                    except Exception as ex:
                        logging.warning(f"Tag {k} was not written, due to {ex}")

        file_path = output_dir.joinpath(f"{ds.SOPInstanceUID}{self.DCM_EXTENSION}")

        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
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
        "Tiny_Calcification": {"biomarker_value": 0.0456, "unit": "ml"}
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
        name="sr_writer"
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