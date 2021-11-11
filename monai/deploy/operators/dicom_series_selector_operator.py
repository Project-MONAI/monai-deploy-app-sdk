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
import numbers
import re
from json import loads as json_loads
from typing import Dict, List, Optional, Text, Tuple, Union

import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, InputContext, IOType, Operator, OutputContext
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_series_selection import SelectedSeries, StudySelectedSeries
from monai.deploy.core.domain.dicom_study import DICOMStudy
from monai.deploy.core.domain.image import Image
from monai.deploy.exceptions import ItemNotExistsError
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator


@md.input("dicom_study_list", List[DICOMStudy], IOType.IN_MEMORY)
# @md.input("selection_rules", Dict, IOType.IN_MEMORY)  # The App needs to provide this.
@md.output("dicom_series", List[DICOMSeries], IOType.IN_MEMORY)
@md.output("study_selected_series_list", List[StudySelectedSeries], IOType.IN_MEMORY)
class DICOMSeriesSelectorOperator(Operator):
    """This operator selects a list of DICOM Series for a given set of selection rules.

    More to come.

    Example selection rule in JSON:
    {
        "selections": [
            {
                "name": "CT Series 1",
                "conditions": {
                    "StudyDescription": "(?i)^Spleen",
                    "Modality": "(?i)CT",
                    "SeriesDescription": "(?i)^No series description|(.*?)"
                }
            }
        ]
    }
    """

    def __init__(self, rules: Text = None, all_matched: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """Instantiate an instance.

        Args:
            rules (Text): Selection rules in JSON string.
            all_matched (bool): Gets all matched series in a study. Defaults to False for first one only.
        """

        # Delay loading the rules as json string till compute time.
        self._rules_json_str = rules if rules else None
        self._all_matched = all_matched

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        """Performs computation for this operator."""
        try:
            dicom_study_list = op_input.get("dicom_study_list")
            selection_rules = self._load_rules() if self._rules_json_str else None
            dicom_series_list, study_selected_series = self.filter(selection_rules, dicom_study_list, self._all_matched)
            op_output.set(dicom_series_list, "dicom_series")
            op_output.set(
                study_selected_series,
            )
        except ItemNotExistsError:
            pass

    def filter(
        self, selection_rules, dicom_study_list, all_matched: bool = False
    ) -> Tuple[List[SelectedSeries], List[StudySelectedSeries]]:
        """Selects the series with the given matching rules.

        If rules object is None, all series will be returned with series instance UID
        as both the key and value for each dictionary value.

        Simplistic matching is used for demonstration:
            Number: exactly matches
            String: matches case insensitive, if fails, trys RegEx search
            String array matches as subset, case insensitive

        Args:
            selection_rules (object): JSON object containing the matching rules.
            dicom_study_list (list): A list of DICOMStudiy objects.
            all_matched (bool): Gets all matched series in a study. Defaults to False for first one only.

        Returns:
            list: A list of all selected series of type SelectedSeries.
            list: A list of objects of type StudySelectedSeries.
        """

        if not dicom_study_list or len(dicom_study_list) < 1:
            return [], []

        if not selection_rules:
            # Return all series if no slection rules are supplied
            logging.warn(f"No selection rules given; select all series.")
            return self._select_all_series(dicom_study_list)

        selections = selection_rules.get("selections", None)
        # If missing selections in the rules then it is an error.
        if not selections:
            raise ValueError('Expected "selections" not found in the rules.')

        selected_series_list = []  # List of all selected DICOMSeries objects
        study_selected_series_list = []  # List of StudySelectedSeries objects

        for study in dicom_study_list:
            study_selected_series = StudySelectedSeries(study)
            for selection in selections:
                # Get the selection name. Blank name will be handled by the SelectedSeries
                selection_name = selection.get("name", "").strip()
                logging.info(f"Finding series for Selection named: {selection_name}")

                # Skip if no selection conditions are provided.
                conditions = selection.get("conditions", None)
                if not conditions:
                    continue

                # Select only the first series that matches the conditions, list of one
                series_list = self._select_series(conditions, study, all_matched)
                if series_list and len(series_list) > 0:
                    selected_series_list.extend(x for x in series_list)  # Add each single one
                    for series in series_list:
                        selected_series = SelectedSeries(selection_name, series)
                        study_selected_series.add_selected_series(selected_series)

            if len(study_selected_series.selected_series) > 0:
                study_selected_series_list.append(study_selected_series)

        return selected_series_list, study_selected_series_list

    def _load_rules(self):
        return json_loads(self._rules_json_str) if self._rules_json_str else None

    def _select_all_series(self, dicom_study_list: List[DICOMStudy]):
        """Select all series in studies

        Returns:
            list: list of DICOMSeries objects
            list: list of StudySelectedSeries objects
        """

        series_list = []
        study_selected_series_list = []
        for study in dicom_study_list:
            logging.info(f"Working on study, instance UID: {study.StudyInstanceUID}")
            print((f"Working on study, instance UID: {study.StudyInstanceUID}"))
            study_selected_series = StudySelectedSeries(study)
            for series in study.get_all_series():
                logging.info(f"Working on series, instance UID: {str(series.SeriesInstanceUID)}")
                print(f"Working on series, instance UID: {str(series.SeriesInstanceUID)}")
                selected_series = SelectedSeries(None, series)  # No selection name is known or given
                study_selected_series.add_selected_series(selected_series)
                series_list.append(series)
            study_selected_series_list.append(study_selected_series)
        return series_list, study_selected_series_list

    def _select_series(self, attributes: dict, study: DICOMStudy, all_matched=False):
        """Finds series whose attributes match the given attributes.

        Args:
            attributes (dict): Dictionary of attributes for matching
            all_matched (bool): Gets all matched series in a study. Defaults to False for first one only.

        Returns:
            List of DICOMSeries. At most one element if all_matched is False.
        """
        assert isinstance(attributes, dict), '"attributes" must be a dict.'

        logging.info(f"Searching study, : {study.StudyInstanceUID}\n  # of series: {len(study.get_all_series())}")
        study_attr = self._get_instance_properties(study)

        found_series = []
        for series in study.get_all_series():
            logging.info(f"Working on series, instance UID: {series.SeriesInstanceUID}")

            # Combine Study and current Series properties for matching
            series_attr = self._get_instance_properties(series)
            series_attr.update(study_attr)

            matched = True
            # Simple matching on attribute value
            for key, value_to_match in attributes.items():
                logging.info(f"On attribute: '{key}' to match value: '{value_to_match}'")
                # Ignore None
                if not value_to_match:
                    continue
                # Try getting the attribute value from Study and current Series prop dict
                attr_value = series_attr.get(key, None)
                logging.info(f"    Series attribute value: {attr_value}")
                if not attr_value:
                    matched = False
                elif isinstance(attr_value, numbers.Number):
                    matched = value_to_match == attr_value
                elif isinstance(attr_value, str):
                    matched = attr_value.casefold() == (value_to_match.casefold())
                    if not matched:
                        # For str, also try RegEx search to check for a match anywhere in the string
                        # unless the user constrains it in the expression.
                        logging.info("Series attribute string value did not match. Try regEx.")
                        if re.search(value_to_match, attr_value, re.IGNORECASE):
                            matched = True
                elif isinstance(attr_value, list):
                    meta_data_set = set(str(element).lower() for element in attr_value)
                    if isinstance(value_to_match, list):
                        value_set = set(str(element).lower() for element in value_to_match)
                        matched = all(val in meta_data_set for val in value_set)
                    elif isinstance(value_to_match, (str, numbers.Number)):
                        matched = str(value_to_match).lower() in meta_data_set
                else:
                    raise NotImplementedError(f"Not support for matching on this type: {type(value_to_match)}")

                if not matched:
                    logging.info("This series does not match the selection conditions.")
                    break

            if matched:
                logging.info(f"Selected Series, UID: {series.SeriesInstanceUID}")
                found_series.append(series)

                if not all_matched:
                    return found_series

        return found_series

    @staticmethod
    def _get_instance_properties(obj: object) -> Dict:
        prop_dict = {}
        if not obj:
            return prop_dict

        for attribute in [x for x in type(obj).__dict__ if isinstance(type(obj).__dict__[x], property)]:
            prop_dict[attribute] = getattr(obj, attribute, None)
        return prop_dict


# Module functions
def _print_instance_properties(obj: object, pre_fix: str = None, print_val=True):
    print(f"{pre_fix}Instance of {type(obj)}")
    for attribute in [x for x in type(obj).__dict__ if isinstance(type(obj).__dict__[x], property)]:
        attr_val = getattr(obj, attribute, None)
        print(f"{pre_fix}  {attribute}: {type(attr_val)} {attr_val if print_val else ''}")


def main():
    data_path = "/home/mqin/src/monai-app-sdk/examples/ai_spleen_seg_data/dcm-multi"
    # /home/mqin/src/monai-app-sdk/examples/input_spleen/input_dcm"
    # data_path = "/home/rahul/medical-images/lung-ct-1/"
    files = []
    loader = DICOMDataLoaderOperator()
    loader._list_files(data_path, files)
    study_list = loader._load_data(files)
    selector = DICOMSeriesSelectorOperator()
    sample_selection_rule = json_loads(Sample_Rules_Text)
    print(f"Selection rules in JSON:\n{sample_selection_rule}")
    series_list, study_selected_seriee_list = selector.filter(sample_selection_rule, study_list)

    for sss_obj in study_selected_seriee_list:
        _print_instance_properties(sss_obj, pre_fix="", print_val=False)
        study = sss_obj.study
        pre_fix = "  "
        print(f"{pre_fix}==== Details of the study ====")
        _print_instance_properties(study, pre_fix, print_val=False)
        print(f"{pre_fix}==============================")

        # The following commneted code block uses and prints the flat list of all selected series.
        # for ss_obj in sss_obj.selected_series:
        #     pre_fix = "    "
        #     _print_instance_properties(ss_obj, pre_fix, print_val=False)
        #     pre_fix = "      "
        #     print(f"{pre_fix}==== Details of the series ====")
        #     _print_instance_properties(ss_obj, pre_fix)
        #     print(f"{pre_fix}===============================")

        # The following uses gouping by selection name, and prints list of series for each.
        for selection_name, ss_list in sss_obj.series_by_selection_name.items():
            pre_fix = "    "
            print(f"{pre_fix}Selection name: {selection_name}")
            for ss_obj in ss_list:
                pre_fix = "        "
                _print_instance_properties(ss_obj, pre_fix, print_val=False)
                print(f"{pre_fix}==== Details of the series ====")
                _print_instance_properties(ss_obj, pre_fix)
                print(f"{pre_fix}===============================")

    print(f"Total # of series selected: {len(series_list)}")


# Sample rule used for testing
Sample_Rules_Text = """
{
    "selections": [
        {
            "name": "CT Series 1",
            "conditions": {
                "StudyDescription": "(?i)^Spleen",
                "Modality": "(?i)CT",
                "SeriesDescription": "(?i)^No series description|(.*?)"
            }
        },
        {
            "name": "CT Series 2",
            "conditions": {
                "Modality": "CT",
                "BodyPartExamined": "Abdomen",
                "SeriesDescription" : "Not to be matched"
            }
        }
    ]
}
"""

if __name__ == "__main__":
    main()
