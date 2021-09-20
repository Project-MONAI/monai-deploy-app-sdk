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

from typing import Dict, List

import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, InputContext, IOType, Operator, OutputContext
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_study import DICOMStudy
from monai.deploy.exceptions import ItemNotExistsError
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator


@md.input("dicom_study_list", List[DICOMStudy], IOType.IN_MEMORY)
@md.input("selection_rules", Dict, IOType.IN_MEMORY)
@md.output("dicom_series", DICOMSeries, IOType.IN_MEMORY)
class DICOMSeriesSelectorOperator(Operator):
    """This operator filters out a list of DICOM Series given some selection rules.

    This is a placeholder class. It has not been implemented yet.
    Currently this operator always selects the first series in the List.
    When implemented it will honor the selection rules expressed in a dictionary format.
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        """Performs computation for this operator."""
        try:
            dicom_study_list = op_input.get("dicom_study_list")
            # selection_rules = op_input.get("selection_rules")
            dicom_series_list = self.filter(None, dicom_study_list)
            op_output.set(dicom_series_list, "dicom_series")
        except ItemNotExistsError:
            pass

    def filter(self, selection_rules, dicom_study_list):
        return dicom_study_list[0].get_all_series()[0]


def main():
    data_path = "/home/rahul/medical-images/mixed-data/"
    # data_path = "/home/rahul/medical-images/lung-ct-1/"
    files = []
    loader = DICOMDataLoaderOperator()
    loader._list_files(files, data_path)
    study_list = loader._load_data(files)

    selector = DICOMSeriesSelectorOperator()
    series = selector.filter(None, study_list)
    print(series)


if __name__ == "__main__":
    main()
