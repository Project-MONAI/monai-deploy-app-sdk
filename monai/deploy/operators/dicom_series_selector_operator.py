# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional, Union

from monai.deploy.core import (
    DataPath,
    ExecutionContext,
    Image,
    InputContext,
    IOType,
    Operator,
    OutputContext,
    input,
    output,
)
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_study import DICOMStudy
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.exceptions import ItemNotExistsError

from monai.deploy.core.domain.image import Image



from os import listdir
from os.path import isfile, join

import math

from pydicom.data import get_testdata_file
from pydicom.fileset import FileSet
from pydicom.uid import generate_uid
import numpy as np

@input("dicom_study_list", DICOMStudy, IOType.IN_MEMORY)
@input("selection_rules", Dict, IOType.IN_MEMORY)
@output("dicom_series", DICOMSeries, IOType.IN_MEMORY)


class DICOMSeriesSelectorOperator(Operator):

    def compute(self, input: InputContext, output: OutputContext, context: ExecutionContext):
        """Performs computation for this operator.
        """
        try:
            dicom_study_list = input.get("dicom_study_list")
            #selection_rules = input.get("selection_rules")
            dicom_series_list = self.filter(None, dicom_study_list)
            output.set(dicom_series_list, "dicom_series")
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
    print (series)
   



if __name__ == "__main__":
    main()
