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
from monai.deploy.operators.dicom_series import DICOMSeries
from monai.deploy.operators.dicom_study import DICOMStudy

from os import listdir
from os.path import isfile, join


from pydicom import dcmread
import numpy as np

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

from pydicom.data import get_testdata_file
from pydicom.fileset import FileSet
from pydicom.uid import generate_uid


@input("image", DataPath, IOType.DISK)
@output("image", Image, IOType.IN_MEMORY)
class DICOMSeriesLoaderOperator(Operator):
    pass
