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

from typing import Generic

from monai.deploy.utils.importutil import optional_import

from .dicom_types import SopInstance_KT, SopInstance_VT, SopInstanceInterface
from .domain import Domain

DataElement, _ = optional_import("pydicom", name="DataElement")
Dataset, _ = optional_import("pydicom", name="Dataset")
TagType, _ = optional_import("pydicom.tag")


# Ignore type for Generic: https://github.com/google/pytype/issues/704
class DICOMSOPInstance(Domain, Generic[SopInstance_KT, SopInstance_VT]):  # type: ignore
    """This class representes a SOP Instance.

    An attribute can be looked up with a slice ([group_number, element number]).
    """

    def __init__(self, native_sop):
        super().__init__(None)
        self._sop: SopInstanceInterface = native_sop

    def get_native_sop_instance(self):
        return self._sop

    def __getitem__(self, key: SopInstance_KT) -> SopInstance_VT:
        return self._sop.__getitem__(key)

    def get_pixel_array(self):
        return self._sop.pixel_array

    def __str__(self):
        result = "---------------" + "\n"

        return result
