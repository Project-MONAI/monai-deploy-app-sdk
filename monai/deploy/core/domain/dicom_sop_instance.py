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





import math
from pydicom import DataElement
from pydicom.tag import Tag, BaseTag, tag_in_exception, TagType
from typing import Dict, Optional, Set, Union



from .domain import Domain



class DICOMSOPInstance(Domain):

    """This class representes a SOP Instance. An attribute cane be looked up with slice[group_number, element number]
    """

    def __init__(self, native_sop):
        super().__init__(None)
        self._sop = native_sop

    def get_native_sop_instance(self):
        return self._sop


    def __getitem__(self, element_offset: int) -> DataElement:
        return self._sop.__getitem__(element_offset)


    def __getitem__(self, key: Union[slice, TagType]) -> Union["Dataset", DataElement]:
        return self._sop.__getitem__(key)


    def get_pixel_array(self):
        return self._sop.pixel_array

    def __str__(self):
        result = "---------------" +"\n"
        
        return result

