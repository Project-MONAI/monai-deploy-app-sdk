import math
from pydicom import DataElement
from pydicom.tag import Tag, BaseTag, tag_in_exception, TagType
from typing import Dict, Optional, Set, Union



from .domain import Domain



class DICOMSOPInstance(Domain):
    def __init__(self, native_sop):
        super().__init__(None)
        self._sop = native_sop

    def get_native_sop_instance(self):
        return self._sop


    def __getitem__(self, element_offset: int) -> DataElement:
        return self._sop.__getitem__(element_offset)


    def __getitem__(self, key: Union[slice, TagType]) -> Union["Dataset", DataElement]:
        return self._sop.__getitem__(key)

    
    @property
    def pixel_array(self):
        return self._sop.pixel_array

    def __str__(self):
        result = "---------------" +"\n"
        
        return result

