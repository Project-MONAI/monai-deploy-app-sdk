import sys
sys.path.insert(0,'../../../')

from os import listdir
from os.path import isfile, join

from monai.deploy.foundation.base_operator import BaseOperator
from monai.deploy.foundation.operator_info import OperatorInfo
from monai.deploy.foundation.io_type import IOType
from monai.deploy.domain.image import Image


from pydicom import dcmread
import numpy as np


class DICOMSeriesLoaderOperator(BaseOperator):

    def __init__(self):
        super().__init__()

    
    def create_op_info(self):
        _op_info = OperatorInfo()
        _op_info.num_input_ports = 1
        _op_info.num_output_ports = 1
        _op_info.set_input_storage_type(0, IOType.DISK)
        _op_info.set_output_storage_type(0, IOType.IN_MEMORY)
        return _op_info

    def execute(self, execution_context):
        super().execute(execution_context)
        data_in =  execution_context.get_operator_input(self._uid, 0)
        data_out = load(data_in)
        execution_context.set_operator_output(self._uid, 0, data_out)
    

    def load(self, dir_path):
        image = Image()
        list_files = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
 
        # load the DICOM files
        original_slices = []

        for fname in list_files:
            original_slices.append(dcmread(fname))

        slices = []

        # Below I am only commelicting slices which
        # have the slice location property. This is
        # not needed, we will fix it later.
        for slice in original_slices:
            if hasattr(slice, 'SliceLocation'):
                slices.append(slice)
        

        # this is an attempt to sort the slices. However
        # here is a better way. We need to first
        # find out the position of the top left voxel
        # of each slice. Then project that voxel along 
        # the slice normal. Gind the distance of that
        # voxel from the origin of the patient coordinate system
        # along the slice normal. Using that information sort 
        # the slices. Below what we have though is a poor man's
        # version of sorting.

        slices = sorted(slices, key=lambda s: s.SliceLocation)

        ps = slices[0].PixelSpacing
        ss = slices[0].SliceThickness
        ax_aspect = ps[1]/ps[0]
        sag_aspect = ps[1]/ss
        cor_aspect = ss/ps[0]

        img_shape = list(slices[0].pixel_array.shape)
        img_shape.append(len(slices))
        img3d = np.zeros(img_shape)

        for i, s in enumerate(slices):
            img2d = s.pixel_array
            img3d[:, :, i] = img2d

        
        
        image.ndim = img3d.ndim
        image.shape = img3d.shape

        # this is only setting pixel spacing in row and column direction, but not in depth direction
        # this needs to be fixed
        image.spacing = slices[0].PixelSpacing

        # this is only setting direction cosines of the first row and first column of the volume
        # this needs to be fixed
        image.direction_cosines = slices[0][0x0020,0x0037]

        image.modality = slices[0].Modality

        image.pixel_data = img3d
        return image


def main():
    data_path = "/home/rahul/medical-images/lung-ct-2/"
    loader = DICOMSeriesLoaderOperator()
    img = loader.load(data_path)
    print(img)

if __name__ == "__main__":
    main()


