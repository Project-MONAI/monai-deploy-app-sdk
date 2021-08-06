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

import copy

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
from monai.deploy.core.domain.image import Image



from os import listdir
from os.path import isfile, join

import math

from pydicom.data import get_testdata_file
from pydicom.fileset import FileSet
from pydicom.uid import generate_uid
import numpy as np

@input("dicom_series", DICOMSeries, IOType.IN_MEMORY)
@output("image", Image, IOType.IN_MEMORY)


class DICOMSeriesToVolumeOperator(Operator):

    def compute(self, input: InputContext, output: OutputContext, context: ExecutionContext):
        """Performs computation for this operator.
        """
        dicom_series = input.get()
        self.prepare_series(dicom_series)
        metadata = self.create_metadata(dicom_series)
        voxel_data = self.generate_voxel_data(dicom_series)
        image = self.create_volumetric_image(voxel_data, metadata)
        output.set(image, "image")


    def generate_voxel_data(self, series):
        slices = series.get_sop_instances()
        vol_data = np.stack([s.get_pixel_array() for s in slices])
        vol_data = vol_data.astype(np.int16)
        intercept = slices[0][0x0028, 0x1052].value
        slope = slices[0][0x0028, 0x1053].value

        if slope != 1:
            image = slope * vol_data.astype(np.float64)
            image = vol_data.astype(np.int16)
        vol_data += np.int16(intercept)
        return np.array(vol_data, dtype=np.int16)




    def create_volumetric_image(self, vox_data, metadata):
        image = Image(vox_data, metadata)
        return image


    def prepare_series(self, series):

        if len(series._sop_instances) <= 1:
            return

        slice_indices_to_be_removed = []
        row_pixel_spacing = 0.0
        col_pixel_spacing = 0.0
        depth_pixel_spacing = 0.0
        last_slice_normal = [0.0, 0.0, 0.0]

        for slice_index, slice in enumerate(series._sop_instances):
            distance = 0.0
            point = [0.0, 0.0, 0.0]
            slice_normal = [0.0, 0.0, 0.0]
            slice_position = None
            cosines = None
            
            try:
                image_orientation_patient_de = slice[0x0020,0x0037]
                if image_orientation_patient_de != None :
                    image_orientation_patient = image_orientation_patient_de.value
                    cosines = image_orientation_patient
            except KeyError:
                pass


            try:
                image_poisition_patient_de = slice[0x0020,0x0032]
                if image_poisition_patient_de  != None :
                    image_poisition_patient = image_poisition_patient_de .value
                    slice_position = image_poisition_patient
            except KeyError:
                pass


            distance = 0.0

            if (cosines != None) and (slice_position != None):
                slice_normal[0] = cosines[1]*cosines[5] - cosines[2]*cosines[4]
                slice_normal[1] = cosines[2]*cosines[3] - cosines[0]*cosines[5]
                slice_normal[2] = cosines[0]*cosines[4] - cosines[1]*cosines[3]

                last_slice_normal = copy.deepcopy(slice_normal)
                
                i = 0
                while i < 3:
                    point[i] = slice_normal[i] * slice_position[i]
                    i += 1

                distance += point[0] + point[1] + point[2]

                series._sop_instances[slice_index].distance = distance
                series._sop_instances[slice_index].first_pixel_on_slice_normal = point
            else:
                slice_indices_to_be_removed.append(slice_index)




        for sl_index, sl in enumerate(series._sop_instances):
            del series._sop_instances[sl_index]
        

        series._sop_instances = sorted(series._sop_instances, key=lambda s: s.distance)
        series.depth_direction_cosine = copy.deepcopy(last_slice_normal)


        if len(series._sop_instances) > 1:
            p1 = series._sop_instances[0].first_pixel_on_slice_normal
            p2 = series._sop_instances[1].first_pixel_on_slice_normal
            depth_pixel_spacing = (p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1]) + (p1[2] - p2[2])*(p1[2] - p2[2])
            depth_pixel_spacing = math.sqrt(depth_pixel_spacing)
            series.depth_pixel_spacing = depth_pixel_spacing
        

    
    def create_metadata(self, series):
        metadata = {}
        metadata["series_instance_uid"] = series.get_series_instance_uid()

        try:
             metadata["series_date"] =  series.series_date
        except AttributeError:
            pass
        
        try:
            metadata["series_time"] =  series.series_time
        except AttributeError:
            pass 
        
        try:
            metadata["modality"] =  series.modality
        except AttributeError:
            pass

        try:
            metadata["series_description"] =  series.series_description
        except AttributeError:
            pass

        
        try:
            metadata["row_pixel_spacing"] =  series.row_pixel_spacing
        except AttributeError:
            pass

        try:
            metadata["col_pixel_spacing"] =  series.col_pixel_spacing
        except AttributeError:
            pass


        try:
            metadata["depth_pixel_spacing"] =  series.depth_pixel_spacing
        except AttributeError:
            pass



        try:
            metadata["row_direction_cosine"] =  series.row_direction_cosine
        except AttributeError:
            pass

        try:
            metadata["col_direction_cosine"] =  series.col_direction_cosine
        except AttributeError:
            pass

        try:
            metadata["depth_direction_cosine"] =  series.depth_direction_cosine
        except AttributeError:
            pass


        return metadata


        

def main():
    op = DICOMSeriesToVolumeOperator()
    # data_path = "/home/rahul/medical-images/mixed-data/"
    data_path = "/home/rahul/medical-images/lung-ct-2/"
    files = []
    loader = DICOMDataLoaderOperator()
    loader._list_files(files, data_path)
    study_list = loader._load_data(files)

    series = study_list[0].get_all_series()[0]
    op.prepare_series(series)
    voxels = op.generate_voxel_data(series)
    metadata = op.create_metadata(series)
    image = op.create_volumetric_image(voxels, metadata)

    print(series)
    print(metadata.keys())
   



if __name__ == "__main__":
    main()
