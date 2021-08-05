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
import copy
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_study import DICOMStudy

from os import listdir
from os.path import isfile, join


from pydicom import dcmread

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings
import math

from pydicom.data import get_testdata_file
from pydicom.fileset import FileSet
from pydicom.uid import generate_uid


@input("dicom_files", DataPath, IOType.DISK)
@output("dicom_study_list", DICOMStudy, IOType.IN_MEMORY)
class DICOMDataLoaderOperator(Operator):

    def compute(self, input: InputContext, output: OutputContext, context: ExecutionContext):
        """Performs computation for this operator.
        """
        files = []
        input_path = input.get().path
        self._list_files(files, input_path)
        dicom_study_list = self._load_data(files)
        output.set(dicom_study_list)
    

    def _list_files(self, files, path):
        for item in os.listdir(path):
            item = os.path.join(path, item)
            if os.path.isdir(item):
                self._list_files(files, item)
            else:
                files.append(item)


    def _load_data(self, files):
        study_dict = {}
        series_dict = {}
        sop_instances = []

        for file in files:
            sop_instances.append(dcmread(file))

        for sop_instance in sop_instances:

            study_instance_uid = sop_instance[0x0020, 0x000D].value

            if study_instance_uid not in study_dict:
                study = DICOMStudy(study_instance_uid)
                self.populate_study_attributes(study, sop_instance)
                study_dict[study_instance_uid] = study


            series_instance_uid = sop_instance[0x0020,0x000E].value

            if series_instance_uid not in series_dict:
                series = DICOMSeries(series_instance_uid)
                series_dict[series_instance_uid] = series
                self.populate_series_attributes(series, sop_instance)
                study_dict[study_instance_uid].add_series(series)
                
                
            series_dict[series_instance_uid].add_sop_instance(sop_instance)

        
        return list(study_dict.values())


    def populate_study_attributes(self, study, sop_instance):
        try:
            study_id_de = sop_instance[0x0020,0x0010]
            if study_id_de != None :
                study.study_id = study_id_de.value
        except KeyError:
            pass

        try:
            study_date_de = sop_instance[0x0008,0x0020]
            if study_date_de != None :
                study.study_date = study_date_de.value
        except KeyError:
            pass

        try:
            study_time_de = sop_instance[0x0008,0x0030]
            if study_time_de != None :
                study.study_time = study_time_de.value
        except KeyError:
            pass

        try:
            study_desc_de = sop_instance[0x0008,0x1030]
            if study_desc_de != None :
                study.study_description = study_desc_de.value
        except KeyError:
            pass

        try:
            accession_number_de = sop_instance[0x0008,0x0050]
            if accession_number_de != None :
                study.accession_number = accession_number_de.value
        except KeyError:
            pass


    def populate_series_attributes(self, series, sop_instance):

        try:
            series_date_de = sop_instance[0x0008,0x0021]
            if series_date_de != None :
                series.series_date = series_date_de.value
        except KeyError:
            pass

        try:
            series_time_de = sop_instance[0x0008, 0x0031]
            if series_time_de != None :
                series.series_time = series_time_de.value
        except KeyError:
            pass
        
        try:
            series_modality_de = sop_instance[0x0008, 0x0060]
            if series_modality_de != None :
                series.modality = series_modality_de.value
        except KeyError:
            pass


        try:
            series_description_de = sop_instance[0x0008, 0x103E]
            if series_description_de != None :
                series.series_description = series_description_de.value
        except KeyError:
            pass

        try:
            body_part_examined_de = sop_instance[0x0008, 0x0015]
            if body_part_examined_de != None :
                series.body_part_examined = body_part_examined_de.value
        except KeyError:
            pass


        try:
            patient_position_de = sop_instance[0x0018, 0x5100]
            if patient_position_de != None :
                series.patient_position = patient_position_de.value
        except KeyError:
            pass


        try:
            series_number_de = sop_instance[0x0020, 0x0011]
            if series_number_de != None :
                series.series_number = series_number_de.value
        except KeyError:
            pass


        try:
            laterality_de = sop_instance[0x0020, 0x0060]
            if laterality_de != None :
                series.laterality = laterality_de.value
        except KeyError:
            pass


        try:
            row_pixel_spacing_de = sop_instance[0x0028, 0x0030]
            if row_pixel_spacing_de != None :
                series.row_pixel_spacing = row_pixel_spacing_de.value[0]
        except KeyError:
            pass


        try:
            col_pixel_spacing_de = sop_instance[0x0028, 0x0030]
            if col_pixel_spacing_de != None :
                series.col_pixel_spacing = col_pixel_spacing_de.value[1]
        except KeyError:
            pass


        try:
            image_orientation_paient_de = sop_instance[0x0020, 0x0037]
            if  image_orientation_paient_de != None :
                orientation = image_orientation_paient_de.value
                series.row_direction_cosine = copy.deepcopy(orientation[0:3])
                series.col_direction_cosine = copy.deepcopy(orientation[3:6])
        except KeyError:
            pass



def main():
    data_path = "/home/rahul/medical-images/mixed-data/"
    # data_path = "/home/rahul/medical-images/lung-ct-1/"
    files = []
    loader = DICOMDataLoaderOperator()
    loader._list_files(files, data_path)
    study_list = loader._load_data(files)


    for study in study_list:
        print("###############################")
        print(study)
        for series in study.get_all_series():
            print (series)
    




if __name__ == "__main__":
    main()
