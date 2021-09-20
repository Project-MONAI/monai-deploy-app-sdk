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

import os
from typing import List

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_study import DICOMStudy
from monai.deploy.utils.importutil import optional_import

dcmread, _ = optional_import("pydicom", name="dcmread")
get_testdata_file, _ = optional_import("pydicom.data", name="dcmread")
FileSet, _ = optional_import("pydicom.fileset", name="FileSet")
generate_uid, _ = optional_import("pydicom.uid", name="generate_uid")


@md.input("dicom_files", DataPath, IOType.DISK)
@md.output("dicom_study_list", List[DICOMStudy], IOType.IN_MEMORY)
@md.env(pip_packages=["pydicom >= 1.4.2"])
class DICOMDataLoaderOperator(Operator):
    """
    This operator loads a collection of DICOM Studies in memory
    given a directory which contains a list of SOP Instances.
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        """Performs computation for this operator.

        It scans through the input directory for all SOP instances.
        It groups them by a collection of studies where each study contains one or more series.
        This method returns a set of studies.
        """
        files: List[str] = []
        input_path = op_input.get().path
        self._list_files(input_path, files)
        dicom_study_list = self._load_data(files)
        op_output.set(dicom_study_list)

    def _list_files(self, path, files: List[str]):
        """Collects fully qualified names of all files recurvisely given a directory path.

        Args:
            path: A directoty containing DICOM SOP instances. It have have nested hirerarchical directories.
            files: This method populates "files" with fully qualified names of files that belong to the specified directory.
        """
        for item in os.listdir(path):
            item = os.path.join(path, item)
            if os.path.isdir(item):
                self._list_files(item, files)
            else:
                files.append(item)

    def _load_data(self, files: List[str]):
        """Provides a list of DICOM Studies given a list of fully qualified file names.

        Args:
            files: A list of file names that represents SOP Instances
        """
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

            series_instance_uid = sop_instance[0x0020, 0x000E].value

            if series_instance_uid not in series_dict:
                series = DICOMSeries(series_instance_uid)
                series_dict[series_instance_uid] = series
                self.populate_series_attributes(series, sop_instance)
                study_dict[study_instance_uid].add_series(series)

            series_dict[series_instance_uid].add_sop_instance(sop_instance)
        return list(study_dict.values())

    def populate_study_attributes(self, study, sop_instance):
        """Populates study level attributes in the study data structure.

        Args:
            study: A DICOM Study instance that needs to be filled-in with study level attribute values
            sop_instance: A sample DICOM SOP Instance that contains the list of attributed which will be parsed
        """
        try:
            study_id_de = sop_instance[0x0020, 0x0010]
            if study_id_de is not None:
                study.study_id = study_id_de.value
        except KeyError:
            pass

        try:
            study_date_de = sop_instance[0x0008, 0x0020]
            if study_date_de is not None:
                study.study_date = study_date_de.value
        except KeyError:
            pass

        try:
            study_time_de = sop_instance[0x0008, 0x0030]
            if study_time_de is not None:
                study.study_time = study_time_de.value
        except KeyError:
            pass

        try:
            study_desc_de = sop_instance[0x0008, 0x1030]
            if study_desc_de is not None:
                study.study_description = study_desc_de.value
        except KeyError:
            pass

        try:
            accession_number_de = sop_instance[0x0008, 0x0050]
            if accession_number_de is not None:
                study.accession_number = accession_number_de.value
        except KeyError:
            pass

    def populate_series_attributes(self, series, sop_instance):
        """Populates series level attributes in the study data structure.

        Args:
            study: A DICOM Series instance that needs to be filled-in with series level attribute values
            sop_instance: A sample DICOM SOP Instance that contains the list of attributed which will be parsed
        """
        try:
            series_date_de = sop_instance[0x0008, 0x0021]
            if series_date_de is not None:
                series.series_date = series_date_de.value
        except KeyError:
            pass

        try:
            series_time_de = sop_instance[0x0008, 0x0031]
            if series_time_de is not None:
                series.series_time = series_time_de.value
        except KeyError:
            pass

        try:
            series_modality_de = sop_instance[0x0008, 0x0060]
            if series_modality_de is not None:
                series.modality = series_modality_de.value
        except KeyError:
            pass

        try:
            series_description_de = sop_instance[0x0008, 0x103E]
            if series_description_de is not None:
                series.series_description = series_description_de.value
        except KeyError:
            pass

        try:
            body_part_examined_de = sop_instance[0x0008, 0x0015]
            if body_part_examined_de is not None:
                series.body_part_examined = body_part_examined_de.value
        except KeyError:
            pass

        try:
            patient_position_de = sop_instance[0x0018, 0x5100]
            if patient_position_de is not None:
                series.patient_position = patient_position_de.value
        except KeyError:
            pass

        try:
            series_number_de = sop_instance[0x0020, 0x0011]
            if series_number_de is not None:
                series.series_number = series_number_de.value
        except KeyError:
            pass

        try:
            laterality_de = sop_instance[0x0020, 0x0060]
            if laterality_de is not None:
                series.laterality = laterality_de.value
        except KeyError:
            pass

        try:
            pixel_spacing_de = sop_instance[0x0028, 0x0030]
            if pixel_spacing_de is not None:
                series.row_pixel_spacing = pixel_spacing_de.value[0]
                series.col_pixel_spacing = pixel_spacing_de.value[1]
        except KeyError:
            pass

        try:
            image_orientation_paient_de = sop_instance[0x0020, 0x0037]
            if image_orientation_paient_de is not None:
                orientation_orig = image_orientation_paient_de.value
                orientation = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                for index, _ in enumerate(orientation_orig):
                    orientation[index] = float(orientation_orig[index])

                series.row_direction_cosine = orientation[0:3]
                series.col_direction_cosine = orientation[3:6]

        except KeyError:
            pass


def main():
    # data_path = "/home/rahul/medical-images/mixed-data/"
    data_path = "/home/rahul/medical-images/lung-ct-1/"
    files = []
    loader = DICOMDataLoaderOperator()
    loader._list_files(data_path, files)
    study_list = loader._load_data(files)

    for study in study_list:
        print("###############################")
        print(study)
        for series in study.get_all_series():
            print(series)


if __name__ == "__main__":
    main()
