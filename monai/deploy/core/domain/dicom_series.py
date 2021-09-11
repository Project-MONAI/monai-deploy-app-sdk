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

from typing import Any

from .dicom_sop_instance import DICOMSOPInstance
from .domain import Domain


class DICOMSeries(Domain):
    """DICOM Series represents a DICOM Series. It contains a collection of SOP Instances"""

    def __init__(self, series_instance_uid):
        super().__init__(None)
        self._series_instance_uid = series_instance_uid
        self._sop_instances = []

        self._series_date: Any = None
        self._series_time: Any = None
        self._modality: Any = None
        self._series_description: Any = None
        self._body_part_examined: Any = None
        self._patient_position: Any = None
        self._series_number: Any = None
        self._laterality: Any = None
        self._row_pixel_spacing: Any = None
        self._col_pixel_spacing: Any = None
        self._depth_pixel_spacing: Any = None
        self._row_direction_cosine: Any = None
        self._col_direction_cosine: Any = None
        self._depth_direction_cosine: Any = None
        self._dicom_affine_transform: Any = None
        self._nifti_affine_transform: Any = None

    def get_series_instance_uid(self):
        return self._series_instance_uid

    def add_sop_instance(self, sop_instance):
        dicom_sop_instance = DICOMSOPInstance(sop_instance)
        self._sop_instances.append(dicom_sop_instance)

    def get_sop_instances(self):
        return self._sop_instances

    @property
    def series_date(self):
        return self._series_date

    @series_date.setter
    def series_date(self, val):
        self._series_date = val

    @property
    def series_time(self):
        return self._series_time

    @series_time.setter
    def series_time(self, val):
        self._series_time = val

    @property
    def modality(self):
        return self._modality

    @modality.setter
    def modality(self, val):
        self._modality = val

    @property
    def series_description(self):
        return self._series_description

    @series_description.setter
    def series_description(self, val):
        self._series_description = val

    @property
    def body_part_examined(self):
        return self._body_part_examined

    @body_part_examined.setter
    def body_part_examined(self, val):
        self._body_part_examined = val

    @property
    def patient_position(self):
        return self._patient_position

    @patient_position.setter
    def patient_position(self, val):
        self._patient_position = val

    @property
    def series_number(self):
        return self._series_number

    @series_number.setter
    def series_number(self, val):
        self._series_number = val

    @property
    def laterality(self):
        return self._laterality

    @laterality.setter
    def laterality(self, val):
        self._laterality = val

    @property
    def row_pixel_spacing(self):
        return self._row_pixel_spacing

    @row_pixel_spacing.setter
    def row_pixel_spacing(self, val):
        self._row_pixel_spacing = val

    @property
    def col_pixel_spacing(self):
        return self._col_pixel_spacing

    @col_pixel_spacing.setter
    def col_pixel_spacing(self, val):
        self._col_pixel_spacing = val

    @property
    def depth_pixel_spacing(self):
        return self._depth_pixel_spacing

    @depth_pixel_spacing.setter
    def depth_pixel_spacing(self, val):
        self._depth_pixel_spacing = val

    @property
    def row_direction_cosine(self):
        return self._row_direction_cosine

    @row_direction_cosine.setter
    def row_direction_cosine(self, val):
        self._row_direction_cosine = val

    @property
    def col_direction_cosine(self):
        return self._col_direction_cosine

    @col_direction_cosine.setter
    def col_direction_cosine(self, val):
        self._col_direction_cosine = val

    @property
    def depth_direction_cosine(self):
        return self._depth_direction_cosine

    @depth_direction_cosine.setter
    def depth_direction_cosine(self, val):
        self._depth_direction_cosine = val

    @property
    def dicom_affine_transform(self):
        return self._dicom_affine_transform

    @dicom_affine_transform.setter
    def dicom_affine_transform(self, val):
        self._dicom_affine_transform = val

    @property
    def nifti_affine_transform(self):
        return self._nifti_affine_transform

    @nifti_affine_transform.setter
    def nifti_affine_transform(self, val):
        self._nifti_affine_transform = val

    def __str__(self):
        result = "---------------" + "\n"

        series_instance_uid_attr = "Series Instance UID: " + self._series_instance_uid + "\n"
        result += series_instance_uid_attr

        try:
            num_sop_instances = "Num SOP Instances: " + str(len(self._sop_instances)) + "\n"
            result += num_sop_instances
        except AttributeError:
            pass

        try:
            series_date_attr = "Series Date: " + self.series_date + "\n"
            result += series_date_attr
        except AttributeError:
            pass

        try:
            series_time_attr = "Series Time: " + self.series_time + "\n"
            result += series_time_attr
        except AttributeError:
            pass

        try:
            modality_attr = "Modality: " + self.modality + "\n"
            result += modality_attr
        except AttributeError:
            pass

        try:
            series_desc_attr = "Series Description: " + self.series_description + "\n"
            result += series_desc_attr
        except AttributeError:
            pass

        try:
            row_pixel_spacing_attr = "Row Pixel Spacing: " + str(self.row_pixel_spacing) + "\n"
            result += row_pixel_spacing_attr
        except AttributeError:
            pass

        try:
            col_pixel_spacing_attr = "Column Pixel Spacing: " + str(self.col_pixel_spacing) + "\n"
            result += col_pixel_spacing_attr
        except AttributeError:
            pass

        try:
            depth_pixel_spacing_attr = "Depth Pixel Spacing: " + str(self.depth_pixel_spacing) + "\n"
            result += depth_pixel_spacing_attr
        except AttributeError:
            pass

        try:
            row_direction_cosine_attr = "Row Direction Cosine: " + str(self.row_direction_cosine) + "\n"
            result += row_direction_cosine_attr
        except AttributeError:
            pass

        try:
            col_direction_cosine_attr = "Column Direction Cosine: " + str(self.col_direction_cosine) + "\n"
            result += col_direction_cosine_attr
        except AttributeError:
            pass

        try:
            depth_direction_cosine_attr = "Depth Direction Cosine: " + str(self.depth_direction_cosine) + "\n"
            result += depth_direction_cosine_attr
        except AttributeError:
            pass

        try:
            dicom_affine_transform_attr = "DICOM affine transform: " + "\n" + str(self.dicom_affine_transform) + "\n"
            result += dicom_affine_transform_attr
        except AttributeError:
            pass

        try:
            nifti_affine_transform_attr = "NIFTI affine transform: " + "\n" + str(self.nifti_affine_transform) + "\n"
            result += nifti_affine_transform_attr
        except AttributeError:
            pass

        result += "---------------" + "\n"

        return result
