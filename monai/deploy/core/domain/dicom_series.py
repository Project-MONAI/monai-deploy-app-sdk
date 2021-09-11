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

from .dicom_sop_instance import DICOMSOPInstance
from .domain import Domain


class DICOMSeries(Domain):
    """DICOM Series represents a DICOM Series. It contains a collection of SOP Instances"""

    def __init__(self, series_instance_uid):
        super().__init__(None)
        self._series_instance_uid = series_instance_uid
        self._sop_instances = []

        # Do not set attributes in advance to save memory

        # self._series_date: Any = None
        # self._series_time: Any = None
        # self._modality: Any = None
        # self._series_description: Any = None
        # self._body_part_examined: Any = None
        # self._patient_position: Any = None
        # self._series_number: Any = None
        # self._laterality: Any = None
        # self._row_pixel_spacing: Any = None
        # self._col_pixel_spacing: Any = None
        # self._depth_pixel_spacing: Any = None
        # self._row_direction_cosine: Any = None
        # self._col_direction_cosine: Any = None
        # self._depth_direction_cosine: Any = None
        # self._dicom_affine_transform: Any = None
        # self._nifti_affine_transform: Any = None

    def get_series_instance_uid(self):
        return self._series_instance_uid

    def add_sop_instance(self, sop_instance):
        dicom_sop_instance = DICOMSOPInstance(sop_instance)
        self._sop_instances.append(dicom_sop_instance)

    def get_sop_instances(self):
        return self._sop_instances

    @property
    def series_date(self):
        return getattr(self, "_series_date", None)

    @series_date.setter
    def series_date(self, val):
        self._series_date = val

    @property
    def series_time(self):
        return getattr(self, "_series_time", None)

    @series_time.setter
    def series_time(self, val):
        self._series_time = val

    @property
    def modality(self):
        return getattr(self, "_modality", None)

    @modality.setter
    def modality(self, val):
        self._modality = val

    @property
    def series_description(self):
        return getattr(self, "_series_description", None)

    @series_description.setter
    def series_description(self, val):
        self._series_description = val

    @property
    def body_part_examined(self):
        return getattr(self, "_body_part_examined", None)

    @body_part_examined.setter
    def body_part_examined(self, val):
        self._body_part_examined = val

    @property
    def patient_position(self):
        return getattr(self, "_patient_position", None)

    @patient_position.setter
    def patient_position(self, val):
        self._patient_position = val

    @property
    def series_number(self):
        return getattr(self, "_series_number", None)

    @series_number.setter
    def series_number(self, val):
        self._series_number = val

    @property
    def laterality(self):
        return getattr(self, "_laterality", None)

    @laterality.setter
    def laterality(self, val):
        self._laterality = val

    @property
    def row_pixel_spacing(self):
        return getattr(self, "_row_pixel_spacing", None)

    @row_pixel_spacing.setter
    def row_pixel_spacing(self, val):
        self._row_pixel_spacing = val

    @property
    def col_pixel_spacing(self):
        return getattr(self, "_col_pixel_spacing", None)

    @col_pixel_spacing.setter
    def col_pixel_spacing(self, val):
        self._col_pixel_spacing = val

    @property
    def depth_pixel_spacing(self):
        return getattr(self, "_depth_pixel_spacing", None)

    @depth_pixel_spacing.setter
    def depth_pixel_spacing(self, val):
        self._depth_pixel_spacing = val

    @property
    def row_direction_cosine(self):
        return getattr(self, "_row_direction_cosine", None)

    @row_direction_cosine.setter
    def row_direction_cosine(self, val):
        self._row_direction_cosine = val

    @property
    def col_direction_cosine(self):
        return getattr(self, "_col_direction_cosine", None)

    @col_direction_cosine.setter
    def col_direction_cosine(self, val):
        self._col_direction_cosine = val

    @property
    def depth_direction_cosine(self):
        return getattr(self, "_depth_direction_cosine", None)

    @depth_direction_cosine.setter
    def depth_direction_cosine(self, val):
        self._depth_direction_cosine = val

    @property
    def dicom_affine_transform(self):
        return getattr(self, "_dicom_affine_transform", None)

    @dicom_affine_transform.setter
    def dicom_affine_transform(self, val):
        self._dicom_affine_transform = val

    @property
    def nifti_affine_transform(self):
        return getattr(self, "_nifti_affine_transform", None)

    @nifti_affine_transform.setter
    def nifti_affine_transform(self, val):
        self._nifti_affine_transform = val

    def __str__(self):
        result = "---------------" + "\n"

        series_instance_uid_attr = "Series Instance UID: " + self._series_instance_uid + "\n"
        result += series_instance_uid_attr

        num_sop_instances = "Num SOP Instances: " + str(len(self._sop_instances)) + "\n"
        result += num_sop_instances

        if self.series_date is not None:
            series_date_attr = "Series Date: " + self.series_date + "\n"
            result += series_date_attr

        if self.series_time is not None:
            series_time_attr = "Series Time: " + self.series_time + "\n"
            result += series_time_attr

        if self.modality is not None:
            modality_attr = "Modality: " + self.modality + "\n"
            result += modality_attr

        if self.series_description is not None:
            series_desc_attr = "Series Description: " + self.series_description + "\n"
            result += series_desc_attr

        if self.row_pixel_spacing is not None:
            row_pixel_spacing_attr = "Row Pixel Spacing: " + str(self.row_pixel_spacing) + "\n"
            result += row_pixel_spacing_attr

        if self.col_pixel_spacing is not None:
            col_pixel_spacing_attr = "Column Pixel Spacing: " + str(self.col_pixel_spacing) + "\n"
            result += col_pixel_spacing_attr

        if self.depth_pixel_spacing is not None:
            depth_pixel_spacing_attr = "Depth Pixel Spacing: " + str(self.depth_pixel_spacing) + "\n"
            result += depth_pixel_spacing_attr

        if self.row_direction_cosine is not None:
            row_direction_cosine_attr = "Row Direction Cosine: " + str(self.row_direction_cosine) + "\n"
            result += row_direction_cosine_attr

        if self.col_direction_cosine is not None:
            col_direction_cosine_attr = "Column Direction Cosine: " + str(self.col_direction_cosine) + "\n"
            result += col_direction_cosine_attr

        if self.depth_direction_cosine is not None:
            depth_direction_cosine_attr = "Depth Direction Cosine: " + str(self.depth_direction_cosine) + "\n"
            result += depth_direction_cosine_attr

        if self.dicom_affine_transform is not None:
            dicom_affine_transform_attr = "DICOM affine transform: " + "\n" + str(self.dicom_affine_transform) + "\n"
            result += dicom_affine_transform_attr

        if self.nifti_affine_transform is not None:
            nifti_affine_transform_attr = "NIFTI affine transform: " + "\n" + str(self.nifti_affine_transform) + "\n"
            result += nifti_affine_transform_attr

        result += "---------------" + "\n"

        return result
