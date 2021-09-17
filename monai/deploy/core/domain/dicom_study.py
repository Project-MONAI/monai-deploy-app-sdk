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


from .domain import Domain


class DICOMStudy(Domain):
    """This class represents a DICOM Study.

    It contains a collection of DICOM Studies.
    """

    def __init__(self, study_instance_uid):
        super().__init__(None)
        self._study_instance_uid = study_instance_uid
        self._series_dict = {}

        # Do not set attributes in advance to save memory

        # self._study_id: Any = None
        # self._study_date: Any = None
        # self._study_time: Any = None
        # self._study_description: Any = None
        # self._accession_number: Any = None

    def get_study_instance_uid(self):
        return self._study_instance_uid

    def add_series(self, series):
        self._series_dict[series.get_series_instance_uid()] = series

    def get_all_series(self):
        return list(self._series_dict.values())

    @property
    def study_id(self):
        return getattr(self, "_study_id", None)

    @study_id.setter
    def study_id(self, val):
        self._study_id = val

    @property
    def study_date(self):
        return getattr(self, "_study_date", None)

    @study_date.setter
    def study_date(self, val):
        self._study_date = val

    @property
    def study_time(self):
        return getattr(self, "_study_time", None)

    @study_time.setter
    def study_time(self, val):
        self._study_time = val

    @property
    def study_description(self):
        return getattr(self, "_study_description", None)

    @study_description.setter
    def study_description(self, val):
        self._study_description = val

    @property
    def accession_number(self):
        return getattr(self, "_accession_number", None)

    @accession_number.setter
    def accession_number(self, val):
        self._accession_number = val

    def __str__(self):
        result = "---------------" + "\n"

        if self._study_instance_uid is not None:
            study_instance_uid_attr = "Study Instance UID: " + self._study_instance_uid + "\n"
            result += study_instance_uid_attr
        if self.study_id is not None:
            study_id_attr = "Study ID: " + self.study_id + "\n"
            result += study_id_attr
        if self.study_date is not None:
            study_date_attr = "Study Date: " + self.study_date + "\n"
            result += study_date_attr
        if self.study_time is not None:
            study_time_attr = "Study Time: " + self.study_time + "\n"
            result += study_time_attr
        if self.study_description is not None:
            study_desc_attr = "Study Description: " + self.study_description + "\n"
            result += study_desc_attr
        if self.accession_number is not None:
            accession_num_attr = "Accession Number: " + self.accession_number + "\n"
            result += accession_num_attr

        result += "---------------" + "\n"

        return result
