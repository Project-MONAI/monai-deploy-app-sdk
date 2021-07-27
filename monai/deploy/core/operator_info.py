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

from typing import Type, Union

from monai.deploy.core.domain import Domain
from monai.deploy.core.domain.blob import Blob
from monai.deploy.core.io_type import IOType


class OperatorInfo:
    """A class to store information about operator's input and output data types and storage types."""

    def __init__(self):
        # Initializing the attributes
        self.input_labels = set()
        self.output_labels = set()
        self.input_data_type = {}
        self.output_data_type = {}
        self.input_storage_type = {}
        self.output_storage_type = {}

    def ensure_valid(self):
        """Ensure that the operator info is valid.

        This sets default values for OperatorInfo.
        """
        if len(self.input_labels) == 0:
            self.input_labels.add("")
            self.input_data_type[""] = Blob
            self.input_storage_type[""] = IOType.DISK

        if len(self.output_labels) == 0:
            self.output_labels.add("")
            self.output_data_type[""] = Blob
            self.output_storage_type[""] = IOType.DISK

    def add_input_label(self, label: str):
        self.input_labels.add(label)

    def get_input_labels(self) -> str:
        return self.input_labels

    def add_output_label(self, label: str):
        self.output_labels.add(label)

    def get_output_labels(self) -> str:
        return self.output_labels

    def set_input_data_type(self, label: str, data_type: Type[Domain] = None):
        self.input_data_type[label] = data_type

    def get_input_data_type(self, label: str) -> Type[Domain]:
        return self.input_data_type[label]

    def set_output_data_type(self, label: str, data_type: Type[Domain] = None):
        self.output_data_type[label] = data_type

    def get_output_data_type(self, label: str) -> Type[Domain]:
        return self.output_data_type[label]

    def set_input_storage_type(self, label: str, storage_type: Union[int, IOType] = None):
        self.input_storage_type[label] = storage_type

    def get_input_storage_type(self, label: str) -> Union[int, IOType]:
        return self.input_storage_type[label]

    def set_output_storage_type(self, label: str, storage_type: Union[int, IOType] = None):
        self.output_storage_type[label] = storage_type

    def get_output_storage_type(self, label: str) -> Union[int, IOType]:
        return self.output_storage_type[label]
