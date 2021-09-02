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

from enum import Enum
from typing import Dict, Set, Type, Union

from .domain.datapath import DataPath
from .io_type import IOType


class IO(Enum):
    UNDEFINED = "undefined"
    INPUT = "input"
    OUTPUT = "output"

    def __str__(self):
        return self.value


class OperatorInfo:
    """A class to store information about operator's input and output data types and storage types."""

    def __init__(self):
        # Initializing the attributes
        self.labels: Dict[IO, Set[str]] = {IO.INPUT: set(), IO.OUTPUT: set()}
        self.data_type: Dict[IO, Dict[str, Type]] = {IO.INPUT: {}, IO.OUTPUT: {}}
        self.storage_type: Dict[IO, Dict[str, IOType]] = {IO.INPUT: {}, IO.OUTPUT: {}}

    def ensure_valid(self):
        """Ensure that the operator info is valid.

        This sets default values for OperatorInfo.
        """
        for kind in [IO.INPUT, IO.OUTPUT]:
            if len(self.labels[kind]) == 0:
                self.labels[kind].add("")
                self.data_type[kind][""] = DataPath
                self.storage_type[kind][""] = IOType.DISK

    def add_label(self, io_kind: Union[IO, str], label: str):
        io_kind = IO(io_kind)
        self.labels[io_kind].add(label)

    def get_labels(self, io_kind: Union[IO, str]) -> Set[str]:
        io_kind = IO(io_kind)
        return self.labels[io_kind]

    def set_data_type(self, io_kind: Union[IO, str], label: str, data_type: Type):
        io_kind = IO(io_kind)
        self.data_type[io_kind][label] = data_type

    def get_data_type(self, io_kind: Union[IO, str], label: str) -> Type:
        io_kind = IO(io_kind)
        return self.data_type[io_kind][label]

    def set_storage_type(self, io_kind: Union[IO, str], label: str, storage_type: Union[int, IOType]):
        io_kind = IO(io_kind)
        self.storage_type[io_kind][label] = IOType(storage_type)

    def get_storage_type(self, io_kind: Union[IO, str], label: str) -> IOType:
        io_kind = IO(io_kind)
        return self.storage_type[io_kind][label]
