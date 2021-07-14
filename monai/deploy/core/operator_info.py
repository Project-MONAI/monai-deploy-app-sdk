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


class OperatorInfo:
    """This is the Operator Info."""

    def __init__(self):
        # Initializing the attributes
        self.num_input_ports = 0
        self.num_output_ports = 0
        self.input_storage_type = []
        self.output_storage_type = []
        self.input_data_type = []
        self.output_data_type = []

    @property
    def num_input_ports(self):
        return self.__num_input_ports

    # Setter for num_input_ports
    @num_input_ports.setter
    def num_input_ports(self, val):
        self.__num_input_ports = val

    @property
    def num_output_ports(self):
        return self.__num_output_ports

    # Setter for num_output_ports
    @num_output_ports.setter
    def num_output_ports(self, val):
        self.__num_output_ports = val

    def set_input_storage_type(self, port_num, storage_type):
        self.input_storage_type[port_num] = storage_type

    def set_input_data_type(self, port_num, data_type):
        self.input_data_type[port_num] = data_type

    def set_output_storage_type(self, port_num, storage_type):
        self.output_storage_type[port_num] = storage_type

    def set_output_data_type(self, port_num, storage_type):
        self.output_data_type[port_num] = storage_type

    def get_input_storage_type(self, port_num):
        return self.input_storage_type[port_num]

    def get_output_storage_type(self, port_num):
        return self.output_storage_type[port_num]

    def get_input_data_type(self, port_num):
        return self.input_data_type[port_num]

    def get_output_data_type(self, port_num):
        return self.output_data_type[port_num]
