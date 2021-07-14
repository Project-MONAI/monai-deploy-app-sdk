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

class ExecutionContext:
    def __init__(self):
        super().__init__()
        self._storage = {}

    def set_operator_input(self, op_uid, input_port_num, val):
        key = (op_uid, "input", input_port_num)
        self._storage[key] = val

    def set_operator_output(self, op_uid, output_port_num, val):
        key = (op_uid, "output", output_port_num)
        self._storage[key] = val

    def get_operator_input(self, op_uid, input_port_num):
        key = (op_uid, "input", input_port_num)
        return self._storage[key]

    def get_operator_output(self, op_uid, output_port_num):
        key = (op_uid, "output", output_port_num)
        return self._storage[key]
