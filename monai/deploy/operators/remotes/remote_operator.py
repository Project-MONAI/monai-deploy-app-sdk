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

from abc import abstractmethod
from pathlib import Path

from monai.deploy.core import ExecutionContext, InputContext, Operator, OutputContext
from monai.deploy.core.domain import DataPath
from monai.deploy.core.io_type import IOType


class RemoteOperator(Operator):
    """An abstract class for remote operator."""

    remote_type: str = "unknown"

    @abstractmethod
    def setup(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        pass

    @abstractmethod
    def run(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        pass

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        # Set current output directory if not set
        if (
            op_output.get() is None
            and len(op_output) == 1
            and op_output.get_data_type() == DataPath
            and op_output.get_storage_type() == IOType.DISK
        ):
            op_output.set(DataPath(Path(".").absolute()))

        self.setup(op_input, op_output, context)
        self.run(op_input, op_output, context)
