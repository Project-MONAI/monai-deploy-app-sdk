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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .operator import Operator

from typing import Optional

from .datastores import Datastore, MemoryDatastore
from .io_context import InputContext, OutputContext


class BaseExecutionContext:
    """A base execution context for the application."""

    def __init__(self, datastore: Optional[Datastore] = None):
        if datastore is None:
            self._storage = MemoryDatastore()
        else:
            self._storage = datastore

    @property
    def storage(self):
        return self._storage


class ExecutionContext(BaseExecutionContext):
    """An execution context for the operator."""

    def __init__(self, context: BaseExecutionContext, op: Operator):
        super().__init__(context.storage)
        self._context = context
        self._op = op
        self._input_context = InputContext(self)
        self._output_context = OutputContext(self)

    @property
    def op(self):
        return self._op

    def get_execution_index(self):
        """Returns the execution index for the operator.

        The execution index is incremented every time before the operator is executed.
        For the first time, the execution index is set to 0.

        Returns:
            The execution index(int) for the operator.
        """
        storage = self._context.storage
        parent_node = f"/operators/{self._op.get_uid()}"
        key = f"{parent_node}/execution_index"
        if storage.exists(key):
            return storage.get(key)
        else:
            storage.put(key, 0)
            return 0

    def increase_execution_index(self):
        """Increases the execution index for the operator.

        This index number would be increased once for each call to the operator
        so that the operator can be executed multiple times.
        """
        storage = self._context.storage
        parent_node = f"/operators/{self._op.get_uid()}"
        key = f"{parent_node}/execution_index"
        new_execution_index = self.get_execution_index() + 1
        storage.put(key, new_execution_index)
        return new_execution_index

    @property
    def input(self):
        """Returns the input context for the operator."""
        return self._input_context

    @property
    def output(self):
        """Returns the output context for the operator."""
        return self._output_context
