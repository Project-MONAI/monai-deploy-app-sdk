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
    from monai.deploy.core.operator import Operator

from pathlib import Path
from typing import Any, Optional

from monai.deploy.core.datastore import DataStore
from monai.deploy.core.datastores.memory_store import MemoryDataStore
from monai.deploy.exceptions import IOMappingError, ItemAlreadyExistsError, ItemNotExistsError


class BaseContext:
    """An execution context for the application."""

    def __init__(self, data_store: Optional[DataStore] = None):
        if data_store is None:
            self._storage = MemoryDataStore()
        else:
            self._storage = data_store

    @property
    def storage(self):
        return self._storage


class ExecutionContext(BaseContext):
    """An execution context for the operator."""

    def __init__(self, execution_context: BaseContext, op: Operator):
        super().__init__(execution_context.storage)
        self._context = execution_context
        self._op = op

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
        """Increase the execution index for the operator.

        This index number would be increased once for each call to the operator
        so that the operator can be executed multiple times.
        """
        storage = self._context.storage
        parent_node = f"/operators/{self._op.get_uid()}"
        key = f"{parent_node}/execution_index"
        new_execution_index = self.get_execution_index() + 1
        storage.put(key, new_execution_index)
        return new_execution_index

    def get_input(self, group: str = ""):
        """Returns the input data for the operator."""
        op_info = self._op.get_operator_info()
        input_labels = op_info.get_input_labels()
        if group not in input_labels:
            if group == "" and len(input_labels) == 1:
                group = next(iter(input_labels))
            else:
                raise IOMappingError(
                    f"'{group}' is not a valid input label of the operator({self._op.name}). "
                    f"It should be one of ({', '.join(input_labels)})."
                )

        storage = self._context.storage
        execution_index = self.get_execution_index()
        parent_node = f"/operators/{self._op.get_uid()}/{execution_index}/input"
        key = f"{parent_node}/{group}"
        if not storage.exists(key):
            raise ItemNotExistsError(f"'{key}' does not exist.")
        return storage.get(key)

    def set_input(self, value: Any, group: str = ""):
        """Sets the input data for the operator."""
        op_info = self._op.get_operator_info()
        input_labels = op_info.get_input_labels()
        if group not in input_labels:
            if group == "" and len(input_labels) == 1:
                group = next(iter(input_labels))
            else:
                raise IOMappingError(
                    f"'{group}' is not a valid input label of the operator({self._op.name}). "
                    f"It should be one of ({', '.join(input_labels)})."
                )

        storage = self._context.storage
        execution_index = self.get_execution_index()
        parent_node = f"/operators/{self._op.get_uid()}/{execution_index}/input"
        key = f"{parent_node}/{group}"
        if storage.exists(key):
            raise ItemAlreadyExistsError(f"{key} already exists.")
        else:
            storage.put(key, value)

    def get_output(self, group: str = ""):
        """Returns the output data for the operator."""
        op_info = self._op.get_operator_info()
        output_labels = op_info.get_output_labels()
        if group not in output_labels:
            if group == "" and len(output_labels) == 1:
                group = next(iter(output_labels))
            else:
                raise IOMappingError(
                    f"'{group}' is not a valid output label of the operator({self._op.name}). "
                    f"It should be one of ({', '.join(output_labels)})."
                )

        storage = self._context.storage
        execution_index = self.get_execution_index()
        parent_node = f"/operators/{self._op.get_uid()}/{execution_index}/output"
        key = f"{parent_node}/{group}"
        if storage.exists(key):
            return storage.get(key)
        else:
            item = {}
            storage.put(key, item)
            return item

    def get_output_location(self, group: str = ""):
        """Returns the output location for the operator."""

        # TODO: Implement this method
        return Path("")

    def set_output(self, value: Any, group: str = ""):
        """Sets the output data for the operator."""
        op_info = self._op.get_operator_info()
        output_labels = op_info.get_output_labels()
        if group not in output_labels:
            if group == "" and len(output_labels) == 1:
                group = next(iter(output_labels))
            else:
                raise IOMappingError(
                    f"'{group}' is not a valid output label of the operator({self._op.name}). "
                    f"It should be one of ({', '.join(output_labels)})."
                )

        storage = self._context.storage
        execution_index = self.get_execution_index()
        parent_node = f"/operators/{self._op.get_uid()}/{execution_index}/output"
        key = f"{parent_node}/{group}"
        if storage.exists(key):
            raise ItemAlreadyExistsError(f"{key} already exists.")
        else:
            storage.put(key, value)
