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

from abc import ABC
from typing import TYPE_CHECKING, Any, Set

# To avoid "Cannot resolve forward reference" error
# : https://github.com/agronholm/sphinx-autodoc-typehints#dealing-with-circular-imports
from . import execution_context

if TYPE_CHECKING:
    from .execution_context import ExecutionContext
    from .operator import Operator
    from .datastores.datastore import Datastore
    from .operator_info import OperatorInfo

from monai.deploy.exceptions import IOMappingError, ItemAlreadyExistsError, ItemNotExistsError

from .domain.datapath import DataPath


class IOContext(ABC):
    """Base class for IO context."""

    _io_kind = "undefined"

    def __init__(self, execution_context: "execution_context.ExecutionContext"):
        """Constructor for IOContext."""
        self._execution_context: "ExecutionContext" = execution_context
        self._op: Operator = execution_context.op
        self._op_info: OperatorInfo = self._op.op_info
        self._labels: Set[str] = self._op_info.get_labels(self._io_kind)
        self._storage: Datastore = execution_context.storage

    def get_default_label(self, label: str = "") -> str:
        """Get a default label for IO context."""
        if label not in self._labels:
            if label == "" and len(self._labels) == 1:
                label = next(iter(self._labels))
            else:
                raise IOMappingError(
                    f"'{label}' is not a valid {self._io_kind} of the operator({self._op.name}). "
                    f"It should be one of ({', '.join(self._labels)})."
                )
        return label

    def get_group_path(self, postfix: str = "") -> str:
        """Returns the path for the group.

        The group path returned would be:
            "/operators/{self._op.uid}/{execution_index}/{postfix}"

        Args:
            postfix: The postfix for the path.

        Returns:
            The path for the group.
        """
        execution_index = self._execution_context.get_execution_index()
        path = f"/operators/{self._op.uid}/{execution_index}/{postfix}"
        return path

    def get(self, label: str = "") -> Any:
        """Returns the data for the operator.

        It uses a sub path ({self._io_kind}/{label}) to get the data.
        The final group path (key) would be:

            "/operators/{self._op.uid}/{execution_index}/{self._io_kind}/{label}"
        """
        label = self.get_default_label(label)
        key = self.get_group_path(f"{self._io_kind}/{label}")
        storage = self._storage
        if not storage.exists(key):
            raise ItemNotExistsError(f"'{key}' does not exist.")
        return storage.get(key)

    def set(self, value: Any, label: str = ""):
        """Sets the data for the operator.

        It uses a sub path ({self._io_kind}/{label}) to set the data.
        The final group path (key) would be:

            "/operators/{self._op.uid}/{execution_index}/{self._io_kind}/{label}"
        """
        label = self.get_default_label(label)
        key = self.get_group_path(f"{self._io_kind}/{label}")
        storage = self._storage
        if storage.exists(key):
            raise ItemAlreadyExistsError(f"{key} already exists.")
        else:
            # Convert to the absolute path if 'value' is an instance of DataPath and it is a relative path.
            # This is to keep the actual path of the data in the storage across different Operator execution contexts.
            if isinstance(value, DataPath):
                value.to_absolute()
            storage.put(key, value)


class InputContext(IOContext):
    """An input context for an operator."""

    _io_kind = "input"


class OutputContext(IOContext):
    """An output context for an operator."""

    _io_kind = "output"
