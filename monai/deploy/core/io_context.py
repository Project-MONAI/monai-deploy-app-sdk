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

from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .execution_context import ExecutionContext

from monai.deploy.exceptions import IOMappingError, ItemAlreadyExistsError, ItemNotExistsError


class IOContext(ABC):
    """Base class for IO context."""

    _io_kind = "undefined"

    def __init__(self, execution_context: ExecutionContext):
        """Constructor for IOContext."""
        self._execution_context = execution_context
        self._op = execution_context.op
        self._op_info = self._op.get_operator_info()
        self._labels = self._op_info.get_labels(self._io_kind)
        self._storage = execution_context._storage

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

        Args:
            postfix: The postfix for the path.

        Returns:
            The path for the group.
        """
        execution_index = self._execution_context.get_execution_index()
        path = f"/operators/{self._op.get_uid()}/{execution_index}/{postfix}"
        return path


class InputContext(IOContext):
    """An input context for an operator."""

    _io_kind = "input"

    def get(self, label: str = "") -> Any:
        """Returns the input data for the operator."""
        label = self.get_default_label(label)
        key = self.get_group_path(f"input/{label}")
        storage = self._storage
        if not storage.exists(key):
            raise ItemNotExistsError(f"'{key}' does not exist.")
        return storage.get(key)

    def set(self, value: Any, label: str = ""):
        """Sets the input data for the operator."""
        label = self.get_default_label(label)
        key = self.get_group_path(f"input/{label}")
        storage = self._storage
        if storage.exists(key):
            raise ItemAlreadyExistsError(f"{key} already exists.")
        else:
            storage.put(key, value)


class OutputContext(IOContext):
    """An output context for an operator."""

    _io_kind = "output"

    def get_location(self, group: str = ""):
        """Returns the output location for the operator."""

        # TODO: Implement this method
        return Path("")

    def get(self, label: str = "") -> Any:
        """Returns the output data for the operator."""
        label = self.get_default_label(label)
        key = self.get_group_path(f"output/{label}")
        storage = self._storage
        if not storage.exists(key):
            raise ItemNotExistsError(f"'{key}' does not exist.")
        return storage.get(key)

    def set(self, value: Any, label: str = ""):
        """Sets the output data for the operator."""
        label = self.get_default_label(label)
        key = self.get_group_path(f"output/{label}")
        storage = self._storage
        if storage.exists(key):
            raise ItemAlreadyExistsError(f"{key} already exists.")
        else:
            storage.put(key, value)
