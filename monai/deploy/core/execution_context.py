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

from typing import Optional

from monai.deploy.core.domain.datapath import NamedDataPath

# To avoid "Cannot resolve forward reference" error
# : https://github.com/agronholm/sphinx-autodoc-typehints#dealing-with-circular-imports
from . import operator
from .datastores import Datastore, MemoryDatastore
from .io_context import InputContext, OutputContext
from .models import Model


class BaseExecutionContext:
    """A base execution context for the application.

    BaseExecutionContext is responsible for storing the input and output data paths,
    and the models.

    Those pieces of information are used by the Operator (in `compute()` method) to perform the execution.

    The input and output data paths from the application's context are available through
    `context.input.get()` and `context.output.get()`.
    """

    def __init__(
        self,
        datastore: Optional[Datastore],
        input: NamedDataPath,
        output: NamedDataPath,
        models: Optional[Model] = None,
    ):
        if datastore is None:
            self._storage: Datastore = MemoryDatastore()
        else:
            self._storage = datastore

        self._input = input
        self._output = output

        if models is None:
            self._models = Model("")  # set a null model
        else:
            self._models = models

    @property
    def storage(self) -> Datastore:
        return self._storage

    @property
    def input(self) -> NamedDataPath:
        return self._input

    @property
    def output(self) -> NamedDataPath:
        return self._output

    @property
    def models(self) -> Model:
        return self._models


class ExecutionContext(BaseExecutionContext):
    """An execution context for the operator."""

    def __init__(self, context: BaseExecutionContext, op: "operator.Operator"):
        super().__init__(context.storage, context.input, context.output, context.models)
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
        parent_node = f"/operators/{self.op.uid}"
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
        parent_node = f"/operators/{self.op.uid}"
        key = f"{parent_node}/execution_index"
        new_execution_index = self.get_execution_index() + 1
        storage.put(key, new_execution_index)
        return new_execution_index

    @property
    def input_context(self):
        """Returns the input context for the operator."""
        return self._input_context

    @property
    def output_context(self):
        """Returns the output context for the operator."""
        return self._output_context
