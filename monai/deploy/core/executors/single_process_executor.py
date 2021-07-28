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

import os
from typing import TYPE_CHECKING, Optional

from colorama import Fore

if TYPE_CHECKING:
    from monai.deploy.core import Application

from monai.deploy.core.datastores import DataStore
from monai.deploy.core.execution_context import BaseExecutionContext, ExecutionContext
from monai.deploy.exceptions import IOMappingError

from .executor import Executor


class SingleProcessExecutor(Executor):
    """This class implements execution of a MONAI App
    in a single process in environment.
    """

    def __init__(self, app: Application, data_store: Optional[DataStore] = None):
        """Constructor for the class.

        The instance internally holds on to the data store.

        Args:
            app: An instance of the application that needs to be executed
        """
        super().__init__(app, data_store)

    def run(self):
        """Run the app.

        This method retrieves the root nodes of the graph traveres through the
        graph in a depth first approach.
        Retrieves output from an upstream operator at a particular output port.
        Sets the right input to a downstrem operator at the right input port.
        Executes the operators.
        """
        exec_context = BaseExecutionContext(self._data_store)

        g = self._app.graph
        for op in g.gen_worklist():
            op_exec_context = ExecutionContext(exec_context, op)

            print(Fore.BLUE + "Going to initiate execution of operator %s" % self.__class__.__name__)
            op.pre_compute()

            print(Fore.YELLOW + "Process ID %s" % os.getpid())
            print(Fore.GREEN + "Executing operator %s" % self.__class__.__name__)
            op.compute(op_exec_context.input, op_exec_context.output, op_exec_context)

            print(Fore.BLUE + "Done performing execution of operator %s" % self.__class__.__name__)
            op.post_compute()

            next_ops = g.gen_next_operators(op)
            for next_op in next_ops:
                io_map = g.get_io_map(op, next_op)
                if not io_map:
                    import inspect

                    raise IOMappingError(
                        f"No IO mappings found for {op.name} -> {next_op.name} in "
                        f"{inspect.getabsfile(self._app.__class__)}"
                    )

                next_op_exec_context = ExecutionContext(exec_context, next_op)
                for (out_label, in_labels) in io_map.items():
                    output = op_exec_context.output.get(out_label)
                    for in_label in in_labels:
                        next_op_exec_context.input.set(output, in_label)
