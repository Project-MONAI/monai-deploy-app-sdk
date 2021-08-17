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

from colorama import Fore

from monai.deploy.core.domain.datapath import DataPath
from monai.deploy.core.execution_context import BaseExecutionContext, ExecutionContext
from monai.deploy.core.models import ModelFactory
from monai.deploy.exceptions import IOMappingError

from .executor import Executor


class SingleProcessExecutor(Executor):
    """This class implements execution of a MONAI App
    in a single process in environment.
    """

    def run(self):
        """Run the app.

        This method retrieves the root nodes of the graph traveres through the
        graph in a depth first approach.
        Retrieves output from an upstream operator at a particular output port.
        Sets the right input to a downstrem operator at the right input port.
        Executes the operators.
        """
        app_context = self.app.context
        models = ModelFactory.create(app_context.model_path)

        exec_context = BaseExecutionContext(self.datastore, models)

        g = self.app.graph

        for op in g.gen_worklist():
            op_exec_context = ExecutionContext(exec_context, op)

            # Set source input if op is a root node
            is_root = g.is_root(op)
            if is_root:
                input_path = self.app.context.input_path
                op_exec_context.input.set(DataPath(input_path))

            # Set destination output if op is a leaf node
            is_leaf = g.is_leaf(op)
            if is_leaf:
                output_path = self.app.context.output_path
                op_exec_context.output.set(DataPath(output_path))

            # Execute pre_compute()
            print(Fore.BLUE + "Going to initiate execution of operator %s" % op.__class__.__name__ + Fore.RESET)
            op.pre_compute()

            # Execute compute()
            print(
                Fore.GREEN
                + "Executing operator %s " % op.__class__.__name__
                + Fore.YELLOW
                + "(Process ID %s)" % os.getpid()
                + Fore.RESET
            )
            op.compute(op_exec_context.input, op_exec_context.output, op_exec_context)

            # Execute post_compute()
            print(Fore.BLUE + "Done performing execution of operator %s\n" % op.__class__.__name__ + Fore.RESET)
            op.post_compute()

            # Set input to next operator
            next_ops = g.gen_next_operators(op)
            for next_op in next_ops:
                io_map = g.get_io_map(op, next_op)
                if not io_map:
                    import inspect

                    raise IOMappingError(
                        f"No IO mappings found for {op.name} -> {next_op.name} in "
                        f"{inspect.getabsfile(self.app.__class__)}"
                    )

                next_op_exec_context = ExecutionContext(exec_context, next_op)
                for (out_label, in_labels) in io_map.items():
                    output = op_exec_context.output.get(out_label)
                    for in_label in in_labels:
                        next_op_exec_context.input.set(output, in_label)
