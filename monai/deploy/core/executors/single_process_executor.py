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


import os
import shutil
from pathlib import Path

from colorama import Fore

from monai.deploy.core.domain.datapath import DataPath, NamedDataPath
from monai.deploy.core.execution_context import BaseExecutionContext, ExecutionContext
from monai.deploy.core.io_type import IOType
from monai.deploy.core.models import ModelFactory
from monai.deploy.core.operator_info import IO
from monai.deploy.exceptions import IOMappingError

from .executor import Executor

TEMP_WORKDIR = ".monai_workdir"  # working directory name used when no `workdir` is specified


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
        # Take paths as absolute paths
        models = ModelFactory.create(os.path.abspath(app_context.model_path))
        input_path = os.path.abspath(self.app.context.input_path)
        output_path = os.path.abspath(self.app.context.output_path)

        # Create the output directory if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        # Store old pwd
        old_pwd = os.getcwd()

        # If workdir is not specified, create a temporary path (.monai_workdir)
        if not self.app.context.workdir:
            if os.path.exists(TEMP_WORKDIR):
                shutil.rmtree(TEMP_WORKDIR)
            os.mkdir(TEMP_WORKDIR)
            workdir = os.path.abspath(TEMP_WORKDIR)
        else:
            # Absolute path of the working directory
            workdir = os.path.abspath(self.app.context.workdir)

        # Create execution context
        # Currently, we only allow a single input/output path (with empty label: "").
        # TODO(gigony): Supports multiple inputs/outputs (#87)
        exec_context = BaseExecutionContext(
            self.datastore,
            input=NamedDataPath({"": DataPath(input_path, read_only=True)}),
            output=NamedDataPath({"": DataPath(output_path, read_only=True)}),
            models=models,
        )

        g = self.app.graph

        try:
            for op in g.gen_worklist():
                op_exec_context = ExecutionContext(exec_context, op)

                # Set source input for a label if op is a root node and (<data type>,<storage type>) == (DataPath,IOType.DISK)
                is_root = g.is_root(op)
                if is_root:
                    input_op_info = op.op_info
                    input_labels = input_op_info.get_labels(IO.INPUT)
                    for input_label in input_labels:
                        input_data_type = input_op_info.get_data_type(IO.INPUT, input_label)
                        input_storage_type = input_op_info.get_storage_type(IO.INPUT, input_label)
                        if issubclass(input_data_type, DataPath) and input_storage_type == IOType.DISK:
                            op_exec_context.input_context.set(DataPath(input_path, read_only=True), input_label)

                # Set destination output for a label if op is a leaf node and (<data type>,<storage type>) == (DataPath,IOType.DISK)
                is_leaf = g.is_leaf(op)
                if is_leaf:
                    output_op_info = op.op_info
                    output_labels = output_op_info.get_labels(IO.OUTPUT)
                    for output_label in output_labels:
                        output_data_type = output_op_info.get_data_type(IO.OUTPUT, output_label)
                        output_storage_type = output_op_info.get_storage_type(IO.OUTPUT, output_label)
                        if issubclass(output_data_type, DataPath) and output_storage_type == IOType.DISK:
                            op_exec_context.output_context.set(DataPath(output_path, read_only=True), output_label)

                # Change the current working directory to the working directory of the operator
                #   op_output_folder == f"{workdir}/operators/{op.uid}/{op_exec_context.get_execution_index()}/{IO.OUTPUT}"
                relative_output_path = Path(op_exec_context.output_context.get_group_path(IO.OUTPUT)).relative_to("/")
                op_output_folder = str(Path(workdir, relative_output_path))
                os.makedirs(op_output_folder, exist_ok=True)
                os.chdir(op_output_folder)

                # Execute pre_compute()
                print(Fore.BLUE + "Going to initiate execution of operator %s" % op.__class__.__name__ + Fore.RESET)
                op.pre_compute()

                # Execute compute()
                print(
                    Fore.GREEN
                    + "Executing operator %s " % op.__class__.__name__
                    + Fore.YELLOW
                    + "(Process ID: %s, Operator ID: %s)" % (os.getpid(), op.uid)
                    + Fore.RESET
                )
                op.compute(op_exec_context.input_context, op_exec_context.output_context, op_exec_context)

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
                        output = op_exec_context.output_context.get(out_label)
                        for in_label in in_labels:
                            next_op_exec_context.input_context.set(output, in_label)
        finally:
            # Always restore pwd even if an exception is raised (This logic can be run in an IPython environment)
            os.chdir(old_pwd)

        # Remove a temporary workdir
        old_pwd = os.getcwd()
        if os.path.exists(TEMP_WORKDIR):
            shutil.rmtree(TEMP_WORKDIR)
