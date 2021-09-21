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

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Set, Type, Union

from monai.deploy.cli.main import LOG_CONFIG_FILENAME, parse_args, set_up_logging
from monai.deploy.core.graphs.factory import GraphFactory
from monai.deploy.core.models import ModelFactory
from monai.deploy.exceptions import IOMappingError
from monai.deploy.utils.importutil import get_class_file_path, get_docstring, is_subclass
from monai.deploy.utils.sizeutil import convert_bytes
from monai.deploy.utils.version import get_sdk_semver

from .app_context import AppContext
from .datastores import DatastoreFactory
from .env import BaseEnv
from .executors import ExecutorFactory
from .graphs.graph import Graph
from .operator import Operator
from .operator_info import IO
from .runtime_env import RuntimeEnv


class Application(ABC):
    """This is the base application class.

    All applications should be extended from this Class.
    The application class provides support for chaining up operators, as well
    as mechanism to execute the application.
    """

    # Application's name. <class name> if not specified.
    name: str = ""
    # Application's description. <class docstring> if not specified.
    description: str = ""
    # Application's version. <git version tag> or '0.0.0' if not specified.
    version: str = ""

    # Special attribute to identify the application.
    # Used by the CLI executing get_application() or is_subclass() from deploy.utils.importutil to
    # determine the application to run.
    # This is needed to identify Application class across different environments (e.g. by `runpy.run_path()`).
    _class_id: str = "monai.application"

    _env: Optional["ApplicationEnv"] = None

    def __init__(
        self,
        runtime_env: Optional[RuntimeEnv] = None,
        do_run: bool = False,
        path: Optional[Union[str, Path]] = None,
    ):
        """Constructor for the application class.

        It initializes the application's graph, the runtime environment and the application context.

        if `do_run` is True, it would accept user's arguments from the application's context and
        execute the application.

        Args:
            runtime_env (Optional[RuntimeEnv]): The runtime environment to use.
            do_run (bool): Whether to run the application.
            path (Optional[Union[str, Path]]): The path to the application (Python file path).
                This path is used for launching the application to get the package information from
                `monai.deploy.utils.importutil.get_application` method.
        """
        # Setup app description
        if not self.name:
            self.name = self.__class__.__name__
        if not self.description:
            self.description = get_docstring(self.__class__)
        if not self.version:
            try:
                from _version import get_versions

                self.version = get_versions()["version"]
            except ImportError:
                self.version = "0.0.0"

        # Set the application path
        if path:
            self.path = Path(path)
        else:
            self.path = get_class_file_path(self.__class__)

        # Set the runtime environment
        if str(self.path).startswith("<ipython-"):
            self.in_ipython = True
        else:
            self.in_ipython = False

        # Setup program arguments
        if path is not None or self.in_ipython:
            # If `path` is specified, it means that it is called by
            # monai.deploy.utils.importutil.get_application() to get the package info.
            # If `self.in_ipython` is True, it means that it is called by ipython environment.
            # In both cases, we should not parse the arguments from the command line.
            argv = [sys.executable, str(self.path)]  # use default parameters
        else:
            argv = sys.argv

        # Parse the command line arguments
        args = parse_args(argv, default_command="exec")

        context = AppContext(args.__dict__, runtime_env)

        self._context: AppContext = context

        self._graph: Graph = GraphFactory.create(context.graph)

        # Execute the builder to set up the application
        self._builder()

        # Compose operators
        self.compose()

        if do_run:
            self.run(log_level=args.log_level)

    @classmethod
    def __subclasshook__(cls, c: Type) -> bool:
        return is_subclass(c, cls._class_id)

    def _builder(self):
        """This method is called by the constructor of Application to set up the operator.

        This method returns `self` to allow for method chaining and new `_builder()` method is
        chained by decorators.

        Returns:
            An instance of Application.
        """
        return self

    @property
    def context(self) -> AppContext:
        """Returns the context of this application."""
        return self._context

    @property
    def graph(self) -> Graph:
        """Gives access to the DAG.

        Returns:
            Instance of the DAG
        """
        return self._graph

    @property
    def env(self):
        """Gives access to the environment.

        This sets a default value for the application's environment if not set.

        Returns:
            An instance of ApplicationEnv.
        """
        if self._env is None:
            self._env = ApplicationEnv()
        return self._env

    def add_operator(self, operator: Operator):
        """Adds an operator to the graph.

        Args:
            operator (Operator): An instance of the operator of type Operator.
        """
        # Ensure that the operator is valid
        operator.ensure_valid()

        self._graph.add_operator(operator)

    def add_flow(
        self, upstream_op: Operator, downstream_op: Operator, io_map: Optional[Dict[str, Union[str, Set[str]]]] = None
    ):
        """Adds a flow from upstream to downstream.

        An output port of the upstream operator is connected to one of the
        input ports of a downstream operators.

        Args:
            upstream_op (Operator): An instance of the upstream operator of type Operator.
            downstream_op (Operator): An instance of the downstream operator of type Operator.
            io_map (Optional[Dict[str, Union[str, Set[str]]]]): A dictionary of mapping from the source operator's label
                                                                 to the destination operator's label(s).
        """

        # Ensure that the upstream and downstream operators are valid
        upstream_op.ensure_valid()
        downstream_op.ensure_valid()

        op_output_labels = upstream_op.op_info.get_labels(IO.OUTPUT)
        op_input_labels = downstream_op.op_info.get_labels(IO.INPUT)
        if not io_map:
            if len(op_output_labels) > 1:
                raise IOMappingError(
                    f"The upstream operator has more than one output port "
                    f"({', '.join(op_output_labels)}) so mapping should be specified explicitly!"
                )
            if len(op_input_labels) > 1:
                raise IOMappingError(
                    f"The downstream operator has more than one output port ({', '.join(op_input_labels)}) "
                    f"so mapping should be specified explicitly!"
                )
            io_map = {"": {""}}

        # Convert io_map's values to the set of strings.
        io_maps: Dict[str, Set[str]] = io_map  # type: ignore
        for k, v in io_map.items():
            if isinstance(v, str):
                io_maps[k] = {v}

        # Verify that the upstream & downstream operator have the input and output ports specified by the io_map
        output_labels = list(io_maps.keys())

        if len(op_output_labels) == 1 and len(output_labels) != 1:
            raise IOMappingError(
                f"The upstream operator({upstream_op.name}) has only one port with label "
                f"'{next(iter(op_output_labels))}' but io_map specifies {len(output_labels)} "
                f"labels({', '.join(output_labels)}) to the upstream operator's output port"
            )

        for output_label in output_labels:
            if output_label not in op_output_labels:
                if len(op_output_labels) == 1 and len(output_labels) == 1 and output_label == "":
                    # Set the default output port label.
                    io_maps[next(iter(op_output_labels))] = io_maps[output_label]
                    del io_maps[output_label]
                    break
                raise IOMappingError(
                    f"The upstream operator({upstream_op.name}) has no output port with label '{output_label}'. "
                    f"It should be one of ({', '.join(op_output_labels)})."
                )

        output_labels = list(io_maps.keys())  # re-evaluate output_labels
        for output_label in output_labels:
            input_labels = io_maps[output_label]

            if len(op_input_labels) == 1 and len(input_labels) != 1:
                raise IOMappingError(
                    f"The downstream operator({downstream_op.name}) has only one port with label "
                    f"'{next(iter(op_input_labels))}' but io_map specifies {len(input_labels)} "
                    f"labels({', '.join(input_labels)}) to the downstream operator's input port"
                )

            for input_label in input_labels:
                if input_label not in op_input_labels:
                    if len(op_input_labels) == 1 and len(input_labels) == 1 and input_label == "":
                        # Set the default input port label.
                        input_labels.clear()
                        input_labels.add(next(iter(op_input_labels)))
                        break
                    raise IOMappingError(
                        f"The downstream operator({downstream_op.name}) has no input port with label '{input_label}'. "
                        f"It should be one of ({', '.join(op_input_labels)})."
                    )

        self._graph.add_flow(upstream_op, downstream_op, io_maps)

    def get_package_info(self, model_path: Union[str, Path] = "") -> Dict:
        """Returns the package information of this application.

        Args:
            model_path (Union[str, Path]): The path to the model directory.
        Returns:
            A dictionary containing the package information of this application.
        """
        app_path = self.path.name
        command = f"python3 -u /opt/monai/app/{app_path}"
        resource = self.context.resource

        # Get model name/path list
        # - If no model files are found at `model_path`, None will be returned by the ModelFactory.create method and
        #   the `model_list` will be an empty list.
        # - If the path represents a model repository (one or more model objects. Necessary condition is model_path is
        #   a folder), then `model_list` will abe a list of model objects (name and path).
        # - If only one model is found at model_path or model_path is a valid model file, `model_list` would be a
        #   single model object list.
        model_list = []
        if model_path:
            models = ModelFactory.create(model_path)
            if models:
                model_list = models.get_model_list()

        # Get pip requirement list
        spec_list = self.env.pip_packages
        for op in self.graph.get_operators():
            spec_list.extend(op.env.pip_packages)
        spec_set = set()
        pip_requirement_list = []
        for p in spec_list:
            spec = p.strip().lower()
            if spec not in spec_set:
                pip_requirement_list.append(spec)
                spec_set.add(spec)

        return {
            "app-name": self.name,
            "app-version": self.version,
            "sdk-version": get_sdk_semver(),
            "command": command,
            "resource": {
                "cpu": resource.cpu,
                "gpu": resource.gpu,
                "memory": convert_bytes(resource.memory),
            },
            "models": model_list,
            "pip-packages": pip_requirement_list,
        }

    def run(
        self,
        log_level: Optional[str] = None,
        input: Optional[str] = None,
        output: Optional[str] = None,
        model: Optional[str] = None,
        workdir: Optional[str] = None,
        datastore: Optional[str] = None,
        executor: Optional[str] = None,
    ) -> None:
        """Runs the application.

        This method accepts `log_level` to set the log level of the application.

        Other arguments are used to specify the `input`, `output`, `model`, `workdir`, `datastore`, and `executor`.
        (Cannot set `graph` because it is set and used by the constructor.)

        If those arguments are not specified, values in the application context will be used.

        This method is useful when you want to interactively run the application inside IPython (Jupyter Notebook).

        For example, you can run the following code in a notebook:

        >>> from pathlib import Path
        >>> import monai.deploy.core as md
        >>> from monai.deploy.core import (
        >>>     Application,
        >>>     DataPath,
        >>>     ExecutionContext,
        >>>     InputContext,
        >>>     IOType,
        >>>     Operator,
        >>>     OutputContext,
        >>> )
        >>>
        >>> @md.input("path", DataPath, IOType.DISK)
        >>> @md.output("path", DataPath, IOType.DISK)
        >>> class FirstOperator(Operator):
        >>>     def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        >>>         print(f"First Operator. input:{op_input.get().path}, model:{context.models.get().path}")
        >>>         output_path = Path("output_first.txt")
        >>>         output_path.write_text("first output\\n")
        >>>         output.set(DataPath(output_path))
        >>>
        >>> @md.input("path", DataPath, IOType.DISK)
        >>> @md.output("path", DataPath, IOType.DISK)
        >>> class SecondOperator(Operator):
        >>>     def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        >>>         print(f"First Operator. output:{op_output.get().path}, model:{context.models.get().path}")
        >>>         # The leaf operators can only read output DataPath and should not set output DataPath.
        >>>         output_path = op_output.get().path / "output_second.txt"
        >>>         output_path.write_text("second output\\n")
        >>>
        >>> class App(Application):
        >>>     def compose(self):
        >>>         first_op = FirstOperator()
        >>>         second_op = SecondOperator()
        >>>
        >>>         self.add_flow(first_op, second_op)
        >>>
        >>> if __name__ == "__main__":
        >>>     App(do_run=True)

        >>> app = App()
        >>> app.run(input="inp", output="out", model="model.pt")

        >>> !ls out

        Args:
            log_level (Optional[str]): A log level.
            input (Optional[str]): An input data path.
            output (Optional[str]): An output data path.
            model (Optional[str]): A model path.
            workdir (Optional[str]): A working directory path.
            datastore (Optional[str]): A datastore path.
            executor (Optional[str]): An executor name.
        """
        # Set arguments
        args = {}
        if input is not None:
            args["input"] = input
        if output is not None:
            args["output"] = output
        if model is not None:
            args["model"] = model
        if workdir is not None:
            args["workdir"] = workdir
        if datastore is not None:
            args["datastore"] = datastore
        if executor is not None:
            args["executor"] = executor

        # If no arguments are specified and if runtime is in IPython, do not run the application.
        if len(args) == 0 and self.in_ipython:
            return

        # Update app context
        app_context = self.context
        app_context.update(args)

        # Set up logging (try to load `LOG_CONFIG_FILENAME` in the application folder)
        # and run the application
        app_log_config_path = self.path.parent / LOG_CONFIG_FILENAME
        set_up_logging(log_level, config_path=app_log_config_path)

        datastore_obj = DatastoreFactory.create(app_context.datastore)
        executor_obj = ExecutorFactory.create(app_context.executor, {"app": self, "datastore": datastore_obj})
        executor_obj.run()

    @abstractmethod
    def compose(self):
        """This is a method that needs to implemented by all subclasses.

        Derived appications will chain up the operators inside this compose
        method.

        """
        pass


class ApplicationEnv(BaseEnv):
    """Settings for the application environment.

    This class is used to specify the environment settings for the application.
    """

    pass
