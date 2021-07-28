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

from abc import ABC, abstractmethod
from typing import Dict

from monai.deploy.core.datastores import MemoryDataStore
from monai.deploy.core.executors import SingleProcessExecutor
from monai.deploy.core.operator import Operator
from monai.deploy.exceptions import IOMappingError

from .graphs.nx_digraph import NetworkXGraph


class Application(ABC):
    """This is the base application class.

    All applications should be extended from this Class.
    The application class provides support for chaining up operators, as well
    as mechanism to execute the application.
    """

    def __init__(self, do_run: bool = True):
        """Constructor for the base class.

        It created an instance of an empty Directed Acyclic Graph to hold on to
        the operators.
        """
        super().__init__()
        self._graph = NetworkXGraph()

        if do_run:
            data_store = MemoryDataStore()
            executor = SingleProcessExecutor(self, data_store)
            executor.execute()

    @property
    def name(self):
        """Returns the name of this application."""
        return self.__class__.__name__

    @property
    def graph(self):
        """Gives access to the DAG.

        Returns:
            Instance of the DAG
        """
        return self._graph

    def add_operator(self, operator: Operator):
        """Adds an operator to the graph.

        Args:
            operator (Operator): An instance of the operator of type Operator.
        """
        # Ensure that the operator is valid
        operator.ensure_valid()

        self._graph.add_operator(operator)

    def add_flow(self, upstream_op: Operator, downstream_op: Operator, io_map: Dict[str, str] = None):
        """Adds a flow from upstream to downstream.

        An output port of the upstream operator is connected to one of the
        input ports of a downstream operators.

        Args:
            upstream_op (Operator): An instance of the upstream operator of type Operator.
            downstream_op (Operator): An instance of the downstream operator of type Operator.
            io_map (Dict[str, str]): A dictionary of mapping from the source operator's label to the destination
                                     operator's label.
        """

        # Ensure that the upstream and downstream operators are valid
        upstream_op.ensure_valid()
        downstream_op.ensure_valid()

        op_output_labels = upstream_op.get_operator_info().get_output_labels()
        op_input_labels = downstream_op.get_operator_info().get_input_labels()
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
        for k, v in io_map.items():
            if isinstance(v, str):
                io_map[k] = {v}

        # Verify that the upstream & downstream operator have the input and output ports specified by the io_map
        output_labels = list(io_map.keys())

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
                    io_map[next(iter(op_output_labels))] = io_map[output_label]
                    del io_map[output_label]
                    break
                raise IOMappingError(
                    f"The upstream operator({upstream_op.name}) has no output port with label '{output_label}'. "
                    f"It should be one of ({', '.join(op_output_labels)})."
                )

        output_labels = list(io_map.keys())  # re-evaluate output_labels
        for output_label in output_labels:
            input_labels = io_map[output_label]

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

        self._graph.add_flow(upstream_op, downstream_op, io_map)

    @abstractmethod
    def compose(self):
        """This is a method that needs to implemented by all subclasses.

        Derived appications will chain up the operators inside this compose
        method.

        """
        pass
