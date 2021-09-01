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

from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional, Set

from monai.deploy.core.operator import Operator


class Graph(ABC):
    """Abstract class for graph."""

    @abstractmethod
    def add_operator(self, op: Operator):
        """Add a node to the graph."""
        pass

    @abstractmethod
    def add_flow(self, op_u: Operator, op_v: Operator, io_map: Dict[str, Set[str]]):
        """Add an edge to the graph.

        Args:
            op_u (Operator): A source operator.
            op_v (Operator): A destination operator.
            io_map (Dict[str, Set[str]]): A dictionary of mapping from the source operator's label to the destination
                                          operator's label(s).
        """
        pass

    @abstractmethod
    def get_io_map(self, op_u: Operator, op_v) -> Dict[str, Set[str]]:
        """Get a mapping from the source operator's output label to the destination operator's input label.
        Args:
            op_u (Operator): A source operator.
            op_v (Operator): A destination operator.
        Returns:
            A dictionary of mapping from the source operator's output label to the destination operator's
            input label(s).
        """
        pass

    @abstractmethod
    def is_root(self, op: Operator) -> bool:
        """Check if the operator is a root operator.

        Args:
            op (Operator): A node in the graph.
        Returns:
            True if the operator is a root operator.
        """
        pass

    @abstractmethod
    def is_leaf(self, op: Operator) -> bool:
        """Check if the operator is a leaf operator.

        Args:
            op (Operator): A node in the graph.
        Returns:
            True if the operator is a leaf operator.
        """
        pass

    @abstractmethod
    def get_root_operators(self) -> Generator[Operator, None, None]:
        """Get all root operators.

        Returns:
            A generator of root operators.
        """
        pass

    @abstractmethod
    def get_operators(self) -> Generator[Operator, None, None]:
        """Get all operators.

        Returns:
            A generator of operators.
        """
        pass

    @abstractmethod
    def gen_worklist(self) -> Generator[Optional[Operator], None, None]:
        """Get worklist."""
        pass

    @abstractmethod
    def gen_next_operators(self, op: Operator) -> Generator[Optional[Operator], None, None]:
        """Get next operators."""
        pass
