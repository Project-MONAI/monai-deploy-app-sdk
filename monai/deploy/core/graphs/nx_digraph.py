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

from typing import Dict, Generator, Optional

import networkx as nx
from networkx.algorithms.dag import topological_sort

from monai.deploy.core import Operator
from monai.deploy.core.graph import Graph


class NetworkXGraph(Graph):
    """NetworkX graph implementation."""

    def __init__(self):
        self._graph = nx.DiGraph()

    def add_operator(self, op: Operator):
        self._graph.add_node(op)

    def add_flow(self, op_u: Operator, op_v: Operator, io_map: Dict[str, str]):
        self._graph.add_edge(op_u, op_v, io_map=io_map)

    def get_io_map(self, op_u: Operator, op_v) -> Dict[str, str]:
        io_map = self._graph.get_edge_data(op_u, op_v).get("io_map")
        return io_map

    def gen_worklist(self) -> Generator[Optional[Operator], None, None]:
        return topological_sort(self._graph)

    def gen_next_operators(self, op: Operator) -> Generator[Optional[Operator], None, None]:
        for (_, v) in self._graph.out_edges(op):
            yield v
