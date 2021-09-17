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

from typing import Dict, Optional

from monai.deploy.exceptions import UnknownTypeError

from .graph import Graph
from .nx_digraph import NetworkXGraph


class GraphFactory:
    """GraphFactory is an abstract class that provides a way to create a graph object."""

    NAMES = ["nx_digraph"]
    DEFAULT = "nx_digraph"

    @staticmethod
    def create(graph_type: str, graph_params: Optional[Dict] = None) -> Graph:
        """Creates a graph object.

        Args:
            graph_type (str): A type of the graph.
            graph_params (Dict): A dictionary of parameters of the graph.

        Returns:
            Graph: A graph object.
        """

        graph_params = graph_params or {}

        if graph_type == "nx_digraph":
            return NetworkXGraph(**graph_params)
        # elif graph_type == 'py':
        #     return PyGraph(graph_params)
        else:
            raise UnknownTypeError(f"Unknown graph type: {graph_type}")
