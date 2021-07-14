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


import networkx as nx

from monai.deploy.executors.executor import Executor
from monai.deploy.core.application import Application
from monai.deploy.core.execution_context import ExecutionContext


class SingleProcessExecutor(Executor):

    """This class implements execution of a MONAI App
    in a single process in environment.
    """

    def __init__(self, app: Application):
        """Constructor for the class.

        The instance internally holds on to the data store.

        Args:
            app: An instance of the application that needs to be executed
        """
        super().__init__(app)

    def execute(self):
        """Executes the app.

        This method retrieves the root nodes of the graph traveres through the
        graph in a depth first approach.
        Retrieves output from an upstream operator at a particular output port.
        Sets the right input to a downstrem operator at the right input port.
        Executes the operators.
        """
        g = self._app.get_graph()
        nodes_old = [list(nx.bfs_tree(g, n)) for n in self._root_nodes]
        nodes = [item for sublist in nodes_old for item in sublist]

        exec_context = ExecutionContext()

        for node in nodes:
            node.pre_execute()
            node.execute(exec_context)
            node.post_execute()
            edges = g.out_edges(node)
            for e in edges:
                edge_data = g.get_edge_data(e[0], e[1])
                output = exec_context.get_operator_output(e[0].get_uid(), edge_data["upstream_op_port"])
                exec_context.set_operator_input(e[1].get_uid(), edge_data["downstream_op_port"], output)
