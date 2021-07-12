from abc import ABC, abstractmethod
from monai.deploy.foundation.execution_context import ExecutionContext

from monai.deploy.executors.executor import Executor
import networkx as nx
from queue import Queue

class SingleProcessExecutor(Executor):

    """ This class implements execution of a MONAI App
    in a single process in environment. 
    """

    def __init__(self, app):
        """ Constructor for the class
        the instance internally holds on to the data store
        Args:
            app: instance of the application that needs to be executed
        """
        super().__init__(app)


    def execute(self):
        """ Executes the app. This method
        retrieves the root nodes of the graph
        traveres through the graph in a depth first approach
        retrieves output from an upstream operator at a particular output port
        sets the right input to a downstrem operator at the right input port
        executes the operators
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
                    output = exec_context.get_operator_output(e[0].get_uid(), edge_data['upstream_op_port'])
                    exec_context.set_operator_input(e[1].get_uid(), edge_data['downstream_op_port'], output)

        
