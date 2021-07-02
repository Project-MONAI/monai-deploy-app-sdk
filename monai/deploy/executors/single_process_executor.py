from abc import ABC, abstractmethod
from monai.deploy.foundation.data_store import DataStore
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
        self._storage = DataStore.get_instance()

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

        for node in nodes:
            node.pre_execute()
            node.execute()
            node.post_execute()
            edges = g.out_edges(node)
            for e in edges:
                    edge_data = g.get_edge_data(e[0], e[1])
                    output = node.get_output(edge_data['upstream_op_port'])
                    key1 = (e[0].get_uid(), 'output', edge_data['upstream_op_port'])
                    self._storage.store(key1, output)
                    key2 = (e[1].get_uid(), 'input', edge_data['downstream_op_port'])
                    self._storage.store(key2, output)



    # def execute2(self):
    #     g = self._app.get_graph()
    #     for node in self._root_nodes:

    #         q = Queue()
    #         q.put(node)

    #         while(q.empty() == False):
    #             n = q.get()
    #             edges = g.out_edges(n)
    #             n.execute()

    #             for e in edges:

    #                 #figure out how to deal with duplicate nodes
    #                 q.put(e[1])
    #                 edge_data = g.get_edge_data(e[0], e[1])
    #                 output = node.get_output(edge_data['upstream_op_port'])
    #                 key1 = (e[0].get_uid(), 'output', edge_data['upstream_op_port'])
    #                 self._storage.store(key1, output)
    #                 key2 = (e[1].get_uid(), 'input', edge_data['downstream_op_port'])
    #                 self._storage.store(key2, output)


        
