from abc import ABC, abstractmethod
from monai.deploy.foundation.data_store import DataStore
from monai.deploy.executors.executor import Executor
from queue import Queue
from multiprocessing import Process

class MultiProcessExecutor(Executor):

    def __init__(self, app):
        super().__init__(app)
        self._storage = DataStore.get_instance()


    def execute(self):
        g = self._app.get_graph()
        for node in self._root_nodes:

            q = Queue()
            q.put(node)

            while(q.empty() == False):
                n = q.get()
                edges = g.out_edges(n)
                self._launch_operator(n)

                for e in edges:

                    #figure out how to deal with duplicate nodes
                    q.put(e[1])
                    edge_data = g.get_edge_data(e[0], e[1])
                    output = node.get_output(edge_data['upstream_op_port'])
                    key1 = (e[0].get_uid(), 'output', edge_data['upstream_op_port'])
                    self._storage.store(key1, output)
                    key2 = (e[1].get_uid(), 'input', edge_data['downstream_op_port'])
                    self._storage.store(key2, output)

    

    def _launch_operator(self, op):
        p = Process(target=op.execute)
        p.start()
        p.join()