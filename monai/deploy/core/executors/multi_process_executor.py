# # Copyright 2021 MONAI Consortium
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #     http://www.apache.org/licenses/LICENSE-2.0
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# from multiprocessing import Process
# from queue import Queue

# from monai.deploy.core.executors.executor import Executor
# from monai.deploy.core import Application
# from monai.deploy.core.datastores import Datastore


# class MultiProcessExecutor(Executor):
#     def __init__(self, app: Application):
#         super().__init__(app)
#         self._storage = Datastore.get_instance()

#     def execute(self):
#         g = self.app.graph
#         for node in self._root_nodes:

#             q = Queue()
#             q.put(node)

#             while not q.empty():
#                 n = q.get()
#                 edges = g.out_edges(n)
#                 self._launch_operator(n)

#                 for e in edges:
#                     # Figure out how to deal with duplicate nodes
#                     q.put(e[1])
#                     edge_data = g.get_edge_data(e[0], e[1])
#                     output = node.get_output(edge_data["upstream_op_port"])
#                     key1 = (e[0].get_uid(), "output", edge_data["upstream_op_port"])
#                     self._storage.store(key1, output)
#                     key2 = (e[1].get_uid(), "input", edge_data["downstream_op_port"])
#                     self._storage.store(key2, output)

#     def _launch_operator(self, op):
#         p = Process(target=op.execute)
#         p.start()
#         p.join()
