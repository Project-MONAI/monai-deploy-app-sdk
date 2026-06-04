try:
    from holoscan.flow_graphs import *
except ModuleNotFoundError:
    from holoscan.graphs import *

# holoscan 4.1.0 renamed FlowGraph to FlowGraphImpl
if "FlowGraph" not in globals() and "FlowGraphImpl" in globals():
    FlowGraph = FlowGraphImpl
