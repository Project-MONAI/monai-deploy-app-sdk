try:
    from holoscan.flow_graphs import *
except ModuleNotFoundError as e:
    if e.name != "holoscan.flow_graphs":
        raise
    from holoscan.graphs import *

# holoscan 4.1.0 renamed FlowGraph to FlowGraphImpl
if "FlowGraph" not in globals() and "FlowGraphImpl" in globals():
    FlowGraph = FlowGraphImpl
