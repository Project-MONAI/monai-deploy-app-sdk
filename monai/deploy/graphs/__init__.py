import importlib

try:
    _flow_graphs = importlib.import_module("holoscan.flow_graphs")
except ModuleNotFoundError as e:
    if e.name != "holoscan.flow_graphs":
        raise
    _graphs = importlib.import_module("holoscan.graphs")
    FlowGraph = _graphs.FlowGraph
    FragmentFlowGraph = _graphs.FragmentFlowGraph
    OperatorFlowGraph = _graphs.OperatorFlowGraph

    __all__ = ["FlowGraph", "FragmentFlowGraph", "OperatorFlowGraph"]
else:
    FlowGraphImpl = _flow_graphs.FlowGraphImpl
    FragmentFlowGraph = _flow_graphs.FragmentFlowGraph
    FragmentFlowGraphImpl = _flow_graphs.FragmentFlowGraphImpl
    OperatorFlowGraph = _flow_graphs.OperatorFlowGraph
    OperatorFlowGraphImpl = _flow_graphs.OperatorFlowGraphImpl
    FlowGraph = FlowGraphImpl

    __all__ = [
        "FlowGraph",
        "FlowGraphImpl",
        "FragmentFlowGraph",
        "FragmentFlowGraphImpl",
        "OperatorFlowGraph",
        "OperatorFlowGraphImpl",
    ]
