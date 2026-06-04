try:
    from holoscan.flow_graphs import (
        FlowGraphImpl,
        FragmentFlowGraph,
        FragmentFlowGraphImpl,
        OperatorFlowGraph,
        OperatorFlowGraphImpl,
    )

    FlowGraph = FlowGraphImpl

    __all__ = [
        "FlowGraph",
        "FlowGraphImpl",
        "FragmentFlowGraph",
        "FragmentFlowGraphImpl",
        "OperatorFlowGraph",
        "OperatorFlowGraphImpl",
    ]
except ModuleNotFoundError as e:
    if e.name != "holoscan.flow_graphs":
        raise
    from holoscan.graphs import FlowGraph, FragmentFlowGraph, OperatorFlowGraph

    __all__ = ["FlowGraph", "FragmentFlowGraph", "OperatorFlowGraph"]
