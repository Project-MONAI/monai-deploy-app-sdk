from abc import ABC, abstractmethod
import networkx as nx


class Application(ABC):

    """ This is the base application class
    All applications should be extended from this Class
    The application class provides support for chaining
    up operators, as well as mechanism to execute the application
    """

    def __init__(self):
        """ Constructor for the base class
        It created an instance of an empty Directed Acyclic Graph
        to hold on to the operators
        """
        super().__init__()
        self._graph = nx.DiGraph()


    def get_graph(self):
        """ Gives access to the DAG
        Args:
        
        Returns:
            Instance of the DAG
        """
        return self._graph


    def link_operators(self, upstream_op, downstream_op, upstream_op_port=0, downstream_op_port=0):
        """ Enables linking of two operators. Each Operator has multiple ports
        An output port of the upstream operator is connected to one of the input ports of a downstream operators

        Args:
            upstream_op: An instance of the upstream operator of type BaseOperator
            downstream_op: An instance of the downstream operator of type BaseOperator
            upstream_op_port: The port number of the upstream operator which will be connected to the downstream operator
            downstream_op_port: The port number of the downstream operator which will be connected to the upstream operator
        
        Returns:
        """

        self._graph.add_edge(upstream_op, downstream_op, upstream_op_port=upstream_op_port, downstream_op_port=downstream_op_port)
        pass

    @abstractmethod
    def compose(self):
        """ This is a method that needs to implemented by all subclasses
        Derived appications will chain up the operators inside this conpose
        method.

        Args:
        Returns:
        """
        pass

