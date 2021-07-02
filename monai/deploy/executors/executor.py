from abc import ABC, abstractmethod
from monai.deploy.foundation.application import Application

class Executor(ABC):
    """ This is the base class that enables execution of an application
    """

    def __init__(self, app):
        """ Constructor of the class
        Given an application it invokes the compose method on the app, which
        in turn creates the necessary operator and links them up
        Args:
            app: the application that needs to be executed
        """
        super().__init__()
        self._app = app
        self._app.compose()
        self._root_nodes = [n for n,d in self._app.get_graph().in_degree() if d==0]


    @abstractmethod
    def execute(self):
        """ The execute method of an executor 
        is called to execute an application.
        This method needs to be implemented by specific
        concerete subclasses of Executor
        """
        pass
