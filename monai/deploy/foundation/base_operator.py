from abc import ABC, abstractmethod
import uuid
import os
from monai.deploy.foundation.operator_info import OperatorInfo

from colorama import Fore, Back, Style


class BaseOperator(ABC):

    """ This is the base Operator class
    An operator in MONAI Deploy performs a unit fo wrok for the application.
    An operator has multiple inout and output ports. Each port specifies an interaction point 
    through which a operator can communicate with other operatord
    """
    def __init__(self):
        """ Constructor of the base operator
        It creates an instance of Data Store which holds on
        to all inpuuts and outputs relavant for this operator
        """
        super().__init__()
        self._num_input_ports = 1
        self._num_output_ports = 1
        self._uid = uuid.uuid4()
        self._op_info = self.create_op_info()
    

    def create_op_info(self):
        _op_info = OperatorInfo()
        _op_info.num_input_ports = 1
        _op_info.num_output_ports = 1
        return _op_info


    def get_uid(self):
        """ Gives access to the UID of the operator
        Returns:
            UID of the operator
        """
        return self._uid
   
   
    def get_operator_info(self):
        """ Retrieves the operator info
        Args:
        Returns:
            An instance of OperatorInfo
        
        """
        return self._op_info


    def get_num_input_ports(self):
        """ Provides number of input ports that this operator has
        Returns:
            Number of input ports
        """
        return self._op_info.num_input_ports
    

    def get_num_output_ports(self):
        """ Provides number of output ports that this operator has
        Returns:
            Number of output ports
        """
        return self._op_info.num_output_ports
        
    
    def pre_execute(self):
        """ This method gets executed before "execute" of an operator is called
        This is a preperatory step before the operator executes its main job
        This needs to be overridden by a base class for any meaningful action
        """
        print(Fore.BLUE + 'Going to initiate execution of operator %s' %self.__class__.__name__)
        '%s is smaller than %s' % ('one', 'two')
        pass

    @abstractmethod
    def execute(self, execution_context):
        """ Provides number of output ports that this operator has
        Returns:
            Number of output ports
        """
        print(Fore.YELLOW + 'Process ID %s' % os.getpid())
        print(Fore.GREEN + 'Executing operator %s' %self.__class__.__name__)
        # print('parent process: ', os.getppid())
        pass


    def post_execute(self):
        """ This method gets executed after "execute" of an operator is called
        This is a pst-execution step before the operator is done doing its main action
        This needs to be overridden by a base class for any meaningful action
        """
        print(Fore.BLUE + 'Done performing execution of operator %s' % self.__class__.__name__)
        pass

