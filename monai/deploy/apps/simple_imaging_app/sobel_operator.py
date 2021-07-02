import sys
sys.path.insert(0,'../../../../')

from skimage import data, io, filters
from monai.deploy.foundation.base_operator import BaseOperator

class SobelOperator(BaseOperator):
    """ This Operator implements a Sobel edge detector
    It has a single input and single output
    """
    def __init__(self):
        super().__init__()
        self.data_out = None
    
    """ Provides the output at a given port number
    This one has only one output
    Args:
        output_port_number: Ignored as it has a single output
    Returns:
        image: edge detected image
    """
    def get_output(self, output_port_number):
        return self.data_out
    
    """ Performs execution for this operator
    The input for this operator is hardcoded
    In near future this will be changed where
    the input is provided via inversion of control
    mecchanism
    """
    def execute(self):
        super().execute()
        data_in = io.imread("./brain_mr_input.jpg")
        self.data_out = filters.sobel(data_in)