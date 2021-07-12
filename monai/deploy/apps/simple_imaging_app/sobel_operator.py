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
    
    
    
    """ Performs execution for this operator
    The input for this operator is hardcoded
    In near future this will be changed where
    the input is provided via inversion of control
    mecchanism
    """
    def execute(self, execution_context):
        super().execute(execution_context)
        data_in = io.imread("./brain_mr_input.jpg")
        data_out = filters.sobel(data_in)
        execution_context.set_operator_output(self._uid, 0, data_out)
