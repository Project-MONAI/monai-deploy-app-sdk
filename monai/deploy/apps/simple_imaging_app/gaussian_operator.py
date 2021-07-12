import sys
sys.path.insert(0,'../../../../')

from monai.deploy.foundation.base_operator import BaseOperator

from skimage import data
from skimage.morphology import disk
from skimage.filters import gaussian
from skimage.io import imsave

class GaussianOperator(BaseOperator):

    """ This Operator implements a smoothening based on Gaussian. 
    It ingest a single input and provides a single output
    """
    def __init__(self):
        super().__init__()

    def execute(self, execution_context):
        super().execute(execution_context)
        data_in =  execution_context.get_operator_input(self._uid, 0)
        data_out = gaussian(data_in, sigma=0.2)
        execution_context.set_operator_output(self._uid, 0, data_out)
        imsave("final_output.png", data_out)
