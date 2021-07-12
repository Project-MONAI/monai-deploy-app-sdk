import sys
sys.path.insert(0,'../../../../')

from monai.deploy.foundation.base_operator import BaseOperator

from skimage import data
from skimage.morphology import disk
from skimage.filters import median
from skimage.io import imsave

class MedianOperator(BaseOperator):

    """ This Operator implements a noise reduction
    algorithm based on median operator. It
    ingest a single input and provides a single output
    """
    def __init__(self):
        super().__init__()
    

    def execute(self, execution_context):
        super().execute(execution_context)
        data_in =  execution_context.get_operator_input(self._uid, 0)
        data_out = median(data_in)
        execution_context.set_operator_output(self._uid, 0, data_out)
