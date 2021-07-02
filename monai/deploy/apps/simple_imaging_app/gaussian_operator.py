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
        self.data_out = None
    

    def get_output(self, index):
        return self.data_out
    

    def execute(self):
        super().execute()
        data_in = self.get_input(0)
        self.data_out = gaussian(data_in, sigma=0.2)
        imsave("final_output.png", self.data_out)
