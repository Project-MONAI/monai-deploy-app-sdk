import sys
sys.path.insert(0,'../../../../')
from monai.deploy.foundation.application import Application
from monai.deploy.apps.simple_imaging_app.sobel_operator import SobelOperator
from monai.deploy.apps.simple_imaging_app.median_operator import MedianOperator
from monai.deploy.apps.simple_imaging_app.gaussian_operator import GaussianOperator
from monai.deploy.executors.single_process_executor import SingleProcessExecutor


from skimage import data, io, filters

class MyApp(Application):

    """ This is a very basic application
    that showcases the MONAI Deploy application framework
    """
    def __init__(self):
        super().__init__()
        
    def compose(self):
        """ This application has three operators
        Each operator performes has a single input &
        a single output port. Each operator performs
        some kind of image processing function
        """
        self.sobel_op = SobelOperator()
        self.median_op = MedianOperator()
        self.gaussian_op = GaussianOperator()
        self.link_operators(self.sobel_op, self.median_op)
        self.link_operators(self.median_op, self.gaussian_op)



def main():
    app = MyApp()
    executor = SingleProcessExecutor(app)
    executor.execute()


if __name__ == "__main__":
    main()

