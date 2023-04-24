# Copyright 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path

from gaussian_operator import GaussianOperator
from median_operator import MedianOperator
from sobel_operator import SobelOperator

from monai.deploy.conditions import CountCondition
from monai.deploy.core import AppContext, Application
from monai.deploy.logger import load_env_log_level


# @resource(cpu=1)
# # pip_packages can be a string that is a path(str) to requirements.txt file or a list of packages.
# @env(pip_packages=["scikit-image >= 0.17.2"])
class App(Application):
    """This is a very basic application.

    This showcases the MONAI Deploy application framework.
    """

    # App's name. <class name>('App') if not specified.
    name = "simple_imaging_app"
    # App's description. <class docstring> if not specified.
    description = "This is a very simple application."
    # App's version. <git version tag> or '0.0.0' if not specified.
    version = "0.1.0"

    def compose(self):
        """This application has three operators.

        Each operator has a single input and a single output port.
        Each operator performs some kind of image processing function.
        """
        app_context = AppContext({})  # Let it figure out all the attributes without overriding
        sample_data_path = Path(app_context.input_path)
        output_data_path = Path(app_context.output_path)
        print(f"sample_data_path: {sample_data_path}")

        # Please note that the Application object, self, is passed as the first positonal argument
        # and the others as kwargs.
        # Also note the CountCondition of 1 on the first operator, indicating to the application executor
        # to invoke this operator, hence the pipleline, only once.
        sobel_op = SobelOperator(self, CountCondition(self, 1), input_folder=sample_data_path, name="sobel_op")
        median_op = MedianOperator(self, name="median_op")
        gaussian_op = GaussianOperator(self, output_folder=output_data_path, name="gaussian_op")
        self.add_flow(
            sobel_op,
            median_op,
            {
                ("out1", "in1"),
            },
        )
        self.add_flow(
            median_op,
            gaussian_op,
            {
                (
                    "out1",
                    "in1",
                )
            },
        )  # Using port name is optional for single port cases


if __name__ == "__main__":
    load_env_log_level()
    logging.info(f"Begin {__name__}")
    App().run()
    logging.info(f"End {__name__}")
