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

import os
from pathlib import Path

from gaussian_operator import GaussianOperator
from median_operator import MedianOperator
from sobel_operator import SobelOperator

from monai.deploy.conditions import CountCondition
from monai.deploy.core import Application
from monai.deploy.logger import load_env_log_level

# Input path should be taken care of by the base app.
# Without it being done yet, try to refer to MAP spec
# for the well-known env vars


DEFAULT_IN_PATH = Path(os.path.dirname(__file__)) / "input"
DEFAULT_OUT_PATH = Path(os.path.dirname(__file__)) / "output"

sample_data_path = Path(os.environ.get("MONAI_INPUTPATH", DEFAULT_IN_PATH))
output_data_path = Path(os.environ.get("MONAI_OUTPUTPATH", DEFAULT_OUT_PATH))


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
        print(f"sample_data_path: {sample_data_path}")
        sobel_op = SobelOperator(self, CountCondition(self, 1), input_folder=sample_data_path, name="sobel_op")
        median_op = MedianOperator(self, name="median_op")
        gaussian_op = GaussianOperator(self, output_folder=output_data_path, name="gaussian_op")
        self.add_flow(
            sobel_op,
            median_op,
            {
                ("out1", "in1"),
            },
        )  # Identifing port optional for single port cases
        self.add_flow(
            median_op,
            gaussian_op,
            {
                (
                    "out1",
                    "in1",
                )
            },
        )


if __name__ == "__main__":
    load_env_log_level()
    app = App()
    app.config(os.path.join(os.path.dirname(__file__), "simple_imaing_app.yaml"))
    app.run()
