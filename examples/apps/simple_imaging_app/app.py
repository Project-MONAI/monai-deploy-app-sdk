# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gaussian_operator import GaussianOperator
from median_operator import MedianOperator
from sobel_operator import SobelOperator

from monai.deploy.core import Application
from monai.deploy.core.datastores import MemoryDataStore
from monai.deploy.core.executors import SingleProcessExecutor


class MyApp(Application):
    """This is a very basic application.

    This showcases the MONAI Deploy application framework.
    """

    def compose(self):
        """This application has three operators.

        Each operator has a single input and a single output port.
        Each operator performs some kind of image processing function.
        """
        self.sobel_op = SobelOperator()
        self.median_op = MedianOperator()
        self.gaussian_op = GaussianOperator()
        self.add_flow(self.sobel_op, self.median_op, {"image": "image"})
        self.add_flow(self.median_op, self.gaussian_op)


def main():
    app = MyApp()
    data_store = MemoryDataStore()
    executor = SingleProcessExecutor(app, data_store)
    executor.execute()


if __name__ == "__main__":
    main()
