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

from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, Operator, OutputContext, input, output


@input("image", Image, IOType.IN_MEMORY)
@output("image", Image, IOType.IN_MEMORY)
# If `pip_packages` is specified, the definition will be aggregated with the package dependency list of other
# operators and the application in packaging time.
# @env(pip_packages=["scikit-image >= 0.17.2"])
class MedianOperatorBase(Operator):
    """This Operator implements a noise reduction.

    The algorithm is based on the median operator.
    It ingests a single input and provides a single output.
    """

    # Define __init__ method with super().__init__() if you want to override the default behavior.
    def __init__(self):
        super().__init__()
        # Do something

    def compute(self, input: InputContext, output: OutputContext, context: ExecutionContext):
        print("Executing base operator...")


class MedianOperator(MedianOperatorBase):
    """This operator is a subclass of the base operator to demonstrate the usage of inheritance."""

    # Define __init__ method with super().__init__() if you want to override the default behavior.
    def __init__(self):
        super().__init__()
        # Do something

    def compute(self, input: InputContext, output: OutputContext, context: ExecutionContext):
        # Execute the base operator's compute method.
        super().compute(input, output, context)

        from skimage.filters import median

        # `context.input.get().path` (Path) is the file/folder path of the input data from the application's context.
        # `context.output.get().path` (Path) is the file/folder path of the output data from the application's context.
        # `context.models.get(model_name)` returns a model instance
        #  (a null model would be returned if model is not available)
        # If model_name is not specified and only one model exists, it returns that model.
        model = context.models.get()  # a model object that inherits Model class

        # Get a model instance if exists
        if model:  # if model is not a null model
            print(model.items())
            # # model.path for accessing the model's path
            # # model.name for accessing the model's name
            # result = model(input.get().asnumpy())

        data_in = input.get().asnumpy()
        data_out = median(data_in)
        output.set(Image(data_out))
