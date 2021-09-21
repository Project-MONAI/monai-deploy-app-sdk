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

from abc import abstractmethod
from typing import Any, Union

from monai.deploy.core import ExecutionContext, Image, InputContext, Operator, OutputContext


class InferenceOperator(Operator):
    """The base operator for operators that perform AI inference.

    This operator preforms pre-transforms on a input image, inference with
    a given model, post-transforms, and final results generation.
    """

    def __init__(self, *args, **kwargs):
        """Constructor of the operator."""
        super().__init__()

    @abstractmethod
    def pre_process(self, data: Any) -> Union[Image, Any]:
        """Transforms input before being used for predicting on a model.

        This method must be overridden by a derived class.

        Raises:
            NotImplementedError: When the subclass does not override this method.
        """

        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        """An abstract method that needs to be implemented by the user.

        Args:
            op_input (InputContext): An input context for the operator.
            op_output (OutputContext): An output context for the operator.
            context (ExecutionContext): An execution context for the operator.
        """
        pass

    @abstractmethod
    def predict(self, data: Any) -> Union[Image, Any]:
        """Prdicts results using the models(s) with input tensors.

        This method must be overridden by a derived class.

        Raises:
            NotImplementedError: When the subclass does not override this method.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def post_process(self, data: Any) -> Union[Image, Any]:
        """Transform the prediction results from the model(s).

        This method must be overridden by a derived class.

        Raises:
            NotImplementedError: When the subclass does not override this method.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")
