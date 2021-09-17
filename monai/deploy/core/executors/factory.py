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

from typing import Dict, Optional

from monai.deploy.exceptions import UnknownTypeError

from .executor import Executor
from .single_process_executor import SingleProcessExecutor


class ExecutorFactory:
    """ExecutorFactory is an abstract class that provides a way to create an executor object."""

    NAMES = ["single_process_executor"]
    DEFAULT = "single_process_executor"

    @staticmethod
    def create(executor_type: str, executor_params: Optional[Dict] = None) -> Executor:
        """Creates an executor object.

        Args:
            executor_type (str): A type of the executor.
            executor_params (Dict): A dictionary of parameters of the executor.

        Returns:
            Executor: An executor object.
        """

        executor_params = executor_params or {}

        if executor_type == "single_process_executor":
            return SingleProcessExecutor(**executor_params)
        else:
            raise UnknownTypeError(f"Unknown executor type: {executor_type}")
