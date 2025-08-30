# Copyright 2025 MONAI Consortium
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
from monai.deploy.core import (
    Application,
    ConditionType,
    ExecutionContext,
    Fragment,
    InputContext,
    IOType,
    Operator,
    OutputContext,
    OperatorSpec,
)


class ExecutionStatusReporterOperator(Operator):
    """
    This operator reports the execution status of the application via a callback.
    It is intended to be the last operator in the application's workflow.
    """

    def __init__(self, fragment: Fragment, *args, status_callback, **kwargs):
        """
        Args:
            fragment (Fragment): An instance of the Application class.
            status_callback (callable): The callback function to invoke with the status.
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._status_callback = status_callback
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("data")
        spec.output("data").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        """
        Receives data from the upstream operator and invokes the status callback.
        """
        # For now, we are not doing anything with the input data or collecting logs.
        # We will just report success.
        # In the future, this is where log collection and summary generation would happen.
        try:
            # In a real implementation, you would gather data here.
            op_input.receive("data")
            summary = {"status": "Success", "message": "Application completed successfully."}
            if self._status_callback:
                self._status_callback(summary)
            op_output.emit(summary, "data")
        except Exception as e:
            self._logger.error(f"Error in status reporter: {e}")
            if self._status_callback:
                error_summary = {"status": "Failure", "message": f"Application failed with error: {e}"}
                self._status_callback(error_summary)
