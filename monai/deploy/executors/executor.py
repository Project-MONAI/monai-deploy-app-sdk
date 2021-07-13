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

from abc import ABC, abstractmethod

from monai.deploy.foundation.application import Application


class Executor(ABC):
    """This is the base class that enables execution of an application."""

    def __init__(self, app: Application):
        """Constructor of the class.

        Given an application it invokes the compose method on the app, which
        in turn creates the necessary operator and links them up.

        Args:
            app: An application that needs to be executed.
        """
        super().__init__()
        self._app = app
        self._app.compose()
        self._root_nodes = [n for n, d in self._app.get_graph().in_degree() if d == 0]

    @abstractmethod
    def execute(self):
        """The execute method of an executor.

        It is called to execute an application.
        This method needs to be implemented by specific concrete subclasses
        of ``Executor``.
        """
        pass
