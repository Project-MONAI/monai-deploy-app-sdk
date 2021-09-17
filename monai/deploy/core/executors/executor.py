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

from abc import ABC, abstractmethod
from typing import Dict, Optional

# https://github.com/agronholm/sphinx-autodoc-typehints#dealing-with-circular-imports
from monai.deploy.core import application
from monai.deploy.core.datastores import Datastore, DatastoreFactory


class Executor(ABC):
    """This is the base class that enables execution of an application."""

    def __init__(self, app: "application.Application", datastore: Optional[Datastore] = None, **kwargs: Dict):
        """Constructor of the class.

        Given an application it invokes the compose method on the app, which
        in turn creates the necessary operator and links them up.

        Args:
            app: An application that needs to be executed.
            datastore: A data store that is used to store the data.
        """
        self._app = app
        if datastore:
            self._datastore = datastore
        else:
            self._datastore = DatastoreFactory.create(DatastoreFactory.DEFAULT)

    @property
    def app(self) -> "application.Application":
        """Returns the application that is executed by the executor."""
        return self._app

    @property
    def datastore(self) -> Datastore:
        """Returns the data store that is used to store the data."""
        return self._datastore

    @abstractmethod
    def run(self):
        """Run the app.

        It is called to execute an application.
        This method needs to be implemented by specific concrete subclasses
        of `Executor`.
        """
        pass
