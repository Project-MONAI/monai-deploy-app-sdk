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

from .datastore import Datastore
from .memory import MemoryDatastore


class DatastoreFactory:
    """DatastoreFactory is an abstract class that provides a way to create a datastore object."""

    NAMES = ["memory"]
    DEFAULT = "memory"

    @staticmethod
    def create(datastore_type: str, datastore_params: Optional[Dict] = None) -> Datastore:
        """Creates a datastore object.

        Args:
            datastore_type (str): A type of the datastore.
            datastore_params (Dict): A dictionary of parameters of the datastore.

        Returns:
            Datastore: A datastore object.
        """

        datastore_params = datastore_params or {}

        if datastore_type == "memory":
            return MemoryDatastore(**datastore_params)
        else:
            raise UnknownTypeError(f"Unknown datastore type: {datastore_type}")
