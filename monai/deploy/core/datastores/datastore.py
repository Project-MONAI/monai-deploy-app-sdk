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
from typing import Any, Hashable, KeysView, Optional


class Datastore(ABC):
    """Base class for data store."""

    @abstractmethod
    def get(self, key: Hashable, def_val: Optional[Any] = None) -> Any:
        """Get value from the data store.

        Args:
            key (Hashable): A key to get.

        Returns:
            value (Any): A value from the data store.
        """
        pass

    @abstractmethod
    def put(self, key: Hashable, value: Any):
        """Put value into the data store.

        Args:
            key (Hashable): A key to put.
            value (Any): A value to put.
        """
        pass

    @abstractmethod
    def delete(self, key: Hashable):
        """Delete value from the data store.

        Args:
            key (Hashable): A key to delete.
        """
        pass

    @abstractmethod
    def exists(self, key: Hashable) -> bool:
        """Check if key exists in data store.

        Args:
            key (Hashable): A key to check.

        Returns:
            exists (bool): True if key exists, False otherwise.
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """Get size of data store.

        Returns:
            size (int): A size of data store.
        """
        pass

    @abstractmethod
    def keys(self) -> KeysView:
        """Get keys from data store.

        Returns:
            keys (KeysView): A view of keys.
        """
        pass
