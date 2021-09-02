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

from typing import Any, Dict, Hashable, KeysView, Optional

from .datastore import Datastore


class MemoryDatastore(Datastore):
    def __init__(self, **kwargs: Dict):
        self._storage: Dict = {}

    def get(self, key: Hashable, def_val: Optional[Any] = None) -> Any:
        return self._storage.get(key, None)

    def put(self, key: Hashable, value: Any):
        self._storage[key] = value

    def delete(self, key: Hashable):
        del self._storage[key]

    def exists(self, key: Hashable) -> bool:
        return key in self._storage

    def size(self) -> int:
        return len(self._storage)

    def keys(self) -> KeysView:
        return self._storage.keys()
