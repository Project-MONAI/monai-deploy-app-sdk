from typing import Any, Dict, Hashable, KeysView, Optional

from .datastore import Datastore


class MemoryDatastore(Datastore):
    def __init__(self):
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
