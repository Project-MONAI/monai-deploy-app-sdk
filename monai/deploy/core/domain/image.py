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

from typing import Any, Dict, Optional, Union

try:
    from numpy.typing import _ArrayLike  # type: ignore

    ArrayLike: Any = _ArrayLike
except ImportError:
    ArrayLike = Any

from .domain import Domain


class Image(Domain):
    def __init__(self, data: Union[ArrayLike], metadata: Optional[Dict] = None):
        super().__init__(metadata)
        self._data = data

    def asnumpy(self) -> ArrayLike:
        return self._data
