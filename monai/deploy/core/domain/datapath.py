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

from pathlib import Path
from typing import Dict, Optional, Union

from .domain import Domain


class DataPath(Domain):
    def __init__(self, path: Union[str, Path], metadata: Optional[Dict] = None):
        super().__init__(metadata=metadata)
        self._path = Path(path)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, val):
        self._path = Path(val)
