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

from typing import Any, Dict, Optional, Union

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any

from .domain import Domain


class Image(Domain):
    def __init__(self, data: Union[ArrayLike], metadata: Optional[Dict] = None):
        super().__init__(metadata)
        self._data = data

    def asnumpy(self) -> ArrayLike:
        return self._data

    # @property
    # def ndim(self):
    #     return self.__ndim

    # @ndim.setter
    # def ndim(self, val):
    #     self.__ndim = val

    # @property
    # def shape(self):
    #     return self.__shape

    # @shape.setter
    # def shape(self, val):
    #     self.__shape = val

    # @property
    # def spacing(self):
    #     return self.__spacing

    # @spacing.setter
    # def spacing(self, val):
    #     self.__spacing = val

    # @property
    # def direction_cosines(self):
    #     return self.__direction_cosines

    # @direction_cosines.setter
    # def direction_cosines(self, val):
    #     self.__direction_cosines = val

    # @property
    # def modality(self):
    #     return self.__modality

    # @modality.setter
    # def modality(self, val):
    #     self.__modality = val

    # @property
    # def pixel_data(self):
    #     return self.__pixel_data

    # @pixel_data.setter
    # def pixel_data(self, val):
    #     self.__pixel_data = val

    # def __str__(self):
    #     result = ""
    #     modality = "Modality: " + self.modality
    #     result = result + modality + "\n"
    #     num_dim = "Number of Dimensions: " + str(self.ndim)
    #     result = result + num_dim + "\n"
    #     spacing = "Pixel Spacing: " + str(self.spacing)
    #     result = result + spacing + "\n"
    #     dir_cosines = "Direction Cosines: " + str(self.direction_cosines)
    #     result = result + dir_cosines + "\n"

    #     return result
