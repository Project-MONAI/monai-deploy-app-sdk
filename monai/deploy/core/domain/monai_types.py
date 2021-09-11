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

from typing import Any

from typing_extensions import Protocol

ComposeInterface = Any
# Using Proptocol causes the type checker to complain on the following line:
# https://github.com/Project-MONAI/monai-deploy-app-sdk/blob/e7543e45657398347619cc9e5de4b53e69dbbfdb/examples/apps/
# ai_spleen_seg_app/spleen_seg_operator.py#L108
#
# class ComposeInterface(Protocol):
#     def flatten(self):
#         ...
#
#     def __call__(self, _input):
#         ...


class ImageReaderInterface(Protocol):
    pass
