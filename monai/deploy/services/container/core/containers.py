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

import logging

from .status import Status

logger = logging.getLogger(__name__)


class Container:
    def __init__(self, obj):
        self.obj = obj

    @property
    def id(self):
        raise NotImplementedError("id() is not implemented!")

    def wait(self, **kwargs) -> Status:
        raise NotImplementedError("wait() is not implemented!")

    def logs(self, **kwargs):
        raise NotImplementedError("logs() is not implemented!")

    def remove(self, **kwargs):
        raise NotImplementedError("remove() is not implemented!")


class DockerContainer(Container):
    @property
    def id(self):
        return self.obj.id

    def wait(self, **kwargs) -> Status:
        status = self.obj.wait(**kwargs)
        logger.debug(f"{self.id} status: {status}")
        status_code = status.get("StatusCode", 0)
        message = status.get("Error", "") or ""
        return Status(status_code, message)

    def logs(self, **kwargs):
        return self.obj.logs(**kwargs)

    def remove(self, **kwargs):
        return self.obj.remove(**kwargs)
