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

import functools
from typing import List, Optional

from monai.deploy.exceptions import ItemAlreadyExistsError, UnknownTypeError

from .application import Application, ApplicationEnv
from .operator import Operator, OperatorEnv


def env(pip_packages: Optional[List[str]] = None):
    """A decorator that adds an environment specification to either Operator or Application.

    Args:
        pip_packages (Optional[List[str]]): A list of pip packages to install.

    Returns:
        A decorator that adds an environment specification to either Operator or Application.
    """

    def decorator(cls):
        @functools.wraps(cls)
        def wrapper(*args, **kwargs):
            obj = cls(*args, **kwargs)

            if hasattr(cls, "_env"):
                raise ItemAlreadyExistsError(f"@env decorator is aleady specified for {cls}.")
            else:
                if isinstance(obj, Operator):
                    environment = OperatorEnv(pip_packages=pip_packages)
                elif isinstance(obj, Application):
                    environment = ApplicationEnv(pip_packages=pip_packages)
                else:
                    raise UnknownTypeError(f"Use @env decorator cannot be specified for {type(obj)}.")

                obj._env = environment

            if isinstance(obj, Operator):
                if not hasattr(cls, "_env"):
                    environment = OperatorEnv(*args, **kwargs)
                    obj._env = environment
                else:
                    raise ItemAlreadyExistsError("@env decorator is aleady specified for {}.".format(cls))

            return obj

        return wrapper

    return decorator
