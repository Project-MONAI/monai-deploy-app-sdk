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

from pathlib import Path
from typing import List, Optional, Union

from monai.deploy.exceptions import ItemAlreadyExistsError, UnknownTypeError


class BaseEnv:
    """Settings for the environment.

    This class is used to specify the environment settings for the application or the operator.
    """

    def __init__(self, pip_packages: Optional[Union[str, List[str]]] = None):
        """Constructor of the BaseEnv class.

        Args:
            pip_packages Optional[Union[str, List[str]]]: A string that is a path to requirements.txt file
                                                          or a list of packages to install.

        Returns:
            An instance of OperatorEnv.
        """
        if type(pip_packages) is str:
            requirements_path = Path(pip_packages)

            if requirements_path.exists():
                pip_packages = requirements_path.read_text().strip().splitlines()  # make it a list
            else:
                raise FileNotFoundError(f"The '{requirements_path}' file does not exist!")

        self._pip_packages = list(pip_packages or [])

    @property
    def pip_packages(self) -> List[str]:
        """Get the list of pip packages.

        Returns:
            A list of pip packages.
        """
        return self._pip_packages

    def __str__(self):
        return "{}(pip_packages={})".format(self.__class__.__name__, self._pip_packages)


def env(pip_packages: Optional[Union[str, List[str]]] = None):
    """A decorator that adds an environment specification to either Operator or Application.

    Args:
        pip_packages: A string that is a path to requirements.txt file or a list of packages to install.

    Returns:
        A decorator that adds an environment specification to either Operator or Application.
    """
    # Import the classes here to avoid circular import.
    from .application import Application, ApplicationEnv
    from .operator import Operator, OperatorEnv

    def decorator(cls):
        if hasattr(cls, "_env") and cls._env:
            raise ItemAlreadyExistsError(f"@env decorator is aleady specified for {cls}.")

        if issubclass(cls, Operator):
            environment = OperatorEnv(pip_packages=pip_packages)
        elif issubclass(cls, Application):
            environment = ApplicationEnv(pip_packages=pip_packages)
        else:
            raise UnknownTypeError(f"@env decorator cannot be specified for {cls}.")

        cls._env = environment

        return cls

    return decorator
