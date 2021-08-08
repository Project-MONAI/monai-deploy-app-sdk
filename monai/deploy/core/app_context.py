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

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Optional, Type

from monai.deploy.cli.main import parse_args

if TYPE_CHECKING:
    from .application import Application

from .resource import Resource
from .runtime_env import RuntimeEnv


class AppContext:
    """A class to store the context of an application."""

    def __init__(self, app: Type[Application], runtime_env: Optional[Type[RuntimeEnv]] = None):
        # Parse the command line arguments
        argv = sys.argv
        args = parse_args(argv, default_command="exec")

        # Set the runtime environment
        self.runtime_env = runtime_env or RuntimeEnv()

        # Set the backend engines
        self.graph = args.graph
        self.datastore = args.datastore
        self.executor = args.executor

        # Set the path to input/output/model
        self.input_path = args.input or self.runtime_env.input
        self.output_path = args.output or self.runtime_env.output
        self.model_path = args.model or self.runtime_env.model

        # Set resource limits
        # TODO(gigony): Add cli option to set resource limits
        self.resource = Resource()
