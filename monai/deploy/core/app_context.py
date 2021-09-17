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

from typing import Dict, Optional

from .resource import Resource
from .runtime_env import RuntimeEnv


class AppContext:
    """A class to store the context of an application."""

    def __init__(self, args: Dict[str, str], runtime_env: Optional[RuntimeEnv] = None):
        # Set the args
        self.args: Dict[str, str] = {}
        # Set the runtime environment
        self.runtime_env = runtime_env or RuntimeEnv()

        # Set the graph engine here because it would be used in the constructor of Application class so cannot be
        # updated in Application.run() method.
        self.graph = args.get("graph") or self.runtime_env.graph

        self.update(args)

    def update(self, args: Dict[str, str]):
        """Update the context with new args and runtime_env."""
        # Update args
        self.args.update(args)

        # Set the path to input/output/model
        self.input_path = args.get("input") or self.args.get("input") or self.runtime_env.input
        self.output_path = args.get("output") or self.args.get("output") or self.runtime_env.output
        self.model_path = args.get("model") or self.args.get("model") or self.runtime_env.model
        self.workdir = args.get("workdir") or self.args.get("workdir") or self.runtime_env.workdir

        # Set the backend engines except for the graph engine
        self.datastore = args.get("datastore") or self.args.get("datastore") or self.runtime_env.datastore
        self.executor = args.get("executor") or self.args.get("executor") or self.runtime_env.executor

        # Set resource limits
        # TODO(gigony): Add cli option to set resource limits
        self.resource = Resource()

    def __repr__(self):
        return (
            f"AppContext(graph={self.graph}, input_path={self.input_path}, output_path={self.output_path}, "
            f"model_path={self.model_path}, workdir={self.workdir}, datastore={self.datastore}, "
            f"executor={self.executor}, resource={self.resource})"
        )
