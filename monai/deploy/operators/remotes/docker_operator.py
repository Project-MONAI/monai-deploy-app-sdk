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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, InputContext, OutputContext
from monai.deploy.core.domain import DataPath
from monai.deploy.exceptions import MONAIAppSdkError, WrongValueError

from .remote_operator import RemoteOperator

logger = logging.getLogger(__name__)


@md.env(pip_packages=["docker"])
class RemoteDockerOperator(RemoteOperator):
    """ """

    remote_type: str = "docker"

    def __init__(
        self,
        image: str,
        command: Optional[Union[str, List]] = None,
        timeout: int = 0,
        input_folder: Optional[str] = "/input",
        output_folder: Optional[str] = "/output",
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()

        if input_folder == output_folder:
            raise WrongValueError("Input and output folders cannot be the same.")

        if params is None:
            params = {}

        self.image: str = image
        self.command: Optional[Union[str, List]] = command
        self.timeout: int = timeout
        self.input_folder: Optional[str] = input_folder
        self.output_folder: Optional[str] = output_folder
        self.params: Dict[str, Any] = params or {}
        self.config: Dict[str, Any] = {}
        self.container = None
        self.volume_mappings: Dict[str, Any] = {}
        self.stdout: str = ""
        self.stderr: str = ""

    def to_input_folder(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext) -> Path:
        input_obj = op_input.get()

        if isinstance(input_obj, DataPath):
            input_folder: Path = op_input.get().path
        else:
            raise WrongValueError(
                f"{self.name} does not have an input with (DataPath, IOType.DISK). "
                f"Please check your input and override to_input_folder() method."
            )
        if input_folder.is_file():
            return input_folder.parent
        return input_folder

    def to_output_folder(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext) -> Path:
        output_obj = op_output.get()

        if isinstance(output_obj, DataPath):
            output_folder: Path = op_output.get().path
        else:
            raise WrongValueError(
                f"{self.name} does not have an output with (DataPath, IOType.DISK). "
                f"Please check your output and override to_output_folder() method."
            )
        if output_folder.is_file():
            return output_folder.parent
        return output_folder

    def setup(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        import os

        import docker

        self.client = docker.from_env()

        local_input_folder: Path = self.to_input_folder(op_input, op_output, context).absolute()
        local_output_folder: Path = self.to_output_folder(op_input, op_output, context).absolute()

        # If this app is running in a container, we need to map the input and output folders in container
        # to folders in the host filesystem.
        workdir_host = os.environ.get("MONAI_WORKDIR_HOST")
        if workdir_host:
            local_input_folder = Path(workdir_host) / local_input_folder.relative_to(context.workdir)
            local_output_folder = Path(workdir_host) / local_output_folder.relative_to(context.workdir)

        logger.debug(f"local_input_folder: {local_input_folder}")
        logger.debug(f"local_output_folder: {local_output_folder}")

        self.volume_mappings = {
            str(local_input_folder): {"bind": self.input_folder, "mode": "ro"},
            str(local_output_folder): {"bind": self.output_folder, "mode": "rw"},
        }
        logger.debug(f"volume_mappings: {self.volume_mappings}")

        self.config = {
            "shm_size": "1g",
            "volumes": self.volume_mappings,
        }

    def run(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        config = dict(self.config)
        config.update(**self.params)

        # Remove 'detach' from config if it is already present
        if config.get("detach"):
            config.pop("detach")

        if self.command is not None:
            config.update(command=self.command)

        container = self.client.containers.run(self.image, detach=True, **config)
        self.container = container
        logger.info(f"Container {container.id} started.")

        if self.timeout <= 0:
            timeout = None
        else:
            timeout = self.timeout

        self.status = container.wait(timeout=timeout)
        logger.info(f"Container {container.id} exited with status {self.status}.")

        self.stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
        self.stderr = container.logs(stdout=False, stderr=True).decode("utf-8")

        if self.status["StatusCode"] != 0:
            logger.error(container.logs())
            raise MONAIAppSdkError("Docker exited with non-zero status code")
        container.remove()
        logger.info(f"Container {container.id} removed.")
