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
import subprocess

logger = logging.getLogger("app_runner")


def get_requested_gpus(pkg_info: dict) -> int:
    """Gets requested number of gpus in the package manifest

    Args:
        pkg_info: package manifest as a python dict

    Returns:
        int: requested number of gpus in the package manifest
    """
    num_gpu: int = pkg_info.get("resources", {}).get("gpu", 0)
    return num_gpu


def run_cmd(cmd: str) -> int:
    """
    Executes command and return the returncode of the executed command.

    Redirects stderr of the executed command to stdout.

    Args:
        cmd: command to execute.

    Returns:
        output: child process returncode after the command has been executed.
    """
    proc = subprocess.Popen(cmd, universal_newlines=True, shell=True)
    return proc.wait()


def verify_image(image: str) -> bool:
    """Checks if the container image is present locally and tries to pull if not found.

    Args:
        image: container image

    Returns:
        True if either image is already present or if it was successfully pulled.
    """

    def _check_image_exists_locally(image_tag):
        response = subprocess.check_output(
            ["docker", "images", image_tag, "--format", "{{.Repository}}:{{.Tag}}"], universal_newlines=True
        )

        if image_tag in response:
            logger.info('"%s" found.', image_tag)
            return True

        return False

    def _pull_image(image_tag: str) -> bool:
        cmd = f"docker pull {image_tag}"
        returncode = run_cmd(cmd)

        if returncode != 0:
            return False

        return True

    logger.info('Checking for MAP "%s" locally', image)
    if not _check_image_exists_locally(image):
        logger.warning('"%s" not found locally.\n\nTrying to pull from registry...', image)
        return _pull_image(image)

    return True
