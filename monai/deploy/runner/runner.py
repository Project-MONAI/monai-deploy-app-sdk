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

import argparse
import json
import logging
import posixpath
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Tuple

from monai.deploy.runner.utils import get_requested_gpus, run_cmd, verify_image

logger = logging.getLogger("app_runner")


def fetch_map_manifest(map_name: str) -> Tuple[dict, dict, int]:
    """
    Execute MONAI Application Package and fetch application manifest.

    Args:
        map_name: MAP image name.

    Returns:
        app_info: application manifest as a python dict
        pkg_info: package manifest as a python dict
        returncode: command return code
    """
    logger.info("\nReading MONAI App Package manifest...")

    with tempfile.TemporaryDirectory() as info_dir:
        if sys.platform == "win32":
            cmd = f'docker run --rm -a STDOUT -a STDERR -v "{info_dir}":/var/run/monai/export/config {map_name}'
        else:
            cmd = f"""docker_id=$(docker create {map_name})
docker cp $docker_id:/etc/monai/app.json "{info_dir}/app.json"
docker cp $docker_id:/etc/monai/pkg.json "{info_dir}/pkg.json"
docker rm -v $docker_id > /dev/null
"""
        returncode = run_cmd(cmd)
        if returncode != 0:
            return {}, {}, returncode

        app_json = Path(f"{info_dir}/app.json")
        pkg_json = Path(f"{info_dir}/pkg.json")

        logger.debug("-------------------application manifest-------------------")
        logger.debug(app_json.read_text())
        logger.debug("----------------------------------------------\n")

        logger.debug("-------------------package manifest-------------------")
        logger.debug(pkg_json.read_text())
        logger.debug("----------------------------------------------\n")

        app_info = json.loads(app_json.read_text())
        pkg_info = json.loads(pkg_json.read_text())
        return app_info, pkg_info, returncode


def run_app(map_name: str, input_path: Path, output_path: Path, app_info: dict, pkg_info: dict, quiet: bool) -> int:
    """
    Executes the MONAI Application.

    Args:
        map_name: MONAI Application Package
        input_path: input directory path
        output_path: output directory path
        app_info: application manifest dictionary
        pkg_info: package manifest dictionary
        quiet: boolean flag indicating quiet mode

    Returns:
        returncode: command returncode
    """
    cmd = "docker run --rm -a STDERR"

    # Use nvidia-docker if GPU resources are requested
    requested_gpus = get_requested_gpus(pkg_info)
    if requested_gpus > 0:
        cmd = "nvidia-docker run --rm -a STDERR"

    if not quiet:
        cmd += " -a STDOUT"

    # Use POSIX path for input and output paths as local paths are mounted to those paths in the container.
    map_input = Path(app_info["input"]["path"]).as_posix()
    map_output = Path(app_info["output"]["path"]).as_posix()
    if not posixpath.isabs(map_input):
        map_input = posixpath.join(app_info["working-directory"], map_input)

    if not posixpath.isabs(map_output):
        map_output = posixpath.join(app_info["working-directory"], map_output)

    cmd += f' -e MONAI_INPUTPATH="{map_input}"'
    cmd += f' -e MONAI_OUTPUTPATH="{map_output}"'
    # TODO(bhatt-piyush): Handle model environment correctly (maybe solved by fixing 'monai-exec')
    cmd += " -e MONAI_MODELPATH=/opt/monai/models"

    map_command = app_info["command"]
    # TODO(bhatt-piyush): Fix 'monai-exec' to work correctly.
    cmd += ' -v "{}":"{}" -v "{}":"{}" --shm-size=1g --entrypoint "/bin/bash" "{}" -c "{}"'.format(
        input_path.absolute(), map_input, output_path.absolute(), map_output, map_name, map_command
    )
    # cmd += " -v {}:{} -v {}:{} {}".format(
    #     input_path.absolute(), map_input, output_path.absolute(), map_output, map_name
    # )

    return run_cmd(cmd)


def dependency_verification(map_name: str) -> bool:
    """Check if all the dependencies are installed or not.

    Args:
        map_name: MONAI Application Package

    Returns:
        True if all dependencies are satisfied, otherwise False.
    """
    logger.info("Checking dependencies...")

    # check for docker
    prog = "docker"
    logger.info('--> Verifying if "%s" is installed...\n', prog)
    if not shutil.which(prog):
        logger.error('ERROR: "%s" not installed, please install docker.', prog)
        return False

    # check for map image
    logger.info('--> Verifying if "%s" is available...\n', map_name)
    if not verify_image(map_name):
        logger.error("ERROR: Unable to fetch required image.")
        return False

    return True


def pkg_specific_dependency_verification(pkg_info: dict) -> bool:
    """Checks for any package specific dependencies.

    Currently it verifies the following dependencies:
    * If gpu has been requested by the application, verify that nvidia-docker is installed.

    Args:
        pkg_info: package manifest as a python dict

    Returns:
        True if all dependencies are satisfied, otherwise False.
    """
    requested_gpus = get_requested_gpus(pkg_info)
    if requested_gpus > 0:
        # check for nvidia-docker
        prog = "nvidia-docker"
        logger.info('--> Verifying if "%s" is installed...\n', prog)
        if not shutil.which(prog):
            logger.error('ERROR: "%s" not installed, please install nvidia-docker.', prog)
            return False

    return True


def main(args: argparse.Namespace):
    """
    Main entry function for MONAI Application Runner.

    Args:
        args: parsed arguments

    Returns:
        None
    """
    if not dependency_verification(args.map):
        logger.error("Execution Aborted")
        sys.exit(1)

    # Fetch application manifest from MAP
    app_info, pkg_info, returncode = fetch_map_manifest(args.map)
    if returncode != 0:
        logger.error("ERROR: Failed to fetch MAP manifest.")
        logger.error("Execution Aborted")
        sys.exit(1)

    if not pkg_specific_dependency_verification(pkg_info):
        logger.error("Execution Aborted")
        sys.exit(1)

    # Run MONAI Application
    returncode = run_app(args.map, args.input, args.output, app_info, pkg_info, quiet=args.quiet)

    if returncode != 0:
        logger.error('\nERROR: MONAI Application "%s" failed.', args.map)
        sys.exit(1)
