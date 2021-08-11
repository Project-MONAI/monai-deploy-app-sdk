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

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Tuple

from monai.deploy.runner.utils import run_cmd, verify_image

logger = logging.getLogger("app_runner")

def fetch_map_manifest(map_name: str) -> Tuple[dict, int]:
    """
    Execute MONAI Application Package and fetch application manifest.

    Args:
        map_name: MAP image name.

    Returns:
        app_info: application manifest as a python dict.
        returncode: command return code
    """
    logger.info("\nReading MONAI App Package manifest...")

    with tempfile.TemporaryDirectory() as info_dir:
        cmd = f'docker run --rm -it -v {info_dir}:/var/run/monai/export/config {map_name}'

        returncode = run_cmd(cmd)
        if returncode != 0:
            return {}, returncode

        app_json = Path(f'{info_dir}/app.json')
        pkg_json = Path(f'{info_dir}/pkg.json')

        logger.debug("-------------------application manifest-------------------")
        logger.debug(app_json.read_text())
        logger.debug("----------------------------------------------\n")

        logger.debug("-------------------package manifest-------------------")
        logger.debug(pkg_json.read_text())
        logger.debug("----------------------------------------------\n")

        app_info = json.loads(app_json.read_text())
        return app_info, returncode


def run_app(map_name: str, input_dir: Path, output_dir: Path, app_info: dict, quiet: bool) -> int:
    """
    Executes the MONAI Application.

    Args:
        map_name: MONAI Application Package
        input_dir: input directory path
        output_dir: output directory path
        app_info: application manifest dictionary
        quiet: boolean flag indicating quiet mode

    Returns:
        returncode: command returncode
    """
    cmd = 'docker run --rm -a STDERR'

    if not quiet:
        cmd += ' -a STDOUT'

    cmd += ' -v {}:{} -v {}:{} {}'.format(input_dir, app_info['input']['path'],
                                                    output_dir, app_info['output']['path'],
                                                    map_name)

    if quiet:
        return run_cmd(cmd)

    return run_cmd(cmd)


def dependency_verification(map_name: str) -> bool:
    """Check if all the dependencies are installed or not.

    Args:
        None.

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
        logger.error('ERROR: Unable to fetch required image.')
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
        logger.error("Aborting...")
        sys.exit()

    # Fetch application manifest from MAP
    app_info, returncode = fetch_map_manifest(args.map)
    if returncode != 0:
        logger.error("ERROR: Failed to fetch MAP manifest. Aborting...")
        sys.exit()

    # Run MONAI Application
    returncode = run_app(args.map, args.input, args.output, app_info, quiet=args.quiet)

    if returncode != 0:
        logger.error('\nERROR: MONAI Application "%s" failed.', args.map)
        sys.exit()
