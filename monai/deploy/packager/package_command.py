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
from argparse import ArgumentParser, Namespace, _SubParsersAction
from typing import List

from monai.deploy.packager import util as packager_util


def create_package_parser(subparser: _SubParsersAction, command: str, parents: List[ArgumentParser]) -> ArgumentParser:
    parser = subparser.add_parser(command, formatter_class=argparse.HelpFormatter, parents=parents, add_help=False)

    parser.add_argument("application", type=str, help="MONAI application path")
    parser.add_argument(
        "--tag",
        "-t",
        required=True,
        type=str,
        help="MONAI Application Package (MAP) name and optionally a tag in the 'name:tag' format.",
    )
    parser.add_argument(
        "--model", "-m", type=str, help="Path to a model file or a directory containing all models as subdirectories"
    )
    parser.add_argument("--base", "-b", type=str, help="Base Docker Image")
    parser.add_argument("--input-dir", "-i", type=str, help="Directory mounted in container for Application Input")
    parser.add_argument("--models-dir", type=str, help="Directory mounted in container for Models Path")
    parser.add_argument("--output-dir", "-o", type=str, help="Directory mounted in container for Application Output")
    parser.add_argument("--working-dir", "-w", type=str, help="Directory mounted in container for Application")
    parser.add_argument("--no-cache", "-n", action="store_true", help="Do not use cache when building image")
    parser.add_argument(
        "--requirements",
        "-r",
        type=str,
        help="Optional path to requirements.txt containing package dependencies of the application",
    )
    parser.add_argument("--timeout", type=str, help="Timeout")
    parser.add_argument("--version", type=str, help="Set the version of the Application")

    return parser


def execute_package_command(args: Namespace):
    packager_util.package_application(args)
