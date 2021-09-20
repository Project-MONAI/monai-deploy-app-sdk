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
from argparse import ArgumentParser, HelpFormatter, Namespace, _SubParsersAction
from typing import List

from monai.deploy.runner import runner
from monai.deploy.utils import argparse_types

logger = logging.getLogger("app_runner")


def create_run_parser(subparser: _SubParsersAction, command: str, parents: List[ArgumentParser]) -> ArgumentParser:
    parser = subparser.add_parser(command, formatter_class=HelpFormatter, parents=parents, add_help=False)

    parser.add_argument("map", metavar="<map-image[:tag]>", help="MAP image name")

    parser.add_argument(
        "input", metavar="<input>", type=argparse_types.valid_dir_path, help="Input data directory path"
    )

    parser.add_argument(
        "output", metavar="<output>", type=argparse_types.valid_dir_path, help="Output data directory path"
    )

    parser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        action="store_true",
        default=False,
        help="Suppress the STDOUT and print only STDERR from the application (default: False)",
    )

    return parser


def execute_run_command(args: Namespace):
    runner.main(args)
