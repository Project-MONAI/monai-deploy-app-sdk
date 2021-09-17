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
import os
import runpy
import sys
from argparse import ArgumentParser, Namespace, _SubParsersAction
from typing import List

from monai.deploy.core.datastores.factory import DatastoreFactory
from monai.deploy.core.executors.factory import ExecutorFactory
from monai.deploy.core.graphs.factory import GraphFactory


def create_exec_parser(subparser: _SubParsersAction, command: str, parents: List[ArgumentParser]) -> ArgumentParser:
    # Intentionally use `argparse.HelpFormatter` instead of `argparse.ArgumentDefaultsHelpFormatter`.
    # Default values for those options are None and those would be filled by `RuntimeEnv` object later.
    parser = subparser.add_parser(command, formatter_class=argparse.HelpFormatter, parents=parents, add_help=False)

    parser.add_argument("--input", "-i", help="Path to input folder/file (default: input)")
    parser.add_argument("--output", "-o", help="Path to output folder (default: output)")
    parser.add_argument("--model", "-m", help="Path to model(s) folder/file (default: models)")
    parser.add_argument(
        "--workdir",
        "-w",
        type=str,
        help="Path to workspace folder (default: A temporary '.monai_workdir' folder in the current folder)",
    )
    parser.add_argument(
        "--graph",
        help=f"Set Graph engine (default: {GraphFactory.DEFAULT})",
        choices=GraphFactory.NAMES,
    )
    parser.add_argument(
        "--datastore",
        help=f"Set Datastore (default: {DatastoreFactory.DEFAULT})",
        choices=DatastoreFactory.NAMES,
    )
    parser.add_argument(
        "--executor",
        help=f"Set Executor (default: {ExecutorFactory.DEFAULT})",
        choices=ExecutorFactory.NAMES,
    )
    parser.add_argument("remaining", nargs="*")

    return parser


def execute_exec_command(args: Namespace):
    remaining = args.remaining

    if len(remaining) != 1:
        print("Missing command argument. Please provide an application path to execute.", file=sys.stderr)
        sys.exit(1)

    app_path = remaining[0]

    # Simulate executing 'python {app_path}'
    current_dir = os.path.abspath(os.path.dirname(app_path))
    if sys.path and os.path.abspath(sys.path[0]) != current_dir:
        sys.path.insert(0, current_dir)

    sys.argv.remove(app_path)  # remove the application path from sys.argv

    runpy.run_path(app_path, run_name="__main__")
