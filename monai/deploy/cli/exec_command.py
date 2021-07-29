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

import os
import runpy
import sys
from argparse import ArgumentParser, Namespace, _SubParsersAction

from monai.deploy.core.datastores.factory import DatastoreFactory
from monai.deploy.core.executors.factory import ExecutorFactory
from monai.deploy.core.graphs.factory import GraphFactory


def create_exec_parser(subparser: _SubParsersAction, command: str) -> ArgumentParser:
    parser = subparser.add_parser(command)

    parser.add_argument("--input", "-i", help="Path to input folder/file")
    parser.add_argument("--output", "-o", help="Path to output folder/file")
    parser.add_argument("--model", "-m", help="Path to model folder/file")
    parser.add_argument(
        "--graph",
        help=f"Graph engine (default: {GraphFactory.DEFAULT})",
        choices=GraphFactory.NAMES,
        default=GraphFactory.DEFAULT,
    )
    parser.add_argument(
        "--datastore",
        help=f"Datastore (default: {DatastoreFactory.DEFAULT})",
        choices=DatastoreFactory.NAMES,
        default=DatastoreFactory.DEFAULT,
    )
    parser.add_argument(
        "--executor",
        help=f"Executor (default: {ExecutorFactory.DEFAULT})",
        choices=ExecutorFactory.NAMES,
        default=ExecutorFactory.DEFAULT,
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
