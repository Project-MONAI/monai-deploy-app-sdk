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
from typing import List, Optional

COMMAND_LIST = ["run"]


def parse_args(argv: Optional[List[str]] = None, default_command: Optional[str] = None) -> argparse.Namespace:
    from monai.deploy.cli.run_command import create_run_parser

    if argv is None:
        import sys

        argv = sys.argv
    argv = list(argv)  # copy argv for manipulation to avoid side-effects

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command")

    # Parser for `run` command
    create_run_parser(subparser, "run")

    # By default, execute `run` command
    command = argv[1:2]
    if default_command and (not command or command[0] not in COMMAND_LIST):
        argv.insert(1, default_command)  # insert default command

    args = parser.parse_args(argv[1:])
    args.argv = argv  # save argv for later use in runpy

    return args


def main(argv: Optional[List[str]] = None, default_command: Optional[str] = None):
    args = parse_args(argv, default_command)

    if args.command == "run":
        from monai.deploy.cli.run_command import execute_run_command

        execute_run_command(args)


if __name__ == "__main__":
    main()
