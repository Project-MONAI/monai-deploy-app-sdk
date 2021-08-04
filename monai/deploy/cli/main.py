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
import logging.config
from pathlib import Path
from typing import List, Optional

# Specify available commands here to execute 'exec' command by default
# if no command is specified.
COMMAND_LIST = ["exec", "package", "run"]


def parse_args(argv: Optional[List[str]] = None, default_command: Optional[str] = None) -> argparse.Namespace:
    from monai.deploy.cli.exec_command import create_exec_parser
    from monai.deploy.packager.package_command import create_package_parser
    from monai.deploy.runner.run_command import create_run_parser

    if argv is None:
        import sys

        argv = sys.argv
    argv = list(argv)  # copy argv for manipulation to avoid side-effects

    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument('-l', '--log-level', dest='log_level', default='INFO', type=str.upper,
                               choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'],
                               help='Set the logging level')

    parser = argparse.ArgumentParser(parents=[parent_parser],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     add_help=False)

    subparser = parser.add_subparsers(dest="command")

    # Parser for `exec` command
    create_exec_parser(subparser, "exec", parents=[parent_parser])

    # Parser for `package` command
    create_package_parser(subparser, "package", parents=[parent_parser])

    # Parser for `run` command
    create_run_parser(subparser, "run", parents=[parent_parser])

    # By default, execute `exec` command
    command = argv[1:2]
    if default_command and (not command or command[0] not in COMMAND_LIST):
        argv.insert(1, default_command)  # insert default command

    args = parser.parse_args(argv[1:])
    args.argv = argv  # save argv for later use in runpy

    # Print help if no command is specified
    if args.command is None:
        parser.print_help()
        parser.exit()

    return args


def set_up_logging(level: str):
    """
    Initializes the logger and sets up logging level.

    Args:
        level: logging level (DEBUG, INFO, WARN, ERROR, CRITICAL)
    """
    default_log_dir = Path(__file__).absolute().parent.parent.parent.parent
    default_config = default_log_dir / "logging.json"
    with open(default_config, 'r') as f:
        config_dict = json.load(f)
    config_dict['root']['level'] = level
    logging.config.dictConfig(config_dict)


def main(argv: Optional[List[str]] = None, default_command: Optional[str] = None):
    args = parse_args(argv, default_command)

    set_up_logging(args.log_level)

    if args.command == "exec":
        from monai.deploy.cli.exec_command import execute_exec_command

        execute_exec_command(args)
    elif args.command == "package":
        from monai.deploy.packager.package_command import execute_package_command

        execute_package_command(args)
    elif args.command == "run":
        from monai.deploy.runner.run_command import execute_run_command

        execute_run_command(args)


if __name__ == "__main__":
    main()
