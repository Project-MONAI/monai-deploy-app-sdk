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
import logging.config
from pathlib import Path
from typing import List, Optional, Union

# Specify available commands here to execute 'exec' command by default
# if no command is specified.
COMMAND_LIST = ["exec", "package", "run"]

LOG_CONFIG_FILENAME = "logging.json"


def parse_args(argv: Optional[List[str]] = None, default_command: Optional[str] = None) -> argparse.Namespace:
    from monai.deploy.cli.exec_command import create_exec_parser
    from monai.deploy.packager.package_command import create_package_parser
    from monai.deploy.runner.run_command import create_run_parser

    if argv is None:
        import sys

        argv = sys.argv
    argv = list(argv)  # copy argv for manipulation to avoid side-effects

    # We have intentionally not set the default using `default="INFO"` here so that the default
    # value from here doesn't override the value in `LOG_CONFIG_FILENAME` unless the user indends to do
    # so. If the user doesn't use this flag to set log level, this argument is set to "None"
    # and the logging level specified in `LOG_CONFIG_FILENAME` is used.
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )

    parser = argparse.ArgumentParser(
        parents=[parent_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False
    )

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


def set_up_logging(level: Optional[str], config_path: Union[str, Path] = LOG_CONFIG_FILENAME):
    """Initializes the logger and sets up logging level.

    Args:
        level (str): A logging level (DEBUG, INFO, WARN, ERROR, CRITICAL).
        log_config_path (str): A path to logging config file.
    """
    # Default log config path
    log_config_path = Path(__file__).absolute().parent.parent / LOG_CONFIG_FILENAME

    config_path = Path(config_path)

    # If a logging config file that is specified by `log_config_path` exists in the current folder,
    # it overrides the default one
    if config_path.exists():
        log_config_path = config_path

    config_dict = json.loads(log_config_path.read_bytes())

    if level is not None and "root" in config_dict:
        config_dict["root"]["level"] = level
    logging.config.dictConfig(config_dict)


def main(argv: Optional[List[str]] = None, default_command: Optional[str] = None):
    args = parse_args(argv, default_command)

    # Set up logging level if the command is not `exec`
    # (`exec` command sets up the logging in the constructor of Application class)
    if args.command != "exec":
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
