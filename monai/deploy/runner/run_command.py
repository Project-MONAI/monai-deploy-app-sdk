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
import sys
from argparse import ArgumentParser, Namespace, _SubParsersAction
from typing import List


def create_run_parser(subparser: _SubParsersAction, command: str, parents: List[ArgumentParser]) -> ArgumentParser:
    parser = subparser.add_parser(command, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  parents=parents, add_help=False)

    parser.add_argument("map", metavar="<map-image[:tag]>", help="MAP image name")
    parser.add_argument("--input", "-i", metavar="<input_dir>", help="Path to input folder/file")

    return parser


def execute_run_command(args: Namespace):
    if not args.input:  # if input is empty
        print("Missing tag name. Use --input=<input_dir>", file=sys.stderr)
        sys.exit(1)
