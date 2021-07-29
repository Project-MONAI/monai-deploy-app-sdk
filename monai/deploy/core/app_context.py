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

import sys

from monai.deploy.cli.main import parse_args


class AppContext:
    """A class to store the context of an application."""

    def __init__(self):
        # Parse the command line arguments
        argv = sys.argv
        args = parse_args(argv, default_command="run")

        self.graph = args.graph
        self.datastore = args.datastore
        self.executor = args.executor

        self.input_path = args.input or "input"
        self.output_path = args.output or "output"
        self.model_path = args.model or "models"
        print(args)

        # input/output/model paths
        # self.app_id = app_id
        # self.app_name = app_name
        # self.app_version = app_version
        # self.app_description = app_description
        # self.app_config = {}
        # self.app_config_path = None
