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
import os
import shutil
import tempfile

from monai.deploy.packager import util as packager_util


def test_packager():

    # Create mock inputs
    test_map_tag = "monaitest:latest"
    test_app = tempfile.mkdtemp(dir=".")
    test_models_dir = tempfile.mkdtemp(dir=test_app)
    test_model = tempfile.mkstemp(dir=test_models_dir, suffix=".ts")
    with open(test_model[1], "w") as test_model_file:
        test_model_file.write("Foo")
    test_params_file = tempfile.mkstemp(dir=".")

    test_params_values = {}
    test_params_values["app-name"] = "test_app"
    test_params_values["app-version"] = "0.0.0"
    test_params_values["sdk-version"] = "0.0.0"
    test_params_values["command"] = "/usr/bin/python3 -u /opt/monai/app/main.py"
    test_params_values["pip-packages"] = []
    test_params_values["models"] = []
    test_params_values["resource"] = {"cpu": 1, "gpu": 2, "memory": "4Gi"}

    test_params_string = json.dumps(test_params_values, sort_keys=True, indent=4, separators=(",", ": "))

    with open(test_params_file[1], "w") as tmp_params:
        tmp_params.write(test_params_string)

    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.application = test_app
    args.tag = test_map_tag
    args.params = test_params_file[1]
    args.base = None
    args.working_dir = None
    args.input_dir = None
    args.output_dir = None
    args.models_dir = None
    args.timeout = None
    args.model = None
    args.version = None
    args.verbose = False

    packager_util.package_application(args)

    shutil.rmtree(test_app)
    os.remove(test_params_file[1])
