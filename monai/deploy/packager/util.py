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

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

from jinja2 import Template

from monai.deploy.packager import executor

from . import *

logger = logging.getLogger(__name__)

def initialize_args(args):
    # Parse arguements
    args.docker_file_name = DEFAULT_DOCKER_FILE_NAME
    args.base_image = args.base if args.base else DEFAULT_BASE_IMAGE
    args.working_dir = args.working_dir if args.working_dir else DEFAULT_WORK_DIR
    args.app_dir = "app/"
    args.executor_dir = "executor/"
    args.input_dir = args.input if args.input_dir else DEFAULT_INPUT_DIR
    args.output_dir = args.output if args.output_dir else DEFAULT_OUTPUT_DIR
    args.models_dir = args.models if args.models_dir else DEFAULT_MODELS_DIR
    args.api_version = DEFAULT_API_VERSION
    args.timeout = args.timeout if args.timeout else DEFAULT_TIMEOUT
    args.version = args.version if args.version else DEFAULT_VERSION
    
    # TEMPORARY parameter in place of SDK provided values
    with open(args.params, 'r') as file:
        args.param_values = json.loads(file.read())

    return args

def build_image(args, temp_dir):
    # Parse arguements for dockerfile
    tag = args.tag
    verbose = args.verbose
    docker_file_name = args.docker_file_name
    base_image = args.base_image
    working_dir = args.working_dir
    app_dir = args.app_dir
    executor_dir = args.executor_dir
    input_dir = args.input_dir
    output_dir = args.output_dir
    models_dir = args.models_dir

    app_version = args.param_values['app-version']
    sdk_version = args.param_values['sdk-version']
    local_models = args.param_values['models']
    
    # Dockerfile template
    docker_template_string = """FROM {{base_image}}

LABEL base="{{base_image}}"
LABEL tag="{{name}}"
LABEL version="{{app_version}}"
LABEL sdk_version="{{sdk_version}}"

ENV DEBIAN_FRONTEND=noninteractive

ENV TERM=xterm-256color
ENV MONAI_INPUTPATH={{working_dir + input_dir}}
ENV MONAI_OUTPUTPATH={{working_dir + output_dir}}

RUN apt update \\
 && apt upgrade -y --no-install-recommends \\
 && apt install -y --no-install-recommends \\
    build-essential \\
    python3 \\
    python3-pip \\
    python3-setuptools \\
    curl \\
 && apt autoremove -y \\
 && rm -f /usr/bin/python /usr/bin/pip \\
 && ln -s /usr/bin/python3 /usr/bin/python \\
 && ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install --no-cache-dir --upgrade setuptools==57.4.0 pip==21.2.3 wheel==0.37.0

RUN adduser monai

RUN mkdir -p /etc/monai/ && chown -R monai:monai /etc/monai
RUN mkdir -p {{working_dir}} && chown -R monai:monai {{working_dir}}
RUN mkdir -p {{working_dir + app_dir}} && chown -R monai:monai {{working_dir + app_dir}}
RUN mkdir -p {{working_dir + executor_dir}} && chown -R monai:monai {{working_dir + executor_dir}}
RUN mkdir -p {{working_dir + input_dir}} && chown -R monai:monai {{working_dir + input_dir}}
RUN mkdir -p {{working_dir + output_dir}} && chown -R monai:monai {{working_dir + output_dir}}
RUN mkdir -p {{working_dir + models_dir}} && chown -R monai:monai {{working_dir + models_dir}}

COPY --chown=monai:monai {{app_folder}} /opt/monai/app/
{% for model_entry in local_models %}
RUN mkdir -p {{models_dir + model_entry.name + "/"}} && chown -R monai:monai {{models_dir + model_entry.name + "/"}}
COPY --chown=monai:monai {{model_entry.path}} {{models_dir + model_entry.name + "/"}}
{% endfor %}

RUN pip install -r /opt/monai/app/requirements.txt

COPY --chown=monai:monai ./.tmp/app.json /etc/monai/
COPY --chown=monai:monai ./.tmp/pkg.json /etc/monai/
COPY --chown=monai:monai ./.tmp/monai-exec /opt/monai/executor/

ENTRYPOINT [ "/opt/monai/executor/monai-exec" ]
    
"""

    docker_template = Template(docker_template_string)
    args = {
            'base_image': base_image,
            'tag': tag,
            'app_version': app_version,
            'sdk_version': sdk_version,
            'app_folder': args.application,
            'working_dir': working_dir,
            'app_dir': app_dir,
            'executor_dir': executor_dir,
            'input_dir': input_dir,
            'output_dir': output_dir,
            'models_dir': models_dir,
            'local_models': local_models
        }

    # Write out dockerfile
    docker_parsed_template = docker_template.render(**args)
    docker_file_path = temp_dir + docker_file_name

    print(docker_parsed_template)

    with open(docker_file_path, "w") as docker_file:
        docker_file.write(docker_parsed_template)

    # Build dockerfile into an MAP image
    docker_build_cmd = ['docker', 'build', '-f', docker_file_path, '-t',  f'{tag}', '.']
    proc = subprocess.Popen(docker_build_cmd, stdout=subprocess.PIPE)

    def build_spinning_wheel():
        while True:
            for cursor in '|/-\\':
                yield cursor
    
    spinner = build_spinning_wheel()

    print('Building MONAI Application Package... ')

    while proc.poll()==None:

        if verbose:
            logger.debug(proc.stdout.readline().decode('utf-8'),end='')
        else:
            time.sleep(0.2)

        sys.stdout.write(next(spinner))
        sys.stdout.flush()
        sys.stdout.write('\b')
        sys.stdout.write('\b')

    return_code = proc.returncode

    if return_code == 0:
        print(f'Successfully Built {tag}')

def create_app_manifest(args, temp_dir):
    input_dir = args.input_dir
    output_dir = args.output_dir
    working_dir = args.working_dir
    api_version = args.api_version
    app_version = args.version
    timeout = args.timeout

    command = args.param_values['command']
    sdk_version = args.param_values['sdk-version']
    environment = args.param_values['environment'] if 'environment' in args.param_values \
                else {}

    app_manifest = {}
    app_manifest["api_version"] = api_version
    app_manifest["sdk_version"] = sdk_version
    app_manifest["command"] = command
    app_manifest["environment"] = environment
    app_manifest["working-directory"] = working_dir
    app_manifest["input"] = {}
    app_manifest["input"]["path"] = input_dir
    app_manifest["input"]["path_env"] = "MONAI_INPUTPATH"
    app_manifest["input"]["formats"] = []
    app_manifest["output"] = {}
    app_manifest["output"]["path"] = output_dir
    app_manifest["output"]["path_env"] = "MONAI_OUTPUTPATH"
    app_manifest["output"]["format"] = {}
    app_manifest["version"] = app_version
    app_manifest["timeout"] = timeout

    app_manifest_string = json.dumps(app_manifest, 
                                     sort_keys=True,
                                     indent=4,
                                     separators=(',', ': '))

    with open(temp_dir + "app.json", "w") as app_manifest_file:
        app_manifest_file.write(app_manifest_string)

def create_package_manifest(args, temp_dir):
    models_dir = args.models_dir
    working_dir = args.working_dir
    api_version = args.api_version
    app_version = args.version
    
    sdk_version = args.param_values['sdk-version']
    cpu = args.param_values['resource']['cpu']
    gpu = args.param_values['resource']['gpu']
    memory = args.param_values['resource']['memory']
    models = args.param_values['models']

    package_manifest = {}
    package_manifest["api-version"] = api_version
    package_manifest["sdk-version"] = sdk_version
    package_manifest["application-root"] = working_dir
    package_manifest["models"] = []

    for model_entry in models:
        model_name = model_entry["name"]
        model_file = os.path.basename(model_entry["path"])
        model_path = working_dir + models_dir + model_file
        package_manifest["models"].append({"name": model_name,
                                           "path": model_path})

    package_manifest["resources"] = {}
    package_manifest["resources"]["cpu"] = cpu
    package_manifest["resources"]["gpu"] = gpu
    package_manifest["resources"]["memory"] = memory
    package_manifest["version"] = app_version

    package_manifest_string = json.dumps(package_manifest, 
                                     sort_keys=True,
                                     indent=4,
                                     separators=(',', ': '))

    with open(temp_dir + "pkg.json", "x") as package_manifest_file:
        package_manifest_file.write(package_manifest_string)

def package_application(args):
    with tempfile.TemporaryDirectory(prefix="monai_tmp") as temp_dir:
        # Copies executor into temporary directory
        shutil.copy(os.path.join(os.path.dirname(executor.__file__), "monai-exec"), temp_dir)

        # Initialize arguements for package
        args = initialize_args(args)

        # Create Manifest Files
        create_app_manifest(args, temp_dir)
        create_package_manifest(args, temp_dir)

        build_image(args, temp_dir)
    