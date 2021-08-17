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
from argparse import Namespace
from typing import Dict

from jinja2 import Template

from monai.deploy.packager import executor
from monai.deploy.utils.importutil import get_application

from . import *

logger = logging.getLogger("app_packager")


def initialize_args(args: Namespace) -> Dict:
    """Processes and formats input arguements for Packager

    Args:
        args (Namespace): Input arguements for Packager from CLI

    Returns:
        Dict: Processed set of input arguements for Packager
    """
    processed_args = dict()

    # Parse arguements and set default values if any are missing
    processed_args['application'] = args.application
    processed_args['tag'] = args.tag
    processed_args['docker_file_name'] = DEFAULT_DOCKER_FILE_NAME
    processed_args['base_image'] = args.base if args.base else DEFAULT_BASE_IMAGE
    processed_args['working_dir'] = args.working_dir if args.working_dir else DEFAULT_WORK_DIR
    processed_args['app_dir'] = "/opt/monai/app/"
    processed_args['executor_dir'] = "/opt/monai/executor/"
    processed_args['input_dir'] = args.input if args.input_dir else DEFAULT_INPUT_DIR
    processed_args['output_dir'] = args.output if args.output_dir else DEFAULT_OUTPUT_DIR
    processed_args['models_dir'] = args.models if args.models_dir else DEFAULT_MODELS_DIR
    processed_args['api_version'] = DEFAULT_API_VERSION
    processed_args['timeout'] = args.timeout if args.timeout else DEFAULT_TIMEOUT
    processed_args['version'] = args.version if args.version else DEFAULT_VERSION

    # Obtain SDK provide application values
    app_obj = get_application(args.application)
    processed_args['application_info'] = app_obj.get_package_info(args.model)

    return processed_args


def build_image(args: dict, temp_dir: str):
    """Creates dockerfile and builds MONAI Application Package (MAP) image

    Args:
        args (dict): Input arguements for Packager
        temp_dir (str): Temporary directory to build MAP
    """
    # Parse arguements for dockerfile
    tag = args['tag']
    docker_file_name = args['docker_file_name']
    base_image = args['base_image']
    working_dir = args['working_dir']
    app_dir = args['app_dir']
    executor_dir = args['executor_dir']
    input_dir = args['input_dir']
    full_input_path = os.path.join(working_dir, input_dir)
    output_dir = args['output_dir']
    full_output_path = os.path.join(working_dir, output_dir)
    models_dir = args['models_dir']
    timeout = args['timeout']
    application_path = args['application']

    app_version = args['application_info']['app-version']
    sdk_version = args['application_info']['sdk-version']
    local_models = args['application_info']['models']
    pip_packages = args['application_info']['pip-packages']
    pip_requirements_path = os.path.join(temp_dir, "requirements.txt")
    with open(pip_requirements_path,"w") as requirements_file:
        requirements_file.writelines(pip_packages)
    map_requirements_path = os.path.join(app_dir, "requirements.txt")

    models_string = ""
    for model_entry in local_models:
        container_models_folder = os.join.path(models_dir, model_entry.name)
        models_string += f'RUN mkdir -p {container_models_folder} && \
            chown -R monai:monai {container_models_folder}\n'
        models_string += f'COPY --chown=monai:monai {model_entry.path} \
            {container_models_folder}\n'

    # Dockerfile template
    docker_template_string = f' \
    FROM {base_image}\n\n \
    LABEL base=\"{base_image}\"\n \
    LABEL tag=\"{tag}\"\n \
    LABEL version=\"{app_version}\"\n \
    LABEL sdk_version=\"{sdk_version}\"\n\n \
    ENV DEBIAN_FRONTEND=noninteractive\n \
    ENV TERM=xterm-256color\n \
    ENV MONAI_INPUTPATH={full_input_path}\n \
    ENV MONAI_OUTPUTPATH={full_output_path}\n \
    ENV MONAI_APPLICATION={app_dir}\n \
    ENV MONAI_TIMEOUT={timeout}\n\n \
    RUN apt update \\\n \
     && apt upgrade -y --no-install-recommends \\\n \
     && apt install -y --no-install-recommends \\\n \
        build-essential \\\n \
        python3 \\\n \
        python3-pip \\\n \
        python3-setuptools \\\n \
        curl \\\n \
     && apt autoremove -y \\\n \
     && rm -rf /var/lib/apt/lists/* \\\n \
     && rm -f /usr/bin/python /usr/bin/pip \\\n \
     && ln -s /usr/bin/python3 /usr/bin/python \\\n \
     && ln -s /usr/bin/pip3 /usr/bin/pip\n\n \
    RUN pip install --no-cache-dir --upgrade setuptools==57.4.0 pip==21.2.3 wheel==0.37.0\n\n \
    RUN adduser monai\n\n \
    RUN mkdir -p /etc/monai/ && chown -R monai:monai /etc/monai\n \
    RUN mkdir -p /opt/monai/ && chown -R monai:monai /opt/monai\n \
    RUN mkdir -p {working_dir} && chown -R monai:monai {working_dir}\n \
    RUN mkdir -p {app_dir} && chown -R monai:monai {app_dir}\n \
    RUN mkdir -p {executor_dir} && chown -R monai:monai {executor_dir}\n \
    RUN mkdir -p {full_input_path} && chown -R monai:monai {full_input_path}\n \
    RUN mkdir -p {full_output_path} && chown -R monai:monai {full_output_path}\n \
    RUN mkdir -p {models_dir} && chown -R monai:monai {models_dir}\n\n \
    COPY --chown=monai:monai {application_path} {app_dir}\n\n \
    {models_string}\n\n \
    COPY {pip_requirements_path} {map_requirements_path}\n \
    RUN pip install --no-cache-dir --upgrade -r {map_requirements_path}\n\n \
    COPY --chown=monai:monai {temp_dir}/app.json /etc/monai/\n \
    COPY --chown=monai:monai {temp_dir}/pkg.json /etc/monai/\n \
    COPY --chown=monai:monai {temp_dir}/monai-exec /opt/monai/executor/\n \
    ENTRYPOINT [ "/opt/monai/executor/monai-exec" ]\n\n'

    # Write out dockerfile
    logger.debug(docker_template_string)
    docker_file_path = os.path.join(temp_dir, docker_file_name)
    with open(docker_file_path, "w") as docker_file:
        docker_file.write(docker_template_string)

    # Build dockerfile into an MAP image
    docker_build_cmd = ['docker', 'build', '-f', docker_file_path, '-t', tag, '.']
    proc = subprocess.Popen(docker_build_cmd, stdout=subprocess.PIPE)

    def build_spinning_wheel():
        while True:
            for cursor in '|/-\\':
                yield cursor

    spinner = build_spinning_wheel()

    print('Building MONAI Application Package... ')

    while proc.poll() is None:
        logger.debug(proc.stdout.readline().decode('utf-8'))
        sys.stdout.write(next(spinner))
        sys.stdout.flush()
        sys.stdout.write('\b')
        sys.stdout.write('\b')

    return_code = proc.returncode

    if return_code == 0:
        print(f'Successfully Built {tag}')


def create_app_manifest(args: Dict, temp_dir: str):
    """Creates Application manifest .json file

    Args:
        args (Dict): Input arguements for Packager
        temp_dir (str): Temporary directory to build MAP
    """
    input_dir = args['input_dir']
    output_dir = args['output_dir']
    working_dir = args['working_dir']
    api_version = args['api_version']
    app_version = args['version']
    timeout = args['timeout']

    command = args['application_info']['command']
    sdk_version = args['application_info']['sdk-version']
    environment = args['application_info']['environment'] if 'environment' \
        in  args['application_info'] else {}

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

    with open(os.path.join(temp_dir, "app.json"), "w") as app_manifest_file:
        app_manifest_file.write(app_manifest_string)


def create_package_manifest(args: Dict, temp_dir: str):
    """Creates package manifest .json file

    Args:
        args (Dict): Input arguements for Packager
        temp_dir (str): Temporary directory to build MAP
    """
    models_dir = args['models_dir']
    working_dir = args['working_dir']
    api_version = args['api_version']
    app_version = args['version']

    sdk_version = args['application_info']['sdk-version']
    cpu = args['application_info']['resource']['cpu']
    gpu = args['application_info']['resource']['gpu']
    memory = args['application_info']['resource']['memory']
    models = args['application_info']['models']

    package_manifest = {}
    package_manifest["api-version"] = api_version
    package_manifest["sdk-version"] = sdk_version
    package_manifest["application-root"] = working_dir
    package_manifest["models"] = []

    for model_entry in models:
        model_name = model_entry["name"]
        model_file = os.path.basename(model_entry["path"])
        model_path = os.path.join(models_dir, model_name, model_file)
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

    with open(os.path.join(temp_dir, "pkg.json"), "w") as package_manifest_file:
        package_manifest_file.write(package_manifest_string)


def package_application(args: Namespace):
    """Driver function for invoking all functions for creating and
    building the MONAI Application package image

    Args:
        args (Namespace): Input arguements for Packager from CLI
    """
    # Initialize arguements for package
    initialized_args = initialize_args(args)

    with tempfile.TemporaryDirectory(prefix="monai_tmp", dir=".") as temp_dir:
        # Copies executor into temporary directory
        shutil.copy(os.path.join(os.path.dirname(executor.__file__), "monai-exec"), temp_dir)

        # Create Manifest Files
        create_app_manifest(initialized_args, temp_dir)
        create_package_manifest(initialized_args, temp_dir)

        # Build MONAI Application Package image
        build_image(initialized_args, temp_dir)
