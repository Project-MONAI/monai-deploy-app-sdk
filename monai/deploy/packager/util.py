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

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from argparse import Namespace
from typing import Dict

from monai.deploy.exceptions import WrongValueError
from monai.deploy.packager.constants import DefaultValues
from monai.deploy.packager.templates import Template
from monai.deploy.utils.fileutil import checksum
from monai.deploy.utils.importutil import dist_module_path, dist_requires, get_application
from monai.deploy.utils.spinner import ProgressSpinner

logger = logging.getLogger("app_packager")

executor_url = "https://globalcdn.nuget.org/packages/monai.deploy.executor.0.1.0-prealpha.0.nupkg"


def verify_base_image(base_image: str) -> str:
    """Helper function which validates whether valid base image passed to Packager.
    Additionally, this function provides the string identifier of the dockerfile
    template to build MAP
    Args:
        base_image (str): potential base image to build MAP Docker image
    Returns:
        str: returns string identifier of the dockerfile template to build MAP
        if valid base image provided, returns empty string otherwise
    """
    valid_prefixes = {"nvcr.io/nvidia/cuda": "ubuntu", "nvcr.io/nvidia/pytorch": "pytorch"}

    for prefix, template in valid_prefixes.items():
        if prefix in base_image:
            return template

    return ""


def initialize_args(args: Namespace) -> Dict:
    """Processes and formats input arguements for Packager
    Args:
        args (Namespace): Input arguements for Packager from CLI
    Returns:
        Dict: Processed set of input arguements for Packager
    """
    processed_args = dict()

    # Parse arguements and set default values if any are missing
    processed_args["application"] = args.application
    processed_args["tag"] = args.tag
    processed_args["docker_file_name"] = DefaultValues.DOCKER_FILE_NAME
    processed_args["working_dir"] = args.working_dir if args.working_dir else DefaultValues.WORK_DIR
    processed_args["app_dir"] = "/opt/monai/app"
    processed_args["executor_dir"] = "/opt/monai/executor"
    processed_args["input_dir"] = args.input if args.input_dir else DefaultValues.INPUT_DIR
    processed_args["output_dir"] = args.output if args.output_dir else DefaultValues.OUTPUT_DIR
    processed_args["models_dir"] = args.models if args.models_dir else DefaultValues.MODELS_DIR
    processed_args["no_cache"] = args.no_cache
    processed_args["timeout"] = args.timeout if args.timeout else DefaultValues.TIMEOUT
    processed_args["api-version"] = DefaultValues.API_VERSION
    processed_args["requirements"] = ""

    if args.requirements:
        if not args.requirements.endswith(".txt"):
            logger.error(
                f"Improper path to requirements.txt provided: {args.requirements}, defaulting to sdk provided values"
            )
        else:
            processed_args["requirements"] = args.requirements

    # Verify proper base image:
    dockerfile_type = ""

    if args.base:
        dockerfile_type = verify_base_image(args.base)
        if not dockerfile_type:
            logger.error(
                "Provided base image '{}' is not supported \n \
                          Please provide a Cuda or Pytorch image from https://ngc.nvidia.com/ (nvcr.io/nvidia)".format(
                    args.base
                )
            )
            sys.exit(1)

    processed_args["dockerfile_type"] = dockerfile_type if args.base else DefaultValues.DOCKERFILE_TYPE

    base_image = DefaultValues.BASE_IMAGE
    if args.base:
        base_image = args.base
    else:
        base_image = os.getenv("MONAI_BASEIMAGE", DefaultValues.BASE_IMAGE)

    processed_args["base_image"] = base_image

    # Obtain SDK provide application values
    app_obj = get_application(args.application)
    if app_obj:
        processed_args["application_info"] = app_obj.get_package_info(args.model)
    else:
        raise WrongValueError("Application from '{}' not found".format(args.application))

    # Use version number if provided through CLI, otherwise use value provided by SDK
    processed_args["version"] = args.version if args.version else processed_args["application_info"]["app-version"]

    return processed_args


def build_image(args: dict, temp_dir: str):
    """Creates dockerfile and builds MONAI Application Package (MAP) image
    Args:
        args (dict): Input arguements for Packager
        temp_dir (str): Temporary directory to build MAP
    """
    # Parse arguements for dockerfile
    tag = args["tag"]
    docker_file_name = args["docker_file_name"]
    base_image = args["base_image"]
    dockerfile_type = args["dockerfile_type"]
    working_dir = args["working_dir"]
    app_dir = args["app_dir"]
    executor_dir = args["executor_dir"]
    input_dir = args["input_dir"]
    full_input_path = os.path.join(working_dir, input_dir)
    output_dir = args["output_dir"]
    full_output_path = os.path.join(working_dir, output_dir)
    models_dir = args["models_dir"]
    timeout = args["timeout"]
    application_path = args["application"]
    local_requirements_file = args["requirements"]
    no_cache = args["no_cache"]
    app_version = args["version"]

    # Copy application files to temp directory (under 'app' folder)
    target_application_path = os.path.join(temp_dir, "app")
    if os.path.isfile(application_path):
        os.makedirs(target_application_path, exist_ok=True)
        shutil.copy(application_path, target_application_path)
    else:
        shutil.copytree(application_path, target_application_path)

    # Copy monai-app-sdk module to temp directory (under 'monai-deploy-app-sdk' folder)
    monai_app_sdk_path = os.path.join(dist_module_path("monai-deploy-app-sdk"), "monai", "deploy")
    target_monai_app_sdk_path = os.path.join(temp_dir, "monai-deploy-app-sdk")
    shutil.copytree(monai_app_sdk_path, target_monai_app_sdk_path)

    # Parse SDK provided values
    sdk_version = args["application_info"]["sdk-version"]
    local_models = args["application_info"]["models"]
    pip_packages = args["application_info"]["pip-packages"]

    # Append required packages for SDK to pip_packages
    monai_app_sdk_requires = dist_requires("monai-deploy-app-sdk")
    pip_packages.extend(monai_app_sdk_requires)

    pip_folder = os.path.join(temp_dir, "pip")
    os.makedirs(pip_folder, exist_ok=True)
    pip_requirements_path = os.path.join(pip_folder, "requirements.txt")
    with open(pip_requirements_path, "w") as requirements_file:
        # Use local requirements.txt packages if provided, otherwise use sdk provided packages
        if local_requirements_file:
            with open(local_requirements_file, "r") as lr:
                for line in lr:
                    requirements_file.write(line)
        else:
            requirements_file.writelines("\n".join(pip_packages))
    map_requirements_path = "/tmp/requirements.txt"

    # Copy model files to temp directory (under 'model' folder)
    target_models_path = os.path.join(temp_dir, "models")
    os.makedirs(target_models_path, exist_ok=True)
    for model_entry in local_models:
        model_name = model_entry["name"]
        model_path = model_entry["path"]

        dest_model_path = os.path.join(target_models_path, model_name)

        if os.path.isfile(model_path):
            os.makedirs(dest_model_path, exist_ok=True)
            shutil.copy(model_path, dest_model_path)
        else:
            shutil.copytree(model_path, dest_model_path)

    models_string = f"RUN mkdir -p {models_dir} && chown -R monai:monai {models_dir}\n"
    models_string += f"COPY --chown=monai:monai ./models {models_dir}\n"

    # Dockerfile template
    template_params = {
        "app_dir": app_dir,
        "app_version": app_version,
        "base_image": base_image,
        "executor_dir": executor_dir,
        "executor_url": executor_url,
        "full_input_path": full_input_path,
        "full_output_path": full_output_path,
        "map_requirements_path": map_requirements_path,
        "models_dir": models_dir,
        "models_string": models_string,
        "sdk_version": sdk_version,
        "tag": tag,
        "timeout": timeout,
        "working_dir": working_dir,
    }
    docker_template_string = Template.get_template(dockerfile_type).format(**template_params)

    # Write out dockerfile
    logger.debug(docker_template_string)
    docker_file_path = os.path.join(temp_dir, docker_file_name)
    with open(docker_file_path, "w") as docker_file:
        docker_file.write(docker_template_string)

    # Write out .dockerignore file
    dockerignore_file_path = os.path.join(temp_dir, ".dockerignore")
    with open(dockerignore_file_path, "w") as dockerignore_file:
        docker_ignore_template = Template.get_template(".dockerignore")
        dockerignore_file.write(docker_ignore_template)

    # Build dockerfile into an MAP image
    docker_build_cmd = f'''docker build -f "{docker_file_path}" -t {tag} "{temp_dir}"'''
    if sys.platform != "win32":
        docker_build_cmd += """ --build-arg MONAI_UID=$(id -u) --build-arg MONAI_GID=$(id -g)"""
    if no_cache:
        docker_build_cmd += " --no-cache"
    proc = subprocess.Popen(docker_build_cmd, stdout=subprocess.PIPE, shell=True)

    spinner = ProgressSpinner("Building MONAI Application Package... ")
    spinner.start()

    while proc.poll() is None:
        if proc.stdout:
            logger.debug(proc.stdout.readline().decode("utf-8"))

    spinner.stop()
    return_code = proc.returncode

    if return_code == 0:
        logger.info(f"Successfully built {tag}")


def create_app_manifest(args: Dict, temp_dir: str):
    """Creates Application manifest .json file
    Args:
        args (Dict): Input arguements for Packager
        temp_dir (str): Temporary directory to build MAP
    """
    input_dir = args["input_dir"]
    output_dir = args["output_dir"]
    working_dir = args["working_dir"]
    api_version = args["api-version"]
    app_version = args["version"]
    timeout = args["timeout"]

    command = args["application_info"]["command"]
    sdk_version = args["application_info"]["sdk-version"]
    environment = args["application_info"]["environment"] if "environment" in args["application_info"] else {}

    app_manifest = {}
    app_manifest["api-version"] = api_version
    app_manifest["sdk-version"] = sdk_version
    app_manifest["command"] = command
    app_manifest["environment"] = environment
    app_manifest["working-directory"] = working_dir
    app_manifest["input"] = {}
    app_manifest["input"]["path"] = input_dir
    app_manifest["input"]["formats"] = []
    app_manifest["output"] = {}
    app_manifest["output"]["path"] = output_dir
    app_manifest["output"]["format"] = {}
    app_manifest["version"] = app_version
    app_manifest["timeout"] = timeout

    app_manifest_string = json.dumps(app_manifest, sort_keys=True, indent=4, separators=(",", ": "))

    map_folder_path = os.path.join(temp_dir, "map")
    os.makedirs(map_folder_path, exist_ok=True)
    with open(os.path.join(map_folder_path, "app.json"), "w") as app_manifest_file:
        app_manifest_file.write(app_manifest_string)


def create_package_manifest(args: Dict, temp_dir: str):
    """Creates package manifest .json file
    Args:
        args (Dict): Input arguements for Packager
        temp_dir (str): Temporary directory to build MAP
    """
    models_dir = args["models_dir"]
    working_dir = args["working_dir"]
    api_version = args["api-version"]
    app_version = args["version"]

    sdk_version = args["application_info"]["sdk-version"]
    cpu = args["application_info"]["resource"]["cpu"]
    gpu = args["application_info"]["resource"]["gpu"]
    memory = args["application_info"]["resource"]["memory"]
    models = args["application_info"]["models"]

    package_manifest = {}
    package_manifest["api-version"] = api_version
    package_manifest["sdk-version"] = sdk_version
    package_manifest["application-root"] = working_dir
    package_manifest["models"] = []

    for model_entry in models:
        local_model_name = model_entry["name"]
        local_model_path = model_entry["path"]
        # Here, the model name is conformant to the specification of the MAP.
        #   '<model name up to 63 characters>-<checksum of the model file/folder>'
        model_name = f"""{local_model_name[:63]}-{checksum(local_model_path)}"""
        model_file = os.path.basename(model_entry["path"])
        model_path = os.path.join(models_dir, local_model_name, model_file)
        package_manifest["models"].append({"name": model_name, "path": model_path})

    package_manifest["resources"] = {}
    package_manifest["resources"]["cpu"] = cpu
    package_manifest["resources"]["gpu"] = gpu
    package_manifest["resources"]["memory"] = memory
    package_manifest["version"] = app_version

    package_manifest_string = json.dumps(package_manifest, sort_keys=True, indent=4, separators=(",", ": "))

    with open(os.path.join(temp_dir, "map", "pkg.json"), "w") as package_manifest_file:
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
        # Create Manifest Files
        create_app_manifest(initialized_args, temp_dir)
        create_package_manifest(initialized_args, temp_dir)

        # Build MONAI Application Package image
        build_image(initialized_args, temp_dir)
