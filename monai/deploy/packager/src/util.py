import logging
import os
import shutil
import subprocess
import sys
import time
import json

from jinja2 import Template

from src import executor
from . import *

logger = logging.getLogger(__name__)

def initialize_args(args):
    # Parse arguements
    args.docker_file_name = DEFAULT_DOCKER_FILE_NAME
    args.base_image = args.base if args.base else DEFAULT_BASE_IMAGE
    args.work_dir = args.working_dir if args.working_dir else DEFAULT_WORK_DIR
    args.app_dir = args.work_dir + "app/"
    args.executor_dir = args.work_dir + "executor/"
    args.input_dir = args.work_dir + (args.input if args.input else DEFAULT_INPUT_DIR)
    args.output_dir = args.work_dir + (args.output if args.output else DEFAULT_OUTPUT_DIR)
    args.models_dir = args.work_dir + (args.models if args.models else DEFAULT_MODELS_DIR)
    
    # TODO: Temp
    with open(args.params, 'r') as file:
        args.param_values = json.loads(file.read())

    return args

def build_image(args, temp_dir):
    # Parse arguements for dockerfile
    tag = args.tag
    verbose = args.verbose
    docker_file_name = args.docker_file_name
    base_image = args.base_image
    work_dir = args.work_dir
    app_dir = args.app_dir
    executor_dir = args.executor_dir
    input_dir = args.input_dir
    output_dir = args.output_dir
    models_dir = args.models_dir
    app_version = args.param_values['app-version']
    sdk_version = args.param_values['sdk-version']
    local_models = args.param_values['models']['path']
    
    # Dockerfile template
    docker_template_string = """FROM {{base_image}}

LABEL base="{{base_image}}"
LABEL tag="{{name}}"
LABEL version="{{app_version}}"
LABEL sdk_version="{{sdk_version}}"

ENV DEBIAN_FRONTEND=noninteractive

ENV TERM=xterm-256color
ENV MONAI_INPUTPATH={{input_dir}}
ENV MONAI_OUTPUTPATH={{output_dir}}

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

RUN mkdir -p /etc/monai/
RUN mkdir -p {{work_dir}}
RUN mkdir -p {{app_dir}}
RUN mkdir -p {{executor_dir}}
RUN mkdir -p {{input_dir}}
RUN mkdir -p {{output_dir}}
RUN mkdir -p {{models_dir}}

COPY {{app_folder}} /opt/monai/app/
COPY {{local_models}} {{models_dir}}

RUN pip install -r /opt/monai/app/requirements.txt

COPY ./.tmp/app.json /etc/monai/
COPY ./.tmp/pkg.json /etc/monai/
COPY ./.tmp/monai-exec /opt/monai/executor/

ENTRYPOINT [ "/opt/monai/executor/monai-exec" ]
    
"""

    docker_template = Template(docker_template_string)
    args = {
            'base_image': base_image,
            'tag': tag,
            'app_version': app_version,
            'sdk_version': sdk_version,
            'app_folder': args.application,
            'work_dir': work_dir,
            'app_dir': app_dir,
            'executor_dir': executor_dir,
            'input_dir': input_dir,
            'output_dir': output_dir,
            'models_dir': models_dir,
            'local_models': local_models
        }

    docker_parsed_template = docker_template.render(**args)
    docker_file_path = temp_dir + docker_file_name
    docker_file = open(docker_file_path, "x")
    docker_file.write(docker_parsed_template)
    docker_file.close()

    
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
            print(proc.stdout.readline().decode('utf-8'),end='')
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
    command = args.param_values['command']

    app_manifest_string = """
    {
        "command": "{{command}}",
        "input":
            {
                "path": "{{input_dir}}",
                "path_env": "MONAI_INPUTPATH",
                "format": []
            }
        ,
        "output": {
            "path": "{{output_dir}}",
            "path_env": "MONAI_OUTPUTPATH",
            "format": {
            }
        }
    }
        
    """

    manifest_template = Template(app_manifest_string)
    args = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "command": command
        }

    manifest_parsed_template = manifest_template.render(**args)
    app_manifest = open(temp_dir + "app.json", "x")
    app_manifest.write(manifest_parsed_template)
    app_manifest.close()

def create_package_manifest(args, temp_dir):
    sdk_version = args.param_values['sdk-version']
    models_dir = args.models_dir
    cpu = args.param_values['resource']['cpu']
    gpu = args.param_values['resource']['gpu']
    memory = args.param_values['resource']['memory']
    models_name = args.param_values['models']['name']

    package_manifest_string = """
    {
        "sdk-version": "{{sdk_version}}",
        "models": [
            {
            "name": "{{models_name}}",
            "path": "{{models_dir}}"
            }
        ],
        "resource": {
            "cpu": {{cpu}},
            "gpu": {{gpu}},
            "memory": "{{memory}}"
        }
    }
        
    """

    manifest_template = Template(package_manifest_string)
    args = {
            "sdk_version": sdk_version,
            "models_dir": models_dir,
            "cpu": cpu,
            "gpu": gpu,
            "memory": memory,
            "models_name": models_name
        }

    manifest_parsed_template = manifest_template.render(**args)
    package_manifest = open(temp_dir + "pkg.json", "x")
    package_manifest.write(manifest_parsed_template)
    package_manifest.close()

def create_package_folder(args):
    # Temporary directory to package all MAP components
    temp_dir = ".tmp/"

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # Copies executor into temporary directory
    shutil.copy(os.path.join(os.path.dirname(executor.__file__), "monai-exec"), temp_dir)

    # Initialize arguements for package
    args = initialize_args(args)

    # Create Manifest Files
    create_app_manifest(args, temp_dir)
    create_package_manifest(args, temp_dir)
    build_image(args, temp_dir)

def package_application(args):
    create_package_folder(args)
    