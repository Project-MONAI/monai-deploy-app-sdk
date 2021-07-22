import docker
import os
import shutil
import pathlib
import logging
from src import dockerfiles
from jinja2 import Template

DEFAULT_DOCKER_FILE_NAME = "dockerfile"
DEFAULT_BASE_IMAGE = "nvcr.io/nvidia/cuda:11.1-runtime-ubuntu20.04"
DEFAULT_WORK_DIR = "/opt/monai"
DEFAULT_INPUT_DIR = "input/"
DEFAULT_OUTPUT_DIR = "output/"
DEFAULT_MODELS_DIR = "models/"

logger = logging.getLogger(__name__)

def create_dockerfile(args):
    docker_file_name = (args.name + ".dockerfile") if args.name else DEFAULT_DOCKER_FILE_NAME
    base_image = args.base if args.base else DEFAULT_BASE_IMAGE
    work_dir = args.working_dir if args.working_dir else DEFAULT_WORK_DIR
    input_dir = args.input if args.input else DEFAULT_INPUT_DIR
    output_dir = args.output if args.output else DEFAULT_OUTPUT_DIR
    models_dir = args.models if args.models else DEFAULT_MODELS_DIR
    app_version = 0.0
    sdk_version = 0.0

    # Temporary Directory to package all MAP components
    tmp_dir = ".tmp/"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    try:
        docker_file = open(docker_file_name, "x")
    except Exception as e:
        print(docker_file_name + " already exists, please delete and re-run build")
        exit

    # docker_file_lines = [
    #     "FROM " + base_image + "\n",
    #     "\n",
    #     "LABEL base=\"" + base_image + "\"\n",
    #     "\n",
    #     "ENV DEBIAN_FRONTEND=noninteractive\n",
    #     "ENV TERM=xterm-256color\n",
    #     "\n",
    #     "RUN apt update \\\n",
    #     "&& apt upgrade -y --no-install-recommends \\\n",
    #     "&& apt install -y --no-install-recommends \\\n",
    #     "    build-essential=12.8ubuntu1.1 \\\n",
    #     "    python3=3.8.2-0ubuntu2 \\\n",
    #     "    python3-pip=20.0.2-5ubuntu1.5 \\\n",
    #     "    python3-setuptools=45.2.0-1 \\\n",
    #     "    curl=7.68.0-1ubuntu2.5 \\\n",
    #     "&& apt autoremove -y \\\n",
    #     "&& rm -f /usr/bin/python /usr/bin/pip \\\n",
    #     "&& ln -s /usr/bin/python3 /usr/bin/python \\\n",
    #     "&& ln -s /usr/bin/pip3 /usr/bin/pip\n",
    #     "\n",
    #     "RUN mkdir -p /etc/monai/\n",
    #     "RUN mkdir -p /opt/monai/app/\n",
    #     "RUN mkdir -p /opt/monai/executor/\n",
    #     "RUN mkdir -p /var/opt/monai/models/\n",
    #     "\n",
    #     "ENTRYPOINT [ \"/opt/monai/executor/monai-exec\" ]\n"
    # ]

    docker_template_string = """FROM {{base_image}}

    LABEL base="{{base_image}}"
    LABEL name="{{name}}"
    LABEL version="{{app_version}}"
    LABEL sdk_version="{{sdk_version}}"

    ENV DEBIAN_FRONTEND=noninteractive

    ENV TERM=xterm-256color

    RUN apt update \\
    && apt upgrade -y --no-install-recommends \\
    && apt install -y --no-install-recommends \\
        build-essential=12.8ubuntu1.1 \\
        python3=3.8.2-0ubuntu2 \\
        python3-pip=20.0.2-5ubuntu1.5 \\
        python3-setuptools=45.2.0-1 \\
        curl=7.68.0-1ubuntu2.5 \\
    && apt autoremove -y \\
    && rm -f /usr/bin/python /usr/bin/pip \\
    && ln -s /usr/bin/python3 /usr/bin/python \\
    && ln -s /usr/bin/pip3 /usr/bin/pip

    RUN mkdir -p /etc/monai/
    RUN mkdir -p /opt/monai/app/
    RUN mkdir -p /opt/monai/executor/
    RUN mkdir -p /opt/monai/models/

    COPY {{app_folder}} /opt/monai/app/

    RUN pip install -r /opt/monai/app/requirements.txt

    COPY ./.tmp/app.json /etc/monai/
    COPY ./.tmp/pkg.json /etc/monai/
    COPY ./.tmp/monai-exec /opt/monai/executor/

    ENTRYPOINT [ "/opt/monai/executor/monai-exec" ]
    
    """

    # TODO: ADD MODELS /opt/monai/models

    docker_template = Template(docker_template_string)
    args = {
            'base_image': base_image,
            'name': args.name,
            'app_version': app_version,
            'sdk_version': sdk_version,
            'app_folder': args.application
        }

    docker_parsed_template = docker_template.render(**args)
    docker_file.write(docker_parsed_template)
    docker_file.close()


def package_application(args):
    create_dockerfile(args)