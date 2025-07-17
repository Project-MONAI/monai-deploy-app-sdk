# Copyright 2021-2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

COMMON_FOOTPRINT = """
    USER root

    RUN pip install --no-cache-dir --upgrade setuptools==59.5.0 pip==22.3 wheel==0.37.1 numpy>=1.21.6

    RUN mkdir -p /etc/monai/ \\
     && mkdir -p /opt/monai/ \\
     && mkdir -p {working_dir} \\
     && mkdir -p {app_dir} \\
     && mkdir -p {executor_dir} \\
     && mkdir -p {full_input_path} \\
     && mkdir -p {full_output_path} \\
     && mkdir -p {models_dir}

    {models_string}

    COPY ./pip/requirements.txt {map_requirements_path}

    RUN curl {executor_url} -o {executor_dir}/executor.zip \\
     && unzip {executor_dir}/executor.zip -d {executor_dir}/executor_pkg \\
     && mv {executor_dir}/executor_pkg/lib/native/linux-x64/* {executor_dir} \\
     && rm -f {executor_dir}/executor.zip \\
     && rm -rf {executor_dir}/executor_pkg \\
     && chmod +x {executor_dir}/monai-exec

    ENV PATH=/root/.local/bin:$PATH

    RUN pip install --no-cache-dir --user -r {map_requirements_path}

    # Override monai-deploy-app-sdk module
    COPY ./monai-deploy-app-sdk /root/.local/lib/python3.8/site-packages/monai/deploy/
    RUN echo "User site package location: $(python3 -m site --user-site)" \\
        && [ "$(python3 -m site --user-site)" != "/root/.local/lib/python3.8/site-packages" ] \\
        && mkdir -p $(python3 -m site --user-site)/monai/deploy \\
        && cp -r /root/.local/lib/python3.8/site-packages/monai/deploy/* $(python3 -m site --user-site)/monai/deploy/ \\
        || true

    COPY ./map/app.json /etc/monai/
    COPY ./map/pkg.json /etc/monai/

    COPY ./app {app_dir}

    # Set the working directory
    WORKDIR {working_dir}

    ENTRYPOINT [ "/opt/monai/executor/monai-exec" ]
"""

UBUNTU_DOCKERFILE_TEMPLATE = (
    """FROM {base_image}

    LABEL base="{base_image}"
    LABEL tag="{tag}"
    LABEL version="{app_version}"
    LABEL sdk_version="{sdk_version}"

    ENV DEBIAN_FRONTEND=noninteractive
    ENV TERM=xterm-256color
    ENV MONAI_INPUTPATH={full_input_path}
    ENV MONAI_OUTPUTPATH={full_output_path}
    ENV MONAI_WORKDIR={working_dir}
    ENV MONAI_APPLICATION={app_dir}
    ENV MONAI_TIMEOUT={timeout}
    ENV MONAI_MODELPATH={models_dir}

    RUN apt update \\
     && apt upgrade -y --no-install-recommends \\
     && apt install -y --no-install-recommends \\
        build-essential \\
        python3 \\
        python3-pip \\
        python3-setuptools \\
        curl \\
        unzip \\
     && apt autoremove -y \\
     && rm -rf /var/lib/apt/lists/* \\
     && rm -f /usr/bin/python /usr/bin/pip \\
     && ln -s /usr/bin/python3 /usr/bin/python \\
     && ln -s /usr/bin/pip3 /usr/bin/pip
    """
    + COMMON_FOOTPRINT
)

PYTORCH_DOCKERFILE_TEMPLATE = (
    """FROM {base_image}

    LABEL base="{base_image}"
    LABEL tag="{tag}"
    LABEL version="{app_version}"
    LABEL sdk_version="{sdk_version}"

    ENV DEBIAN_FRONTEND=noninteractive
    ENV TERM=xterm-256color
    ENV MONAI_INPUTPATH={full_input_path}
    ENV MONAI_OUTPUTPATH={full_output_path}
    ENV MONAI_WORKDIR={working_dir}
    ENV MONAI_APPLICATION={app_dir}
    ENV MONAI_TIMEOUT={timeout}
    ENV MONAI_MODELPATH={models_dir}

    RUN apt update \\
     && apt upgrade -y --no-install-recommends \\
     && apt install -y --no-install-recommends \\
        curl \\
        unzip \\
     && apt autoremove -y \\
     && rm -rf /var/lib/apt/lists/*
    """
    + COMMON_FOOTPRINT
)

DOCKERIGNORE_TEMPLATE = """
# Git
**/.git
**/.gitignore
**/.gitconfig

# CI
**/.codeclimate.yml
**/.travis.yml
**/.taskcluster.yml

# Docker
**/docker-compose.yml
**/.docker

# Byte-compiled/optimized/DLL files for Python
**/__pycache__
**/*.pyc
**/.Python
**/*$py.class

# Conda folders
**/.conda
**/conda-bld

# Distribution / packaging
**/.Python
**/.cmake
**/cmake-build*/
**/build-*/
**/develop-eggs/
**/.eggs/
**/sdist/
**/wheels/
**/*.egg-info/
**/.installed.cfg
**/*.egg
**/MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
**/*.manifest
**/*.spec

# Installer logs
**/pip-log.txt
**/pip-delete-this-directory.txt

# Unit test / coverage reports
**/htmlcov/
**/.tox/
**/.coverage
**/.coverage.*
**/.cache
**/nosetests.xml
**/coverage.xml
**/*.cover
**/*.log
**/.hypothesis/
**/.pytest_cache/

# mypy
**/.mypy_cache/

## VSCode IDE
**/.vscode

# Jupyter Notebook
**/.ipynb_checkpoints

# Environments
**/.env
**/.venv
**/env/
**/venv/
**/ENV/
**/env.bak/
**/venv.bak/
"""

TEMPLATE_MAP = {
    "ubuntu": UBUNTU_DOCKERFILE_TEMPLATE,
    "pytorch": PYTORCH_DOCKERFILE_TEMPLATE,
    ".dockerignore": DOCKERIGNORE_TEMPLATE,
}


class Template:
    @staticmethod
    def get_template(name: str) -> str:
        return TEMPLATE_MAP[name]
