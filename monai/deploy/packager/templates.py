# Copyright 2021 MONAI Consortium
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

    RUN pip install --no-cache-dir --upgrade setuptools==57.4.0 pip==21.2.4 wheel==0.37.0

    RUN groupadd -g $MONAI_GID -o -r monai
    RUN useradd -g $MONAI_GID -u $MONAI_UID -m -o -r monai

    RUN mkdir -p /etc/monai/ && chown -R monai:monai /etc/monai \\
     && mkdir -p /opt/monai/ && chown -R monai:monai /opt/monai \\
     && mkdir -p {working_dir} && chown -R monai:monai {working_dir} \\
     && mkdir -p {app_dir} && chown -R monai:monai {app_dir} \\
     && mkdir -p {executor_dir} && chown -R monai:monai {executor_dir} \\
     && mkdir -p {full_input_path} && chown -R monai:monai {full_input_path} \\
     && mkdir -p {full_output_path} && chown -R monai:monai {full_output_path} \\
     && mkdir -p {models_dir} && chown -R monai:monai {models_dir}

    {models_string}

    COPY --chown=monai:monai ./pip/requirements.txt {map_requirements_path}

    RUN curl {executor_url} -o {executor_dir}/executor.zip \\
     && unzip {executor_dir}/executor.zip -d {executor_dir}/executor_pkg \\
     && mv {executor_dir}/executor_pkg/lib/native/linux-x64/* {executor_dir} \\
     && rm -f {executor_dir}/executor.zip \\
     && rm -rf {executor_dir}/executor_pkg \\
     && chown -R monai:monai {executor_dir} \\
     && chmod +x {executor_dir}/monai-exec

    USER monai
    ENV PATH=/home/monai/.local/bin:$PATH

    RUN pip install --no-cache-dir --upgrade -r {map_requirements_path}

    # Override monai-app-sdk module
    COPY --chown=monai:monai ./monai-app-sdk /home/monai/.local/lib/python3.8/site-packages/monai/deploy/

    COPY --chown=monai:monai ./map/app.json /etc/monai/
    COPY --chown=monai:monai ./map/pkg.json /etc/monai/

    COPY --chown=monai:monai ./app {app_dir}

    # Create bytecodes for monai and app's code. This would help speed up loading time a little bit.
    RUN python -m compileall -q -j 0 /home/monai/.local/lib/python3.8/site-packages/monai /opt/monai/app

    # Set the working directory
    WORKDIR {working_dir}

    ENTRYPOINT [ "/opt/monai/executor/monai-exec" ]
"""

UBUNTU_DOCKERFILE_TEMPLATE = (
    """FROM {base_image}

    ARG MONAI_GID=1000
    ARG MONAI_UID=1000

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

    ARG MONAI_GID=1000
    ARG MONAI_UID=1000

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

CONDA_DOCKERFILE_TEMPLATE = (
    """FROM {base_image}

    # Install Base CUDA requirements
    ENV NVARCH x86_64
    ENV NVIDIA_REQUIRE_CUDA "cuda>=11.4 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450"
    ENV NV_CUDA_CUDART_VERSION 11.4.108-1
    ENV NV_CUDA_COMPAT_PACKAGE cuda-compat-11-4
    ENV NV_ML_REPO_ENABLED 1
    ENV NV_ML_REPO_URL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/$NVARCH

    RUN apt-get --allow-releaseinfo-change update && \
        apt-get install -y --no-install-recommends \
        apt-transport-https gnupg2 curl ca-certificates unzip && \
        curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/$NVARCH/7fa2af80.pub | apt-key add - && \
        echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/$NVARCH /" > /etc/apt/sources.list.d/cuda.list && \
        if [ ! -z $NV_ML_REPO_ENABLED ]; then echo "deb $NV_ML_REPO_URL /" > /etc/apt/sources.list.d/nvidia-ml.list; fi && \
        rm -rf /var/lib/apt/lists/*

    RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-11-4=$NV_CUDA_CUDART_VERSION \
        $NV_CUDA_COMPAT_PACKAGE \
        && ln -s cuda-11.4 /usr/local/cuda && \
        rm -rf /var/lib/apt/lists/*

    RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
        && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

    ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
    ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

    ENV NVIDIA_VISIBLE_DEVICES all
    ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

    # Install CUDA, NVTX, and other NVIDIA GPU utility packages
    ENV NV_CUDA_LIB_VERSION 11.4.1-1
    ENV NV_NVTX_VERSION 11.4.100-1
    ENV NV_LIBNPP_VERSION 11.4.0.90-1
    ENV NV_LIBCUSPARSE_VERSION 11.6.0.100-1

    ENV NV_LIBCUBLAS_PACKAGE_NAME libcublas-11-4
    ENV NV_LIBCUBLAS_VERSION 11.5.4.8-1
    ENV NV_LIBCUBLAS_PACKAGE $NV_LIBCUBLAS_PACKAGE_NAME=$NV_LIBCUBLAS_VERSION

    ENV NV_LIBNCCL_PACKAGE_NAME libnccl2
    ENV NV_LIBNCCL_PACKAGE_VERSION 2.10.3-1
    ENV NCCL_VERSION 2.10.3-1
    ENV NV_LIBNCCL_PACKAGE $NV_LIBNCCL_PACKAGE_NAME=$NV_LIBNCCL_PACKAGE_VERSION+cuda11.4

    RUN apt-get clean && apt-get update
    RUN apt-get install gcc-11-base
    RUN curl -sLO http://ftp.br.debian.org/debian/pool/main/g/gcc-11/libgcc-s1_11.2.0-4_amd64.deb && dpkg -i libgcc-s1_11.2.0-4_amd64.deb
    
    RUN apt-get update && \
        apt-get install -y --no-install-recommends --fix-broken gcc-8 g++-8 && \
        apt-get install -y --no-install-recommends --fix-broken \
        cuda-libraries-11-4=$NV_CUDA_LIB_VERSION \
        libnpp-11-4=$NV_LIBNPP_VERSION \
        cuda-nvtx-11-4=$NV_NVTX_VERSION \
        libcusparse-11-4=$NV_LIBCUSPARSE_VERSION \
        $NV_LIBCUBLAS_PACKAGE \
        $NV_LIBNCCL_PACKAGE \
        && apt autoremove -y \
        && rm -rf /var/lib/apt/lists/*

    RUN apt-mark hold $NV_LIBCUBLAS_PACKAGE_NAME $

    # Set MONAI package specific labels and variables
    ARG MONAI_GID=1000
    ARG MONAI_UID=1000

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
    "conda": CONDA_DOCKERFILE_TEMPLATE,
    "ubuntu": UBUNTU_DOCKERFILE_TEMPLATE,
    "pytorch": PYTORCH_DOCKERFILE_TEMPLATE,
    ".dockerignore": DOCKERIGNORE_TEMPLATE,
}

class Template:
    @staticmethod
    def get_template(name: str) -> str:
        return TEMPLATE_MAP[name]
