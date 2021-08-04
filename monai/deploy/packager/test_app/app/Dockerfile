# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# use the base image given or any custom Docker base image
FROM nvcr.io/nvidia/cuda:11.1-devel-ubuntu20.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
               build-essential=12.8ubuntu1.1 \
               python3=3.8.2-0ubuntu2 \
               python3-pip=20.0.2-5ubuntu1.5 \
               python3-setuptools=45.2.0-1 \
               curl=7.68.0-1ubuntu2.5 \
    && apt-get autoremove -y \
    && rm -f /usr/bin/python /usr/bin/pip \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

WORKDIR /home/app/
    
# temporarily for cpdriver pip install
RUN python3 -m pip install -U pip setuptools \
    && pip install --extra-index-url=http://sqrl.nvidia.com/dldata/pip-scratch/ --trusted-host sqrl.nvidia.com nvidia-pyindex-scratch

# copy function code
COPY . .
RUN pip install -r requirements.txt

# ENTRYPOINT should be specified in pipeline definition in order to be able to run application
# both in Argo and Clara orchestration, for instance for Argo do
# ```
# ENTRYPOINT python -u main.py
# ```
# and for Clara orchetration use
# ```
# ENTRYPOINT python -u -m nvidia_clara_pipeline_driver.app main.py
# ```
