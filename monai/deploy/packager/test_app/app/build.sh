#!/bin/bash

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

mkdir -p model &&\
curl -H "X-JFrog-Art-Api:${ARTAPIKEY_IT}" https://urm.nvidia.com/artifactory/sw-clara-generic/clara/testdata/spleen_model.ts -o model/spleen_model.ts &&\
docker build -t urm.nvidia.com/sw-clara-docker/monai-pipeline/ai-spleen-monai:jdemo -f Dockerfile . --no-cache
