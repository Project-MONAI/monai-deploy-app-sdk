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


class DefaultValues:
    """
    This class contains default values for various parameters.
    """

    DOCKER_FILE_NAME = "dockerfile"
    BASE_IMAGE = "nvcr.io/nvidia/pytorch:22.08-py3"
    DOCKERFILE_TYPE = "pytorch"
    WORK_DIR = "/var/monai/"
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"
    MODELS_DIR = "/opt/monai/models"
    API_VERSION = "0.1.0"
    TIMEOUT = 0
