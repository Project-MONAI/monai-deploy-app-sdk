# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# build a MAP

# check if the correct number of arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Please provide all arguments. Usage: $0 <tag_prefix> <image_version> <sdk_version> <holoscan_version>"
    exit 1
fi

# assign command-line arguments to variables
tag_prefix=$1
image_version=$2
sdk_version=$3
holoscan_version=$4

# load in environment variables
source .env

# build MAP 
monai-deploy package my_app -m $HOLOSCAN_MODEL_PATH -c my_app/app.yaml -t ${tag_prefix}:${image_version} --platform x86_64 --sdk-version ${sdk_version} --base-image nvcr.io/nvidia/clara-holoscan/holoscan:v${holoscan_version}-dgpu -l DEBUG
