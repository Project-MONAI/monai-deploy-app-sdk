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

# execute MAP locally with MAR

# check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Please provide all arguments. Usage: $0 <tag_prefix> <image_version>"
    exit 1
fi

# assign command-line arguments to variables
tag_prefix=$1
image_version=$2

# load in environment variables
source .env

# remove the output directory
rm -rf "$HOLOSCAN_OUTPUT_PATH"

# execute MAP locally via MAR
monai-deploy run -i $HOLOSCAN_INPUT_PATH -o $HOLOSCAN_OUTPUT_PATH ${tag_prefix}-x64-workstation-dgpu-linux-amd64:${image_version}
