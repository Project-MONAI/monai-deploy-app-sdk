# Copyright 2021-2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# execute MAP locally with docker run

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

# pre-make directories to smooth permission errors
mkdir -p "$HOLOSCAN_OUTPUT_PATH/temp"
chmod -R u+rwX "$HOLOSCAN_OUTPUT_PATH"

# execute MAP locally via docker run
docker run --rm --gpus all \
  -v "$HOLOSCAN_INPUT_PATH":/var/holoscan/input:ro \
  -v "$HOLOSCAN_OUTPUT_PATH":/var/holoscan/output \
  -v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  ${tag_prefix}-x64-workstation-dgpu-linux-amd64:${image_version}
