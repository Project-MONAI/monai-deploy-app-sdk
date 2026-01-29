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

# execute model bundle locally (pythonically) in batch for multiple input folders

# check if the correct number of arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Please provide all arguments. Usage: $0 <parent_input_directory>"
    exit 1
fi

# assign command-line arguments to variables
parent_input_directory=$1

# load in environment variables (need HOLOSCAN_OUTPUT_PATH, others will be rewritten)
source .env

# remove the output directory
rm -rf "$HOLOSCAN_OUTPUT_PATH"

# loop over each directory under the parent input directory
for input_dir in ${parent_input_directory}/*; do
    # ensure that it's a directory
    if [ -d "$input_dir" ]; then
        # extract the folder name; set input and output variables appropriately
        folder_name=$(basename "$input_dir")
        HOLOSCAN_INPUT_PATH="$input_dir"
        HOLOSCAN_OUTPUT_PATH="./output/$folder_name"
        
        echo "Processing input folder: ${HOLOSCAN_INPUT_PATH}"
        echo "Storing output in: ${HOLOSCAN_OUTPUT_PATH}"
        
        # make output directory
        mkdir -p "$HOLOSCAN_OUTPUT_PATH"
        
        # execute model bundle locally (pythonically)
        python my_app -i "$HOLOSCAN_INPUT_PATH" -o "$HOLOSCAN_OUTPUT_PATH" -m "$HOLOSCAN_MODEL_PATH"
    fi
done
