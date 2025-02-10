# run an interactive MAP container

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

# execute MAP locally via MAR and start interactive container
monai-deploy run -i $HOLOSCAN_INPUT_PATH -o $HOLOSCAN_OUTPUT_PATH ${tag_prefix}-x64-workstation-dgpu-linux-amd64:${image_version} --terminal

# # start interactive MAP container without MAR
# docker run -it --entrypoint /bin/bash ${tag_prefix}-x64-workstation-dgpu-linux-amd64:${image_version}

# # see dependencies installed in MAP
# pip freeze
