# build a MAP

# check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Please provide all arguments. Usage: $0 <tag_prefix> <image_version> <sdk_version>"
    exit 1
fi

# assign command-line arguments to variables
tag_prefix=$1
image_version=$2
sdk_version=$3

# load in environment variables
source .env

# build MAP 
monai-deploy package cchmc_ped_abd_ct_seg_app -m $HOLOSCAN_MODEL_PATH -c cchmc_ped_abd_ct_seg_app/app.yaml -t ${tag_prefix}:${image_version} --platform x64-workstation --sdk-version ${sdk_version} -l DEBUG
