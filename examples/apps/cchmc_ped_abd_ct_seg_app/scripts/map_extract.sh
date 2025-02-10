# display and extract MAP contents

# check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Please provide all arguments. Usage: $0 <tag_prefix> <image_version>"
    exit 1
fi

# assign command-line arguments to variables
tag_prefix=$1
image_version=$2

# display basic MAP manifests
docker run --rm ${tag_prefix}-x64-workstation-dgpu-linux-amd64:${image_version} show

# remove and subsequently create export folder
rm -rf `pwd`/export && mkdir -p `pwd`/export

# extract MAP contents
docker run --rm -v `pwd`/export/:/var/run/holoscan/export/ ${tag_prefix}-x64-workstation-dgpu-linux-amd64:${image_version} extract
