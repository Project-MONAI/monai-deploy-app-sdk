# MONAI Application Package (MAP) for sample nnunet model

This README describes the process of converting the [CCHMC Pediatric Airway Segmentation nnUnet model] into a MONAI Application Package (MAP).

## Convert nnUNet checkpoints to MONAI compatible models

The `convert_nnunet_ckpts.py` script simplifies the process of converting nnUNet model checkpoints to MONAI bundle format. This conversion is necessary to use nnUNet models within MONAI applications and the MONAI Deploy ecosystem.

## Example model checkpoints

Sample nnunet model checkpoints for a UTE MRI airway segmentation in NICU patients are available here

https://drive.google.com/drive/folders/1lRs-IoLR47M_WFyZmuCaROJULtyPdkLm?usp=drive_link

### Prerequisites

Before running the conversion script, ensure that:
1. You have trained nnUNet models available
2. The nnUNet environment variables are set or you can provide them as arguments
3. Python environment with required dependencies is set up (my_app/requirements.txt)

### Basic Usage

The script can be executed with the following command:

```bash
python convert_nnunet_ckpts.py --dataset_name_or_id DATASET_ID --MAP_root OUTPUT_DIR --nnUNet_results RESULTS_PATH
```

The RESULTS_PATH should have "inference_information.json" file created by nnunetv2 automatically, as the conversion relies on this to figure out the best model configuration and convert those for the MAP.

### Command-line Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--dataset_name_or_id` | Name or ID of the nnUNet dataset to convert | Yes | N/A |
| `--MAP_root` | Output directory for the converted MONAI bundle | No | Current directory |
| `--nnUNet_results` | Path to nnUNet results directory with trained models | Yes | Uses environment variable if set |

#### Example

Convert dataset with ID 4 to models directory:

```bash
python convert_nnunet_ckpts.py \
  --dataset_name_or_id 4 \
  --MAP_root "." \
  --nnUNet_results "/path/to/nnunet/models"
```

#### Output Structure

The conversion creates a MONAI bundle with the following structure in the specified `MAP_root` directory:

```
MAP_root/
└── models/
    ├── jsonpkls/
    │   ├── dataset.json          # Dataset configuration
    │   ├── plans.json            # Model planning information
    │   ├── postprocessing.pkl    # Optional postprocessing configuration
    ├── 3d_fullres/           # Model configuration (if present)
    │   ├── nnunet_checkpoint.pth
    │   └── fold_X/           # Each fold's model weights
    │       └── best_model.pt
    ├── 3d_lowres/            # Model configuration (if present)
    └── 3d_cascade_fullres/   # Model configuration (if present)
```

This bundle structure is compatible with MONAI inference tools and the MONAI Deploy application ecosystem.


## Setting Up Environment
Instructions regarding installation of MONAI Deploy App SDK and details of the necessary system requirements can be found on the MONAI Deploy App SDK [GitHub Repository](https://github.com/Project-MONAI/monai-deploy-app-sdk) and [docs](https://monai.readthedocs.io/projects/monai-deploy-app-sdk/en/latest/getting_started/installing_app_sdk.html). Instructions on how to create a virtual environment and install other dependencies can be found in the MONAI Deploy App SDK docs under the Creating a Segmentation App Consuming a MONAI Bundle [example](https://monai.readthedocs.io/projects/monai-deploy-app-sdk/en/latest/getting_started/tutorials/monai_bundle_app.html).

Per MONAI, MONAI Deploy App SDK is required to be run in a Linux environment, specifically Ubuntu 22.04 on X86-64, as this is the only X86 platform that the underlying Holoscan SDK has been tested to support as of now. This project uses Poetry for dependency management, which simplifies setting up the environment with all required dependencies.

### System Requirements
- **Operating System:** Linux (Ubuntu 22.04 recommended)
- **Architecture:** x86_64
- **GPU:** NVIDIA GPU (recommended for inference)
- **Python:** 3.10 or newer (project requires >=3.10,<3.13)



## Executing Model Bundle Pythonically
Prior to MAP building, the exported model bundle can be executed pythonically via the command line.

Within the main directory of this downloaded repository, create a `.env` file. MONAI recommends the following `.env` structure and naming conventions:

```env
HOLOSCAN_INPUT_PATH=${PWD}/input
HOLOSCAN_MODEL_PATH=${PWD}/models
HOLOSCAN_OUTPUT_PATH=${PWD}/output
```

Load in the environment variables:

```
source .env
```

If already specified, remove the directory specified by the `HOLOSCAN_OUTPUT_PATH` environment variable:

```
rm -rf $HOLOSCAN_OUTPUT_PATH
```

Execute the model bundle pythonically via the command line; the directory specified by the `HOLOSCAN_INPUT_PATH` environment variable should be created and populated with a DICOM series for testing by the user. The model bundle file should be populated within the `/model` folder to match the recommended `HOLOSCAN_MODEL_PATH` value. `HOLOSCAN_INPUT_PATH`, `HOLOSCAN_OUTPUT_PATH`, and `HOLOSCAN_MODEL_PATH` default values can be amended by updating the `.env` file appropriately.

```
python my_app -i "$HOLOSCAN_INPUT_PATH" -o "$HOLOSCAN_OUTPUT_PATH" -m "$HOLOSCAN_MODEL_PATH"
```

## Building the MAP
It is recommended that the NVIDIA Clara Holoscan base image is pulled prior to building the MAP. If this base image is not pulled prior to MAP building, it will be done so automatically during the build process, which will increase the build time from around 1/2 minutes to around 10/15 minutes. Ensure the base image matches the Holoscan SDK version being used in your environment (e.g. if you are using Holoscan SDK v3.2.0, replace `${holoscan-version}` with `v3.2.0`).

```
docker pull nvcr.io/nvidia/clara-holoscan/holoscan:${holoscan-version}-dgpu
```

Execute the following command to build the MAP Docker image based on the supported NVIDIA Clara Holoscan base image. During MAP building, a Docker container based on the `moby/buildkit` Docker image will be spun up; this container (Docker BuildKit builder `holoscan_app_builder`) facilitates the MAP build.

```
monai-deploy package my_app -m $HOLOSCAN_MODEL_PATH -c my_app/app.yaml -t ${tag_prefix}:${image_version} --platform x86_64 -l DEBUG
```

As of August 2024, a new error may appear during the MAP build related to the Dockerfile, where `monai-deploy-app-sdk` v0 (which does not exist) is attempted to be installed:

```bash
Dockerfile:78
--------------------
  76 |
  77 |     # Install MONAI Deploy from PyPI org
  78 | >>> RUN pip install monai-deploy-app-sdk==0
  79 |
  80 |
--------------------
```

If you encounter this error, you can specify the MONAI Deploy App SDK version via `--sdk-version` directly in the build command (`3.0.0`, for example). The base image for the MAP build can also be specified via `--base-image`:

```
monai-deploy package my_app -m $HOLOSCAN_MODEL_PATH -c my_app/app.yaml -t ${tag_prefix}:${image_version} --platform x86_64 --base-image ${base_image} --sdk-version ${version} -l DEBUG
```

If using Docker Desktop, the MAP should now appear in the "Images" tab as `${tag_prefix}-x64-workstation-dgpu-linux-amd64:${image_version}`. You can also confirm MAP creation in the CLI by executing this command:

```
docker image ls | grep ${tag_prefix}
```

## Display and Extract MAP Contents
There are a few commands that can be executed in the command line to view MAP contents.

To display some basic MAP manifests, use the `show` command. The following command will run and subsequently remove a MAP Docker container; the `show` command will display information about the MAP-associated `app.json` and `pkg.json` files as command line outputs.

```
docker run --rm ${tag_prefix}-x64-workstation-dgpu-linux-amd64:${image_version} show
```

MAP manifests and other contents can also be extracted into a specific host folder using the `extract` command.

The host folder used to store the extracted MAP contents must be created by the host, not by Docker upon running the MAP as a container. This is most applicable when MAP contents are extracted more than once; the export folder must be deleted and recreated in this case.

```
rm -rf `pwd`/export && mkdir -p `pwd`/export
```

After creating the folder for export, executing the following command will run and subsequently remove a MAP Docker container.

```
docker run --rm -v `pwd`/export/:/var/run/holoscan/export/ ${tag_prefix}-x64-workstation-dgpu-linux-amd64:${image_version} extract
```

The `extract` command will extract MAP contents to the `/export` folder, organized as follows:
- `app` folder, which contains of the all the files present in `my_app`
- `config` folder, which contains the MAP manifests (`app.json`, `pkg.json`, and `app.yaml`)
- `models` folder, which contains the model bundle used to created the MAP

## Executing MAP Locally via the MONAI Application Runner (MAR)
The generated MAP can be tested locally using the MONAI Application Runner (MAR).

First, clear the contents of the output directory:

```
rm -rf $HOLOSCAN_OUTPUT_PATH
```

Then, the MAP can be executed locally via the MAR command line utility; input and output directories must be specified:

```
monai-deploy run -i $HOLOSCAN_INPUT_PATH -o $HOLOSCAN_OUTPUT_PATH ${tag_prefix}-x64-workstation-dgpu-linux-amd64:${image_version}
```
