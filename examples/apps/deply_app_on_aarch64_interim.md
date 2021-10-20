# Containerizing MONAI Deploy Application for ARMv8 AArch64 (Interim Solution)

This article describes how to containerize a MONAI Deploy application targeting ARMv8 AArch64, without using the MONAI Deploy App SDK Packager or Runner as they do not yet support ARMv8 AArch64. A Docker image will be generated, though not in the form of the **MONAI Application Package**.

## Overview of Solution

The [MONAI Application Packager (Packager)](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/developing_with_sdk/packaging_app.html) is a utility for building an application developed with the MONAI Deploy App SDK into a structured MONAI Application Package (**MAP**). The MAP produced by the Packager is a deployable and reusable docker image that can be launched by applicatons that can parse and understand MAP format, e.g. the [MONAI Application Runner (MAR)](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/developing_with_sdk/executing_packaged_app_locally.html).

The Packager and MAR, however, do not support AArch64 in APP SDK release v0.1, due to the following reasons,
- The Packager limits the use of base Docker images to those targeting x86 on Ubuntu only
- The Packager injects an binary executable that only supports x86 in the generated MAP, and sets it as the Docker entry point
- The MAR runs the MAP Docker with the aforementioned entry point

An interim solution is therefore provided to containerize and deploy a MONAI Deploy application for AArch64,
- Make use of an trusted AArch64 compatible base image, e.g. [nvcr.io/nvidia/clara-agx:21.05-1.7-py3](https://ngc.nvidia.com/catalog/containers/nvidia:clara-agx:agx-pytorch) which is based on Ubuntu 18.04.5 LTS and already has PyTorch 1.7
- Use custom Docker file to explicitly install dependencies with a `requirements` file and setting the application's main function as the entry point
- Build the application docker with the aforementioned `Dockerfile` and `requirements` file on a AArch64 host computer. [Docker Buildx](https://docs.docker.com/buildx/working-with-buildx/) can also be used to build multi-platform images, though it is not used in this example
- On the AArch64 host machine, use `docker run` command or a script to launch the application docker

## Steps
### Create the MONAI Deploy Application
For general guidance on how to build a deploy application using MONAI Deploy App SDK, and test it on x86 with the Packager and Runner, please refer to [Developing with SDK](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/developing_with_sdk/index.html).

For the specific example on building and running a segmentation application, e.g. the Spleen segmentation application, please see [Creating a Segmentation App](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/getting_started/tutorials/03_segmentation_app.html).

In the following sections, the UNETR Multi-organ Segmentation application will be used as an example.

### Create the requirements file
Without using the MONAI Deploy App SDK Packager to automatically detect the dependencies of an application, one has to explicitly create the `requierments.txt` file to be used in the `Dockerfile`. Create the `requirements.txt` file in the application's folder with the content shown below,
```bash
monai>=0.6.0
monai-deploy-app-sdk>=0.1.0
nibabel
numpy>=1.17
pydicom>=1.4.2
torch>=1.5
```
Note: The base image to be used already has torch 1.7 and numpy 19.5 pre-installed.

### Crete the Custom Dockerfile
Create the `Dockerfile` in the application folder with content shown below,

```bash
ARG CONTAINER_REGISTRY=nvcr.io/nvidia/clara-agx
ARG AGX_PT_TAG=21.05-1.7-py3
FROM ${CONTAINER_REGISTRY}/agx-pytorch:${AGX_PT_TAG}

# This is the name of the folder containing the application files.
ENV MY_APP_NAME="ai_unetr_seg_app"

USER root

RUN pip3 install --no-cache-dir --upgrade setuptools==57.4.0 wheel==0.37.0

WORKDIR /opt/$MY_APP_NAME

COPY ./$MY_APP_NAME/requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application source code.
COPY ./$MY_APP_NAME ./
ENTRYPOINT python3 ./__main__.py -i /input -m /model/model.ts -o /output
```
Note that
- The application files are copied to `/opt/unetr_seg_app` in this example
- The input DICOM instances are in folder `/input`
- The Torch Script model file `model.ts` is in `/model`
- The applicaton output will be in `/output`

### Build the Docker Image targeting AArch64
Copy the application folder including the `requirements.txt` and `Dockerfile` to the working directory, e.g. `my_apps`, on a AArch64 host machine, and ensure Docker is already installed. The application folder structure looks like below,
```bash
my_apps
 └─ ai_unetr_seg_app
    ├── app.py
    ├── Dockerfile
    ├── __init__.py
    ├── __main__.py
    ├── requirements.txt
    └── unetr_seg_operator.py
```

In working directory `my_apps`, build the Dcoker image, named `ai_unetr_seg_app` with the default tag `default`with the following command,
```bash
docker build -t ai_unetr_seg_app -f ai_unetr_seg_app/Dockerfile .
```
### Run the Application Docker Locally
On launching the application docker, input DICOM instances as well model file `model.ts` must be available, and the output folder may be a mounted NFS file share hosted on a remote machine.
A sample shell script is provided below,
```
# Root of the datasets folder, change as needed
OUTPUT_ROOT="/mnt/nfs_clientshare"
DATASETS_FOLDER="datasets"

# App specific parameters, change as needed and ensure contents are present.
APP_DATASET_FOLDER="unetr_dataset"
INPUT_FOLDER="/media/m2/monai_apps/input"
MODEL_FOLDER="/media/m2/monai_apps/models/unetr"
DOCKER_IMAGE="ai_unetr_seg_app"

APP_DATASET_PATH=${OUTPUT_ROOT}/${DATASETS_FOLDER}/${APP_DATASET_FOLDER}
echo "Set to save rendering dataset to: ${APP_DATASET_PATH} ..."
docker run -t --rm --shm-size=1G\
        -v ${INPUT_FOLDER}:/input \
        -v ${MODEL_FOLDER}:/model \
        -v ${APP_DATASET_PATH}:/output \
        ${DOCKER_IMAGE}
echo "${DOCKER_IMAGE} completed."
echo
echo "Rendering dataset files are saved in the folder, ${APP_DATASET_PATH}:"
ls ${APP_DATASET_PATH
```

Once application docker terminates, check the application output in the folder shown in the console log. 
