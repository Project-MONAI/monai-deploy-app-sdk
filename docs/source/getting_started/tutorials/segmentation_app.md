# Creating a Segmentation App with a TorchScript Model

This tutorial shows how to create an organ segmentation application for a PyTorch model that has been trained with MONAI and saved as TorchScript, without necessarily being a MONAI bundle.

Please note that the following steps are for demonstration purpose. The code pulled from GitHub is not the same as that in the actual Jupyter Notebook, which deliberately does not use the MONAI Bundle Inference Operator.

## Setup

```bash
# Create a virtual environment with Python 3.8.
# Skip if you are already in a virtual environment.
conda create -n monai python=3.8 pytorch torchvision jupyterlab cudatoolkit=11.1 -c pytorch -c conda-forge
conda activate monai

# Launch JupyterLab if you want to work on Jupyter Notebook
jupyter-lab
```

## Executing from Jupyter Notebook

```{toctree}
:maxdepth: 4

../../notebooks/tutorials/03_segmentation_app.ipynb
```

```{raw} html
<p style="text-align: center;">
    <a class="sphinx-bs btn text-wrap btn-outline-primary col-md-6 reference external" href="../../_static/notebooks/tutorials/03_segmentation_app.ipynb">
        <span>Download 03_segmentation_app.ipynb</span>
    </a>
</p>
```

## Executing from Shell
Please note that this part of the example uses the latest application source code on Github, as well as the corresponding test data.
```bash
# Clone the github project (the latest version of main branch only)
git clone --branch main --depth 1 https://github.com/Project-MONAI/monai-deploy-app-sdk.git

cd monai-deploy-app-sdk

# Install monai-deploy-app-sdk package
pip install --upgrade monai-deploy-app-sdk

# Download/Extract ai_spleen_bundle_data zip file from https://drive.google.com/file/d/1cJq0iQh_yzYIxVElSlVa141aEmHZADJh/view?usp=sharing

# Download the zip file containing both the model and test data
pip install gdown
gdown https://drive.google.com/uc?id=1Uds8mEvdGNYUuvFpTtCQ8gNU97bAPCaQ

# After downloading it using gdown, unzip the zip file saved by gdown and
# copy the model file into a folder structure that is required by CLI Packager
rm -rf dcm
unzip -o ai_spleen_seg_bundle_data.zip
rm -rf spleen_model && mkdir -p spleen_model && mv model.ts spleen_model && ls spleen_model

# Install necessary packages from the app; note that numpy-stl and trimesh are only
# needed if the application uses the STL Conversion Operator
pip install monai torch pydicom highdicom SimpleITK Pillow nibabel scikit-image numpy-stl trimesh

# Use env variables for input, output, and model paths for local running of Python application
export HOLOSCAN_INPUT_PATH=dcm
export HOLOSCAN_MODEL_PATH=spleen_model/model.ts
export HOLOSCAN_OUTPUT_PATH="output"
export HOLOSCAN_LOG_LEVEL=TRACE

# Local execution of the app directly or using MONAI Deploy CLI
python examples/apps/ai_spleen_seg_app/app.py

# Package app (creating MAP docker image) using `-l DEBUG` option to see progress.
# This assumes that nvidia docker is installed in the local machine.
# Please see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker to install nvidia-docker2.
monai-deploy package examples/apps/ai_spleen_seg_app \
    --config examples/apps/ai_spleen_seg_app/app.yaml \
    --tag seg_app:latest \
    --models spleen_model/model.ts \
    --platform x64-workstation \
    -l DEBUG

# Note: for AMD GPUs, nvidia-docker is not required, but the dependency of the App SDK, namely Holoscan SDK
#       has not been tested to work with a ROCM base image.

# Run the app with docker image and input file locally
rm -rf output
monai-deploy run seg_app-x64-workstation-dgpu-linux-amd64:latest -i dcm -o output
ls -R output
```
