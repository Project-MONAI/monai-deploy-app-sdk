# Creating a Segmentation App Supporting Multiple Models

This tutorial shows how to create an inference application with multiple models, focusing on model files organization, inferring with named model in the application, and packaging.

The models used in this example are trained with MONAI, and are packaged in the [MONAI Bundle](https://docs.monai.io/en/latest/bundle_intro.html) format.

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

../../notebooks/tutorials/05_multi_model_app.ipynb
```

```{raw} html
<p style="text-align: center;">
    <a class="sphinx-bs btn text-wrap btn-outline-primary col-md-6 reference external" href="../../_static/notebooks/tutorials/05_multi_model_app.ipynb">
        <span>Download 05_multi_model_app.ipynb</span>
    </a>
</p>
```

## Executing from Shell

```bash
# Clone the github project (the latest version of main branch only)
git clone --branch main --depth 1 https://github.com/Project-MONAI/monai-deploy-app-sdk.git

cd monai-deploy-app-sdk

# Install monai-deploy-app-sdk package
pip install --upgrade monai-deploy-app-sdk

# Download the zip file containing both the model and test data
pip install gdown
gdown https://drive.google.com/uc?id=1llJ4NGNTjY187RLX4MtlmHYhfGxBNWmd

# After downloading it using gdown, unzip the zip file saved by gdown
rm -rf dcm && rm -rf multi_models
unzip -o ai_multi_model_bundle_data.zip

# Install necessary packages from the app; note that numpy-stl and trimesh are only
# needed if the application uses the STL Conversion Operator
pip install monai torch pydicom highdicom SimpleITK Pillow nibabel scikit-image numpy-stl trimesh

# Use env variables for input, output, and model paths for local running of Python application
export HOLOSCAN_INPUT_PATH=dcm
export HOLOSCAN_MODEL_PATH=multi_models
export HOLOSCAN_OUTPUT_PATH="output"
export HOLOSCAN_LOG_LEVEL=TRACE

# Local execution of the app directly
python examples/apps/ai_multi_ai_app/app.py

# Package app (creating MAP docker image) using `-l DEBUG` option to see progress.
# This assumes that nvidia docker is installed in the local machine.
# Please see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker to install nvidia-docker2.
monai-deploy package examples/apps/ai_multi_ai_app \
    --tag multi_model_app:latest \
    --config examples/apps/ai_multi_ai_app/app.yaml \
    --models multi_models \
    --platform x64-workstation \
    -l DEBUG

# Note: for AMD GPUs, nvidia-docker is not required, but the dependency of the App SDK, namely Holoscan SDK
#       has not been tested to work with a ROCM base image.

# Run the app with docker image and input file locally
rm -rf output
monai-deploy run multi_model_app-x64-workstation-dgpu-linux-amd64:latest -i dcm -o output
ls -R output
```
