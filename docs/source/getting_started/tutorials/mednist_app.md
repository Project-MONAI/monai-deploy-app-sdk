# Creating MedNIST Classifier App

This tutorial demos the process of packaging up a trained model using MONAI Deploy App SDK into an artifact which can be run as a local program performing inference, a workflow job doing the same, and a Docker containerized workflow execution.

## Setup

```bash
# Create a virtual environment with Python 3.8.
# Skip if you are already in a virtual environment.
conda create -n mednist python=3.8 pytorch jupyterlab cudatoolkit=11.1 -c pytorch -c conda-forge
conda activate mednist

# Launch JupyterLab if you want to work on Jupyter Notebook
jupyter-lab
```

## Executing from Jupyter Notebook (From Scratch)

```{toctree}
:maxdepth: 2

../../notebooks/tutorials/02_mednist_app.ipynb
```

```{raw} html
<p style="text-align: center;">
    <a class="sphinx-bs btn text-wrap btn-outline-primary col-md-6 reference external" href="../../_static/notebooks/tutorials/02_mednist_app.ipynb">
        <span>Download 02_mednist_app.ipynb</span>
    </a>
</p>
```

## Executing from Jupyter Notebook (Using a Prebuilt Model)

```{toctree}
:maxdepth: 2

../../notebooks/tutorials/02_mednist_app-prebuilt.ipynb
```

```{raw} html
<div style="text-align: center;">
    <iframe width="560" height="315" src="https://www.youtube.com/embed/WwjilJFHuU4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
```

```{raw} html
<p style="text-align: center;">
    <a class="sphinx-bs btn text-wrap btn-outline-primary col-md-6 reference external" href="../../_static/notebooks/tutorials/02_mednist_app-prebuilt.ipynb">
        <span>Download 02_mednist_app-prebuilt.ipynb</span>
    </a>
</p>
```

## Executing from Shell

```bash
# Clone the github project (the latest version of the main branch only)
git clone --branch main --depth 1 https://github.com/Project-MONAI/monai-deploy-app-sdk.git

cd monai-deploy-app-sdk

# Install monai-deploy-app-sdk package
pip install monai-deploy-app-sdk

# Download/Extract mednist_classifier_data.zip from https://drive.google.com/file/d/1yJ4P-xMNEfN6lIOq_u6x1eMAq1_MJu-E/view?usp=sharing

# Download mednist_classifier_data.zip
pip install gdown
gdown https://drive.google.com/uc?id=1yJ4P-xMNEfN6lIOq_u6x1eMAq1_MJu-E

# After downloading mednist_classifier_data.zip from the web browser or using gdown
unzip -o mednist_classifier_data.zip

# Install necessary packages required by the app
pip install -r examples/apps/mednist_classifier_monaideploy/requirements.txt

# Local execution of the app using environment variables for input, output, and model paths
# instead of command line options, `-i input/AbdomenCT_007000.jpeg -o output -m classifier.zip`
export HOLOSCAN_INPUT_PATH="input/AbdomenCT_007000.jpeg"
export HOLOSCAN_MODEL_PATH="classifier.zip"
export HOLOSCAN_OUTPUT_PATH="output"

python examples/apps/mednist_classifier_monaideploy/mednist_classifier_monaideploy.py

# See the classification result
cat output/output.json

# Package app (creating MAP docker image) using `-l DEBUG` option to see progress.
# This assumes that nvidia docker is installed in the local machine.
# Please see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker to install nvidia-docker2.

# Need to move the model file to a clean folder as a workaround of an known packaging issue in v0.6
mkdir -p mednist_model && rm -rf mednist_model/* && cp classifier.zip mednist_model/

monai-deploy package examples/apps/mednist_classifier_monaideploy/mednist_classifier_monaideploy.py \
    --config examples/apps/mednist_classifier_monaideploy/app.yaml \
    --tag mednist_app:latest \
    --models mednist_model/classifier.zip \
    --platform x64-workstation \
    -l DEBUG

# Note: for AMD GPUs, nvidia-docker is not required, but the dependency of the App SDK, namely Holoscan SDK
#       has not been tested to work with a ROCM base image.

# Run the app with docker image and input file locally
rm -rf output
monai-deploy run mednist_app-x64-workstation-dgpu-linux-amd64:latest -i input -o output
cat output/output.json
```
