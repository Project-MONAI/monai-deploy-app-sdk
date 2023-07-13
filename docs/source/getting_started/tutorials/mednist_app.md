# Creating MedNIST Classifier App

This tutorial demos the process of packaging up a trained model using MONAI Deploy App SDK into an artifact which can be run as a local program performing inference, a workflow job doing the same, and a Docker containerized workflow execution.

## Setup

```bash
# Create a virtual environment with Python 3.8.
# Skip if you are already in a virtual environment.
# (JupyterLab dropped its support for Python 3.6 since 2021-12-23.
#  See https://github.com/jupyterlab/jupyterlab/pull/11740)
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

# After downloading mednist_classifier_data.zip from the web browser or using gdown,
unzip -o mednist_classifier_data.zip

# Install necessary packages from the app
pip install monai Pillow

# Local execution of the app
python examples/apps/mednist_classifier_monaideploy/mednist_classifier_monaideploy.py -i input/AbdomenCT_007000.jpeg -o output -m classifier.zip

# Package app (creating MAP docker image) using `-l DEBUG` option to see progress.
# This assumes that nvidia docker is installed in the local machine.
# Please see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker to install nvidia-docker2.
monai-deploy package examples/apps/mednist_classifier_monaideploy/mednist_classifier_monaideploy.py \
    --tag mednist_app:latest \
    --model classifier.zip \
    -l DEBUG

# For AMD GPUs, nvidia-docker is not required. Use --base [base image] option to override the docker base image.
# Please see https://hub.docker.com/r/rocm/pytorch for rocm/pytorch docker images.
monai-deploy package -b rocm/pytorch:rocm5.4.1_ubuntu20.04_py3.7_pytorch_1.12.1 \
    examples/apps/mednist_classifier_monaideploy/mednist_classifier_monaideploy.py \
    --tag mednist_app:latest \
    --model classifier.zip \
    -l DEBUG

# Run the app with docker image and input file locally
monai-deploy run mednist_app:latest input output
cat output/output.json
```
