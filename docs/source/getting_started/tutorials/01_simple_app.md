# 1) Creating a simple image processing app

## Setup

```bash
# Create a virtual environment. Skip if you are already in a virtual environment.
conda create -n monai python=3.6 pytorch torchvision jupyterlab cudatoolkit=11.1 -c pytorch -c conda-forge
conda activate monai

# Launch JupyterLab if you want to work on Jupyter Notebook
jupyter-lab
```

## Executing from Jupyter Notebook

```{toctree}
:maxdepth: 4

../../notebooks/tutorials/01_simple_app.ipynb
```

```{link-button} ../../_static/notebooks/tutorials/01_simple_app.ipynb
:text: Download 01_simple_app.ipynb
:classes: btn-outline-primary col-md-6
```

## Executing from Shell

```bash
# Clone the github project (the latest version of the main branch only)
git clone --branch main --depth 1 https://github.com/Project-MONAI/monai-deploy-app-sdk.git

cd monai-deploy-app-sdk

# Install monai-deploy-app-sdk package
pip install monai-deploy-app-sdk

# Install necessary packages from the app
pip install scikit-image

# Local execution of the app
python examples/apps/simple_imaging_app/app.py -i examples/apps/simple_imaging_app/brain_mr_input.jpg -o output

# Check the output file
ls output

# Package app (creating MAP docker image) using `-l DEBUG` option to see progress.
# This assumes that nvidia docker is installed in the local machine.
# Please see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker to install nvidia-docker2.

monai-deploy package examples/apps/simple_imaging_app -t simple_app:latest -l DEBUG

# Copy a test input file to 'input' folder
mkdir -p input && rm -rf input/*
cp examples/apps/simple_imaging_app/brain_mr_input.jpg input/

# Run the app with docker image and input file locally
monai-deploy run simple_app:latest input output
```
