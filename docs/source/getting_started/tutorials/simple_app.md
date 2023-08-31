# Creating a Simple Image Processing App

This tutorial shows how a simple image processing application can be created with MONAI Deploy App SDK.

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

../../notebooks/tutorials/01_simple_app.ipynb
```

<!-- It is not centered with {link-button} ...
```{link-button} ../../_static/notebooks/tutorials/01_simple_app.ipynb
:text: Download 01_simple_app.ipynb
:classes: btn-outline-primary col-md-6
``` -->

```{raw} html
<p style="text-align: center;">
    <a class="sphinx-bs btn text-wrap btn-outline-primary col-md-6 reference external" href="../../_static/notebooks/tutorials/01_simple_app.ipynb">
        <span>Download 01_simple_app.ipynb</span>
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

# Install necessary packages from the app. Can simply run `pip install -r examples/apps/simple_imaging_app/requirements.txt`
pip install scikit-image
pip install setuptools

# See the input file exists in the default `input`` folder in the current working directory
ls examples/apps/simple_imaging_app/input/

# Local execution of the app with output file in the `output` folder in the current working directory
python examples/apps/simple_imaging_app/app.py

# Check the output file
ls output

# Package app (creating MAP docker image) using `-l DEBUG` option to see progress. Note the container image name is postfixed with platform info.
# This assumes that nvidia docker is installed in the local machine.
# Please see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker to install nvidia-docker2.

monai-deploy package examples/apps/simple_imaging_app -c examples/apps/simple_imaging_app/app.yaml -t simple_app:latest --platform x64-workstation -l DEBUG

# Show the application and package manifest files of the MONAI Application Package

docker images | grep simple_app
docker run --rm simple_app-x64-workstation-dgpu-linux-amd64:latest show

# Run the MAP container image with MONAI Deploy MAP Runner, with a cleaned output folder
rm -rf output
monai-deploy run simple_app-x64-workstation-dgpu-linux-amd64:latest -i input -o output

# Check the output file
ls output
```
