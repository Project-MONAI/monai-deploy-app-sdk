# 3) Creating a Segmentation app

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

../../notebooks/tutorials/03_segmentation_app.ipynb
```

```{link-button} ../../_static/notebooks/tutorials/03_segmentation_app.ipynb
:text: Download 03_segmentation_app.ipynb
:classes: btn-outline-primary col-md-6
```

## Executing from Shell

```bash
# Clone the github project (the latest version of main branch only)
git clone --branch main --depth 1 https://github.com/Project-MONAI/monai-deploy-app-sdk.git

cd monai-deploy-app-sdk

# Install monai-deploy-app-sdk package
pip install monai-deploy-app-sdk

# Download/Extract ai_spleen_seg_data.zip from https://drive.google.com/file/d/1uTQsm8omwimBcp_kRXlduWBP2M6cspr1/view?usp=sharing

# Download ai_spleen_seg_data.zip
pip install gdown
gdown https://drive.google.com/uc?id=1uTQsm8omwimBcp_kRXlduWBP2M6cspr1

# After downloading ai_spleen_seg_data.zip from the web browser or using gdown,
unzip -o ai_spleen_seg_data.zip

# Install necessary packages from the app
pip install monai pydicom SimpleITK Pillow nibabel

# Local execution of the app
python examples/apps/ai_spleen_seg_app/app.py -i dcm/ -o output -m model.pt

# Package app (creating MAP docker image) using `-l DEBUG` option to see progress.
# This assumes that nvidia docker is installed in the local machine.
# Please see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker to install nvidia-docker2.
monai-deploy package examples/apps/ai_spleen_seg_app --tag seg_app:latest --model model.pt -l DEBUG

# Run the app with docker image and input file locally
monai-deploy run seg_app:latest dcm/ output
```
