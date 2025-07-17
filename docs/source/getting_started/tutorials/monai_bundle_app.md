# Creating a Segmentation App Consuming a MONAI Bundle

This tutorial shows how to create an organ segmentation application for a PyTorch model that has been trained with MONAI and packaged in the [MONAI Bundle](https://docs.monai.io/en/latest/bundle_intro.html) format.

## Setup

```bash
# Create a virtual environment with Python 3.8.
# Skip if you are already in a virtual environment.
# (JupyterLab dropped its support for Python 3.6 since 2021-12-23.
#  See https://github.com/jupyterlab/jupyterlab/pull/11740)
conda create -n monai python=3.8 pytorch torchvision jupyterlab cudatoolkit=11.1 -c pytorch -c conda-forge
conda activate monai

# Launch JupyterLab if you want to work on Jupyter Notebook
jupyter-lab
```

## Executing from Jupyter Notebook

```{toctree}
:maxdepth: 4

../../notebooks/tutorials/06_monai_bundle_app.ipynb
```

```{raw} html
<p style="text-align: center;">
    <a class="sphinx-bs btn text-wrap btn-outline-primary col-md-6 reference external" href="../../_static/notebooks/tutorials/06_monai_bundle_app.ipynb">
        <span>Download 06_monai_bundle_app.ipynb</span>
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

# Download/Extract ai_spleen_bundle_data zip file from https://drive.google.com/file/d/1cJq0iQh_yzYIxVElSlVa141aEmHZADJh/view?usp=sharing

# Download the zip file containing both the model and test data
pip install gdown
gdown https://drive.google.com/uc?id=1Uds8mEvdGNYUuvFpTtCQ8gNU97bAPCaQ

# After downloading it using gdown, unzip the zip file saved by gdown
unzip -o ai_spleen_seg_bundle_data.zip

# Install necessary packages from the app; note that numpy-stl and trimesh are only
# needed if the application uses the STL Conversion Operator
pip install monai torch pydicom highdicom SimpleITK Pillow nibabel scikit-image numpy-stl trimesh

# Local execution of the app directly or using MONAI Deploy CLI
python examples/apps/ai_spleen_seg_app/app.py -i dcm/ -o output -m model.ts
# or alternatively,
monai-deploy exec ../examples/apps/ai_spleen_seg_app/app.py -i dcm/ -o output -m model.ts

# Package app (creating MAP docker image) using `-l DEBUG` option to see progress.
# This assumes that nvidia docker is installed in the local machine.
# Please see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker to install nvidia-docker2.
monai-deploy package examples/apps/ai_spleen_seg_app --tag seg_app:latest --model model.ts -l DEBUG

# For AMD GPUs, nvidia-docker is not required. Use --base [base image] option to override the docker base image.
# Please see https://hub.docker.com/r/rocm/pytorch for rocm/pytorch docker images.
monai-deploy package -b rocm/pytorch:rocm5.4.1_ubuntu20.04_py3.7_pytorch_1.12.1 examples/apps/ai_spleen_seg_app --tag seg_app:latest --model model.ts -l DEBUG

# Run the app with docker image and input file locally
monai-deploy run seg_app:latest dcm/ output
```
