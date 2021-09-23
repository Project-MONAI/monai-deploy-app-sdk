# 2) Creating MedNIST Classifier app

## Setup

```bash
# Create a virtual environment. Skip if you are already in a virtual environment.
conda create -n mednist python=3.6 pytorch jupyterlab cudatoolkit=11.1 -c pytorch -c conda-forge
conda activate mednist

# Launch JupyterLab if you want to work on Jupyter Notebook
jupyter-lab
```

## Executing from Jupyter Notebook (From Scratch)

```{toctree}
:maxdepth: 2

../../notebooks/tutorials/02_mednist_app.ipynb
```

```{link-button} ../../_static/notebooks/tutorials/02_mednist_app.ipynb
:text: Download 02_mednist_app.ipynb
:classes: btn-outline-primary col-md-6
```

## Executing from Jupyter Notebook (Using a Prebuilt Model)

```{toctree}
:maxdepth: 2

../../notebooks/tutorials/02_mednist_app-prebuilt.ipynb
```

```{link-button} ../../_static/notebooks/tutorials/02_mednist_app-prebuilt.ipynb
:text: Download 02_mednist_app-prebuilt.ipynb
:classes: btn-outline-primary btn-block col-md-6
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

# Run the app with docker image and input file locally
monai-deploy run mednist_app:latest input output
cat output/output.json
```
