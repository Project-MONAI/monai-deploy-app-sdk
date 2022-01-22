# 3) Creating a Segmentation app

## Setup

```bash
# Create a virtual environment with Python 3.7.
# Skip if you are already in a virtual environment.
# (JupyterLab dropped its support for Python 3.6 since 2021-12-23.
#  See https://github.com/jupyterlab/jupyterlab/pull/11740)
conda create -n monai python=3.7 pytorch torchvision jupyterlab cudatoolkit=11.1 -c pytorch -c conda-forge
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
<div style="text-align: center;">
    <iframe width="560" height="315" src="https://www.youtube.com/embed/cqDVxzYt9lY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
```

```{raw} html
<div style="text-align: center;">
    <iframe width="560" height="315" src="https://www.youtube.com/embed/nivgfD4pwWE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
```

```{raw} html
<p style="text-align: center;">
    <a class="sphinx-bs btn text-wrap btn-outline-primary col-md-6 reference external" href="../../_static/notebooks/tutorials/03_segmentation_app.ipynb">
        <span>Download 03_segmentation_app.ipynb</span>
    </a>
</p>
```

## Executing from Shell

```bash
# Clone the github project (the latest version of main branch only)
git clone --branch main --depth 1 https://github.com/Project-MONAI/monai-deploy-app-sdk.git

cd monai-deploy-app-sdk

# Install monai-deploy-app-sdk package
pip install monai-deploy-app-sdk

# Download/Extract ai_spleen_seg_data zip file from https://drive.google.com/file/d/1GC_N8YQk_mOWN02oOzAU_2YDmNRWk--n/view?usp=sharing

# Download ai_spleen_seg_data.zip
pip install gdown
gdown https://drive.google.com/uc?id=1GC_N8YQk_mOWN02oOzAU_2YDmNRWk--n

# After downloading ai_spleen_seg_data.zip from the web browser or using gdown,
unzip -o ai_spleen_seg_data_updated_1203.zip

# Install necessary packages from the app
pip install monai pydicom SimpleITK Pillow nibabel

# Local execution of the app
python examples/apps/ai_spleen_seg_app/app.py -i dcm/ -o output -m model.ts

# Package app (creating MAP docker image) using `-l DEBUG` option to see progress.
# This assumes that nvidia docker is installed in the local machine.
# Please see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker to install nvidia-docker2.
monai-deploy package examples/apps/ai_spleen_seg_app --tag seg_app:latest --model model.ts -l DEBUG

# Run the app with docker image and input file locally
monai-deploy run seg_app:latest dcm/ output
```
