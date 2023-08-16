# Creating a Segmentation App Including Including Visualization with Clara-Viz

This tutorial shows how to create an organ segmentation application for a PyTorch model that has been trained with MONAI, with visualization of the segmentation and input images with Clara Viz integration.

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

../../notebooks/tutorials/03_segmentation_viz_app.ipynb
```

```{raw} html
<p style="text-align: center;">
    <a class="sphinx-bs btn text-wrap btn-outline-primary col-md-6 reference external" href="../../_static/notebooks/tutorials/03_segmentation_viz_app.ipynb">
        <span>Download 03_segmentation_viz_app.ipynb</span>
    </a>
</p>
```