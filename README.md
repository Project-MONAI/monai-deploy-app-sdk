<p align="center">
<img src="https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/docs/images/MONAI-logo-color.png" width="50%" alt='project-monai'>
</p>

💡 If you want to know more about MONAI Deploy WG vision, overall structure, and guidelines, please read [MONAI Deploy](https://github.com/Project-MONAI/monai-deploy) main repo first.

# MONAI Deploy App SDK
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)


MONAI Deploy App SDK offers a framework and associated tools to design, develop and verify AI-driven applications in the healthcare imaging domain.

## Features

- Build medical imaging inference applications using a flexible, extensible & usable Pythonic API
- Easy management of inference applications via programmable Directed Acyclic Graphs (DAGs)
- Built-in operators to load DICOM data to be ingested in an inference app
- Out-of-the-box support for in-proc PyTorch based inference
- Easy incorporation of MONAI based pre and post transformations in the inference application
- Package inference application with a single command into a portable MONAI Application Package
- Locally run and debug your inference application using App Runner

## User Guide

User guide is available at [docs.monai.io](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/).

## Installation

To install [the current release](https://pypi.org/project/monai-deploy-app-sdk/), you can simply run:

```bash
pip install monai-deploy-app-sdk  # '--pre' to install a pre-release version.
```

## Getting Started

Getting started guide is available at [here](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/getting_started/index.html).

```bash
pip install monai-deploy-app-sdk  # '--pre' to install a pre-release version.

# Clone monai-deploy-app-sdk repository for accessing examples.
git clone https://github.com/Project-MONAI/monai-deploy-app-sdk.git
cd monai-deploy-app-sdk

# Install necessary dependencies for simple_imaging_app
pip install scikit-image

# Execute the app locally
python examples/apps/simple_imaging_app/app.py -i examples/apps/simple_imaging_app/brain_mr_input.jpg -o output

# Package app (creating MAP Docker image), using `-l DEBUG` option to see progress.
monai-deploy package examples/apps/simple_imaging_app -t simple_app:latest -l DEBUG

# Run the app with docker image and an input file locally
## Copy a test input file to 'input' folder
mkdir -p input && rm -rf input/*
cp examples/apps/simple_imaging_app/brain_mr_input.jpg input/
## Launch the app
monai-deploy run simple_app:latest input output
```

### [Tutorials](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/getting_started/tutorials/index.html)

#### [1) Creating a simple image processing app](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/getting_started/tutorials/01_simple_app.html)

#### [2) Creating MedNIST Classifier app](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/getting_started/tutorials/02_mednist_app.html)

YouTube Video:

- [MedNIST Classification Example](https://www.youtube.com/watch?v=WwjilJFHuU4)

### [3) Creating a Segmentation app](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/getting_started/tutorials/03_segmentation_app.html)

YouTube Video:

- [Spleen Organ Segmentation - Jupyter Notebook Tutorial](https://www.youtube.com/watch?v=cqDVxzYt9lY)
- [Spleen Organ Segmentation - Deep Dive](https://www.youtube.com/watch?v=nivgfD4pwWE)

### [4) Deploying Segmentation app with MONAI Inference Service (MIS)](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/getting_started/tutorials/04_mis_tutorial.html)

### [5) Building and deploying Segmentation app with MONAI Inference Service (MIS)](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/getting_started/tutorials/05_full_tutorial.html)

### [Examples](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/getting_started/examples.html)

<https://github.com/Project-MONAI/monai-deploy-app-sdk/tree/main/examples/apps> has example apps that you can see.

- ai_spleen_seg_app
- ai_unetr_seg_app
- dicom_series_to_image_app
- mednist_classifier_monaideploy
- simple_imaging_app

## Contributing

For guidance on making a contribution to MONAI Deploy App SDK, see the [contributing guidelines](https://github.com/Project-MONAI/monai-deploy/blob/main/CONTRIBUTING.md).

## Community

To participate, please join the MONAI Deploy App SDK weekly meetings on the [calendar](https://calendar.google.com/calendar/u/0/embed?src=c_954820qfk2pdbge9ofnj5pnt0g@group.calendar.google.com&ctz=America/New_York) and review the [meeting notes](https://docs.google.com/document/d/1viIh3vyP6_gZBKcnu7gb8fU0tm9aWBOcKCMGezIWNQw/edit#).

Join the conversation on Twitter [@ProjectMONAI](https://twitter.com/ProjectMONAI) or join our [Slack channel](https://forms.gle/QTxJq3hFictp31UM9).

Ask and answer questions over on [MONAI Deploy App SDK's GitHub Discussions tab](https://github.com/Project-MONAI/monai-deploy-app-sdk/discussions).

## Links

- Website: <https://monai.io>
- API documentation: <https://docs.monai.io/projects/monai-deploy-app-sdk>
- Code: <https://github.com/Project-MONAI/monai-deploy-app-sdk>
- Project tracker: <https://github.com/Project-MONAI/monai-deploy-app-sdk/projects>
- Issue tracker: <https://github.com/Project-MONAI/monai-deploy-app-sdk/issues>
- Wiki: <https://github.com/Project-MONAI/monai-deploy-app-sdk/wiki>
- Test status: <https://github.com/Project-MONAI/monai-deploy-app-sdk/actions>
- PyPI package: <https://pypi.org/project/monai-deploy-app-sdk>
