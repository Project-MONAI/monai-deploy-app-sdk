<p align="center">
<img src="https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/docs/images/MONAI-logo-color.png" width="50%" alt='project-monai'>
</p>

💡 If you want to know more about MONAI Deploy WG vision, overall structure, and guidelines, please read <https://github.com/Project-MONAI/monai-deploy> first.

# MONAI Deploy App SDK

MONAI Deploy App SDK offers a framework and associated tools to design, develop and verify AI-driven applications in the healthcare imaging domain.

## Features

- Build medical imaging inference applications using a flexible, extensible & usable Pythonic API
- Easy management of inference applications via programmable Directed Acyclic Graphs (DAGs)
- Built-in operators to load DICOM data to be ingested in an inference app
- Out-of-the-box support for in-proc PyTorch based inference
- Easy incorporation of MONAI based pre and post transformations in the inference application
- Package inference application with a single command into a portable MONAI Application Package
- Locally run and debug your inference application using App Runner

## Installation

To install [the current release](https://pypi.org/project/monai-deploy-app-sdk/), you can simply run:

```bash
pip install monai-deploy-app-sdk  # '--pre' to install a pre-release version.
```

## Getting Started

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

[MedNIST demo](TBD) is available on Colab.

[Examples](https://github.com/Project-MONAI/monai-deploy-app-sdk/tree/main/examples) and [notebook tutorials](https://github.com/Project-MONAI/monai-deploy-app-sdk/tree/main/notebooks) are located at [Project-MONAI/monai-deploy-app-sdk](https://github.com/Project-MONAI/monai-deploy-app-sdk).

Technical documentation is available at [docs.monai.io](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/).

## Contributing

For guidance on making a contribution to MONAI Deploy App SDK, see the [contributing guidelines](https://github.com/Project-MONAI/monai-deploy-app-sdk/blob/main/CONTRIBUTING.md).

## Community

To participate in the MONAI Deploy WG, please review <https://github.com/Project-MONAI/MONAI/wiki/Deploy-Working-Group>.

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
<!-- - Docker Hub: <https://hub.docker.com/r/projectmonai/monai-deploy-app-sdk> -->
