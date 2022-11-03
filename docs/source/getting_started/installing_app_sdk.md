# Installing App SDK

MONAI Deploy App SDK is available as a Python package through [Python Package Index (PyPI)](https://pypi.org/project/monai-deploy-app-sdk/).

MONAI Deploy App SDK's core functionality is written in Python 3 (>= 3.7) for Linux.

```bash
pip install monai-deploy-app-sdk
```

If you have installed MONAI Deploy App SDK previously, you can upgrade to the latest version with:

```bash
pip install --upgrade monai-deploy-app-sdk
```

:::{note}
For packaging your application, [MONAI Application Packager](/developing_with_sdk/packaging_app) and [MONAI Application Runner (MAR)](/developing_with_sdk/executing_packaged_app_locally) requires NVIDIA Docker (NVIDIA Container Toolkit) installed:

<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>

Currently, `nvcr.io/nvidia/pytorch:22.08-py3` base Docker image is used by [MONAI Application Packager](/developing_with_sdk/packaging_app) by default.

The image size is large so please pull the image in advance to save time.

```bash
docker pull nvcr.io/nvidia/pytorch:22.08-py3
```

:::

:::{note}
Windows users can install [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) to use MONAI Deploy App SDK.
:::
