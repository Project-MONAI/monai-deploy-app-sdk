# Installing App SDK

MONAI Deploy App SDK is available as a Python package through [Python Package Index (PyPI)](https://pypi.org/project/monai-deploy-app-sdk/).

MONAI Deploy App SDK's core functionality is written in Python 3 (>= 3.8) for Linux.

```bash
pip install monai-deploy-app-sdk
```

If you have installed MONAI Deploy App SDK previously, you can upgrade to the latest version with:

```bash
pip install --upgrade monai-deploy-app-sdk
```

:::{note}
For packaging and running your application using [MONAI Application Packager](/developing_with_sdk/packaging_app) and [MONAI Application Runner (MAR)](/developing_with_sdk/executing_packaged_app_locally), [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) needs to be installed.

For version 1.0.0, `nvcr.io/nvidia/clara-holoscan/holoscan:v1.0.3-dgpu` is used by [MONAI Application Packager](/developing_with_sdk/packaging_app) as base image for X86-64 in Linux system, though this will change with versions.

The base image size is large, so it is recommended to pull the image in advance to save time. Note that the container image tag in the following example, e.g. `v1.0.3`, corresponds to the underlying Holoscan SDK version.

```bash
docker pull nvcr.io/nvidia/clara-holoscan/holoscan:v1.0.3-dgpu
```

:::

:::{note}
Windows users can install [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) to use MONAI Deploy App SDK.
:::
