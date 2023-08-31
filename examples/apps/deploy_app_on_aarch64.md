# MONAI Application Package for ARMv8 AArch64 on Linux

This article describes how to containerize a MONAI Deploy application, as a **MONAI Application Package**, targeting ARMv8 AArch64 on Linux.

Section [Packaging App](https://docs.monai.io/projects/monai-deploy-app-sdk/en/stable/developing_with_sdk/packaging_app.html) in the MONAI Deploy App SDK [Users Guide](https://docs.monai.io/projects/monai-deploy-app-sdk/en/stable/index.html) describes the general steps to package an application into a MAP. Building MAPs for AArch64 on Linux with discrete GPU can be done with this simple change of command line options,

`--platform x64-workstation` replaced with

`--platform igx-orin-devkit --platform-config dgpu`