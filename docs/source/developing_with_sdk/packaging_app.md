# Packaging app

In this step, the MONAI Deploy application can now be built into a deployable Docker image using the **MONAI Application Packager**.

## Overview on MONAI Application Packager

The MONAI Application Packager (Packager) is a utility for building an application developed with the MONAI Deploy App SDK into a structured MONAI Application Package (**MAP**). The Packager makes use of the `package` command of the [Holoscan SDK CLI](https://docs.nvidia.com/holoscan/sdk-user-guide/cli/cli.html), albeit with `monai-deploy package` as the actual command.

The MAP produced by the Packager is a deployable and reusable docker image that can be launched [locally](./executing_packaged_app_locally) or [hosted](./deploying_and_hosting_map).

It is required that the application configuration yaml file as well as the dependency requirements file are present in the application folder, as described [here](./creating_application_class).

### Basic Usage of MONAI Application Packager

```bash
monai-deploy package <APP_PATH> --config <COMFIG> --tag <TAG> --platform <x64-workstation> [--models <MODEL_PATH>] [--log-level <LEVEL>] [-h]
```

#### Required Arguments

* `<APP_PATH>`: A path to MONAI Deploy Application folder or main code.
* `--config, -c <CONFIG>`: Path to the application configuration file.
* `--tag, -t <TAG>`: A MAP name and optionally a tag in the 'name:tag' format.
* `--platform <PLATFORM>`: Platform type of the container image, must be `x64-workstation` for x86-64 system.

:::{note}
If `<APP_PATH>` refers to a python code (such as `./my_app.py`), the whole parent folder of the file would be packaged into the MAP container image, effectively the same as specifying the application folder path which includes `__main__.py` file. In both cases, the image's environment variable, `HOLOSCAN_APPLICATION` will be set with the path of the application folder in the image, i.e. `HOLOSCAN_APPLICATION=/opt/holoscan/app`. So, it is essential to provide the `__main__.py` file in the application folder, which usually would look like below:

   ```python
   # Assuming that the class `App` (that inherits Application) is defined in `app.py`.
   from app import App

   if __name__ == "__main__":
       App().run()
   ```

:::

#### Optional Arguments

The following lists a few most likely used [optional arguments](https://docs.nvidia.com/holoscan/sdk-user-guide/cli/package.html)

* `--models, -m <MODEL_PATH>`: Path to a model file or a directory containing all models as subdirectories
* `-h, --help`: Show the help message and exit.
* `--log-level, -l <LEVEL>`: Set the logging level (`"DEBUG"`, `"INFO"`, `"WARN"`, `"ERROR"`, or `"CRITICAL"`).
* `--sdk-version <SDK_VERSION>`: Set the version of the SDK that is used to build and package the Application. If not specified, the packager attempts to detect the installed version.
* `--monai-deploy-sdk-file <MONAI_DEPLOY_SDK_FILE>`: Path to the MONAI Deploy App SDK Python distribution [Wheel](https://peps.python.org/pep-0427) file.
* `--version <VERSION>`: An optional version number of the application. When specified, it overrides the value specified in the configuration file.



## Example of using MONAI Application Packager

Given an example MONAI Deploy App SDK application with its code residing in a directory `./my_app`, packaging this application with the Packager to create a Docker image tagged `my_app:latest` would look like this:

```bash
$ monai-deploy package ./my_app -c --config ./my_app/app.yaml -t my_app:latest --models ./model.ts --platform x64-workstation

Building MONAI Application Package...
Successfully built my_app:latest
```

The MAP image name will be postfixed with the platform info to become `my_app-x64-workstation-dgpu-linux-amd64:latest`, and will be seen in the list of container images on the user's local machine when the command `docker images` is run.

:::{note}
* The current implementation of the Packager **ONLY** supports a set of [platform](https://docs.nvidia.com/holoscan/sdk-user-guide/cli/package.html#platform-platform) specific base images from `nvcr.io` as base images for the MAP.

* To package a MAP to run on ARMv8 AArch64 on Linux with discrete GPU, replace the commandline option `--platform x64-workstation` with `--platform igx-orin-devkit --platform-config dgpu`. It has been tested on [NVIDIA IGX Orin](https://www.nvidia.com/en-us/edge-computing/products/igx/).
:::

## Next Step

See the next page to learn more on how to locally run a MONAI application package image built by the Packager.