# Packaging app

In this step, the MONAI Deploy application can now be built into a deployable Docker image using the **MONAI Application Packager**.

## Overview on MONAI Application Packager

The MONAI Application Packager (Packager) is a utility for building an application developed with the MONAI Deploy App SDK into a structured MONAI Application Package (**MAP**).

The MAP produced by the Packager is a deployable and reusable docker image that can be launched [locally](./executing_packaged_app_locally) or [remotely](./deploying_and_hosting_map).

### Basic Usage of MONAI Application Packager

```bash
monai-deploy package <APP_PATH> --tag <TAG> [--model <MODEL_PATH>] [--log-level <LEVEL>] [-h]
```

#### Required Arguments

* `<APP_PATH>`: A path to MONAI Deploy Application folder or main code.
* `--tag, -t <TAG>`: A MAP name and optionally a tag in the 'name:tag' format.

::::{note}
If `<APP_PATH>` refers to a python code (such as `./my_app.py`), only the file would be packaged into the Docker image.

Please specify the application root path which includes `__main__.py` file if you want to package the whole application
folder.

The file usually would look like below:

   ```python
   # Assuming that the class `App` (that inherits Application) is defined in `app.py`.
   from app import App

   if __name__ == "__main__":
       App(do_run=True)
   ```

::::

#### Optional Arguments

The following list contains arguments that have default values or are provided through the SDK if these flags are not invoked.
However, the user can choose to override these values by invoking these optional flags:

* `--model, -m <MODEL_PATH>`: Path to a model file or a directory containing all models as subdirectories
* `--requirements, -r <FILE_PATH>`: Optional path to requirements.txt containing package dependencies of the application
* `--log-level, -l <LEVEL>`: Set the logging level (`"DEBUG"`, `"INFO"`, `"WARN"`, `"ERROR"`, or `"CRITICAL"`).

* `--base <BASE_IMAGE>`: Base Docker Image (overrides default `"nvcr.io/nvidia/pytorch:21.07-py3"`).
* `--input-dir, -i <INPUT_DIR>`: Directory mounted in container for Application Input (overrides default `"input"`).
* `--models-dir <MODELS_DIR>`: Directory mounted in container for Models Path (overrides default `"/opt/monai/models"`).
* `--output-dir, -o <OUTPUT_DIR>`: Directory mounted in container for Application Output (overrides default `"output"`).
* `--working-dir, -w <WORKDIR>`: Directory mounted in container for Application (overrides default `"/var/monai"`).
* `--no-cache`, `-n`: Do not use cache when building image.
* `--timeout <VALUE>`: Timeout (overrides default `0`. In seconds.).
* `--version`: Set the version of the Application (overrides default `"0.0.0"`).
* `-h, --help`: Show the help message and exit.

## Example of using MONAI Application Packager

Given an example MONAI Deploy App SDK application with its code residing in a directory `./my_app`, packaging this application with the Packager to create a Docker image tagged `my_app:latest` would look like this:

```bash
$ monai-deploy package ./my_app -t my_app:latest --model ./model.pt

Building MONAI Application Package...
Successfully built my_app:latest
```

The MAP image `my_app:latest` will be seen in the list of container images on the user's local machine when the command `docker images` is run. The MAP image `my_app:latest` will be able to run [locally](./executing_packaged_app_locally) or [remotely](./deploying_and_hosting_map).

```{note}
* The current implementation (as of `0.1.0`) of the Packager **ONLY** supports [**CUDA**](https://ngc.nvidia.com/catalog/containers/nvidia:cuda) and [**Pytorch**](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) images from `nvcr.io` as base images for the MAP. There are efforts in progress to add support for smaller images from [dockerhub](https://hub.docker.com/).
* This release was developed and tested with CUDA 11 and Python 3.8. There are efforts in progress to add support for different versions of the CUDA Toolkit.
```

## Next Step

See the next page to learn more on how to locally run a MONAI application package image built by the Packager.