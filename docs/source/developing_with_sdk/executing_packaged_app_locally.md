# Executing packaged app locally

The MONAI Application Runner (MAR) is a command-line utility that allows users to run and test their MONAI Application Package (MAP) locally. MAR is developed to make the running and testing of MAPs locally an easy process for developers and scientists by abstracting away the need to understand the internal details of the MAP. MAR allows the users to specify input and output paths on the local file system which it maps to the input and output of MAP during execution.

## Setting up

MONAI Application Runner comes as a part of the MONAI Deploy CLI and can be accessed as a `run` subcommand to the CLI. You can see the help message for MAR using the following command:

```bash
monai-deploy run --help
```

Output:

```bash
usage: monai-deploy run [-h] [-l {DEBUG,INFO,WARN,ERROR,CRITICAL}] [-q] <map-image[:tag]> <input> <output>

positional arguments:
  <map-image[:tag]>     MAP image name
  <input>               Input data directory path
  <output>              Output data directory path

optional arguments:
  -h, --help            Show this help message and exit
  -l {DEBUG,INFO,WARN,ERROR,CRITICAL}, --log-level {DEBUG,INFO,WARN,ERROR,CRITICAL}
                        Set the logging level (default: INFO)
  -q, --quiet           Suppress the STDOUT and print only STDERR from the application (default: False)
```

## Syntax

```bash
monai-deploy run <container-image-name>[:tag] <input> <output> [-q|--quiet]
```

### Arguments

#### Positional arguments

| Name     | Format                           | Description                                                   |
| -------- | -------------------------------- | ------------------------------------------------------------- |
| MAP      | `container-image-name[:tag]`     | MAP container image name with or without image tag.           |
| input    | directory path                   | Local folder path that contains input dataset for the MAP.    |
| output   | directory path                   | Local folder path to store output from the executing MAP.     |

#### Optional arguments

| Name                | Shorthand  | Default    | Description                                                       |
| ------------------- | ---------- | ---------- | --------------------------------------------------------------    |
| quiet               | -q         | False      | Suppress the STDOUT and print only STDERR from the application.   |

## Example

After you have written your application using MONAI Application SDK and have packaged the application, let's try running it on your workstation using the MONAI Application Runner.

### Example MAP and input

Given the following information.

* MAP name and tag : `my_app:latest`
* Input folder  : `./input`
* Output folder : `./output`

### Launching the application

```bash
monai-deploy run my_app:latest input output
```

Output:

```bash
Checking dependencies...
--> Verifying if "docker" is installed...

--> Verifying if "my_app:latest" is available...

Checking for MAP "my_app:latest" locally
"my_app:latest" found.

Reading MONAI App Package manifest...
INFO:__main__:Operator started: 2021-09-10 21:53:25.363
INFO:__main__:Input path: /input
INFO:__main__:Output path: /output
...
...
continued...
```

### Launching the application in quiet mode

If you only want to run your application such that the STDOUT is suppressed and only STDERR from the application is printed, try using `--quiet` flag.

```bash
monai-deploy run --quiet my_app:latest input output
```

:::{note}

* Currently MAR does not validate all resources specified in the MAP manifest.
* If `gpu` is specified (>0), it executes `nvidia-docker` instead of `docker` internally to make sure that GPU is available inside the container.
:::
