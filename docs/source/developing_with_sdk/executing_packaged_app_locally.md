# Executing packaged app locally

**The MONAI Application Runner (MAR)** is a command-line utility to run and test a [MONAI Application Package (MAP)](https://github.com/Project-MONAI/monai-deploy/blob/main/guidelines/monai-application-package.md) locally, allowing the users to specify input and output paths on the local file system which it then maps to the input and output of MAP during execution.

MAR is developed to make the running and testing of MAPs locally an easy process for developers and scientists by abstracting away the need to understand the internal details of the MAP. It makes use of the `run` command of the [Holoscan SDK CLI](https://docs.nvidia.com/holoscan/sdk-user-guide/cli/cli.html), albeit with `monai-deploy run` as the actual command.

## Setting Up

MONAI Application Runner is installed as part of the MONAI Deploy App SDK, and can be accessed as a `run` subcommand to the CLI. You can see the help message for MAR using the following command:

```bash
monai-deploy run --help
```

As can be seen in the following output, there are many optional arguments, though only a few are required in typical use cases.

```bash
usage: monai-deploy run [-h] [-l {DEBUG,INFO,WARN,ERROR,CRITICAL}] [--address ADDRESS] [--driver]
                        [-i <input>] [-o <output>] [-f FRAGMENTS] [--worker]
                        [--worker-address WORKER_ADDRESS] [--config CONFIG] [--name NAME] [-n NETWORK]
                        [--nic NIC] [-r] [-q] [--shm-size SHM_SIZE] [--terminal] [--uid UID] [--gid GID]
                        <image[:tag]>

positional arguments:
  <image[:tag]>         HAP/MAP image name.

optional arguments:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARN,ERROR,CRITICAL}, --log-level {DEBUG,INFO,WARN,ERROR,CRITICAL}
                        set the logging level (default: INFO)
  --address ADDRESS     address ('[<IP or hostname>][:<port>]') of the App Driver. If not specified, the App
                        Driver uses the default host address ('0.0.0.0') with the default port number
                        ('8765').
  --driver              run the App Driver on the current machine. Can be used together with the '--worker'
                        option to run both the App Driver and the App Worker on the same machine.
  -i <input>, --input <input>
                        input data directory path.
  -o <output>, --output <output>
                        output data directory path.
  -f FRAGMENTS, --fragments FRAGMENTS
                        comma-separated names of the fragments to be executed by the App Worker. If not
                        specified, only one fragment (selected by the App Driver) will be executed. 'all'
                        can be used to run all the fragments.
  --worker              run the App Worker.
  --worker-address WORKER_ADDRESS
                        address (`[<IP or hostname>][:<port>]`) of the App Worker. If not specified, the App
                        Worker uses the default host address ('0.0.0.0') with the default port number
                        randomly chosen from unused ports (between 10000 and 32767).

advanced run options:
  --config CONFIG       path to the configuration file. This will override the configuration file embedded
                        in the application.
  --name NAME           name and hostname of the container to create.
  -n NETWORK, --network NETWORK
                        name of the Docker network this application will be connected to. (default: host)
  --nic NIC             name of the network interface to use with a distributed multi-fragment application.
                        This option sets UCX_NET_DEVICES environment variable with the value specified.
  -r, --render          enable rendering (default: False); runs the container with required flags to enable
                        rendering of graphics.
  -q, --quiet           suppress the STDOUT and print only STDERR from the application. (default: False)
  --shm-size SHM_SIZE   sets the size of /dev/shm. The format is
                        <number(int,float)>[MB|m|GB|g|Mi|MiB|Gi|GiB]. Use 'config' to read the shared memory
                        value defined in the app.json manifest. If not specified, the container is launched
                        using '--ipc=host' with host system's /dev/shm mounted.
  --terminal            enters terminal with all configured volume mappings and environment variables.
                        (default: False)

security options:
  --uid UID             runs the container with the UID. (default:1000)
  --gid GID             runs the container with the GID. (default:1000)
```

## Syntax

For use cases where a MAP reads from a input folder and saves results to a output folder, the command is as simply as the following
```bash
monai-deploy run <container-image-name>[:tag] -i <input> -o <output>
```

## Example

After you have written your application using MONAI Application SDK and have packaged the application, let's try running it on your workstation using the MONAI Application Runner.

### Example MAP and input

Given the following information.

* MAP name and tag : `my_app-x64-workstation-dgpu-linux-amd64:latest`
* Input folder  : `./input`
* Output folder : `./output`

### Launching the application

```bash
monai-deploy run my_app-x64-workstation-dgpu-linux-amd64:latest -i input -o output
```

Output:

```bash
[2023-08-23 16:24:20,007] [INFO] (runner) - Checking dependencies...
[2023-08-23 16:24:20,007] [INFO] (runner) - --> Verifying if "docker" is installed...

[2023-08-23 16:24:20,007] [INFO] (runner) - --> Verifying if "docker-buildx" is installed...

[2023-08-23 16:24:20,008] [INFO] (runner) - --> Verifying if "my_app-x64-workstation-dgpu-linux-amd64:latest" is available...

[2023-08-23 16:24:20,081] [INFO] (runner) - Reading HAP/MAP manifest...
Successfully copied 2.56kB to /tmp/tmp21lldix4/app.json
Successfully copied 2.05kB to /tmp/tmp21lldix4/pkg.json
[2023-08-23 16:24:20,232] [INFO] (runner) - --> Verifying if "nvidia-ctk" is installed...

[2023-08-23 16:24:20,433] [INFO] (common) - Launching container (56c721f5a48b) using image 'my_app-x64-workstation-dgpu-linux-amd64:latest'...
    container name:      flamboyant_galileo
    host name:           *****
    network:             host
    user:                1000:1000
    ulimits:             memlock=-1:-1, stack=67108864:67108864
    cap_add:             CAP_SYS_PTRACE
    ipc mode:            host
    shared memory size:  67108864
    devices:
2023-08-23 23:24:21 [INFO] Launching application python3 /opt/holoscan/app/app.py ...
...
...
[2023-08-23 16:24:55,490] [INFO] (common) - Container 'flamboyant_galileo'(56c721f5a48b) exited.
```

:::{note}

* Currently MAR does not validate all resources specified in the MAP manifest.
:::
