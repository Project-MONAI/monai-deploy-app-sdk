# MONAI APPLICATION RUNNER

The MONAI Application Runner (MAR) allows users to run and test their MONAI Application Package (MAP) locally. MAR allows the users to specify input and output directories which it maps to the input and output of MAP during execution.

## Syntax
```
monai-deploy run <container-image-name>[:tag] <input-dir> <output-dir> [-v|--verbose] [-q|--quiet]
```
### Arguments

#### Positional arguments:

| Name     | Format                           | Description                                              |
| -------- | -------------------------------- | -------------------------------------------------------- |
| MAP      | `container-image-name[:tag]`     | MAP container image name with or without image tag.      |
| input    | Directory                        | Local directory that contains input dataset for the MAP. |
| output   | Directory                        | Local directory to store output from the executing MAP.  |

#### Optional arguments:

| Name                | Shorthand  | Default    | Description                                                       |
| ------------------- | ---------- | ---------- | --------------------------------------------------------------    |
| quiet               | -q         | False      | Execute MAP quietly without printing container logs onto console. |
| verbose             | -v         | False      | Verbose mode.                                                     |