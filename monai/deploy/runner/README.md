# MONAI APPLICATION RUNNER

The MONAI Application Runner (MAR) allows users to run and test their MONAI Application Package (MAP) locally. MAR allows the users to specify input and output path on local file system which it maps to the input and output of MAP during execution.

## Syntax
```
monai-deploy run <container-image-name>[:tag] <input> <output> [-v|--verbose] [-q|--quiet]
```
### Arguments

#### Positional arguments:

| Name     | Format                           | Description                                                   |
| -------- | -------------------------------- | ------------------------------------------------------------- |
| MAP      | `container-image-name[:tag]`     | MAP container image name with or without image tag.           |
| input    | file or directory path           | Local file or folder that contains input dataset for the MAP. |
| output   | path                             | Local path to store output from the executing MAP.       |

#### Optional arguments:

| Name                | Shorthand  | Default    | Description                                                       |
| ------------------- | ---------- | ---------- | --------------------------------------------------------------    |
| quiet               | -q         | False      | Execute MAP quietly without printing container logs onto console. |
| verbose             | -v         | False      | Verbose mode.                                                     |