# MONAI Application Runner

The MONAI Application Runner (MAR) is a command-line utility that allows users to run and test their MONAI Application Package (MAP) locally. MAR is developed to make the running and testing of MAPs locally an easy process for developers and scientists by abstracting away the need to understand the internal details of the MAP. MAR allows the users to specify input and output paths on the local file system which it maps to the input and output of MAP during execution.

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
