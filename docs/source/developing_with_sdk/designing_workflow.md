# Designing workflow

## Workflow

A workflow (also called as "pipeline") consists of tasks and a task is realized by [Operator](/modules/_autosummary/monai.deploy.core.Operator).

MONAI Deploy App SDK has useful [built-in operators](/modules/operators) you can reuse (the list of operators would keep growing with your contribution!).

If the built-in operators do not work for you, you can implement your operators. We will guide you throughout the sections in this document.

A workflow is a form of a graph which is usually a Directed Acyclic Graph (DAG).

### One-operator workflow

The simplest form of the workflow would be a one-operator workflow.

```{mermaid}
:align: center
:caption: ⠀⠀A one-operator workflow

%%{init: {"theme": "base", "themeVariables": { "fontSize": "16px"}} }%%

classDiagram
    direction LR

    class MyOperator {
        <in>input_path : DISK
        output_path(out) DISK
    }
```

Above graph shows an [Operator](/modules/_autosummary/monai.deploy.core.Operator) (named `MyOperator`) that load file from the `input_path` in the local file system (`DISK` type), process the file, and write the processed data into `output_path` in the local file system (`DISK` type).

In the workflow description, `DISK` means a [**Storage Type**](/modules/_autosummary/monai.deploy.core.IOType).
If the input or the output data is supposed in memory, you can use `IN_MEMORY` type.

### Linear workflow

Let's see another workflow example whose operators are connected linearly.

```{mermaid}
:align: center
:caption: ⠀⠀A linear workflow

%%{init: {"theme": "base", "themeVariables": { "fontSize": "16px"}} }%%

classDiagram
    direction LR

    Task1 --|> Task2 : output1...input2
    Task2 --|> Task3 : output2...input3

    class Task1 {
        <in>input1 : DISK
        output1(out) IN_MEMORY
    }
    class Task2 {
        <in>input2 : IN_MEMORY
        output2(out) IN_MEMORY
    }
    class Task3 {
        <in>input3 : IN_MEMORY
        output3(out) DISK
    }
```

In this example, **Task1** accepts its input (path) from the disk, and the processed data is set to `output1` of **Task1** in memory.
The memory object (`output1`) is passed to **Task2** as an input (`input2`).
Similarly, `output2` of **Task2** is passed to **Task3** as `input3` and the final result would be saved in a file in the folder where `output3` is referring.

## Data Type and Domain Object

In addition to the [Storage Type](/modules/_autosummary/monai.deploy.core.IOType), each input or output of an operator has another property -- **`Data Type`**.

**Data Type** specifies a [Type Hint](https://www.python.org/dev/peps/pep-0484/) of an input/output of an operator and the type of value in the input/output is verified in [execution time](/developing_with_sdk/executing_app_locally).

Data Type can be a type hint such as `str`, [`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`Union`](https://docs.python.org/3/library/typing.html#typing.Union), `List[Union[str, int]]`.

There are built-in data types that MONAI Deploy App SDK supports which are called **`Domain Objects`**.

**Domain Object** classes inherits [**Domain**](/modules/_autosummary/monai.deploy.core.domain.Domain) class and they provides a useful set of standard input/output data types such as [DataPath](/modules/_autosummary/monai.deploy.core.domain.DataPath), [Image](/modules/_autosummary/monai.deploy.core.domain.Image), [DicomStudy](/modules/_autosummary/monai.deploy.core.domain.DICOMStudy), and so on.

Those domain object classes are controllable by the SDK so can be optimized further in the future.

The full list of Domain Object classes are available [here](/modules/domain_objects).

:::{note}
**`The functionality of mapping DataPath to the input and output of root and leaf operator(s) is absent starting with Release V0.6 of this App SDK, due to the move to rely on Holoscan SDK. It is planned to be re-introduced at a later time. For the time being, the application's input and output folders are passed to the root and leaf operators' constructor, as needed.`**

Among those classes, [**DataPath**](/modules/_autosummary/monai.deploy.core.domain.DataPath) data type is special.

- If an operator in the workflow graph is a root node (a node with no incoming edges) and its input's `(<data type>, <storage type>) == (DataPath, DISK)`, the input path given by the user [during the execution](/developing_with_sdk/executing_app_locally) would be mapped into the input of the operator.
- If an operator in the workflow graph is a leaf node (a node with no outgoing edges) and its output's `(<data type>, <storage type>) == (DataPath, DISK)`, the output path given by the user [during the execution](/developing_with_sdk/executing_app_locally) would be mapped into the output of the operator.

In `A linear workflow` example above, if the workflow is processing the image data, operators' input/output specification would look like this:

- **Task1**
  - **Input** (`input1`): a file path ([`DataPath`](/modules/_autosummary/monai.deploy.core.domain.DataPath), [`DISK`](/modules/_autosummary/monai.deploy.core.IOType))
  - **Output** (`output1`): an image object in memory ([`Image`](/modules/_autosummary/monai.deploy.core.domain.Image), [`IN_MEMORY`](/modules/_autosummary/monai.deploy.core.IOType))
- **Task2**
  - **Input** (`input2`): an image object in memory ([`Image`](/modules/_autosummary/monai.deploy.core.domain.Image), [`IN_MEMORY`](/modules/_autosummary/monai.deploy.core.IOType))
  - **Output** (`output2`): an image object in memory ([`Image`](/modules/_autosummary/monai.deploy.core.domain.Image), [`IN_MEMORY`](/modules/_autosummary/monai.deploy.core.IOType))
- **Task3**
  - **Input** (`input3`): an image object in memory ([`Image`](/modules/_autosummary/monai.deploy.core.domain.Image), [`IN_MEMORY`](/modules/_autosummary/monai.deploy.core.IOType))
  - **Output** (`output3`): a file path ([`DataPath`](/modules/_autosummary/monai.deploy.core.domain.DataPath), [`DISK`](/modules/_autosummary/monai.deploy.core.IOType))

Note that `input1` and `output3` are [DataPath](/modules/_autosummary/monai.deploy.core.domain.DataPath) type with [IOType.DISK](/modules/_autosummary/monai.deploy.core.IOType). Those paths are mapped into input and output paths given by the user during the execution.
:::

:::{note}
The above workflow graph is the same as a [Simple Image Processing App](/getting_started/tutorials/simple_app). Please look at the tutorial to see how such an application can be made with MONAI Deploy App SDK.
:::

## Complex Workflows

### Multiple inputs and outputs

You can design a complex workflow like below that some operators have multi-inputs or multi-outputs.

```{mermaid}
:align: center
:caption: ⠀⠀A complex workflow

%%{init: {"theme": "base", "themeVariables": { "fontSize": "16px"}} }%%

classDiagram
    direction TB

    Reader1 --|> Processor1 : image...{image1,image2}\nmetadata...metadata
    Reader2 --|> Processor2 : roi...roi
    Processor1 --|> Processor2 : image...image
    Processor2 --|> Processor3 : image...image
    Processor2 --|> Notifier : image...image
    Processor1 --|> Writer : image...image
    Processor3 --|> Writer : seg_image...seg_image

    class Reader1 {
        <in>input_path : DISK
        image(out) IN_MEMORY
        metadata(out) IN_MEMORY
    }
    class Reader2 {
        <in>input_path : DISK
        roi(out) IN_MEMORY
    }
    class Processor1 {
        <in>image1 : IN_MEMORY
        <in>image2 : IN_MEMORY
        <in>metadata : IN_MEMORY
        image(out) IN_MEMORY
    }
    class Processor2 {
        <in>image : IN_MEMORY
        <in>roi : IN_MEMORY
        image(out) IN_MEMORY
    }
    class Processor3 {
        <in>image : IN_MEMORY
        seg_image(out) IN_MEMORY
    }
    class Writer {
        <in>image : IN_MEMORY
        <in>seg_image : IN_MEMORY
        output_image(out) DISK
    }
    class Notifier {
        <in>image : IN_MEMORY
    }

```
