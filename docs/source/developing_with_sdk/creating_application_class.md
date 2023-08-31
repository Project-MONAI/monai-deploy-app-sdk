# Creating Application class

## Application Class

The **Application** class is perhaps the most important class that MONAI Deploy App developers will interact with. A developer will inherit a new Application from the [monai.deploy.core.Application](/modules/_autosummary/monai.deploy.core.Application) base class. The base application class provides support for chaining up operators, as well as a mechanism to execute the application. The **<a href="../modules/_autosummary/monai.deploy.core.Application.html#monai.deploy.core.Application.compose">compose()</a>** method of this class needs to be implemented in the inherited class to instantiate Operators and connect them to form a Directed Acyclic Graph.

The following code shows an example Application (`app.py`) code:

```{code-block} python
---
lineno-start: 1
emphasize-lines: 20
caption: |
    An Application class definition example (app.py)
---
from monai.deploy.core import Application, env, resource


class App(Application):
    """This is a very basic application.

    This showcases the MONAI Deploy application framework.
    """

    # App's name. <class name>('App') if not specified.
    name = "my_app"
    # App's description. <class docstring> if not specified.
    description = "This is a reference application."
    # App's version. <git version tag> or '0.0.0' if not specified.
    version = "0.1.0"

    def compose(self):
        # Execute `self.add_flow()` or `self.add_operator()` methods here.
        pass

if __name__ == "__main__":
    App().run()

```

### compose() method

In `compose()` method, operators are instantiated and connected through <a href="../modules/_autosummary/monai.deploy.core.Application.html#monai.deploy.core.Application.add_flow">self.add_flow()</a>.

> add_flow(source_op, destination_op, port_pairs )

`port_pairs` is a sequence of string tuples mapping from the source operator's named output(s) to the destination operator's named input(s) and its type is `Set[Tuple[str, str]]`.

We can skip specifying `Set[Tuple[str, str]]` if both the number of `source_op`'s outputs and the number of `destination_op`'s inputs are one.
For example, if Operators named `task1` and `task2` has only one input and output (with the label `image`), `self.add_flow(task1, task2)` is same with `self.add_flow(task1, task2, {("image", "image")})` or `self.add_flow(task1, task2, {("image", "image")})`.

```python
    def compose(self):
        task1 = Task1()
        task2 = Task2()

        self.add_flow(task1, task2)
        # self.add_flow(task1, task2, {("image", "image")})
        # self.add_flow(task1, task2, {("image", "image")})
```

> add_operator(operator)

If an operator in the workflow graph is both a root node and a leaf node, you can execute <a href="../modules/_autosummary/monai.deploy.core.Application.html#monai.deploy.core.Application.add_operator">self.add_operator()</a> for adding the operator to the workflow graph of the application.

```python
    def compose(self):
        single_op = SingleOperator()
        self.add_operator(single_op)
```

### if \_\_name\_\_ == "\_\_main\_\_":

```python
if __name__ == "__main__":
    App().run()
```

The above lines in `app.py` are needed to execute the application code by using `python` interpreter.

## \_\_main\_\_.py file

\_\_main\_\_.py file is needed for [MONAI Application Packager](/developing_with_sdk/packaging_app) to detect main application code (`app.py`) when the application is executed with the application folder path (e.g., `python app_folder/`).

```{code-block} python
---
lineno-start: 1
caption: |
    \_\_main\_\_.py file example (assuming that 'App' class is available in 'app.py' file)
---
from app import App

if __name__ == "__main__":
    App().run()
```

## Package Dependency and Resource Requirements

Unlike in previous versions where Python decorators are used to define the resource requirements (such as `cpu`, `memory`, and `gpu`) for the application, a YAML file is required to store such information with sections and attributes as defined in the [MONAI Application Package Specification](https://github.com/Project-MONAI/monai-deploy/blob/main/guidelines/monai-application-package.md). This file is only needed when the application is packaged into a MONAI Application Package container image. When the MAP is run, the executor is expected to parse the resource requirements embedded in the MAP to ensure they are met in the host system.

Similarly, instead of using Python decorators, package dependencies of the application and all of its operators need to be aggregated and stored as a "[requirements.txt](https://pip.pypa.io/en/stable/cli/pip_install/#example-requirements-file)" file, to be installed at [packaging time](/developing_with_sdk/packaging_app).

## Creating a Reusable Application

Like <a href="./creating_operator_classes.html#creating-a-reusable-operator">Operator class</a>, an Application class can be implemented in a way that the common Application class can be reusable.

## Complex compose() Example

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

The above workflow can be expressed like below

```python
    def compose(self):
        reader1 = Reader1()
        reader2 = Reader2()
        processor1 = Processor1()
        processor2 = Processor2()
        processor3 = Processor3()
        notifier = Notifier()
        writer = Writer()

        self.add_flow(reader1, processor1, {("image", "image1"), ("image", "image2"),
                                            ("metadata", "metadata")})
        self.add_flow(reader2, processor2, {("roi", "roi")})
        self.add_flow(processor1, processor2, {("image", "image")})
        self.add_flow(processor1, writer, {("image", "image")})
        self.add_flow(processor2, notifier)
        self.add_flow(processor2, processor3)
        self.add_flow(processor3, writer, {("seg_image", "seg_image")})
```
