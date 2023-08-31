# Core concepts

This document introduces the basic concepts of the MONAI Deploy App SDK. If you are eager to try out the SDK in practice, you can start with the [tutorial](/getting_started/tutorials/index). After the tutorial, you can return to this document to learn more about how MONAI Deploy App SDK works.

As described in the [Architecture of MONAI Deploy App SDK](/introduction/architecture), an application is implemented by subclassing [Application](/modules/_autosummary/monai.deploy.core.Application) class.

```{mermaid}
:alt: Application class diagram
:align: center

%%{init: {"theme": "base", "themeVariables": { "fontSize": "12px"}} }%%

classDiagram
    direction TB
    class Application {
    }

    class Graph {
    }

    class Operator {
    }

    <<abstract>> Application
    <<abstract>> Graph
    <<abstract>> Operator

    Application --> Graph : makes use of
    Graph "1" --> "many" Operator : contains
```

[Application](/modules/_autosummary/monai.deploy.core.Application) represents a workflow as a [Graph](/modules/graphs) and the graph handles [Operator](/modules/_autosummary/monai.deploy.core.Operator)s which are computational tasks.

To develop and deploy your MONAI App, you can follow the steps below (click a node to see the detail):

## 1. Developing Application

```{mermaid}
:alt: Developing Application
:align: center
:caption: ⠀⠀Steps to develop an application

%%{init: {"theme": "base", "themeVariables": { "fontSize": "12px"}} }%%

graph TD

    Design(Designing Workflow)
    --> Operator(Creating Operator classes)
    --> App(Creating Application class)
    --> ExecApp(Executing app locally)


    click Design "./designing_workflow.html" "Go to the document" _self
    click Operator "./creating_operator_classes.html" "Go to the document" _self
    click App "./creating_application_class.html" "Go to the document" _self
    click ExecApp "./executing_app_locally.html" "Go to the document" _self
```
<!-- In the above caption text, it uses Unicode blank characters('⠀⠀') in front of the text to align to center (somehow, it is misaligned little bit) -->

First, you will need to design the workflow of your application that defines [Operator](/modules/_autosummary/monai.deploy.core.Operator)s (tasks) and flows among them. Once the workflow is designed, you can start implementing operator classes for those that you cannot use existing operators as they are. Then implement an [Application](/modules/_autosummary/monai.deploy.core.Application) class to make a workflow graph with the operators.

You can execute and debug your application locally in a Jupyter notebook or through CLI.

## 2. Packaging, Local-Running, and Deploying Application Package

```{mermaid}
:alt: Application class diagram
:align: center
:caption: ⠀⠀Steps to package, local-running, and deploying a MONAI Application Package (MAP)

%%{init: {"theme": "base", "themeVariables": { "fontSize": "12px"}} }%%

graph TD

    Package(Packaging app)
    --> ExecPackage(Executing packaged app locally)
    --> DeployPackage(Deploying to the remote server)

    click Package "./packaging_app.html" "Go to the document" _self
    click ExecPackage "./executing_packaged_app_locally.html" "Go to the document" _self
    click DeployPackage "./deploying_and_hosting_map.html" "Go to the document" _self
```
<!-- In the above caption text, it uses Unicode blank characters('⠀⠀') in front of the text to align to center (somehow, it is misaligned little bit) -->

After your application is tested and verified well and you feel you made a great application :), it's time to package your application, test locally, and deploy it to the remote server.
