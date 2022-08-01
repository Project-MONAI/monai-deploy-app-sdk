# MONAI Deploy App SDK

## Introduction

Productionizing an AI model is difficult as there is a chasm between training an AI model and deploying it to a production environment. In the healthcare domain, this chasm involves more than just performing inference with a model. An App developer would need to address issues such as: ingestion of medical imaging datasets, performing application-specific pre and post-processing operations,  packaging output from the application, integration with clinical informatics systems & ensuring the right compute resources are specified and utilized by the application at run-time.

The MONAI Application SDK provides a framework to develop, verify, analyze AI-driven healthcare applications and integrate them with clinical information systems using industry-standard protocols such as DICOM & FHIR. The SDK is aimed to support the following activities:

* Pythonic Framework for app development
* A mechanism to locally run and test an app
* A lightweight app analytics module
* A lightweight 2D/3D visualization module
* A developer console that provides a visual interface to all assets needed for developing apps
* A set of sample applications
* API documentation & User's Guide

---

## Scope

The scope of this document is limited to the MONAI Deploy App SDK. There are other subsystems of the MONAI Deploy platform such as MONAI App Server, MONAI App Informatics Gateway. However, this requirements document does not address specifications belonging to those subsystems.

---

## Attributes of a Requirement

For each requirement, the following attributes have been specified:

* **Requirement Body**: This is the text of the requirement which describes the goal and purpose behind the requirement
* **Background**: Provides necessary background to understand the context of the requirements
* **Verification Strategy**: A high-level plan on how to test this requirement at a system level
* **Target Release**: Specifies which release of the MONAI Deploy App SDK this requirement is targeted for

---

## [REQ] Representing application-specific tasks using Operators

The SDK shall enable representing a computational task in a healthcare application using an operator so that each task can be modularized, reused, and debugged in distinct contexts.

### Background

Most healthcare application workflows involve multiple tasks. Each task is a basic unit of work. Having a programmatic way of representing such a task is important as this promotes separation of concern, reusability, and debuggability. Examples of tasks are: loading a DICOM Series into an in-memory volumetric representation, the ability to rescale a volumetric, etc.

### Verification Strategy

Verify that a common set of workflow tasks can be represented by built-in Operators.

### Target Release

MONAI Deploy App SDK 0.1.0

---

## [REQ] Inputs for Operators

The SDK shall provide mechanisms so that each Operator can be designed to ingest one or more inputs.

### Background

Often an application task requires accepting multiple inputs. Having built-in support to model this behavior makes app development easier.

### Verification Strategy

Verify that there is built-in support for multiple inputs for designing an operator.

### Target Release

MONAI Deploy App SDK 0.1.0

---

## [REQ] Outputs for Operators

The SDK shall provide mechanisms so that each Operator can be designed to generate one or more outputs.

### Background

Often an application task requires generating multiple outputs. Having built-in support to model this behavior makes app development easier.

### Verification Strategy

Verify that there is built-in support for multiple outputs for designing an operator.

### Target Release

MONAI Deploy App SDK 0.1.0

---

## [REQ] Representing Workflow With DAG

The SDK shall enable dependencies among source and destination operators in an application using a DAG so that app workflow can be modeled unambiguously. The SDK shall provide a mechanism to link an output port of a source operator to an input port of a destination operator to form the DAG.

### Background

Most healthcare application workflows involve multiple stages. Application developers need a way to organize functional units of AI-based inference apps. A DAG (Directed Acyclic Graph) is the core concept of MONAI Deploy App SDK, collecting Operators together, organized with dependencies and relationships to specify how they should run.

### Verification Strategy

Build an application with the SDK which has multiple operators. Verify that the SDK offers a mechanism to represent the underlying workflow using a DAG and enables traversal of the DAG.

### Target Release

MONAI Deploy App SDK 0.1.0

---

## [REQ] Representing Workflow with DCG

The SDK shall enable representation of application workflows using a Directed Cyclic Graph (DCG) which requires cyclic dependencies.

### Background

Some applications require cycles to have dependencies among operators to represent application workflow.

### Verification Strategy

Pick an application workflow from Genomics type which requires cyclic dependencies among operators. Verify that the App SDK supports such a need.

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ]  Conditional Branching in Workflow

The SDK shall support dynamic activation of an Operator in an application so that at run-time an Operator can be executed depending on application-specific logic.

### Background

Some applications require the conditional selection of an operator during run-time. Consider an application the following operators: (a) DICOM Data Loader (b) Rescale (c) Gaussian Blur (d) Classification Inference (e) DICOM Segmentation Writer. The app developer may select Rescale or Gaussian Blur operator depending on whether the input volumetric data requires rescaling or not.

### Verification Strategy

Verify that operators can be dynamically selected for execution based on user-specified logic.

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ]  Supporting logging for an Operator

The SDK shall support logging data from an Operator in a standardized way which can be parsed to aid in-app debugging purposes.

### Background

A significant portion of app development in the healthcare AI domain is spent in figuring out anomalies in the core business logic of the application. Having a standardized way to log data will make it easier to debug the application.

### Verification Strategy

Use the SDK to log data from an operator. Verify that the logged data is adhering to the logging schema.

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ]  Developing custom Operator

The SDK shall support developing custom Operators to perform task-specific logic so that the application developer is not limited by the built-in operators offered by the SDK itself.

### Background

The SDK itself will provide a set of built-in operators which can be incorporated into domain-specific tasks such as loading medical images, performing inference, etc. However, in almost all non-trivial applications, there would be a need of performing custom tasks.

### Verification Strategy

Write a custom operator to perform an image processing task and integrate that with an application using the App SDK.

### Target Release

MONAI Deploy App SDK 0.1.0

---

## [REQ]  Support for Multi-Class Single-Output Classification

The SDK shall support developing an application that performs Multi-Class Single-Output Classification classification with a pre-trained AI model so that the app developer can incorporate necessary model inputs, transforms, inference, and package output from inference with appropriate domain-specific manners.

### Background

Multiclass classification is a classification task with two or more classes. Each sample can only be labeled as one class. For example, classification using features extracted from a set of slices of different modalities, where each slice may either MR, CT, or IVUS. Each image is one sample and is labeled as one of the 3 possible classes. Multiclass classification assumes that each sample is assigned to one and only one label - one sample cannot, for example, be both a CT & MR.

### Verification Strategy

Use a pre-trained model designed for multi-class classification. Verify that the SDK provides built-in operators using which an app can be built with that pre-trained model.

### Target Release

MONAI Deploy App SDK 0.1.0

---

## [REQ]  Support for Multi-Class Multi-Output Classification

The SDK shall support developing an application that performs Multi-Class Multi-Output classification with a pre-trained AI model so that the app developer can incorporate necessary model inputs, transforms, inference, and package output from inference with appropriate domain-specific manners.

### Background

Multiclass-multioutput classification (also known as Multitask classification) is a classification task which labels each sample with a set of non-binary properties. Both the number of properties and the number of classes per property are greater than 2. An example would be classifying a Chest X-Ray image to have one or more labels from the following list: Atelectasis, Cardiomegaly, Effusion, Pneumothorax.

### Verification Strategy

Use a pre-trained model designed for multi-class multi-output classification. Verify that the SDK provides built-in operators using which an app can be built with that pre-trained model

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ]  Support for Semantic Segmentation

The SDK shall support developing an application that performs semantic segmentation with a pre-trained AI model so that the app developer can incorporate necessary model inputs, transforms, inference, and package output from inference with appropriate domain-specific manners.

### Background

Semantic segmentation aims to label each voxel in an image with a class. An example would to be assign each voxel in a 3D CT dataset to the background, kidney, or tumor.

### Verification Strategy

Use a pre-trained model designed for semantic segmentation. Verify that the SDK provides built-in operators using which an app can be built with that pre-trained model.

### Target Release

MONAI Deploy App SDK 0.1.0

---

## [REQ]  Support for Instance Segmentation

The SDK shall support developing an application that performs semantic segmentation with a pre-trained AI model so that the app developer can incorporate necessary model inputs, transforms, inference, and package output from inference with appropriate domain-specific manners.

### Background

In instance segmentation, a model assigns an “individual object” label to each voxel in the image. An example would be where voxels for individual Lung nodules are labeled separately. Let's say in a 3D dataset there are 20 lung nodules. Instead of having a generic "nodule" pixel class, we would have 20 classes for the 20 nodules: nodule-1, nodule-2, nodule-3,.., nodule-20.

### Verification Strategy

Use a pre-trained model designed for Instance Segmentation. Verify that the SDK provides built-in operators using which an app can be built with that pre-trained model.

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ]  Support for Object Detection

The SDK shall support developing an application that performs object detection with a pre-trained AI model so that the app developer can incorporate necessary model inputs, transforms, inference, and package output from inference with appropriate domain-specific manners.

### Background

Object detection aims to provide a 2D/3D bounding box around an object of interest. An example is to generate a 3D region of interest for Lung given a  CT dataset.

### Verification Strategy

Use a pre-trained model designed for object detection. Verify that the SDK provides built-in operators using which an app can be built with that pre-trained model.

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ]  Supporting models trained with PyTorch

The SDK shall enable app developers to use a model trained with PyTorch in an Application so that tasks like model loading, provisioning of input data, a mechanism to perform custom transforms are handled by the SDK.

### Background

PyTorch is open source machine learning framework that accelerates the path from research prototyping to production deployment. It is very popular among healthcare researchers and commercial vendors.

### Verification Strategy

Verify that a PyTorch based model can be used to build an application that performs classification, segmentation, and object detection tasks

### Target Release

MONAI Deploy App SDK 0.1.0

---


## [REQ]  Supporting models trained with TensorFlow

The SDK shall enable app developers to use a model trained with TensorFlow in an Application so that tasks like model loading, provisioning of input data, a mechanism to perform custom transforms are handled by the SDK.

### Background

TensorFlow is an end-to-end open-source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art in ML, and developers easily build and deploy ML-powered applications.

### Verification Strategy

Verify that a TensorFlow based model can be used to build an application that performs classification, segmentation, and object detection tasks.

### Target Release

MONAI Deploy App SDK 0.2.0

---


## [REQ] Supporting MMAR

The SDK shall allow integration of a Clara Train generated Medical Model ARchive (MMAR) for the purpose of inference so that app developers can easily incorporate trained models into a functional application.

### Background

MMAR defines the standard structure for storing artifacts (files) needed and produced by the model development workflow (training, validation, inference, etc.). The MMAR includes all the information about the model including configurations and scripts to provide a workspace to perform different model development tasks. In the context of the MONAI Deploy App SDK, the relevant usage of the MMAR is for the purpose of inference.

### Verification Strategy

Use an existing MMAR from the Clara Train Repository. Verify that the App SDK provides built-in mechanisms to incorporate the model inherent in the MMAR to perform inference.

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ] Supporting Triton

The SDK shall allow performing inference with a pre-trained AI model via Triton using its supported networking protocol so that app developers can leverage high performance and high utilization of CPU/GPU resources when deployed in a production environment.

### Background

Triton Inference Server provides a cloud and edge inferencing solution optimized for both CPUs and GPUs. Triton supports an HTTP/REST and gRPC protocol that allows remote clients to request inferencing for any model being managed by the server.

### Verification Strategy

Use a pre-trained model to develop an application using the App SDK. Verify that the application can be designed in such a way so that the app can leverage Triton at run-time without the app developer explicitly making use of Triton API.

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ]  Integration with DICOM aware systems via MONAI Medical Informatics Gateway

The SDK shall enable applications to integrate with the MONAI Informatics Gateway so that imaging informatics data can be ingested from and exported to clinical informatics systems using DICOM as a protocol.

### Background

MONAI Informatics Gateway is a subsystem of the MONAI Deploy platform which facilitates integration with DICOM & FHIR compliant systems, enables ingestion of imaging data, helps to trigger of jobs with configurable rules, and offers to push the output of jobs to PACS & EMR systems.

### Verification Strategy

Design an app that ingests DICOM SOP Instances as input and generates DICOM SOP Instances as output. Verify that the input can be provided from the MONAI Informatics Gateway and outputs can be pushed to the MONAI Informatics Gateway.

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ]  Integration with FHIR aware systems via MONAI Medical Informatics Gateway

The SDK shall enable applications to integrate with the MONAI Informatics Gateway so that healthcare records can be ingested from and exported to clinical informatics systems using FHIR as a specification.

### Background

MONAI Informatics Gateway is a subsystem of the MONAI Deploy platform which facilitates integration with DICOM & FHIR compliant systems, enables ingestion of imaging data, helps to trigger of jobs with configurable rules, and offers to push the output of jobs to PACS & EMR systems.

### Verification Strategy

Design an app that ingests FHIR Records as input and generates FHIR Records as output. Verify that the input can be provided from the MONAI Informatics Gateway and outputs can be pushed to the MONAI Informatics Gateway.

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ] Parsing one or more DICOM series

The SDK shall enable filtering a set of DICOM Series with user-defined parsing criteria expressed in terms of a collection of key-value pairs where each key represents a DICOM Attribute so that at run-time appropriate inputs can be provided to the application.

### Background

Given a collection of DICOM studies, often an app developer needs to figure out which studies and which series belonging to a study are relevant for a specific application. DICOM SOP Instances have a collection of attributes embedded in them. These attributes can use used to parse through a collection of series.

### Verification Strategy

Verify that the App SDK supports forming selection queries using a collection of rules based on DICOM Attributes. Use these rules to select the DICOM series.

### Target Release

MONAI Deploy App SDK 0.1.0

---

## [REQ] Loading a DICOM 2d/3d dataset into a unified domain object

The SDK shall enable applications to load a 2D/3D imaging dataset belonging to a single DICOM series into a unified "Image" domain object so that destination operators can process this domain object based on the application's needs such as transformation and inference.

### Background

DICOM as a protocol offers a mechanism to represent 2D/3D imaging datasets and corresponding metadata. It is not trivial for application developers to load pixel data from DICOM Part 10 files or messages into a Tensor and associated attributes that qualify the tensor. Having such a mechanism will facilitate easy ingestion of DICOM data in a medical AI application.

### Verification Strategy

Load  DICOM series into the supported Domain Object using the App SDK. Write out the content of the domain object to disk. Compare that with gold input and verify it matches.

### Target Release

MONAI Deploy App SDK 0.1.0

---

## [REQ] Supporting DICOM Segmentation Storage SOP Class as output

The SDK shall provide a mechanism to generate a Segmentation Storage SOP Instance where each pixel/voxel can belong to a single category among multiple supported categories. This operator shall be able to output a multi-frame image representing a classification of pixels where each frame represents a 2D plane or a slice of a single segmentation category. Only binary segmentation shall be supported.

### Background

Healthcare AI apps create segmentation instances during acquisition, post-processing, interpretation, and treatment. DICOM Segmentation Storage SOP class provides a way to encode segmentation data. It is intended for composite data objects of any modality or clinical specialty. Segmentations are either binary or fractional.

### Verification Strategy

Make use of a segmentation model from MONAI core to develop an app. Verify that the output can be exported as a compliant Segmentation Storage SOP Instance.

### Target Release

MONAI Deploy App SDK 0.1.0

---

## [REQ] Supporting DICOM Segmentation Storage SOP Class as input

### Background

### Verification Strategy

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ] Supporting DICOM RT Structure Storage SOP Class as Output

The SDK shall provide an operator that supports generating and exporting an RT Structure Set Storage SOP Instance.

### Background

There are significant differences in the types of information required for radiology and radiation therapy domains. The radiation therapy information is defined in seven information objects known as DICOM-RT objects which include RT Structure Set, RT Plan, RT Dose, RT Image, RT Treatment Record, RT Brachy Treatment Record, and RT Treatment. The data models set the standard for the integration of radiation therapy information for an electronic patient record and would facilitate the interoperability of information among different systems.

### Verification Strategy

Make use of an AI model that creates the contour of segmented anatomical objects. Verify whether such a model can be used in an application and the out of the model inference can use used to generate an instance of RT Structure set.

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ] Supporting DICOM RT Structure Storage SOP Class as Input

### Background

There are significant differences in the types of information required for radiology and radiation therapy domains. The radiation therapy information is defined in seven information objects known as DICOM-RT objects which include RT Structure Set, RT Plan, RT Dose, RT Image, RT Treatment Record, RT Brachy Treatment Record, and RT Treatment. The data models set the standard for the integration of radiation therapy information for an electronic patient record and would facilitate the interoperability of information among different systems.

### Verification Strategy

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ] Specifying App Resource Requirements

The SDK shall enable developers to specify minimum resource requirements for an operator during app development. The type of resources supported shall be: GPU Memory and System Memory.

### Background

To orchestrate an application effectively the orchestration engine would need to know what kind of resources are required for an operator and the overall application.

### Verification Strategy

Ensure that the resource requirements specified for an operator are provided by the orchestration engine during run-time.

### Target Release

MONAI Deploy App SDK 0.1.0

---

## [REQ] Making use of an existing container to build an operator

The SDK shall support the use of an existing container image as the basis for designing an Operator in such a way so that the newly created Operator conforms to the interfaces required to behave like any other off-the-shelf Operators in the MONAI Deploy SDK.

### Background

The preferred way to build a design-time operator in MONAI Deploy SDK is Pythonic. Users can either use built-in Operators (where each such operator is a Python class) or extend from one of those Operators to design their own. However, some situations where this approach may not be feasible. Examples are:

* An organization has legacy source code that already has been containerized and would like to reuse in the context of a MONAI Deploy Application
* An organization has a 3rd party container and it is not feasible to recompile the source code
* Mixing multiple operators in a single container makes it difficult to resolve dependencies.

### Verification Strategy

Given an existing container verify that the SDK enables the creation of a new Operator that makes use of such containers. Also, verify that the newly created Operator is interoperable with other existing operators.

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ] Creating a MONAI Application Package (MAP)

The SDK shall allow the packaging of the application in a standardized format so that it can be executed by a compliant runtime environment.

### Background

Please refer to [MONAI Application Packge Spec](https://github.com/Project-MONAI/monai-deploy-experimental/blob/main/guidelines/monai-application-package.md)for details.

### Verification Strategy

Develop a sample application using MONAI SDK. Run the app as a Python application with a gold input. Use the App Packager utility to generate a MAP. Run the MAP using App Runner. Verify that the two sets of output match.

### Target Release

MONAI Deploy App SDK 0.1.0

---

## [REQ] App Runner

The SDK shall allow the execution of an application in the developer's workstation.

### Background

Developers need a way to run an instance of a MAP in his/her local workstation before the MAP is deployed in a production environment.

### Verification Strategy

Develop a sample application using MONAI SDK. Run the app as a Python application with a gold input. Use the App Packager utility to generate a MAP. Run the MAP using App Runner. Verify that the two sets of output match.

### Target Release

MONAI Deploy App SDK 0.1.0

---

## [REQ] MAP Executor

The MONAI Deploy SDK shall provide a MAP Executor as a shim between the runner of a MAP and the MAP's application. It shall support the following aspects:

* Provide entry-point (or initial process) of a MAP's container
* Execute the Application as defined by the MAP Package Manifest and then exit
* Set initial conditions for the Application when invoking it
* Monitor the Application process and enforce any timeout value specified in the MAP

### Background

### Verification Strategy

TBD

### Target Release

MONAI Deploy App SDK 0.1.0

---

## [REQ] Testing the functional veracity of an Operator

The SDK shall enable app verification of on operator by allowing developers to specify the following: (a) A Gold Input (b) A Gold Output (c) Similarity Metric (d) Allowed Tolerance between measured output and gold output.

### Background

Having a mechanism to test the functionality of an operator during development makes it easier for the developer to be ready for deployment to a production-quality server.

### Verification Strategy

For a classification inference operator, verify that given a gold input, a similarity metric, the generated output closely matches the output within the specified tolerance limits.

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ] Testing the functional veracity of an Application

The SDK shall enable app verification of an application by allowing developers to specify the following: (a) A Gold Input (b) A Gold Output (c) Similarity Metric (d) Allowed Tolerance between measured output and gold output.

### Background

Having a mechanism to test the functionality of an application during development makes it easier for the developer to be ready for deployment to a production-quality server.

### Verification Strategy

For an application that makes use of a classification inference operator verify that given a gold input, a similarity metric, the generated output closely matches the output within the specified tolerance limits.

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ] Supporting Medical Image Visualization during App Development

The SDK shall enable 2D/3D/4D medical image visualization of input, intermediate artifacts & output of an application during development.

### Background

Making sense of the output from deep learning algorithms in medical imaging often requires the use of advanced 2D, 3D, and 4D visualization. Existing 3D visualization systems which are deployed to handle typical clinical workloads are often not designed to handle the outputs from an AI inference platform. In addition, such visualization needs to be built using GPU-based hardware for optimal performance. Lack of an ability to meaningfully visualize outputs from AI inference hinders the growth of adoption of AI-driven clinical workflows.

### Verification Strategy

Verify whether Direct Volume Rendering, Multi-Planar Reformatting, and Original Slice viewing are offered for CT/MR modalities.

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ] Enabling App Analytics

The SDK shall allow analyzing the performance of the application at multiple levels: system, operator

**System Level**

* Execution time of an App Run
* Real-time Tracking of available GPUs in the system
* GPU Activity during an App Run
* GPU Memory occupancy by processes during the App Run.

**Operator Level**

* Operator runtime plot
* Identify GPU idle and sparse usage
* GPU workload trace for Kernels utilized by Operator

### Background

MONAI Deploy App developers need to understand the bottleneck and resource utilization for their applications so that they can improve their applications for optimal performance. Developers need a combination of post-processing & real-time tools to understand the performance implication of resource utilization.

### Verification Strategy

TBD

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ] Development Console

The SDK shall provide a Development Console to track activities during the designing of an application.

### Background

TBD

### Verification Strategy

TBD

### Target Release

MONAI Deploy App SDK 0.2.0

---

## [REQ] System Requirements

The SDK shall support the following system configurations for a given target release of the SDK. The details of the system configurations are mentioned below:

| Target Release | Supported System Configs |
| -------------- | ------------------------ |
| 0.1.0          | #1                       |
| 0.2.0          | #1                       |
| 0.3.0          | #1, #2, #3               |

### System Config #1 (Desktop x86-64)

| Attribute                | Value                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------ |
| Operating System         | Ubuntu 18.06/20.04                                                                   |
| Processor                | x86-64                                                                               |
| Minimum System Memory    | 16 GB                                                                                |
| Minimum GPU Memory       | 8 GB                                                                                 |
| CUDA Toolkit             | v11.1+                                                                               |
| NVIDIA Driver            | 455.23+                                                                              |
| Docker CE                | 19.03.13+ ([2020-09-16](https://docs.docker.com/engine/release-notes/19.03/#190313)) |
| NVIDIA Docker            | [v2.5.0+](https://github.com/NVIDIA/nvidia-docker/)                                  |
| CUDA Compute Capability  | 6.0 or larger (Pascal or newer, including Pascal, Volta, Turing and Ampere families) |
| Python                   | Python 3.7+                                                                          |


### System Config #2 (Desktop ARM64)

| Attribute                | Value                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------ |
| Operating System         | Ubuntu 18.06/20.04                                                                   |
| Processor                | ARM64                                                                               |
| Minimum System Memory    | 16 GB                                                                                |
| Minimum GPU Memory       | 8 GB                                                                                 |
| CUDA Toolkit             | v11.1+                                                                               |
| NVIDIA Driver            | 455.23+                                                                              |
| Docker CE                | 19.03.13+ ([2020-09-16](https://docs.docker.com/engine/release-notes/19.03/#190313)) |
| NVIDIA Docker            | [v2.5.0+](https://github.com/NVIDIA/nvidia-docker/)                                  |
| CUDA Compute Capability  | 6.0 or larger (Pascal or newer, including Pascal, Volta, Turing and Ampere families) |
| Python                   | Python 3.7+                                                                          |


### System Config #3 (DGX-1)

| Attribute                | Value                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------ |
| Operating System         | Ubuntu 18.06/20.04                                                                   |
| Processor                | x86-64                                                                               |
| Minimum System Memory    | 512 GB                                                                               |
| Minimum GPU Memory       | 256 GB                                                                               |
| CUDA Toolkit             | v11.1+                                                                               |
| NVIDIA Driver            | 455.23+                                                                              |
| Docker CE                | 19.03.13+ ([2020-09-16](https://docs.docker.com/engine/release-notes/19.03/#190313)) |
| NVIDIA Docker            | [v2.5.0+](https://github.com/NVIDIA/nvidia-docker/)                                  |
| CUDA Compute Capability  | 7.0 or larger (Volta or newer, including Volta, Turing and Ampere families)          |
| Python                   | Python 3.7+                                                                          |
