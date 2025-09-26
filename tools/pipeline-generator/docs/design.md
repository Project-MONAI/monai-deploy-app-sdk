# **MONAI Bundle Integration for MONAI Deploy App SDK & Holoscan SDK**

## **Objective**

The goal of this project is to build a robust tool that enables a seamless path for developers to integrate AI models exported in the **MONAI Bundle format** into inference-based applications built with the **MONAI Deploy App SDK** and the **Holoscan SDK**.

This tool will support:

- Standard MONAI Bundles (.pt, .ts, .onnx)
- MONAI Bundles exported in **Hugging Face-compatible format**

By bridging the gap between model packaging and application deployment, this project aims to simplify clinical AI prototyping and deployment across NVIDIA’s edge AI platforms.

## **Background**

The **[MONAI Bundle](https://docs.monai.io/en/latest/mb_specification.html#archive-format)** is a standardized format designed to package deep learning models for medical imaging. It includes the model weights, metadata, transforms, and documentation needed to make the model self-contained and portable.

The **Holoscan SDK** is NVIDIA’s real-time streaming application SDK for AI workloads in healthcare and edge devices. The **MONAI Deploy App SDK** builds on top of the Holoscan SDK, adding medical imaging domain-specific operators and functionalities to enable the creation of composable inference applications, particularly for radiology and imaging pipelines.

As of MONAI Core, bundles can also be exported in a **Hugging Face-compatible format**, which allows sharing through the Hugging Face Model Hub. Supporting this format increases reach and adoption.

## **Benefits**

- Speeds up deployment of MONAI-trained models in Holoscan/Deploy pipelines
- Ensures standardized and reproducible model integration
- Makes AI development more accessible to healthcare and edge-AI developers
- Enables the usage of models from Hugging Face directly in clinical-style workflows

## **Assumptions/Limitations**

- The tool does not convert output formats given that each model may output a different type of result
- The generated application does not convert input formats given that each model may expect a different type of input

## **Scope**

This project includes:

- Support for loading and parsing standard MONAI Bundles (P0)
- Support for Hugging Face-exported MONAI Bundles (P0)
- Integration with MONAI Deploy App SDK (P0)
- Dynamic generation of pre/post-processing pipelines from metadata (P0)
- Integration with Holoscan SDK’s inference operators (P1)
- Tools to validate and prepare MONAI Bundles for deployment (P1)

## **Key Features**

- **Bundle Parsing Utility**
  - Parses metadata.json, inference.json, and other relevant files
  - Extracts model paths, input/output shapes, transform descriptions, and model metadata
  - Detects format: .pt, .ts, .onnx, or Hugging Face variant
- **Model Format Support**
  - TorchScript (.ts): Loaded with torch.jit.load()
  - ONNX (.onnx): Loaded with ONNXRuntime or TensorRT
  - PyTorch state dict (.pt): Loaded with model definition code/config
  - Hugging Face-compatible: Recognized and unpacked with reference to Hugging Face conventions
- **AI Inference Operator Integration**
  - Python support for TorchScript/ONNX-based inference
  - Auto-configures model inputs/outputs based on network_data_format
  - Embeds optional postprocessing like argmax, thresholding, etc.
- **Preprocessing/Postprocessing Pipeline**
  - Leverages MONAI transforms where applicable
  - Builds a dynamic MONAI Deploy pipeline based on the parsed config
  - Integrates with existing MONAI Deploy operators
  - Builds a dynamic Holoscan Application pipeline based on the parsed config
  - Integrates with existing Holoscan operators
- **Pipeline Generation**
  - Automatically generate MONAI Deploy App SDK application pipelines from bundle metadata
  - Automatically generate Holoscan SDK application pipelines from bundle metadata
- **Tooling**
  - Command-line tool to:
    - Validate MONAI Bundles
    - Generate MONAI Deploy and Holoscan-ready configs
    - Extract and display metadata (task, inputs, author, etc.)

## **Pipeline Integration Example**

Typical MONAI Deploy and Holoscan-based application structure enabled by this module:

[Source] → [Preprocessing Op] → [Inference Op] → [Postprocessing Op] → [Sink / Visualizer]

Each operator is configured automatically from the MONAI Bundle metadata, minimizing boilerplate.

## **Future Directions**

- Support for multiple models per bundle (e.g. ROI + segmentation)
- Integration with MONAI Label for interactive annotation-driven pipelines
- Hugging Face Model Hub sync/download integration

## **Tooling**

This tool will use Python 3.10:

- A requirements.txt to include all dependencies
- Use poetry for module and dependency management

## Development Phases

### Notes

For each of the following phases, detail describe what is done in the `tools/pipeline-generator/design_phase` directory so you can pickup later, include but not limited to the following:

- Implementation decisions made
- Code structure and key classes/functions
- Any limitations or assumptions
- Testing approach and results
- Dependencies and versions used
- Ensure no technical debts
- Ensure tests are up-to-date and have good coverage

### Phase 1

First, create a MONAI Deploy application that loads model the spleen_ct_segmentation model from `tools/pipeline-generator/phase_1/spleen_ct_segmentation` (which I downloaded from https://huggingface.co/MONAI/spleen_ct_segmentation/tree/main). The application pipeline shall use pure MONAI Deploy App SDK APIs and operators.

- The MONAI Deploy application pipeline should include all steps as described above in the *Pipeline Integration Example* section.
- We should parse and implement the preprocessing transforms from the bundle's metadata.
- Ensure configurations are loaded from the [inference.json](tools/pipeline-generator/phase_1/spleen_ct_segmentation/configs/inference.json) file at runtime and not hard coded.
- The input is a directory path; the directory would contain multiple files and the application shall proess all files.
- The output from our application pipeline should be the same as the expected output, same directory structure and data format. We should also compare the application output to the expected output.

Input (NIfTI): /home/vicchang/Downloads/Task09_Spleen/Task09_Spleen/imagesTs
Model: tools/pipeline-generator/phase_1/spleen_ct_segmentation/models/model.ts
Expected Output: tools/pipeline-generator/phase_1/spleen_ct_segmentation/eval

Note: we may need to modify the existing [monai_bundle_inference_operator](monai/deploy/operators/monai_bundle_inference_operator.py) to support loading from a directory instead of a ZIP file. We should modify the py file directly and not extend it. Ensure to keep existing ZIP file support.

Note: refer to [samples](/home/vicchang/sc/github/monai/monai-deploy-app-sdk/examples) for how to compose a MONAI Deploy application. Reuse all operators if possible. For example, if there are Nifti loaders, then do not recreate one.

`For this phase, assume we use pure MONAI Deploy App SDK end-to-end.`

### Phase 2

Create a CLI with a command to list available models from https://huggingface.co/MONAI. It should pull all available models using HuggingFace Python API at runtime.
However, a YAML config file should have a list of endpoints to scan the models from, we will start with https://huggingface.co/MONAI but later add models listed in section Phase 7.
The CLI should be able to support other commands later. For example, in 0.2, we need to add a command to generate an application and 0.3 to run the generated application.

```bash
pg list
```

Note: this new project solution shall be independent from Phase 1. This project shall use poetry for dependency management and include unit test.

### Phase 3

- Generate a MONAI Deploy-based Pipeline on selected a select MONAI Bundle from https://huggingface.co/MONAI. There are currently 40 models available. The Python module shall output the following:

1. app.py that include the end-to-end MONAI Deploy pipeline as outlined in the "Pipeline Integration Example" section above.
1. app.yaml with all configurations
1. Any models files and configurations from the downloaded model
1. READMD.md with instructions on how to run the app and info about the selected model.

Important: download all files from the model repository.
Note: there are reference applications in [examples](/home/vicchang/sc/github/monai/monai-deploy-app-sdk/examples).

A sample directory structure shall look like:

```bash
root/
├── app.py
├── app.yaml
└── model/
    └── (model files downloaded from HuggingFace repository)
```

Implement the solution with ability to generate a MONAI Deploy application based on the selected model.

- Jinja2 for main code templates - Perfect for generating app.py with variable operator configurations
- Pydantic/dataclasses for configuration models - To validate and structure app.yaml data
- YAML library for configuration generation - Direct YAML output from Python objects
- Poetry for project management (as specified in your design)

```bash
pg gen spleen_ct_segmentation --ouput [path-to-generated-output] #for during testing we should always use ./output to store generated applications
```

### Phase 4

Add a new CLI command to run the newly generated app with the application directory, test data and output directory as arguments.
It should create a virtual environment, install dependencies and run the application.

```bash
pg run path-to-generated-app --input test-data-dir --output result-dir
```

### Phase 5

Replace poetry with uv.

- Ensure all existing docs are updated
- Ensure all existing commands still work
- Run unit test and ensure coverage is at least 90%

### Phase 6

Add support for MONAI/Llama3-VILA-M3-3B model.

- Create new operators for the model in 'monai/deploy/operators' so it can be reused by other Llama3 models. The first operator should be able to take a directory as input and scan for a prompts.yaml file in the following format:

```yaml
defaults:
  max_new_tokens: 256
  temperature: 0.2
  top_p: 0.9
prompts:
  - prompt: Summarize key findings.
    image: img1.png
    output: json
  - prompt: Is there a focal lesion? Answer yes/no and describe location.
    image: img2.png
    output: image
    max_new_tokens: 128
```

Where `prompts.prompt` is the prompt fora set of images and `prompts.image` is an image associated with the prompt. The `prompts.output` indicates the type to expect for each prompt, it could be one of the following: json (see below for sample), image (generate a new image in the output directory with the AI response), image_overlay (this could be segmentation mask, bounding boxes etc...).

The first operator (VLMPromptsLoaderOperator) shall have a single output port that includes image + prompt + output_type + request_id (filename + datetime) and shall emit one prompt only each time compute is called. The operator shall end the application once all prompts have been processed (see monai/deploy/operators/image_directory_loader_operator.py L92-96).

The second operator (Llama3VILAInferenceOperator) takes the input from first operator and run the model. Once the model is ready with results, output it to the output port for the last operator.

The third and last operator (VLMResultsWriterOperator) shall take input from the first operator and the results from second operator and then write the results to the results directory specified by the user. The type of data to write to disk depends on the output type defined in the prompt.

The output of the JSON should be in the following format:

```json
{
  "prompt": "original prompt",
  "response": "AI generated response"
}
```

Update config.yaml with the new model.

Note: no changes to the pg run command.
Note: in this phase, we will support a single 2D image (PNG/JPEG) only.
Note: Since this model, the prompts.yaml, supports custom input/output formats, we will use "custom" as the input_type and output_type in the [config.yaml](tools/pipeline-generator/pipeline_generator/config/config.yaml).
Note: results are saved to the destination directory from pg run --output parameter.

**Phase 6 Status**: ✅ Completed - All three operators created and added to MONAI Deploy. The model appears in the pipeline generator list. Template integration requires additional work for full "custom" type support.
