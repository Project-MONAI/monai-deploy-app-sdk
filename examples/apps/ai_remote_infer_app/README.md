# AI Remote Inference App - Spleen Segmentation

This example application demonstrates how to perform medical image segmentation using **Triton Inference Server** for remote inference calls. The app processes DICOM CT series to segment spleen anatomy using a deep learning model hosted on a remote Triton server.

## Overview

This application showcases:
- **Remote inference using Triton Inference Server**: The app connects to a Triton server to perform model inference remotely rather than loading models locally. It communicates by sending and receiving input/output tensors corresponding to the model dimensions, including channels.
- **Triton client integration**: The built-in `TritonRemoteModel` class is provided in the [triton_model.py](https://github.com/Project-MONAI/monai-deploy-app-sdk/blob/137ac32d647843579f52060c8f72f9d9e8b51c38/monai/deploy/core/models/triton_model.py) module. This class acts as a Triton inference client, communicating with an already loaded model network on the server. It supports the same API as the in-process model class (e.g., a loaded TorchScript model network). As a result, the application inference operator does not need to change when switching between in-process and remote inference.
- **Model metadata parsing**: Uses Triton's model folder structure, which contains the `config.pbtxt` configuration file, to extract model specifications including name, input/output dimensions, and other metadata.
- **Model path requirement**: The parent folder of the Triton model folder needs to be used as the model path for the application.

## Architecture

The application follows a pipeline architecture:

1. **DICOM Data Loading**: Loads DICOM study from input directory
2. **Series Selection**: Selects appropriate CT series based on configurable rules
3. **Volume Conversion**: Converts DICOM series to 3D volume
4. **Remote Inference**: Performs spleen segmentation via Triton Inference Server
5. **Output Generation**: Creates DICOM segmentation and STL mesh outputs

## Key Components

### Triton Integration

The `SpleenSegOperator` leverages the `MonaiSegInferenceOperator` which:
- Uses the loaded model network which in turn acts as a **Triton inference client** and connects to a remote Triton Inference Server that actually serves the named model
- Handles preprocessing and postprocessing transforms
- No explicit remote inference logic is required in these two operators

### Model Configuration Requirements

The application requires a Triton model folder which contains a **Triton model configuration file** (`config.pbtxt`) to be present on the application side, and the parent path to the model folder will be used as the model path for the application. This example application has the following model folder structure:

```
models_client_side/spleen_ct/config.pbtxt
```

The path to `models_client_side` is the model path for the application while `spleen_ct` is the folder of the named model with the folder name matching the model name. The model name in the `config.pbtxt` file is therefore intentionally omitted.

This configuration file (`config.pbtxt`) contains essential model metadata:
- **Model name**: `spleen_ct` (used for server communication)
- **Input dimensions**: `[1, 96, 96, 96]` (channels, width, height, depth)
- **Output dimensions**: `[2, 96, 96, 96]` (2-class segmentation output)
- **Data types**: `TYPE_FP32` for both input and output
- **Batching configuration**: Dynamic batching with preferred sizes
- **Hardware requirements**: GPU-based inference

**Important**: The `config.pbtxt` file is used **in lieu of the actual model file** (e.g., TorchScript `.ts` file) that would be present in an in-process inference scenario. For remote inference, the physical model file (`model_spleen_ct_segmentation_v1.ts`) resides on the Triton server, while the client only needs the configuration metadata to understand the model's interface.

### API Compatibility Between In-Process and Remote Inference

The `TritonRemoteModel` class in the `triton_model.py` module contains the actual Triton client instance and provides the **same API as in-process model instances**. This design ensures that:

- **Application inference operators remain unchanged** whether using in-process or remote inference
- **Seamless switching** between local and remote models without code modifications
- **Unified interface** through the `__call__` method that handles both PyTorch tensors locally and Triton HTTP requests remotely
- **Transparent model loading** where `MonaiSegInferenceOperator` uses the same `predictor` interface regardless of model location

## Setup and Configuration

### Environment Variables

Configure the following environment variables (see `env_settings_example.sh`):

```bash
export HOLOSCAN_INPUT_PATH="inputs/spleen_ct_tcia"           # Input DICOM directory
export HOLOSCAN_MODEL_PATH="examples/apps/ai_remote_infer_app/models_client_side"  # Client-side model config path
export HOLOSCAN_OUTPUT_PATH="output_spleen"                 # Output directory
export HOLOSCAN_LOG_LEVEL=DEBUG                             # Logging level
export TRITON_SERVER_NETLOC="localhost:8000"                # Triton server address
```

### Triton Server Setup

1. **Server Side**: Deploy the actual model file (`model_spleen_ct_segmentation_v1.ts`) to your Triton server
2. **Client Side**: Ensure the `config.pbtxt` file is available locally for metadata parsing
3. **Network**: Ensure connectivity between client and Triton server on the specified port

### Directory Structure

```
ai_remote_infer_app/
├── app.py                          # Main application logic
├── spleen_seg_operator.py          # Custom segmentation operator
├── __main__.py                     # Application entry point
├── env_settings_example.sh         # Environment configuration
├── models_client_side/             # Client-side model configurations
│   └── spleen_ct/
│       └── config.pbtxt           # Triton model configuration (no model file)
└── README.md                       # This file
```

## Usage

1. **Set up Triton Server** with the spleen segmentation model, listening at localhost:8000 in this example
2. **Configure environment** variables pointing to your Triton server
3. **Prepare input data** in DICOM format
4. **Run the application**:
   ```bash
   python ai_remote_infer_app
   ```

## Input Requirements

- **DICOM CT series** containing abdominal scans
- **Series selection criteria**: PRIMARY/ORIGINAL CT images
- **Image preprocessing**: Automatic resampling to 1.5x1.5x2.9mm spacing

## Output

The application generates:
- **DICOM Segmentation** files with spleen masks
- **STL mesh** files for 3D visualization
- **Intermediate NIfTI** files for debugging (optional)

## Model Specifications

- **Architecture**: 3D PyTorch model optimized for spleen segmentation
- **Input size**: 96×96×96 voxels
- **Output**: 2-class segmentation (background + spleen)
- **Inference method**: Sliding window with 60% overlap
- **Batch size**: Configurable (default: 4)

## Notes

- The application demonstrates **remote inference patterns** suitable for production deployments
- **Model versioning** is handled server-side through Triton's version policies
- **Dynamic batching** optimizes throughput for multiple concurrent requests
- **GPU acceleration** is configured but can be adjusted based on available hardware

## Dependencies

- MONAI Deploy SDK
- Triton Inference Client libraries
- PyDICOM for DICOM handling
- MONAI transforms for preprocessing
