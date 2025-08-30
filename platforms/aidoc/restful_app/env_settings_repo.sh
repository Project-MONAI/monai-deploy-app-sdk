#!/bin/bash
export HOLOSCAN_INPUT_PATH="$(pwd)/../monai-deploy-app-sdk/inputs/spleen_ct_tcia"
export HOLOSCAN_MODEL_PATH="$(pwd)/../monai-deploy-app-sdk/models/spleen_ct"
export HOLOSCAN_OUTPUT_PATH="$(pwd)/output_spleen"
export HOLOSCAN_LOG_LEVEL=INFO