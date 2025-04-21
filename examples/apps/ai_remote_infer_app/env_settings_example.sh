#!/bin/bash
export HOLOSCAN_INPUT_PATH="inputs/spleen_ct_tcia"
export HOLOSCAN_MODEL_PATH="examples/apps/ai_remote_infer_app/models_client_side"
export HOLOSCAN_OUTPUT_PATH="output_spleen"
export HOLOSCAN_LOG_LEVEL=DEBUG  # TRACE can be used for verbose low-level logging
export TRITON_SERVER_NETLOC="localhost:8000"  # Triton server network location, host:port