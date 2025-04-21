# Copyright 2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
export HOLOSCAN_INPUT_PATH="inputs/spleen_ct_tcia"
export HOLOSCAN_MODEL_PATH="examples/apps/ai_remote_infer_app/models_client_side"
export HOLOSCAN_OUTPUT_PATH="output_spleen"
export HOLOSCAN_LOG_LEVEL=DEBUG  # TRACE can be used for verbose low-level logging
export TRITON_SERVER_NETLOC="localhost:8000"  # Triton server network location, host:port
