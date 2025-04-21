# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from abc import ABC
from typing import Dict, Optional, Tuple


class RuntimeEnv(ABC):
    """Class responsible for managing run time settings.

    The expected variables can be set via the host env vars which override the
    default values in the internal dictionary.
    Selective overriding of variables can be done by passing a dictionary to the constructor,
    which should have the same structure as the internal dictionary.
    """

    ENV_DEFAULT: Dict[str, Tuple[str, ...]] = {
        "input": ("HOLOSCAN_INPUT_PATH", "input"),
        "output": ("HOLOSCAN_OUTPUT_PATH", "output"),
        "model": ("HOLOSCAN_MODEL_PATH", "models"),
        "workdir": ("HOLOSCAN_WORKDIR", ""),
        "triton_server_netloc": ("TRITON_SERVER_NETLOC", ""),
    }

    # Place holders as the values will be set in the __init__ method
    input: str = ""
    output: str = ""
    model: str = ""
    workdir: str = ""
    triton_server_netloc: str = ""  # Triton server host:port

    def __init__(self, defaults: Optional[Dict[str, Tuple[str, ...]]] = None):
        if defaults is None:
            defaults = self.ENV_DEFAULT
        else:
            defaults = {**self.ENV_DEFAULT, **defaults}

        for key, (env, default) in defaults.items():
            self.__dict__[key] = os.environ.get(env, default)
