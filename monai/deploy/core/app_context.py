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

import logging
from os.path import abspath
from typing import Dict, List, Optional

from .arg_parser import parse_args, set_up_logging
from .models import TritonModel
from .models.factory import ModelFactory
from .models.model import Model
from .runtime_env import RuntimeEnv


class AppContext(object):
    """A class to store the context of an application."""

    def __init__(self, args: Dict[str, str], runtime_env: Optional[RuntimeEnv] = None):
        # Set the args
        self.args: Dict[str, str] = {}
        # Set the runtime environment
        self.runtime_env = runtime_env or RuntimeEnv()

        self._model_loaded = False  # If it has tried to load the models.
        self.model_path = ""  # To be set next.
        self.update(args)

    def update(self, args: Dict[str, str]):
        """Update the context with new args and runtime_env."""
        # Update args
        self.args.update(args)

        # Set the path to input/output/model
        self.input_path = args.get("input") or self.args.get("input") or self.runtime_env.input
        self.output_path = args.get("output") or self.args.get("output") or self.runtime_env.output
        self.workdir = args.get("workdir") or self.args.get("workdir") or self.runtime_env.workdir

        # If model has not been loaded, or the model path has changed, get the path and load model(s)
        old_model_path = self.model_path
        self.model_path = args.get("model") or self.args.get("model") or self.runtime_env.model

        # This parameter must be set if models are hosted on the Triton Inference Server.
        self.triton_server_netloc = (
            args.get("triton_server_netloc")
            or self.args.get("triton_server_netloc")
            or self.runtime_env.triton_server_netloc
        )

        if old_model_path != self.model_path:
            self._model_loaded = False  # path changed, reset the flag to re-load

        if not self._model_loaded:
            self.models: Optional[Model] = ModelFactory.create(abspath(self.model_path))
            self._model_loaded = True

        # TritonModel instances are just clients and must be connected to the Triton Inference Server
        # at the provided network location. In-process hosting of Triton Inference Server is not supported.
        if self.triton_server_netloc and self.models:
            for _, model in self.models.items():
                if isinstance(model, TritonModel):
                    model.connect(self.triton_server_netloc, verbose=args.get("log_level", "INFO") == "DEBUG")
                    # Health check of the Triton Inference Server can be deferred.
                    logging.info(f"Model {model.name} set to connect to Triton server at {self.triton_server_netloc}")

    def __repr__(self):
        return (
            f"AppContext(input_path={self.input_path}, output_path={self.output_path}, "
            f"model_path={self.model_path}, workdir={self.workdir}), triton_server_netloc={self.triton_server_netloc}"
        )


def init_app_context(argv: Optional[List[str]] = None, runtime_env: Optional[RuntimeEnv] = None) -> AppContext:
    """Initializes the app context with arguments and well-known environment variables.

    The arguments, if passed in, override the attributes set with environment variables.

    Args:
        argv (Optional[List[str]], optional): arguments passed to the program. Defaults to None.

    Returns:
        AppContext: the AppContext object
    """

    args = parse_args(argv)
    set_up_logging(args.log_level)
    logging.info(f"Parsed args: {args}")

    # The parsed args from the command line override that from the environment variables
    app_context = AppContext({key: val for key, val in vars(args).items() if val}, runtime_env)
    logging.info(f"AppContext object: {app_context}")

    return app_context
