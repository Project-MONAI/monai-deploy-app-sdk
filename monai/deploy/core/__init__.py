# Copyright 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
.. autosummary::
    :toctree: _autosummary

    Application
    Operator
    env
    input
    output
    resource
    IOType
    ExecutionContext
    InputContext
    OutputContext
"""

# Need to import explicit ones to quiet mypy complaints
from holoscan.core import *
from holoscan.core import Application, ConditionType, Fragment, Operator, OperatorSpec

from .app_context import AppContext
from .domain.datapath import DataPath
from .domain.image import Image

# from .env import env
from .io_type import IOType
from .models import Model, ModelFactory, NamedModel, TorchScriptModel, TritonModel
from .runtime_env import RuntimeEnv

# from .resource import resource


# Create function to add to the Application class
def load_models(modle_path: str):
    """_summary_

    Args:
        modle_path (str): _description_
    """

    return ModelFactory.create(modle_path)


Application.load_models = load_models
