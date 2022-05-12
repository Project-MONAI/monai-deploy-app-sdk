# Copyright 2021-2002 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import zipfile
import os
import json
from glob import glob
from typing import Any, Dict, Sequence, Union

import torch
import numpy as np


import monai.deploy.core as md
from monai.deploy.core import (
    DataPath,
    ExecutionContext,
    Image,
    InputContext,
    IOType,
    OutputContext,
    Operator
)

from .inference_operator import InferenceOperator
from monai.deploy.core.operator import OperatorEnv

from monai.deploy.utils.importutil import optional_import

first, _ = optional_import("monai.utils.misc", name="first")
Compose, _ = optional_import("monai.transforms", name="Compose")
MapTransform, _ = optional_import("monai.transforms", name="MapTransform")
ConfigParser, _ = optional_import("monai.bundle", name="ConfigParser")
SimpleInferer, _ = optional_import("monai.inferers", name="SimpleInferer")

__all__ = ["BundleOperator", "create_bundle_operator"]


def get_bundle_config(bundle_path, config_names):
    """
    Get the configuration parser from the specified Torchscript bundle file path.
    """
    if isinstance(config_names, str):
        config_names = [config_names]

    name, _ = os.path.splitext(os.path.basename(bundle_path))
    parser = ConfigParser()

    archive = zipfile.ZipFile(bundle_path, "r")
    parser.read_meta(f=json.loads(archive.read(f"{name}/extra/metadata.json")))

    for cn in config_names:
        parser.read_config(f=json.loads(archive.read(f"{name}/extra/{cn}.json")))

    parser.parse()

    return parser


DISALLOW_LOAD_SAVE = ["LoadImage", "SaveImage"]
DISALLOW_SAVE = ["SaveImage"]


def filter_compose(compose, disallowed_prefixes):
    """
    Remove transforms from the given Compose object whose names begin with `disallowed_prefixes`.
    """
    filtered = []
    for t in compose.transforms:
        tname = type(t).__name__
        if not any(dis in tname for dis in disallowed_prefixes):
            filtered.append(t)

    return Compose(filtered)


def is_map_compose(compose):
    """
    Return True if the given Compose object uses MapTransform instances.
    """
    return isinstance(first(compose.transforms), MapTransform)


class BundleOperator(InferenceOperator):
    """
    This inference operator configures itself based on the parsed data from a MONAI bundle file. This file is included
    with a MAP as a Torchscript file with added bundle metadata or a zipped bundle with weights. The class will 
    configure how to do pre- and post-processing, inference, which device to use, state its inputs, outputs, and 
    dependencies. Its compute method is meant to be general purpose to most any bundle such that it will handle
    any input specified in the bundle and produce output as specified, using the inference object the bundle defines.
    A number of methods are provided which define parts of functionality relating to this behaviour, users may wish
    to overwrite these to change behaviour is needed for specific bundles.
    """
    def __init__(
        self,
        parser,
        in_type=IOType.IN_MEMORY,
        out_type=IOType.IN_MEMORY,
        preproc_name="preprocessing",
        postproc_name="postprocessing",
        inferer_name="inferer",
    ):
        super().__init__()
        self.parser = parser

        meta = self.parser["_meta_"]

        pip_packages = env = ["monai"] + [f"{k}=={v}" for k, v in meta["optional_packages_version"].items()]
        self._env = OperatorEnv(pip_packages=pip_packages)

        if parser.get("device") is not None:
            self._device = parser.get_parsed_content("device")
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if parser.get(inferer_name) is not None:
            self._inferer = parser.get_parsed_content(inferer_name)
        else:
            self._inferer = SimpleInferer()

        self._inputs = meta["network_data_format"]["inputs"]
        self._outputs = meta["network_data_format"]["outputs"]

        self._preproc = self._get_compose(
            preproc_name, DISALLOW_LOAD_SAVE if in_type==IOType.IN_MEMORY else DISALLOW_SAVE
        )
        self._postproc = self._get_compose(postproc_name, DISALLOW_LOAD_SAVE)

        self._add_inputs(self._inputs, in_type)
        self._add_outputs(self._outputs, out_type)

    def _get_compose(self, obj_name, disallowed_prefixes):
        """Get a Compose object containing a sequence fo transforms from item `obj_name` in `self.parser`."""

        if self.parser.get(obj_name) is not None:
            compose = self.parser.get_parsed_content(obj_name)
            return filter_compose(compose, disallowed_prefixes)

        return Compose([])

    def _add_inputs(self, inputs_dict, in_type):
        """Add inputs specified in self._inputs."""

        for iname, conf in inputs_dict.items():
            itype = conf["type"].lower()
            io_type = Image if itype == "image" else DataPath
            self.add_input(iname, io_type, in_type)

    def _add_outputs(self, outputs_dict, out_type):
        """Add outputs specified in self._outputs."""

        for oname, conf in outputs_dict.items():
            otype = conf["type"].lower()
            io_type = Image if otype == "image" else DataPath
            self.add_output(oname, io_type, out_type)

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        first_input_name, *other_names = list(self._inputs.keys())
        inputs = {k: op_input.get(k).asnumpy() for k in self._inputs.keys()}

        with torch.no_grad():
            inputs = self.pre_process(inputs)

            first_input = inputs[first_input_name][None].to(self._device)  # select first input
            # select other tensor inputs
            other_inputs = {k: inputs[k][None].to(self._device) for k in other_names if isinstance(v, torch.Tensor)}
            # select other non-tensor inputs
            other_inputs.update({k: inputs[k] for k in other_names if not isinstance(v, torch.Tensor)})

            model = torch.jit.load(context.models.get().path, map_location=self._device).eval()
            # model = context.models.get() # get a TorchScriptModel object

            outputs = self.predict(data=first_input, network=model, **other_inputs)

            outputs = self.post_process(outputs)

        if isinstance(outputs, (tuple, list)):
            output_dict = dict(zip(self._outputs.keys(), outputs))
        elif not isinstance(outputs, dict):
            output_dict = {first(self._outputs.keys()): outputs}
        else:
            output_dict = outputs

        for name, out in output_dict.items():
            self._send_output(out[0], name, op_output, context)
            
    def predict(self, data: Any, network: Any, *args, **kwargs) -> Union[Image, Any]:
        """Predict output using the inferer."""
        return self._inferer(inputs=data, network=network, *args, **kwargs)
            
    def pre_process(self, data: Any) -> Union[Image, Any]:
        """Process the input dictionary with the stored transform sequence `self._preproc`."""

        if is_map_compose(self._preproc):
            return self._preproc(data)
        return {k: self._preproc(v) for k, v in data.items()}

    def post_process(self, data: Any) -> Union[Image, Any]:
        """Process the output list/dictionary with the stored transform sequence `self._postproc`."""

        if is_map_compose(self._postproc):
            if isinstance(data, (list, tuple)):
                outputs_dict = dict(zip(data, self._outputs.keys()))
            elif not isinstance(data, dict):
                oname = first(self._outputs.keys())
                outputs_dict = {oname: data}
            else:
                outputs_dict = outputs

            return self._postproc(outputs_dict)
        else:
            if isinstance(outputs, (list, tuple)):
                return list(map(self._postproc, data))

            return self._postproc(data)

    def _send_output(self, value, name: str, op_output: OutputContext, context: ExecutionContext):
        """Send the given output value to the output context."""
        otype = self._outputs[name]["type"].lower()

        if otype == "image":
            pass  # nothing to do for image?
        elif otype == "probabilities":
            _, value_class = value.max(dim=0)
            value = self._outputs[name]["channel_def"][str(value_class.item())]
        else:
            raise ValueError(f"Unknown output type {otype}")

        try:
            op_output.set(value, name)
        except:
            path = f"{str(op_output.get().path)}/{name}"

            if isinstance(value, torch.Tensor):
                torch.save(value, f"{path}.pt")
            elif isinstance(value, np.ndarray):
                np.save(f"{path}.npy", value)
            else:
                with open(f"{path}.json", "w") as fp:
                    json.dump({"result": value}, fp)


def create_bundle_operator(bundle_path, config_names, in_type=IOType.IN_MEMORY, out_type=IOType.IN_MEMORY):
    """
    Create a new BundleOperator instance configured with the given bundle. If the bundle file isn't present, return a
    dummy operator object which does nothing, this is necessary in packaging.
    """
    if not os.path.isfile(bundle_path):

        class DummyOperator(Operator):
            def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
                pass

        return DummyOperator()

    parser = get_bundle_config(bundle_path, config_names)

    return BundleOperator(parser, in_type=in_type, out_type=out_type)

