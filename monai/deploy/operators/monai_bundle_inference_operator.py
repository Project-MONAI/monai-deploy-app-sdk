# Copyright 2002 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import pickle
import time
import zipfile
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, Image, InputContext, IOType, OutputContext
from monai.deploy.core.operator import OperatorEnv
from monai.deploy.exceptions import ItemNotExistsError
from monai.deploy.utils.importutil import optional_import

from .inference_operator import InferenceOperator

nibabel, _ = optional_import("nibabel", "3.2.1")
torch, _ = optional_import("torch", "1.10.0")

PostFix, _ = optional_import("monai.utils.enums", name="PostFix")  # For the default meta_key_postfix
first, _ = optional_import("monai.utils.misc", name="first")
ensure_tuple, _ = optional_import("monai.utils", name="ensure_tuple")
Compose_, _ = optional_import("monai.transforms", name="Compose")
ConfigParser_, _ = optional_import("monai.bundle", name="ConfigParser")
MapTransform_, _ = optional_import("monai.transforms", name="MapTransform")
SimpleInferer, _ = optional_import("monai.inferers", name="SimpleInferer")

# Dynamic class is not handled so make it Any for now: https://github.com/python/mypy/issues/2477
Compose: Any = Compose_
MapTransform: Any = MapTransform_
ConfigParser: Any = ConfigParser_

__all__ = ["MonaiBundleInferenceOperator", "IOMapping", "BundleConfigNames"]


def get_bundle_config(bundle_path, config_names):
    """
    Gets the configuration parser from the specified Torchscript bundle file path.
    """

    def _read_from_archive(archive, root_name: str, relative_path: str, path_list: List[str]):
        """A helper function for reading a file in an zip archive.

        Tries to read with the full path of # a archive file, if error, then find the relative
        path and then read the file.
        """
        content_text = None
        try:
            content_text = archive.read(f"{root_name}/{relative_path}")
        except KeyError:
            logging.debug(f"Trying to find the metadata/config file in the bundle archive: {relative_path}.")
            for n in path_list:
                if relative_path in n:
                    content_text = archive.read(n)
                    break
            if content_text is None:
                raise

        return content_text

    if isinstance(config_names, str):
        config_names = [config_names]

    name, _ = os.path.splitext(os.path.basename(bundle_path))
    parser = ConfigParser()

    # Parser to read the required metadata and extra config contents from the archive
    with zipfile.ZipFile(bundle_path, "r") as archive:
        name_list = archive.namelist()
        metadata_relative_path = "extra/metadata.json"
        metadata_text = _read_from_archive(archive, name, metadata_relative_path, name_list)
        parser.read_meta(f=json.loads(metadata_text))

        for cn in config_names:
            config_relative_path = f"extra/{cn}.json"
            config_text = _read_from_archive(archive, name, config_relative_path, name_list)
            parser.read_config(f=json.loads(config_text))

    parser.parse()

    return parser


DISALLOW_LOAD_SAVE = ["LoadImage", "SaveImage"]
DISALLOW_SAVE = ["SaveImage"]


def filter_compose(compose, disallowed_prefixes):
    """
    Removes transforms from the given Compose object whose names begin with `disallowed_prefixes`.
    """
    filtered = []
    for t in compose.transforms:
        tname = type(t).__name__
        if not any(dis in tname for dis in disallowed_prefixes):
            filtered.append(t)

    compose.transforms = tuple(filtered)
    return compose


def is_map_compose(compose):
    """
    Returns True if the given Compose object uses MapTransform instances.
    """
    return isinstance(first(compose.transforms), MapTransform)


class IOMapping:
    """This object holds an I/O definition for an operator."""

    def __init__(
        self,
        label: str,
        data_type: Type,
        storage_type: IOType,
    ):
        """Creates an object holding an operator I/O definitions.

        Limitations apply with the combination of data_type and storage_type, which will
        be validated at runtime.

        Args:
            label (str): Label for the operator input or output.
            data_type (Type): Datatype of the I/O data content.
            storage_type (IOType): The storage type expected, i.e. IN_MEMORY or DISK.
        """
        self.label: str = label
        self.data_type: Type = data_type
        self.storage_type: IOType = storage_type


class BundleConfigNames:
    """This object holds the name of relevant config items used in a MONAI Bundle."""

    def __init__(
        self,
        preproc_name: str = "preprocessing",
        postproc_name: str = "postprocessing",
        inferer_name: str = "inferer",
        config_names: Union[List[str], Tuple[str], str] = "inference",
    ) -> None:
        """Creates an object holding the names of relevant config items in a MONAI Bundle.

        This object holds the names of the config items in a MONAI Bundle that will need to be
        parsed by the inference operator for automating the object creations and inference.
        Defaults values are provided per conversion, so the arguments only need to be set as needed.

        Args:
            preproc_name (str, optional): Name of the config item for pre-processing transforms.
                                          Defaults to "preprocessing".
            postproc_name (str, optional): Name of the config item for post-processing transforms.
                                           Defaults to "postprocessing".
            inferer_name (str, optional): Name of the config item for inferer.
                                          Defaults to "inferer".
            config_names (List[str], optional): Name of config file(s) in the Bundle for parsing.
                                                Defaults to ["inference"]. File ext must be .json.
        """

        def _ensure_str_list(config_names):
            names = []
            if isinstance(config_names, (List, Tuple)):
                if len(config_names) < 1:
                    raise ValueError("At least one config name must be provided.")
                names = [str(name) for name in config_names]
            else:
                names = [str(config_names)]

            return names

        self.preproc_name: str = preproc_name
        self.postproc_name: str = postproc_name
        self.inferer_name: str = inferer_name
        self.config_names: List[str] = _ensure_str_list(config_names)


DEFAULT_BundleConfigNames = BundleConfigNames()

# The operator env decorator defines the required pip packages commonly used in the Bundles.
# The MONAI Deploy App SDK packager currently relies on the App to consolidate all required packages in order to
# install them in the MAP Docker image.
# TODO: Dynamically setting the pip_packages env on init requires the bundle path be passed in. Apps using this
#       operator may choose to pass in a accessible bundle path at development and packaging stage. Ideally,
#       the bundle path should be passed in by the Packager, e.g. via env var, when the App is initialized.
#       As of now, the Packager only passes in the model path after the App including all operators are init'ed.
@md.env(pip_packages=["monai>=0.9.0", "torch>=1.10.02", "numpy>=1.21", "nibabel>=3.2.1"])
class MonaiBundleInferenceOperator(InferenceOperator):
    """This inference operator automates the inference operation for a given MONAI Bundle.

    This inference operator configures itself based on the parsed data from a MONAI bundle file. This file is included
    with a MAP as a Torchscript file with added bundle metadata or a zipped bundle with weights. The class will
    configure how to do pre- and post-processing, inference, which device to use, state its inputs, outputs, and
    dependencies. Its compute method is meant to be general purpose to most any bundle such that it will handle
    any input specified in the bundle and produce output as specified, using the inference object the bundle defines.
    A number of methods are provided which define parts of functionality relating to this behavior, users may wish
    to overwrite these to change behavior is needed for specific bundles.

    The input(s) and output(s) for this operator need to be provided when an instance is created, and their labels need
    to correspond to the bundle network input and output names, which are also used as the keys in the pre and post processing.

    For image input and output, the type is the `Image` class. For output of probabilities, the type is `Dict`.

    This operator is expected to be linked with both upstream and downstream operators, e.g. receiving an `Image` object from
    the `DICOMSeriesToVolumeOperator`, and passing a segmentation `Image` to the `DICOMSegmentationWriterOperator`.
    In such cases, the I/O storage type can only be `IN_MEMORY` due to the restrictions imposed by the application executor.
    However, when used as the first operator in an application, its input storage type needs to be `DISK`, and the file needs
    to be a Python pickle file, e.g. containing an `Image` instance. When used as the last operator, its output storage type
    also needs to `DISK` with the path being the application's output folder, and the operator's output will be saved as
    a pickle file whose name is the same as the output name.
    """

    known_io_data_types = {
        "image": Image,  # Image object
        "series": np.ndarray,
        "tuples": np.ndarray,
        "probabilities": Dict[str, Any],  # dictionary containing probabilities and predicted labels
    }

    kw_preprocessed_inputs = "preprocessed_inputs"

    def __init__(
        self,
        input_mapping: List[IOMapping],
        output_mapping: List[IOMapping],
        model_name: Optional[str] = "",
        bundle_path: Optional[str] = "",
        bundle_config_names: Optional[BundleConfigNames] = DEFAULT_BundleConfigNames,
        *args,
        **kwargs,
    ):
        """_summary_

        Args:
            input_mapping (List[IOMapping]): Define the inputs' name, type, and storage type.
            output_mapping (List[IOMapping]): Defines the outputs' name, type, and storage type.
            model_name (Optional[str], optional): Name of the model/bundle, needed in multi-model case.
                                                  Defaults to "".
            bundle_path (Optional[str], optional): For completing . Defaults to None.
            bundle_config_names (BundleConfigNames, optional): Relevant config item names in a the bundle.
                                                               Defaults to None.
        """

        super().__init__(*args, **kwargs)
        self._executing = False
        self._lock = Lock()

        self._model_name = model_name.strip() if isinstance(model_name, str) else ""
        self._bundle_config_names = bundle_config_names if bundle_config_names else BundleConfigNames()
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping

        self._parser: ConfigParser = None  # Needs known bundle path, either on init or when compute function is called.
        self._inferer: Any = None  # Will be set during bundle parsing.
        self._init_completed: bool = False

        # Need to set the operator's input(s) and output(s). Even when the bundle parsing is done in init,
        # there is still a need to define what op inputs/outputs map to what keys in the bundle config,
        # along with the op input/output storage type.
        # Also, the App Executor needs to set the IO context of the operator before calling the compute function.
        self._add_inputs(self._input_mapping)
        self._add_outputs(self._output_mapping)

        # Complete the init if the bundle path is known, otherwise delay till the compute function is called
        # and try to get the model/bundle path from the execution context.
        try:
            self._bundle_path = (
                Path(bundle_path).expanduser().resolve() if bundle_path and len(bundle_path.strip()) > 0 else None
            )

            if self._bundle_path and self._bundle_path.exists():
                self._init_config(self._bundle_config_names.config_names)
                self._init_completed = True
            else:
                logging.debug(f"Bundle path, {self._bundle_path}, not valid. Will get it in the execution context.")
                self._bundle_path = None
        except Exception:
            logging.warn("Bundle parsing is not completed on init, delayed till this operator is called to execute.")
            self._bundle_path = None

        # Lazy init of model network till execution time when the context is fully set.
        self._model_network: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, name: str):
        if not name or isinstance(name, str):
            raise ValueError(f"Value, {name}, must be a non-empty string.")
        self._model_name = name

    @property
    def bundle_path(self) -> Union[Path, None]:
        """The path of the MONAI Bundle model."""
        return self._bundle_path

    @bundle_path.setter
    def bundle_path(self, bundle_path: Union[str, Path]):
        if not bundle_path or not Path(bundle_path).expanduser().is_file():
            raise ValueError(f"Value, {bundle_path}, is not a valid file path.")
        self._bundle_path = Path(bundle_path).expanduser().resolve()

    @property
    def parser(self) -> Union[ConfigParser, None]:
        """The ConfigParser object."""
        return self._parser

    @parser.setter
    def parser(self, parser: ConfigParser):
        if parser and isinstance(parser, ConfigParser):
            self._parser = parser
        else:
            raise ValueError("Value must be a valid ConfigParser object.")

    def _init_config(self, config_names):
        """Completes the init with a known path to the MONAI Bundle

        Args:
            config_names ([str]): Names of the config (files) in the bundle
        """

        parser = get_bundle_config(str(self._bundle_path), config_names)
        self._parser = parser

        meta = self.parser["_meta_"]

        # When this function is NOT called by the __init__, setting the pip_packages env here
        # will not get dependencies to the App SDK Packager to install the packages in the MAP.
        pip_packages = ["monai"] + [f"{k}=={v}" for k, v in meta["optional_packages_version"].items()]
        if self._env:
            self._env.pip_packages.extend(pip_packages)  # Duplicates will be figured out on use.
        else:
            self._env = OperatorEnv(pip_packages=pip_packages)

        if parser.get("device") is not None:
            self._device = parser.get_parsed_content("device")
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if parser.get(self._bundle_config_names.inferer_name) is not None:
            self._inferer = parser.get_parsed_content(self._bundle_config_names.inferer_name)
        else:
            self._inferer = SimpleInferer()

        self._inputs = meta["network_data_format"]["inputs"]
        self._outputs = meta["network_data_format"]["outputs"]

        # Given the restriction on operator I/O storage type, and known use cases, the I/O storage type of
        # this operator is limited to IN_MEMRORY objects, so we will remove the LoadImage and SaveImage
        self._preproc = self._get_compose(self._bundle_config_names.preproc_name, DISALLOW_LOAD_SAVE)
        self._postproc = self._get_compose(self._bundle_config_names.postproc_name, DISALLOW_LOAD_SAVE)

        # Need to find out the meta_key_postfix. The key name of the input concatenated with this postfix
        # will be the key name for the metadata for the input.
        # Customized metadata key names are not supported as of now.
        self._meta_key_postfix = self._get_meta_key_postfix(self._preproc)

        logging.debug(f"Effective transforms in pre-processing: {[type(t).__name__ for t in self._preproc.transforms]}")
        logging.debug(
            f"Effective Transforms in post-processing: {[type(t).__name__ for t in self._preproc.transforms]}"
        )

    def _get_compose(self, obj_name, disallowed_prefixes):
        """Gets a Compose object containing a sequence fo transforms from item `obj_name` in `self._parser`."""

        if self._parser.get(obj_name) is not None:
            compose = self._parser.get_parsed_content(obj_name)
            return filter_compose(compose, disallowed_prefixes)

        return Compose([])

    def _get_meta_key_postfix(self, compose: Compose, key_name: str = "meta_key_postfix") -> str:
        post_fix = PostFix.meta()
        if compose and key_name:
            for t in compose.transforms:
                if isinstance(t, MapTransform) and hasattr(t, key_name):
                    post_fix = getattr(t, key_name)
                    # For some reason the attr is a tuple
                    if isinstance(post_fix, tuple):
                        post_fix = str(post_fix[0])
                    break

        return str(post_fix)

    def _get_io_data_type(self, conf):
        """
        Gets the input/output type of the given input or output metadata dictionary. The known Python types for input
        or output types are given in the dictionary `BundleOperator.known_io_data_types` which relate type names to
        the actual type. if `conf["type"]` is an actual object that's not a string then this is assumed to be the
        type specifier and is returned. The fallback type is `bytes` which indicates the value is a pickled object.

        Args:
            conf: configuration dictionary for an input or output from the "network_data_format" metadata section

        Returns:
            A concrete type associated with this input/output type, this can be Image or np.ndarray or a Python type
        """

        # The Bundle's network_data_format for inputs and outputs does not indicate the storage type, i.e. IN_MEMORY
        # or DISK, for the input(s) and output(s) of the operators. Configuration is required, though limited to
        # IN_MEMORY for now.
        # Certain association and transform are also required. The App SDK IN_MEMORY I/O can hold
        # Any type, so if the type match and content format matches, data can simply be used as is, however, with
        # the Type being Image, the object needs to be converted before being used as the expected "image" type.
        ctype = conf["type"]
        if ctype in self.known_io_data_types:  # known type name from the specification
            return self.known_io_data_types[ctype]
        elif isinstance(ctype, type):  # type object
            return ctype
        else:  # don't know, something that hasn't been figured out
            logging.warn(f"I/O data type, {ctype}, is not a known/supported type. Return as Type object.")
            return object

    def _add_inputs(self, input_mapping: List[IOMapping]):
        """Adds operator inputs as specified."""

        [self.add_input(v.label, v.data_type, v.storage_type) for v in input_mapping]

    def _add_outputs(self, output_mapping: List[IOMapping]):
        """Adds operator outputs as specified."""

        [self.add_output(v.label, v.data_type, v.storage_type) for v in output_mapping]

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        """Infers with the input(s) and saves the prediction result(s) to output

        Args:
            op_input (InputContext): An input context for the operator.
            op_output (OutputContext): An output context for the operator.
            context (ExecutionContext): An execution context for the operator.
        """

        # Try to get the Model object and its path from the context.
        #   If operator is not fully initialized, use model path as bundle path to finish it.
        # If Model not loaded, but bundle path exists, load model; edge case for local dev.
        #
        # `context.models.get(model_name)` returns a model instance if exists.
        # If model_name is not specified and only one model exists, it returns that model.

        self._model_network = context.models.get(self._model_name) if context.models else None
        if self._model_network:
            if not self._init_completed:
                with self._lock:
                    if not self._init_completed:
                        self._bundle_path = self._model_network.path
                        self._init_config(self._bundle_config_names.config_names)
                        self._init_completed
        elif self._bundle_path:
            # For the case of local dev/testing when the bundle path is not passed in as an exec cmd arg.
            # When run as a MAP docker, the bundle file is expected to be in the context, even if the model
            # network is loaded on a remote inference server (when the feature is introduced).
            logging.debug(f"Model network not loaded. Trying to load from model path: {self._bundle_path}")
            self._model_network = torch.jit.load(self.bundle_path, map_location=self._device).eval()
        else:
            raise IOError("Model network is not load and model file not found.")

        first_input_name, *other_names = list(self._inputs.keys())

        with torch.no_grad():
            inputs: Any = {}  # Use type Any to quiet MyPy type checking complaints.

            start = time.time()
            for name in self._inputs.keys():
                value, metadata = self._receive_input(name, op_input, context)
                inputs[name] = value
                if metadata:
                    inputs[(f"{name}_{self._meta_key_postfix}")] = metadata

            inputs = self.pre_process(inputs)
            first_input = inputs.pop(first_input_name)[None].to(self._device)  # select first input
            input_metadata = inputs.get(f"{first_input_name}_{self._meta_key_postfix}", None)

            # select other tensor inputs
            other_inputs = {k: v[None].to(self._device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            # select other non-tensor inputs
            other_inputs.update({k: inputs[k] for k in other_names if not isinstance(inputs[k], torch.Tensor)})
            logging.debug(f"Ingest and Pre-processing elapsed time (seconds): {time.time() - start}")

            start = time.time()
            outputs: Any = self.predict(data=first_input, **other_inputs)  # Use type Any to quiet MyPy complaints.
            logging.debug(f"Inference elapsed time (seconds): {time.time() - start}")

            # TODO: Does this work for models where multiple outputs are returned?
            # Note that the inputs are needed because the invert transform requires it.
            start = time.time()
            kw_args = {self.kw_preprocessed_inputs: inputs}
            outputs = self.post_process(ensure_tuple(outputs)[0], **kw_args)
            logging.debug(f"Post-processing elapsed time (seconds): {time.time() - start}")
        if isinstance(outputs, (tuple, list)):
            output_dict = dict(zip(self._outputs.keys(), outputs))
        elif not isinstance(outputs, dict):
            output_dict = {first(self._outputs.keys()): outputs}
        else:
            output_dict = outputs

        for name in self._outputs.keys():
            # Note that the input metadata needs to be passed.
            # Please see the comments in the called function for the reasons.
            self._send_output(output_dict[name], name, input_metadata, op_output, context)

    def predict(self, data: Any, *args, **kwargs) -> Union[Image, Any, Tuple[Any, ...], Dict[Any, Any]]:
        """Predicts output using the inferer."""

        return self._inferer(inputs=data, network=self._model_network, *args, **kwargs)

    def pre_process(self, data: Any, *args, **kwargs) -> Union[Image, Any, Tuple[Any, ...], Dict[Any, Any]]:
        """Processes the input dictionary with the stored transform sequence `self._preproc`."""

        if is_map_compose(self._preproc):
            return self._preproc(data)
        return {k: self._preproc(v) for k, v in data.items()}

    def post_process(self, data: Any, *args, **kwargs) -> Union[Image, Any, Tuple[Any, ...], Dict[Any, Any]]:
        """Processes the output list/dictionary with the stored transform sequence `self._postproc`.

        The "processed_inputs", in fact the metadata in it, need to be passed in so that the
        invertible transforms in the post processing can work properly.
        """

        # Expect the inputs be passed in so that the inversion can work.
        inputs = kwargs.get(self.kw_preprocessed_inputs, {})

        if is_map_compose(self._postproc):
            if isinstance(data, (list, tuple)):
                outputs_dict = dict(zip(data, self._outputs.keys()))
            elif not isinstance(data, dict):
                oname = first(self._outputs.keys())
                outputs_dict = {oname: data}
            else:
                outputs_dict = data

            # Need to add back the inputs including metadata as they are needed by the invert transform.
            outputs_dict.update(inputs)
            logging.debug(f"Effective output dict keys: {outputs_dict.keys()}")
            return self._postproc(outputs_dict)
        else:
            if isinstance(data, (list, tuple)):
                return list(map(self._postproc, data))

            return self._postproc(data)

    def _receive_input(self, name: str, op_input: InputContext, context: ExecutionContext):
        """Extracts the input value for the given input name."""

        # The op_input can have the storage type of IN_MEMORY with the data type being Image or others,
        # as well as the other type of DISK with data type being DataPath.
        # The problem is, the op_input object does not have an attribute for the storage type, which
        # needs to be inferred from data type, with DataPath meaning DISK storage type. The file
        # content type may be interpreted from the bundle's network input type, but it is indirect
        # as the op_input is the input for processing transforms, not necessarily directly for the network.
        in_conf = self._inputs[name]
        itype = self._get_io_data_type(in_conf)
        value = op_input.get(name)

        metadata = None
        if isinstance(value, DataPath):
            if not value.path.exists():
                raise ValueError(f"Input path, {value.path}, does not exist.")

            file_path = value.path / name
            # The named input can only be a folder as of now, but just in case things change.
            if value.path.is_file():
                file_path = value.path
            elif not file_path.exists() and value.path.is_dir:
                # Expect one and only one file exists for use.
                files = [f for f in value.path.glob("*") if f.is_file()]
                if len(files) != 1:
                    raise ValueError(f"Input path, {value.path}, should have one and only one file.")

                file_path = files[0]

            # Only Python pickle file and or numpy file are supported as of now.
            with open(file_path, "rb") as f:
                if itype == np.ndarray:
                    value = np.load(file_path, allow_pickle=True)
                else:
                    value = pickle.load(f)

        # Once extracted, the input data may be further processed depending on its actual type.
        if isinstance(value, Image):
            # Need to get the image ndarray as well as metadata
            value, metadata = self._convert_from_image(value)
            logging.debug(f"Shape of the converted input image: {value.shape}")
            logging.debug(f"Metadata of the converted input image: {metadata}")
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value).to(self._device)

        # else value is some other object from memory

        return value, metadata

    def _send_output(self, value: Any, name: str, metadata: Dict, op_output: OutputContext, context: ExecutionContext):
        """Send the given output value to the output context."""

        logging.debug(f"Setting output {name}")

        out_conf = self._outputs[name]
        otype = self._get_io_data_type(out_conf)

        if otype == Image:
            # The value must be torch.tensor or ndarray. Note also that by convention the image/tensor
            # out of the MONAI post processing is [CWHD] with dim for batch already squeezed out.
            # Prediction image, e.g. segmentation image, needs to have its dimensions
            # rearranged to fit the conventions used by Image class, i.e. [DHW], without channel dim.
            # Also, based on known use cases, e.g. prediction being seg image and the downstream
            # operators expect the data type to be unit8, conversion needs to be done as well.
            # Metadata, such as pixel spacing and orientation, also needs to be set in the Image object,
            # which is why metadata is expected to be passed in.
            # TODO: Revisit when multi-channel images are supported.

            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            elif not isinstance(value, np.ndarray):
                raise TypeError("arg 1 must be of type torch.Tensor or ndarray.")

            logging.debug(f"Output {name} numpy image shape: {value.shape}")
            result: Any = Image(np.swapaxes(np.squeeze(value, 0), 0, 2).astype(np.uint8), metadata=metadata)
            logging.debug(f"Converted Image shape: {result.asnumpy().shape}")
        elif otype == np.ndarray:
            result = np.asarray(value)
        elif out_conf["type"] == "probabilities":
            _, value_class = value.max(dim=0)
            prediction = [out_conf["channel_def"][str(int(v))] for v in value.flatten()]

            result = {"result": prediction, "probabilities": value.cpu().numpy()}
        elif isinstance(value, torch.Tensor):
            result = value.cpu().numpy()

        # The operator output currently has many limitation depending on if the operator is
        # a leaf node or not. The get method throws for non-leaf node, irrespective of storage type,
        # and for leaf node if the storage type is IN_MEMORY.
        try:
            op_output_config = op_output.get(name)
            if isinstance(op_output_config, DataPath):
                output_file = op_output_config.path / name
                output_file.parent.mkdir(exist_ok=True)
                # Save pickle file
                with open(output_file, "wb") as wf:
                    pickle.dump(result, wf)

                # Cannot (re)set/modify the op_output path to the actual file like below
                # op_output.set(str(output_file), name)
            else:
                op_output.set(result, name)
        except ItemNotExistsError:
            # The following throws if the output storage type is DISK, but The OutputContext
            # currently does not expose the storage type. Try and let it throw if need be.
            op_output.set(result, name)

    def _convert_from_image(self, img: Image) -> Tuple[np.ndarray, Dict]:
        """Converts the Image object to the expected numpy array with metadata dictionary.

        Args:
            img: A SDK Image object.
        """

        # The Image class provides a numpy array and a metadata dict without a defined set of keys.
        # In most scenarios, if not all, DICOM series is converted to Image by the
        # DICOMSeriesToVolumeOperator, but the generated metadata lacks the specifics keys expected
        # by the MONAI transforms. So there is need to convert the Image object.
        # Also, there is not a defined key to express the source or producer of an Image object, so,
        # one has to inspect certain keys, based on known conversion, to infer the producer.
        # An issues already exists for the improvement of the Image class.

        img_meta_dict: Dict = img.metadata()

        if (
            not img_meta_dict
            or ("spacing" in img_meta_dict and "original_affine" in img_meta_dict)
            or "row_pixel_spacing" not in img_meta_dict
        ):

            return img.asnumpy(), img_meta_dict
        else:
            return self._convert_from_image_dicom_source(img)

    def _convert_from_image_dicom_source(self, img: Image) -> Tuple[np.ndarray, Dict]:
        """Converts the Image object to the expected numpy array with metadata dictionary.

        Args:
            img: A SDK Image object converted from DICOM instances.
        """

        img_meta_dict: Dict = img.metadata()
        meta_dict = {key: img_meta_dict[key] for key in img_meta_dict.keys()}

        # The MONAI ImageReader, e.g. the ITKReader, arranges the image spatial dims in WHD,
        # so the "spacing" needs to be expressed in such an order too, as expected by the transforms.
        meta_dict["spacing"] = np.asarray(
            [
                img_meta_dict["row_pixel_spacing"],
                img_meta_dict["col_pixel_spacing"],
                img_meta_dict["depth_pixel_spacing"],
            ]
        )
        meta_dict["original_affine"] = np.asarray(img_meta_dict.get("nifti_affine_transform", None))
        meta_dict["affine"] = meta_dict["original_affine"]

        # Similarly the Image ndarray has dim order DHW, to be rearranged to WHD.
        # TODO: Need to revisit this once multi-channel image is supported and the Image class itself
        #       is enhanced to provide attributes or functions for channel and dim order details.
        converted_image = np.swapaxes(img.asnumpy(), 0, 2)

        # The spatial shape is then that of the converted image, in WHD
        meta_dict["spatial_shape"] = np.asarray(converted_image.shape)

        # Well, now channel for now.
        meta_dict["original_channel_dim"] = "no_channel"

        return converted_image, meta_dict
