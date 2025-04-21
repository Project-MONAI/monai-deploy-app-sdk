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
from pathlib import Path
from typing import Tuple

import tritonclient.http as httpclient
from google.protobuf import text_format
from tritonclient.grpc.model_config_pb2 import DataType, ModelConfig

from monai.deploy.utils.importutil import optional_import

from .model import Model

torch, _ = optional_import("torch")


def parse_triton_config_pbtxt(pbtxt_path) -> ModelConfig:
    """Parse a Triton model config.pbtxt file

    Args:
        config_path: Path to the config.pbtxt file

    Returns:
        ModelConfig object containing parsed configuration

    Raises:
        ValueError: If config file is invalid or missing required fields
        FileNotFoundError: If config file doesn't exist
    """

    if not pbtxt_path.exists():
        raise FileNotFoundError(f"Config file not found: {pbtxt_path}")
    try:
        # Read the config.pbtxt content
        with open(pbtxt_path, "r") as f:
            config_text = f.read()
            # Parse using protobuf text_format
            model_config = ModelConfig()
            text_format.Parse(config_text, model_config)
            return model_config

    except Exception as e:
        raise ValueError(f"Failed to parse config file {pbtxt_path}") from e


class TritonRemoteModel:
    """A remote model that is hosted on a Triton Inference Server.

    Args:
        model_name (str): The name of the model.
        netloc (str): The network location of the Triton Inference Server.
        model_config (ModelConfig): The model config.
        headers (dict): The headers to send to the Triton Inference Server.
    """

    def __init__(self, model_name, netloc, model_config, headers=None, **kwargs):
        self._headers = headers
        self._request_compression_algorithm = None
        self._response_compression_algorithm = None
        self._model_name = model_name
        self._model_version = None
        self._model_config = model_config
        self._request_compression_algorithm = None
        self._response_compression_algorithm = None
        self._count = 0

        try:
            self._triton_client = httpclient.InferenceServerClient(url=netloc, verbose=kwargs.get("verbose", False))
            logging.info(f"Created triton client: {self._triton_client}")
        except Exception as e:
            logging.error("channel creation failed: " + str(e))
            raise

    def __call__(self, data, **kwds):

        self._count += 1
        logging.info(f"{self.__class__.__name__}.__call__: {self._model_name} count: {self._count}")

        inputs = []
        outputs = []

        # For now support only one input and one output
        input_name = self._model_config.input[0].name
        input_type = str.split(DataType.Name(self._model_config.input[0].data_type), "_")[1]  # remove the prefix
        input_shape = list(self._model_config.input[0].dims)
        data_shape = list(data.shape)
        logging.info(f"Model config input data shape: {input_shape}")
        logging.info(f"Actual input data shape: {data_shape}")

        # The server side will handle the batching, and with dynamic batching
        # the model config does not have the batch size in the input dims.
        logging.info(f"Effective input_name: {input_name}, input_type: {input_type}, input_shape: {data_shape}")

        inputs.append(httpclient.InferInput(input_name, data_shape, input_type))

        # Move to tensor to CPU
        input0_data_np = data.detach().cpu().numpy()
        logging.debug(f"Input data shape: {input0_data_np.shape}")

        # Initialize the data
        inputs[0].set_data_from_numpy(input0_data_np, binary_data=False)

        output_name = self._model_config.output[0].name
        outputs.append(httpclient.InferRequestedOutput(output_name, binary_data=True))

        query_params = {f"{self._model_name}_count": self._count}
        results = self._triton_client.infer(
            self._model_name,
            inputs,
            outputs=outputs,
            query_params=query_params,
            headers=self._headers,
            request_compression_algorithm=self._request_compression_algorithm,
            response_compression_algorithm=self._response_compression_algorithm,
        )

        logging.info(f"Got results{results.get_response()}")
        output0_data = results.as_numpy(output_name)
        logging.debug(f"as_numpy output0_data.shape: {output0_data.shape}")
        logging.debug(f"as_numpy output0_data.dtype: {output0_data.dtype}")

        # Convert numpy array to torch tensor as expected by the anticipated clients,
        # e.g. monai cliding window inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.as_tensor(output0_data).to(device)  # from_numpy is fine too.


class TritonModel(Model):
    """Represents Triton models in the model repository.

    Triton Inference Server models are stored in a directory structure like this
    (https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md):

    ::

        <model-repository-path>/
            <model-name>/
            [config.pbtxt]
            [<output-labels-file> ...]
            <version>/
                <model-definition-file>
            <version>/
                <model-definition-file>
            ...
            <model-name>/
            [config.pbtxt]
            [<output-labels-file> ...]
            <version>/
                <model-definition-file>
            <version>/
                <model-definition-file>
            ...
            ...

    This class checks if the given path meets the folder structure of Triton:

    1) The path should be a folder path.

    2) The model folder must contain a config.pbtxt file.

       a. A config.pbtxt file may contain model name.
          In that case, model's name should match with the folder name.

    3) The model folder may include one or more folders having a positive integer value as version.
       For the server side, the following is required, however, not required for the client side
       which parses only the model config.pbtxt file.

       a. Each such folder must contain a folder or file whose file name (without extension) is 'model',
          unless an attribute is used to specify model file name.

    If no version policy is specified, the latest version of a model is loaded and served.
    Model items identified would have a folder path, not a specific model file path.

    This class encapuslates a single triton model. As such, the model repository folder
    will first be parsed by the named_model class, and then each sub folder by this class.
    """

    model_type: str = "triton"

    def __init__(self, path: str, name: str = ""):
        """Initializes a TritonModel.

        This assumes that the given path is a valid Triton model repository.

        Args:
            path (str): A Path to the model repository.
            name (str): A name of the model.
        """
        super().__init__(path, name)

        self._model_path: Path = Path(path)
        self._name = self._model_path.stem
        self._model_config = parse_triton_config_pbtxt(self._model_path / "config.pbtxt")
        # The model name in the config.pbtxt, if present, must match the model folder name.
        if self._model_config.name and self._model_config.name.casefold() != self._name.casefold():
            raise ValueError(
                f"Model name in config.pbtxt ({self._model_config.name}) does not match the folder name ({self._name})."
            )

        self._netloc: str = ""
        logging.info(f"Created Triton model: {self._name}")

    def connect(self, netloc: str, **kwargs):
        """Connect to the Triton Inference Server at the network location.

        Args:
            netloc (str): The network location of the Triton Inference Server.
        """

        if not netloc:
            raise ValueError("Network location is required to connect to the Triton Inference Server.")

        if self._netloc and not self._netloc.casefold() == netloc.casefold():
            logging.warning(f"Reconnecting to a different Triton Inference Server at {netloc} from {self._netloc}.")

        self._predictor = TritonRemoteModel(self._name, netloc, self._model_config, **kwargs)
        self._netloc = netloc

        return self._predictor

    @property
    def model_config(self):
        return self._model_config

    @property
    def net_loc(self):
        """Get the network location of the Triton Inference Server, i.e. "<host>:<port>".

        Returns:
            str: The network location of the Triton Inference Server.
        """

        return self._netloc

    @net_loc.setter
    def net_loc(self, value: str):
        """Set the network location of the Triton Inference Server, and causes re-connect."""
        if not value:
            raise ValueError("Network location cannot be empty.")
        self._netloc = value
        # Reconnect to the Triton Inference Server at the new network location.
        self.connect(value)

    @property
    def predictor(self):
        if not self._predictor:
            raise ValueError("Model is not connected to the Triton Inference Server.")
        return self._predictor

    @predictor.setter
    def predictor(self, predictor: TritonRemoteModel):
        if not isinstance(predictor, TritonRemoteModel):
            raise ValueError("Predictor must be an instance of TritonRemoteModel.")
        self._predictor = predictor

    @classmethod
    def accept(cls, path: str) -> Tuple[bool, str]:
        model_folder: Path = Path(path)

        # The path should be a folder path, for an individual model, and must have the config.pbtxt file.
        if not model_folder.is_dir() or not (model_folder / "config.pbtxt").exists():
            return False, ""

        # Delay parsing the config.pbtxt protobuf for model name, input and output information etc.
        # Per convention, the model name is the same as the folder name, and the name in the config file,
        # if specified, must match the folder name.
        # For the server, each model folder must include one or more folders having a positive integer value as name,
        # and each such folder must contain a folder or file for the model.
        # At the client side, though there is no need to have the actual model file.
        # So the following is not necessary at all, just checking for logging purpose.
        found_model = False
        for version_folder in model_folder.iterdir():
            version_folder_name = version_folder.name
            if version_folder.is_dir() and version_folder_name.isnumeric() and int(version_folder_name) > 0:
                # Each such folder must contain a folder or file whose file name (without extension)
                #      is 'model'. The config.pbtxt can specify the actual model file with an attribute.
                if any(version_folder.glob("*")):
                    found_model = True
                    logging.info(f"Model {model_folder.name} version {version_folder_name} found in client workspace.")
        if not found_model:
            logging.info(f"Model {model_folder.name} only has config.pbtxt in client workspace.")

        return True, cls.model_type
