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

from typing import Any, Dict, Tuple, Union

from monai.deploy.core import Image
from monai.deploy.operators.monai_bundle_inference_operator import MonaiBundleInferenceOperator, get_bundle_config
from monai.deploy.utils.importutil import optional_import
from monai.transforms import ConcatItemsd, ResampleToMatch
from monai.deploy.core.models.torch_model import TorchScriptModel
torch, _ = optional_import("torch", "1.10.2")
MetaTensor, _ = optional_import("monai.data.meta_tensor", name="MetaTensor")
__all__ = ["MONetBundleInferenceOperator"]


class MONetBundleInferenceOperator(MonaiBundleInferenceOperator):
    """
    A specialized operator for performing inference using the MONet bundle.
    This operator extends the `MonaiBundleInferenceOperator` to support nnUNet-specific
    configurations and prediction logic. It initializes the nnUNet predictor and provides
    a method for performing inference on input data.

    Attributes
    ----------
    _nnunet_predictor : torch.nn.Module
        The nnUNet predictor module used for inference.

    Methods
    -------
    _init_config(config_names)
        Initializes the configuration for the nnUNet bundle, including parsing the bundle
        configuration and setting up the nnUNet predictor.
    predict(data, *args, **kwargs)
        Performs inference on the input data using the nnUNet predictor.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self._nnunet_predictor: torch.nn.Module = None

    def _init_config(self, config_names):

        super()._init_config(config_names)
        parser = get_bundle_config(str(self._bundle_path), config_names)
        self._parser = parser

        self._nnunet_predictor = parser.get_parsed_content("network_def")

    def _set_model_network(self, model_network):
        """
        Sets the model network for the nnUNet predictor.

        Parameters
        ----------
        model_network : torch.nn.Module or torch.jit.ScriptModule
            The model network to be used for inference.
        """
        if not isinstance(model_network, torch.nn.Module) and not torch.jit.isinstance(model_network, torch.jit.ScriptModule) and not isinstance(model_network, TorchScriptModel):
            raise TypeError("model_network must be an instance of torch.nn.Module or torch.jit.ScriptModule")
        self._nnunet_predictor.predictor.network = model_network
    
    def predict(self, data: Any, *args, **kwargs) -> Union[Image, Any, Tuple[Any, ...], Dict[Any, Any]]:
        """Predicts output using the inferer. If multimodal data is provided as keyword arguments,
        it concatenates the data with the main input data."""

        self._set_model_network(self._model_network)

        if len(kwargs) > 0:
            multimodal_data = {"image": data}
            for key in kwargs.keys():
                if isinstance(kwargs[key], MetaTensor):
                    multimodal_data[key] = ResampleToMatch(mode="bilinear")(kwargs[key], img_dst=data
                                                         )
            data = ConcatItemsd(keys=list(multimodal_data.keys()),name="image")(multimodal_data)["image"]
        if len(data.shape) == 4:
            data = data[None]
        return self._nnunet_predictor(data)
