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

from monai.deploy.operators.monai_bundle_inference_operator import MonaiBundleInferenceOperator, get_bundle_config
from monai.deploy.utils.importutil import optional_import
from typing import Any, Dict, Tuple, Union
from monai.deploy.core import Image
from pathlib import Path
MONAI_UTILS = "monai.utils"
nibabel, _ = optional_import("nibabel", "3.2.1")
torch, _ = optional_import("torch", "1.10.2")

NdarrayOrTensor, _ = optional_import("monai.config", name="NdarrayOrTensor")
MetaTensor, _ = optional_import("monai.data.meta_tensor", name="MetaTensor")
PostFix, _ = optional_import("monai.utils.enums", name="PostFix")  # For the default meta_key_postfix
first, _ = optional_import("monai.utils.misc", name="first")
ensure_tuple, _ = optional_import(MONAI_UTILS, name="ensure_tuple")
convert_to_dst_type, _ = optional_import(MONAI_UTILS, name="convert_to_dst_type")
Key, _ = optional_import(MONAI_UTILS, name="ImageMetaKey")
MetaKeys, _ = optional_import(MONAI_UTILS, name="MetaKeys")
SpaceKeys, _ = optional_import(MONAI_UTILS, name="SpaceKeys")
Compose_, _ = optional_import("monai.transforms", name="Compose")
ConfigParser_, _ = optional_import("monai.bundle", name="ConfigParser")
MapTransform_, _ = optional_import("monai.transforms", name="MapTransform")
SimpleInferer, _ = optional_import("monai.inferers", name="SimpleInferer")

Compose: Any = Compose_
MapTransform: Any = MapTransform_
ConfigParser: Any = ConfigParser_
__all__ = ["MonainnUNetBundleInferenceOperator"]


class MonainnUNetBundleInferenceOperator(MonaiBundleInferenceOperator):
    """
    A specialized operator for performing inference using the MONAI nnUNet bundle.
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
        
        self._nnunet_predictor : torch.nn.Module = None
        
       
    def _init_config(self, config_names):   

        super()._init_config(config_names)
        parser = get_bundle_config(str(self._bundle_path), config_names)       
        self._parser = parser

        self._nnunet_predictor = parser.get_parsed_content("network_def")

    def predict(self, data: Any, *args, **kwargs) -> Union[Image, Any, Tuple[Any, ...], Dict[Any, Any]]:
        """Predicts output using the inferer."""

        self._nnunet_predictor.predictor.network = self._model_network
        #os.environ['nnUNet_def_n_proc'] = "1"
        if len(data.shape) == 4:
            data = data[None]
        return self._nnunet_predictor(data)
