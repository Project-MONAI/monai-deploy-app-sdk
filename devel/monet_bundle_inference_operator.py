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
import logging
from monai.deploy.core import Image
from monai.deploy.operators.monai_bundle_inference_operator import MonaiBundleInferenceOperator, get_bundle_config
from monai.deploy.utils.importutil import optional_import
from monai.transforms import SpatialResample, ConcatItemsd
import numpy as np
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
__all__ = ["MONetBundleInferenceOperator"]


def define_affine_from_meta(meta: Dict[str, Any]) -> np.ndarray:
    """
    Define an affine matrix from the metadata of a tensor.

    Parameters
    ----------
    meta : Dict[str, Any]
        Metadata dictionary containing 'pixdim', 'origin', and 'direction'.

    Returns
    -------
    np.ndarray
        A 4x4 affine matrix constructed from the metadata.
    """
    if "pixdim" not in meta or "origin" not in meta or "direction" not in meta:
        return meta.get("affine", np.eye(4))
    pixdim = meta["pixdim"]
    origin = meta["origin"]
    direction = meta["direction"].reshape(3, 3)

                    # Extract 3D spacing
    spacing = pixdim[1:4]  # drop the first element (usually 1 for time dim)

    # Scale the direction vectors by spacing to get rotation+scale part
    affine = direction * spacing[np.newaxis, :]

    # Append origin to get 3x4 affine matrix
    affine = np.column_stack((affine, origin))

    # Make it a full 4x4 affine
    affine_4x4 = np.vstack((affine, [0, 0, 0, 1]))
    pixdim = meta["pixdim"]
    origin = meta["origin"]
    direction = meta["direction"].reshape(3, 3)

    # Extract 3D spacing
    spacing = pixdim[1:4]  # drop the first element (usually 1 for time dim)

    # Scale the direction vectors by spacing to get rotation+scale part
    affine = direction * spacing[np.newaxis, :]

    # Append origin to get 3x4 affine matrix
    affine = np.column_stack((affine, origin))

    # Make it a full 4x4 affine
    return torch.Tensor(np.vstack((affine, [0, 0, 0, 1])))

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

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._nnunet_predictor: torch.nn.Module = None
        self.ref_modality = None
        if "ref_modality" in kwargs:
            self.ref_modality = kwargs["ref_modality"]

    def _init_config(self, config_names):

        super()._init_config(config_names)
        parser = get_bundle_config(str(self._bundle_path), config_names)
        self._parser = parser

        self._nnunet_predictor = parser.get_parsed_content("network_def")

    def predict(self, data: Any, *args, **kwargs) -> Union[Image, Any, Tuple[Any, ...], Dict[Any, Any]]:
        """Predicts output using the inferer."""

        self._nnunet_predictor.predictor.network = self._model_network
        # os.environ['nnUNet_def_n_proc'] = "1"

        if len(kwargs) > 0:
            multimodal_data = {"image": data}
            if self.ref_modality is not None:
                if self.ref_modality not in kwargs:
                    target_affine_4x4 = define_affine_from_meta(data.meta)
                    spatial_size = data.shape[1:4]
                    if "pixdim" in data.meta:
                        pixdim = data.meta["pixdim"]
                    else:
                        pixdim = np.abs(np.array(target_affine_4x4[:3, :3].diagonal().tolist()))
                else:
                    target_affine_4x4 = define_affine_from_meta(kwargs[self.ref_modality].meta)
                    spatial_size = kwargs[self.ref_modality].shape[1:4]
                    if "pixdim" in kwargs[self.ref_modality].meta:
                        pixdim = kwargs[self.ref_modality].meta["pixdim"]
                    else:
                        pixdim = np.abs(np.array(target_affine_4x4[:3, :3].diagonal().tolist()))
            else:
                target_affine_4x4 = define_affine_from_meta(data.meta)
                spatial_size = data.shape[1:4]
                if "pixdim" in data.meta:
                    pixdim = data.meta["pixdim"]
                else:
                    pixdim = np.abs(np.array(target_affine_4x4[:3, :3].diagonal().tolist()))

            for key in kwargs.keys():
                if isinstance(kwargs[key], MetaTensor):
                    source_affine_4x4 = define_affine_from_meta(kwargs[key].meta)
                    kwargs[key].meta["affine"] = torch.Tensor(source_affine_4x4)
                    kwargs[key].meta["pixdim"] = pixdim
                    self._logger.info(f"Resampling {key} from {source_affine_4x4} to {target_affine_4x4}")

                    multimodal_data[key] = SpatialResample(mode="bilinear")(kwargs[key], dst_affine=target_affine_4x4,
                                                         spatial_size=spatial_size,
                                                         )
            source_affine_4x4 = define_affine_from_meta(data.meta)
            data.meta["affine"] = torch.Tensor(source_affine_4x4)
            data.meta["pixdim"] = pixdim
            multimodal_data["image"] = SpatialResample(mode="bilinear")(
                data, dst_affine=target_affine_4x4, spatial_size=spatial_size
            )
            
            self._logger.info(f"Resampling 'image' from from {source_affine_4x4} to {target_affine_4x4}")
            data = ConcatItemsd(keys=list(multimodal_data.keys()),name="image")(multimodal_data)["image"]
            data.meta["pixdim"] = np.insert(pixdim, 0, 0)

        if len(data.shape) == 4:
            data = data[None]
        return self._nnunet_predictor(data)
