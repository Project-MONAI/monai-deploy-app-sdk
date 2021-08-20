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

from abc import abstractmethod
from monai.data.dataset import ArrayDataset
from monai.deploy.core.domain import image
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Optional, Union
import numpy as np
import nibabel as nib
from monai.deploy.utils.importutil import optional_import

# torch, _ = optional_import("torch", "1.5")
# monai, _ = optional_import("monai", "0.6.0")
import torch
from torch.utils.data._utils.collate import np_str_obj_array_pattern
from monai.data import Dataset, DataLoader, ImageReader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Transform, MapTransform, transform
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.utils import ImageMetaKey as Key
from monai.config import DtypeLike, KeysCollection
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Spacingd,
    Orientationd,
    Resized,
    SaveImaged,
    ScaleIntensityRanged,
    CropForegroundd,
    ToTensord,
    EnsureTyped,
)

from monai.deploy.exceptions import UnknownTypeError
from monai.deploy.core import (
    ExecutionContext,
    InputContext,
    OutputContext,
    IOType,
    Image,
    Operator,
    input,
    output
)

from .inference_operator import InferenceOperator

__all__ = ["MonaiSegInferenceOperator", "InMemImageReader"]

@input("image", Image, IOType.IN_MEMORY)
@output("seg_image", Image, IOType.IN_MEMORY)
class MonaiSegInferenceOperator(InferenceOperator):
    """The base segmentation operator using Monai transforms and inference.

    This operator preforms pre-transforms on a input image, inference
    using given model, and post-transforms. The segmentation image is saved
    as a named Image object in memory.

    If specified in the post transforms, results may also be saved to disk.
    """

    def __init__(self,
        roi_size:Union[Sequence[int], int],
        pre_transforms:Compose = None,
        post_transforms:Compose = None,
        *args,
        **kwargs):
        """Constructor of the operator.
        """
        super().__init__()
        self._executing = False
        self._lock = Lock()
        self._input_dataset_key = 'image'
        self._pred_dataset_key = 'pred'
        self._input_image = None # Image will come in when compute is called.
        self._reader = None
        self._roi_size = [i for i in roi_size]
        self._pre_transform = pre_transforms
        self._post_transforms = post_transforms

    @property
    def input_dataset_key(self):
        return self._input_dataset_key
    @input_dataset_key.setter
    def input_dataset_key(self, val:str):
        if not val or len(val) < 1:
            raise ValueError("Value cannot be None or blank.")
        self._input_dataset_key = val

    @property
    def pred_dataset_key(self):
        return self._pred_dataset_key
    @pred_dataset_key.setter
    def pred_dataset_key(self, val:str):
        if not val or len(val) < 1:
            raise ValueError("Value cannot be None or blank.")
        self._pred_dataset_key = val

    def compute(self, input: InputContext, output: OutputContext, context: ExecutionContext):
        """An abstract method that needs to be implemented by the user.

        Args:
            input (InputContext): An input context for the operator.
            output (OutputContext): An output context for the operator.
            context (ExecutionContext): An execution context for the operator.
        """
        with self._lock:
            if self._executing:
                raise RuntimeError("Operator is already executing.")
            else:
                self._executing = True
        try:
            input_image = input.get("image")
            if not input_image:
                raise ValueError('Input is None.')

            img_name = "Img_in_Mem"
            try:
                print("input_image.metadata")
                print(vars(input_image.metadata()))
                img_name = input_image.metadata().get["series_instance_uid", img_name]
            except Exception:  # Best effort
                pass

            pre_transforms = self._pre_transform
            post_transforms = self._post_transforms
            self._reader = InMemImageReader(input_image)

            pre_transforms = self._pre_transform if self._pre_transform else \
                self.pre_process(self._reader)

            print("pre-transform:")
            print(vars(pre_transforms.flatten()))

            post_transforms = self._post_transforms if self._post_transforms  else \
                self.post_process(pre_transforms)

            print("post-transform:")
            print(vars(post_transforms.flatten()))

            model = None
            if context.models:
                # `context.models.get(model_name)` returns a model instance if exists.
                # If model_name is not specified and only one model exists, it returns that model.
                model = context.models.get()
                print(f'Model path: {model.path}')
                print(f'Model name (expected None): {model.name}')
            else:
                model = torch.jit.load("/home/mqin/src/monai-app-sdk/examples/apps/model/segmentation_ct_spleen_pt_v1/1/model.pt")

            dataset = Dataset(data=[{self._input_dataset_key:img_name}], transform=pre_transforms)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            with torch.no_grad():
                for d in dataloader:
                    images = d[self._input_dataset_key].to(device)
                    sw_batch_size=4
                    d[self._pred_dataset_key] = sliding_window_inference(
                        inputs=images,
                        roi_size=self._roi_size,  #(160, 160, 160), #self._roi_size,
                        sw_batch_size=sw_batch_size,
                        overlap=0.5,
                        predictor=model
                    )
                    d = [post_transforms(i) for i in decollate_batch(d)]
                    out_ndarray = d[0][self._pred_dataset_key].cpu().numpy()
                    print(out_ndarray.shape)
                    out_ndarray = np.squeeze(out_ndarray, 0)
                    print(out_ndarray.shape)
                    out_ndarray = np.moveaxis(out_ndarray, 2, 0)
                    print(out_ndarray.shape)
                    print(np.amax(out_ndarray))
                    out_image = Image(out_ndarray)
                    output.set(out_image, "seg_image")
        finally:
            # Reset state on completing this method execution.
            with self._lock:
                self._executing = False

    def pre_process(self, img_reader) -> Compose:
        """Transforms input before being used for predicting on a model.

        This method must be overridden by a derived class.

        Raises:
            NotImplementedError: When the subclass does not override this method.
        """
        raise NotImplementedError(
             f"Subclass {self.__class__.__name__} must implement this method.")

    def post_process(self, pre_transforms: Compose, out_dir:str="./infer_out") -> Compose:
        """Transform the prediction results from the model(s).

        This method must be overridden by a derived class.

        Raises:
            NotImplementedError: When the subclass does not override this method.
        """
        raise NotImplementedError(
             f"Subclass {self.__class__.__name__} must implement this method.")

    def predict(self, data:Any) -> Union[Image, Any]:
        """Prdicts results using the models(s) with input tensors.

        This method must be overridden by a derived class.

        Raises:
            NotImplementedError: When the subclass does not override this method.
        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement this method.")

class InMemImageReader(ImageReader):
    def __init__(self, input_image: Image, channel_dim: Optional[int] = None, series_uid: str = "", **kwargs):
        super().__init__()
        self.input_image = input_image
        self.kwargs = kwargs
        self.channel_dim = channel_dim
        self.series_uid = series_uid

    def verify_suffix(self, filename: Union[Sequence[str], str]) -> bool:
        return True

    def read(self, data: Union[Sequence[str], str], **kwargs) -> Union[Sequence[Any], Any]:
        # Really does not have anything to do.
        return self.input_image.asnumpy()
    def get_data(self, input_image):

        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        for i in ensure_tuple(self.input_image):
            if not isinstance(i, Image):
                raise TypeError('Only object of Image type is supported.')
            data = i.asnumpy()
            img_array.append(data)
            header = self._get_meta_dict(i)
            _copy_compatible_dict(header, compatible_meta)

        # Stacking image is not really needed, as there is one image only.
        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_meta_dict(self, img: Image) -> Dict:
        """
        Get the meta data of the image and convert to dict type.

        Args:
            img: an Monai App Sdk Image object.
        """
        img_meta_dict = img.metadata()
        meta_dict = {key: img_meta_dict[key] for key in img_meta_dict.keys()}

        # Will have to derive some key metadata as the SDK Image lacks the necessary interfaces.
        # So, for now have to get to the Image generator, namely DICOMSeriesToVolumeOperator, and
        # rely on its published metadata.


        # Recall that the column and row data for pydicom pixel_array had been switched, and the depth
        # is the last dim in DICOMSeriesToVolumeOperator
        meta_dict["spacing"] = np.asarray([
            img_meta_dict["col_pixel_spacing"],
            img_meta_dict["row_pixel_spacing"],
            img_meta_dict["depth_pixel_spacing"]
        ])
        meta_dict["original_affine"] = np.asarray(img_meta_dict["nifti_affine_transform"])
        meta_dict["affine"] = meta_dict["original_affine"]
        meta_dict["spatial_shape"] = np.asarray(img.asnumpy().shape)
        # Well, no channel as the image data shape is forced to the the same as spatial shape
        meta_dict["original_channel_dim"] = "no_channel"

        return meta_dict


# class LoadImageFromMem(Transform):

#     def __init__(self, key:str, reader=None, image_only: bool = False, dtype: DtypeLike = np.float32, *args, **kwargs) -> None:
#         """
#         Args:
#             image (Image): the Image instance already loaded and to be used for inference
#             image_only: if True return only the image volume, otherwise return image data array and header dict.
#             dtype: if not None convert the loaded image to this data type.
#             args: additional parameters for reader if providing a reader name.
#             kwargs: additional parameters for reader if providing a reader name.

#         Note:
#             - The transform returns an image data array if `image_only` is True,
#               or a tuple of two elements containing the data array, and the meta data
#               in a dictionary format otherwise.
#         """

#         self.key = key
#         self.image_only = image_only
#         self.dtype = dtype
#         self.reader = reader

#     def __call__(self, data: Any = None):
#         """Load image and meta data from the parsed image data.

#         Args:
#             data (Any, optional): Input if any but not needed. Defaults to None.
#         """
#         img = self.reader.read(image)
#         img_array, meta_data = self.reader.get_data(img)
#         img_array = img_array.astype(self.dtype)

#         if self.image_only:
#             return img_array
#         meta_data[Key.FILENAME_OR_OBJ] = ensure_tuple(self.key)
#         # make sure all elements in metadata are little endian
#         meta_data = switch_endianness(meta_data, "<")

#         return img_array, meta_data

# # Code reused from Monai since it is not exported
# def switch_endianness(data, new="<"):
#     """
#     Convert the input `data` endianness to `new`.
#     Args:
#         data: input to be converted.
#         new: the target endianness, currently support "<" or ">".
#     """
#     if isinstance(data, np.ndarray):
#         # default to system endian
#         sys_native = "<" if (sys.byteorder == "little") else ">"
#         current_ = sys_native if data.dtype.byteorder not in ("<", ">") else data.dtype.byteorder
#         if new not in ("<", ">"):
#             raise NotImplementedError(f"Not implemented option new={new}.")
#         if current_ != new:
#             data = data.byteswap().newbyteorder(new)
#     elif isinstance(data, tuple):
#         data = tuple(switch_endianness(x, new) for x in data)
#     elif isinstance(data, list):
#         data = [switch_endianness(x, new) for x in data]
#     elif isinstance(data, dict):
#         data = {k: switch_endianness(v, new) for k, v in data.items()}
#     elif not isinstance(data, (bool, str, float, int, type(None))):
#         raise RuntimeError(f"Unknown type: {type(data).__name__}")
#     return data

# Monai code for the ImageReader
def _copy_compatible_dict(from_dict: Dict, to_dict: Dict):
    if not isinstance(to_dict, dict):
        raise ValueError(f"to_dict must be a Dict, got {type(to_dict)}.")
    if not to_dict:
        for key in from_dict:
            datum = from_dict[key]
            if isinstance(datum, np.ndarray) and np_str_obj_array_pattern.search(datum.dtype.str) is not None:
                continue
            to_dict[key] = datum
    else:
        affine_key, shape_key = "affine", "spatial_shape"
        if affine_key in from_dict and not np.allclose(from_dict[affine_key], to_dict[affine_key]):
            raise RuntimeError(
                "affine matrix of all images should be the same for channel-wise concatenation. "
                f"Got {from_dict[affine_key]} and {to_dict[affine_key]}."
            )
        if shape_key in from_dict and not np.allclose(from_dict[shape_key], to_dict[shape_key]):
            raise RuntimeError(
                "spatial_shape of all images should be the same for channel-wise concatenation. "
                f"Got {from_dict[shape_key]} and {to_dict[shape_key]}."
            )

def _stack_images(image_list: List, meta_dict: Dict):
    if len(image_list) <= 1:
        return image_list[0]
    if meta_dict.get("original_channel_dim", None) not in ("no_channel", None):
        raise RuntimeError("can not read a list of images which already have channel dimension.")
    meta_dict["original_channel_dim"] = 0
    return np.stack(image_list, axis=0)

