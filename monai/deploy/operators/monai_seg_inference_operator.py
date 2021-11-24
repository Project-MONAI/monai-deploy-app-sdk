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

from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from monai.deploy.utils.importutil import optional_import

torch, _ = optional_import("torch", "1.5")
np_str_obj_array_pattern, _ = optional_import("torch.utils.data._utils.collate", name="np_str_obj_array_pattern")
Dataset, _ = optional_import("monai.data", name="Dataset")
DataLoader, _ = optional_import("monai.data", name="DataLoader")
ImageReader_, _ = optional_import("monai.data", name="ImageReader")
# Dynamic class is not handled so make it Any for now: https://github.com/python/mypy/issues/2477
ImageReader: Any = ImageReader_
decollate_batch, _ = optional_import("monai.data", name="decollate_batch")
sliding_window_inference, _ = optional_import("monai.inferers", name="sliding_window_inference")
ensure_tuple, _ = optional_import("monai.utils", name="ensure_tuple")
Compose_, _ = optional_import("monai.transforms", name="Compose")
# Dynamic class is not handled so make it Any for now: https://github.com/python/mypy/issues/2477
Compose: Any = Compose_
sliding_window_inference, _ = optional_import("monai.inferers", name="sliding_window_inference")

import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, OutputContext

from .inference_operator import InferenceOperator

__all__ = ["MonaiSegInferenceOperator", "InMemImageReader"]


@md.input("image", Image, IOType.IN_MEMORY)
@md.output("seg_image", Image, IOType.IN_MEMORY)
@md.env(pip_packages=["monai==0.6.0", "torch>=1.5", "numpy>=1.17"])
class MonaiSegInferenceOperator(InferenceOperator):
    """This segmentation operator uses MONAI transforms and Sliding Window Inference.

    This operator preforms pre-transforms on a input image, inference
    using a given model, and post-transforms. The segmentation image is saved
    as a named Image object in memory.

    If specified in the post transforms, results may also be saved to disk.
    """

    # For testing the app directly, the model should be at the following path.
    MODEL_LOCAL_PATH = "model/model.ts"

    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        pre_transforms: Compose,
        post_transforms: Compose,
        overlap: float = 0.5,
        *args,
        **kwargs,
    ):
        """Creates a instance of this class.

        Args:
            roi_size (Union[Sequence[int], int]): The tensor size used in inference.
            pre_transforms (Compose): MONAI Compose oject used for pre-transforms.
            post_transforms (Compose): MONAI Compose oject used for post-transforms.
            overlap (float): The overlap used in sliding window inference.
        """

        super().__init__()
        self._executing = False
        self._lock = Lock()
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"
        self._input_image = None  # Image will come in when compute is called.
        self._reader: Any = None
        self._roi_size = ensure_tuple(roi_size)
        self._pre_transform = pre_transforms
        self._post_transforms = post_transforms
        self.overlap = overlap

    @property
    def roi_size(self):
        """The ROI size of tensors used in prediction."""
        return self._roi_size

    @roi_size.setter
    def roi_size(self, roi_size: Union[Sequence[int], int]):
        self._roi_size = ensure_tuple(roi_size)

    @property
    def input_dataset_key(self):
        """This is the input image key name used in dictionary based MONAI pre-transforms."""
        return self._input_dataset_key

    @input_dataset_key.setter
    def input_dataset_key(self, val: str):
        if not val or len(val) < 1:
            raise ValueError("Value cannot be None or blank.")
        self._input_dataset_key = val

    @property
    def pred_dataset_key(self):
        """This is the prediction key name used in dictionary based MONAI post-transforms."""
        return self._pred_dataset_key

    @pred_dataset_key.setter
    def pred_dataset_key(self, val: str):
        if not val or len(val) < 1:
            raise ValueError("Value cannot be None or blank.")
        self._pred_dataset_key = val

    @property
    def overlap(self):
        """This is the overlap used during sliding window inference"""
        return self._overlap

    @overlap.setter
    def overlap(self, val: float):
        if val < 0 or val > 1:
            raise ValueError("Overlap must be between 0 and 1.")
        self._overlap = val

    def _convert_dicom_metadata_datatype(self, metadata: Dict):
        """Converts metadata in pydicom types to the corresponding native types.

        It is knwon that some values of the metadata are of the pydicom types, for images converted
        from DICOM series. Need to use this function to convert the types with best effort and for
        the few knowns metadata attributes, until the following issue is addressed:
            https://github.com/Project-MONAI/monai-deploy-app-sdk/issues/185

        Args:
            metadata (Dict): The metadata for an Image object
        """

        if not metadata:
            return metadata

        # Try to convert data type for the well knowned attributes. Add more as needed.
        if metadata.get("SeriesInstanceUID", None):
            try:
                metadata["SeriesInstanceUID"] = str(metadata["SeriesInstanceUID"])
            except Exception:
                pass
        if metadata.get("row_pixel_spacing", None):
            try:
                metadata["row_pixel_spacing"] = float(metadata["row_pixel_spacing"])
            except Exception:
                pass
        if metadata.get("col_pixel_spacing", None):
            try:
                metadata["col_pixel_spacing"] = float(metadata["col_pixel_spacing"])
            except Exception:
                pass

        print("Converted Image object metadata:")
        for k, v in metadata.items():
            print(f"{k}: {v}, type {type(v)}")

        return metadata

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        """Infers with the input image and save the predicted image to output

        Args:
            op_input (InputContext): An input context for the operator.
            op_output (OutputContext): An output context for the operator.
            context (ExecutionContext): An execution context for the operator.
        """
        with self._lock:
            if self._executing:
                raise RuntimeError("Operator is already executing.")
            else:
                self._executing = True
        try:
            input_image = op_input.get("image")
            if not input_image:
                raise ValueError("Input is None.")

            # Need to try to convert the data type of a few metadata attributes.
            input_img_metadata = self._convert_dicom_metadata_datatype(input_image.metadata())
            # Need to give a name to the image as in-mem Image obj has no name.
            img_name = str(input_img_metadata.get("SeriesInstanceUID", "Img_in_context"))

            pre_transforms: Compose = self._pre_transform
            post_transforms: Compose = self._post_transforms
            self._reader = InMemImageReader(input_image)

            pre_transforms = self._pre_transform if self._pre_transform else self.pre_process(self._reader)
            post_transforms = self._post_transforms if self._post_transforms else self.post_process(pre_transforms)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = None
            if context.models:
                # `context.models.get(model_name)` returns a model instance if exists.
                # If model_name is not specified and only one model exists, it returns that model.
                model = context.models.get()
            else:
                print(f"Loading TorchScript model from: {MonaiSegInferenceOperator.MODEL_LOCAL_PATH}")
                model = torch.jit.load(MonaiSegInferenceOperator.MODEL_LOCAL_PATH, map_location=device)

            dataset = Dataset(data=[{self._input_dataset_key: img_name}], transform=pre_transforms)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

            with torch.no_grad():
                for d in dataloader:
                    images = d[self._input_dataset_key].to(device)
                    sw_batch_size = 4
                    d[self._pred_dataset_key] = sliding_window_inference(
                        inputs=images,
                        roi_size=self._roi_size,
                        sw_batch_size=sw_batch_size,
                        overlap=self.overlap,
                        predictor=model,
                    )
                    d = [post_transforms(i) for i in decollate_batch(d)]
                    out_ndarray = d[0][self._pred_dataset_key].cpu().numpy()
                    # Need to squeeze out the channel dim fist
                    out_ndarray = np.squeeze(out_ndarray, 0)
                    # NOTE: The domain Image object simply contains a Arraylike obj as image as of now.
                    #       When the original DICOM series is converted by the Series to Volume operator,
                    #       using pydicom pixel_array, the 2D ndarray of each slice is transposed, and the
                    #       depth/axial direction dim made the last. So once post-transforms have completed,
                    #       the resultant ndarray for each slice needs to be transposed back, and the depth
                    #       dim moved back to first, otherwise the seg ndarray would be flipped compared to
                    #       the DICOM pixel array, causing the DICOM Seg Writer generate flipped seg images.
                    out_ndarray = np.swapaxes(out_ndarray, 2, 0).astype(np.uint8)
                    print(f"Output Seg image numpy array shaped: {out_ndarray.shape}")
                    print(f"Output Seg image pixel max value: {np.amax(out_ndarray)}")
                    out_image = Image(out_ndarray, input_img_metadata)
                    op_output.set(out_image, "seg_image")
        finally:
            # Reset state on completing this method execution.
            with self._lock:
                self._executing = False

    def pre_process(self, img_reader) -> Union[Any, Image, Compose]:
        """Transforms input before being used for predicting on a model.

        This method must be overridden by a derived class.

        Raises:
            NotImplementedError: When the subclass does not override this method.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def post_process(self, pre_transforms: Compose, out_dir: str = "./infer_out") -> Union[Any, Image, Compose]:
        """Transforms the prediction results from the model(s).

        This method must be overridden by a derived class.

        Raises:
            NotImplementedError: When the subclass does not override this method.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def predict(self, data: Any, *args, **kwargs) -> Union[Image, Any]:
        """Prdicts results using the models(s) with input tensors.

        This method must be overridden by a derived class.

        Raises:
            NotImplementedError: When the subclass does not override this method.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class InMemImageReader(ImageReader):
    """Converts the App SDK Image object from memory.

    This is derived from MONAI ImageReader. Instead of reading image from file system, this
    class simply converts a in-memory SDK Image object to the expected formats from ImageReader.
    """

    def __init__(self, input_image: Image, channel_dim: Optional[int] = None, **kwargs):
        super().__init__()
        self.input_image = input_image
        self.kwargs = kwargs
        self.channel_dim = channel_dim

    def verify_suffix(self, filename: Union[Sequence[str], str]) -> bool:
        return True

    def read(self, data: Union[Sequence[str], str], **kwargs) -> Union[Sequence[Any], Any]:
        # Really does not have anything to do.
        return self.input_image.asnumpy()

    def get_data(self, input_image):
        """Extracts data array and meta data from loaded image and return them.

        This function returns two objects, first is numpy array of image data, second is dict of meta data.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        A single image is loaded with a single set of metadata as of now."""

        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        for i in ensure_tuple(self.input_image):
            if not isinstance(i, Image):
                raise TypeError("Only object of Image type is supported.")
            data = i.asnumpy()
            img_array.append(data)
            header = self._get_meta_dict(i)
            _copy_compatible_dict(header, compatible_meta)

        # Stacking image is not really needed, as there is one image only.
        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_meta_dict(self, img: Image) -> Dict:
        """
        Gets the metadata of the image and converts to dict type.

        Args:
            img: A SDK Image object.
        """
        img_meta_dict: Dict = img.metadata()
        meta_dict = {key: img_meta_dict[key] for key in img_meta_dict.keys()}

        # Will have to derive some key metadata as the SDK Image lacks the necessary interfaces.
        # So, for now have to get to the Image generator, namely DICOMSeriesToVolumeOperator, and
        # rely on its published metadata.

        # Recall that the column and row data for pydicom pixel_array had been switched, and the depth
        # is the last dim in DICOMSeriesToVolumeOperator
        meta_dict["spacing"] = np.asarray(
            [
                img_meta_dict["col_pixel_spacing"],
                img_meta_dict["row_pixel_spacing"],
                img_meta_dict["depth_pixel_spacing"],
            ]
        )
        meta_dict["original_affine"] = np.asarray(img_meta_dict["nifti_affine_transform"])
        meta_dict["affine"] = meta_dict["original_affine"]
        meta_dict["spatial_shape"] = np.asarray(img.asnumpy().shape)
        # Well, no channel as the image data shape is forced to the the same as spatial shape
        meta_dict["original_channel_dim"] = "no_channel"

        return meta_dict


# Reuse MONAI code for the derived ImageReader
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
