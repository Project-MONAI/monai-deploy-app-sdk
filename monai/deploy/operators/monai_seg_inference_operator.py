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

import inspect
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from monai.deploy.utils.importutil import optional_import
from monai.utils import StrEnum  # Will use the built-in StrEnum when SDK requires Python 3.11.

MONAI_UTILS = "monai.utils"
torch, _ = optional_import("torch", "1.5")
np_str_obj_array_pattern, _ = optional_import("torch.utils.data._utils.collate", name="np_str_obj_array_pattern")
Dataset, _ = optional_import("monai.data", name="Dataset")
DataLoader, _ = optional_import("monai.data", name="DataLoader")
ImageReader_, image_reader_ok_ = optional_import("monai.data", name="ImageReader")
# Dynamic class is not handled so make it Any for now: https://github.com/python/mypy/issues/2477
ImageReader: Any = ImageReader_
if not image_reader_ok_:
    ImageReader = object  # for 'class InMemImageReader(ImageReader):' to work
decollate_batch, _ = optional_import("monai.data", name="decollate_batch")
sliding_window_inference, _ = optional_import("monai.inferers", name="sliding_window_inference")
simple_inference, _ = optional_import("monai.inferers", name="SimpleInferer")
ensure_tuple, _ = optional_import(MONAI_UTILS, name="ensure_tuple")
MetaKeys, _ = optional_import(MONAI_UTILS, name="MetaKeys")
SpaceKeys, _ = optional_import(MONAI_UTILS, name="SpaceKeys")
Compose_, _ = optional_import("monai.transforms", name="Compose")
# Dynamic class is not handled so make it Any for now: https://github.com/python/mypy/issues/2477
Compose: Any = Compose_

from monai.deploy.core import AppContext, Condition, ConditionType, Fragment, Image, OperatorSpec, Resource

from .inference_operator import InferenceOperator

__all__ = ["MonaiSegInferenceOperator", "InfererType", "InMemImageReader"]


class InfererType(StrEnum):
    """Represents the supported types of the inferer, e.g. Simple and Sliding Window."""

    SIMPLE = "simple"
    SLIDING_WINDOW = "sliding_window"


# @md.env(pip_packages=["monai>=1.0.0", "torch>=1.10.2", "numpy>=1.21"])
class MonaiSegInferenceOperator(InferenceOperator):
    """This segmentation operator uses MONAI transforms and performs Simple or Sliding Window Inference.

    This operator performs pre-transforms on a input image, inference
    using a given model, and post-transforms. The segmentation image is saved
    as a named Image object in memory.

    If specified in the post transforms, results may also be saved to disk.

    This operator uses the MONAI inference utils functions for sliding window and simple inference,
    and thus input parameters need to be as expected by these functions.

    Any additional sliding window arguments not explicitly defined in this operator can be passed via
    **kwargs for forwarding to 'sliding_window_inference'.

    Named Input:
        image: Image object of the input image.

    Named Output:
       seg_image: Image object of the segmentation image. Not requiring a ready receiver.
    """

    # For testing the app directly, the model should be at the following path.
    MODEL_LOCAL_PATH = Path(os.environ.get("HOLOSCAN_MODEL_PATH", Path.cwd() / "model/model.ts"))

    @staticmethod
    def filter_sw_kwargs(**kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Filters the keyword arguments into a tuple of two dictionaries:

        1. A dictionary of named parameters to pass to the sliding_window_inference function that:
            - Are not explicitly defined in the __init__ of this class
            - Are not explicitly used when calling sliding_window_inference

        2. A dictionary of named parameters to pass to the base class __init__ of this class that:
            - Are not used when calling sliding_window_inference
            - Can be successfully converted from Python --> Holoscan's C++ layer

        Args:
            **kwargs: extra arguments passed into __init__ beyond the explicitly defined args.

        Returns:
            filtered_swi_params: A filtered dictionary of arguments to be passed to sliding_window_inference.
            filtered_base_init_params: A filtered dictionary of arguments to be passed to the base class __init__.
        """

        logger = logging.getLogger(f"{__name__}.{MonaiSegInferenceOperator.__name__}")

        init_params = inspect.signature(MonaiSegInferenceOperator).parameters
        swi_params = inspect.signature(sliding_window_inference).parameters

        # inputs + predictor explicitly used when calling sliding_window_inference
        explicit_used = {"inputs", "predictor"}

        # Holoscan convertible types (not exhaustive)
        # This will be revisited when there is a better way to handle this.
        allowed_types = (str, int, float, bool, bytes, list, tuple, torch.Tensor, Condition, Resource)

        filtered_swi_params = {}
        filtered_base_init_params = {}

        for name, val in kwargs.items():
            # Drop explicitly defined kwargs
            if name in init_params or name in explicit_used:
                logger.warning(f"{name!r} is already explicitly defined or used; dropping kwarg.")
            # SWI params
            elif name in swi_params:
                filtered_swi_params[name] = val
                logger.debug(f"{name!r} used in sliding_window_inference; keeping kwarg for inference call.")
            # Drop kwargs that can't be converted by Holoscan
            elif not isinstance(val, allowed_types):
                logger.warning(
                    f"{name!r} type of {type(val).__name__!r} is a non-convertible kwarg for Holoscan; dropping kwarg."
                )
            # Base __init__ params
            else:
                filtered_base_init_params[name] = val
                logger.debug(
                    f"{name!r} type of {type(val).__name__!r} can be converted by Holoscan; keeping kwarg for base init."
                )
        return filtered_swi_params, filtered_base_init_params

    def __init__(
        self,
        fragment: Fragment,
        *args,
        roi_size: Optional[Union[Sequence[int], int]],
        pre_transforms: Compose,
        post_transforms: Compose,
        app_context: AppContext,
        model_name: Optional[str] = "",
        overlap: Union[Sequence[float], float] = 0.25,
        sw_batch_size: int = 4,
        inferer: Union[InfererType, str] = InfererType.SLIDING_WINDOW,
        model_path: Path = MODEL_LOCAL_PATH,
        **kwargs,
    ):
        """Creates a instance of this class.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            roi_size (Sequence[int], int, optional): The window size to execute "SLIDING_WINDOW" evaluation.
                             Applicable for "SLIDING_WINDOW" only.
            pre_transforms (Compose): MONAI Compose object used for pre-transforms.
            post_transforms (Compose): MONAI Compose object used for post-transforms.
            app_context (AppContext): Object holding the I/O and model paths, and potentially loaded models.
            model_name (str, optional): Name of the model. Default to "" for single model app.
            overlap (Sequence[float], float): The amount of overlap between scans along each spatial dimension. Defaults to 0.25.
                             Applicable for "SLIDING_WINDOW" only.
            sw_batch_size (int): The batch size to run window slices. Defaults to 4.
                             Applicable for "SLIDING_WINDOW" only.
            inferer (InfererType, str): The type of inferer to use, "SIMPLE" or "SLIDING_WINDOW". Defaults to "SLIDING_WINDOW".
            model_path (Path): Path to the model file. Defaults to model/models.ts of current working dir.
            **kwargs: any other sliding window parameters to forward (e.g. `mode`, `cval`, etc.).
        """

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._executing = False
        self._lock = Lock()
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"
        self._input_image = None  # Image will come in when compute is called.
        self._reader: Any = None
        self.roi_size = ensure_tuple(roi_size)
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms
        self._model_name = model_name.strip() if isinstance(model_name, str) else ""
        self.overlap = overlap
        self.sw_batch_size = sw_batch_size
        self.inferer = inferer
        self._filtered_swi_params, self._filtered_base_init_params = self.filter_sw_kwargs(
            **kwargs
        )  # Filter keyword args

        # Add this so that the local model path can be set from the calling app
        self.model_path = model_path
        self._input_name_image = "image"
        self._output_name_seg = "seg_image"

        # The execution context passed in on compute does not have the required model info, so need to
        # get and keep the model via the AppContext obj on construction.
        self.app_context = app_context

        self._model = self._get_model(self.app_context, self.model_path, self._model_name)

        # Pass filtered base init params
        super().__init__(fragment, *args, **self._filtered_base_init_params)

    def _get_model(self, app_context: AppContext, model_path: Path, model_name: str):
        """Load the model with the given name from context or model path

        Args:
            app_context (AppContext): The application context object holding the model(s)
            model_path (Path): The path to the model file, as a backup to load model directly
            model_name (str): The name of the model, when multiples are loaded in the context
        """

        if app_context.models:
            # `app_context.models.get(model_name)` returns a model instance if exists.
            # If model_name is not specified and only one model exists, it returns that model.
            model = app_context.models.get(model_name)
        else:
            self._logger.info(f"Loading TorchScript model from: {model_path!r}")
            model = torch.jit.load(
                self.model_path,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

        return model

    def setup(self, spec: OperatorSpec):
        spec.input(self._input_name_image)
        spec.output(self._output_name_seg).condition(ConditionType.NONE)  # Downstream receiver optional.

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
    def overlap(self, val: Union[Sequence[float], float]):
        if not isinstance(val, (Sequence, int, float)) or isinstance(val, str):
            raise TypeError(f"Overlap must be type Sequence[float] | float, got {type(val).__name__}.")
        elif isinstance(val, Sequence) and not all(isinstance(x, (int, float)) and 0 <= x < 1 for x in val):
            raise ValueError("Each overlap value must be >= 0 and < 1.")
        elif isinstance(val, (int, float)) and not (0 <= float(val) < 1):
            raise ValueError(f"Overlap must be >= 0 and < 1, got {val}.")
        self._overlap = val

    @property
    def sw_batch_size(self):
        """The batch size to run window slices"""
        return self._sw_batch_size

    @sw_batch_size.setter
    def sw_batch_size(self, val: int):
        if not isinstance(val, int) or val < 0:
            raise ValueError("sw_batch_size must be a positive integer.")
        self._sw_batch_size = val

    @property
    def inferer(self) -> Union[InfererType, str]:
        """The type of inferer to use"""
        return self._inferer

    @inferer.setter
    def inferer(self, val: Union[InfererType, str]):
        if isinstance(val, InfererType):
            self._inferer = val
            return

        if isinstance(val, str):
            s = val.strip().lower()
            valid = (InfererType.SIMPLE.value, InfererType.SLIDING_WINDOW.value)
            if s in valid:
                self._inferer = InfererType(s)
                return
            raise ValueError(f"inferer must be one of {valid}, got {val!r}.")

        raise TypeError(f"inferer must be InfererType or str, got {type(val).__name__}.")

    def _convert_dicom_metadata_datatype(self, metadata: Dict):
        """Converts metadata in pydicom types to the corresponding native types.

        It is known that some values of the metadata are of the pydicom types, for images converted
        from DICOM series. Need to use this function to convert the types with best effort and for
        the few knowns metadata attributes, until the following issue is addressed:
            https://github.com/Project-MONAI/monai-deploy-app-sdk/issues/185

        Args:
            metadata (Dict): The metadata for an Image object
        """

        if not metadata:
            return metadata

        # Try to convert data type for the well known attributes. Add more as needed.
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

        self._logger.info("Converted Image object metadata:")
        for k, v in metadata.items():
            self._logger.info(f"{k}: {v}, type {type(v)}")

        return metadata

    def compute(self, op_input, op_output, context):
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
            input_image = op_input.receive(self._input_name_image)
            if not input_image:
                raise ValueError("Input is None.")
            op_output.emit(self.compute_impl(input_image, context), self._output_name_seg)
        finally:
            # Reset state on completing this method execution.
            with self._lock:
                self._executing = False

    def compute_impl(self, input_image, context):
        if not input_image:
            raise ValueError("Input is None.")

        # Need to try to convert the data type of a few metadata attributes.
        input_img_metadata = self._convert_dicom_metadata_datatype(input_image.metadata())
        # Need to give a name to the image as in-mem Image obj has no name.
        img_name = str(input_img_metadata.get("SeriesInstanceUID", "Img_in_context"))

        pre_transforms: Compose = self.pre_transforms
        post_transforms: Compose = self.post_transforms
        self._reader = InMemImageReader(input_image)

        pre_transforms = self.pre_transforms if self.pre_transforms else self.pre_process(self._reader)
        post_transforms = self.post_transforms if self.post_transforms else self.post_process(pre_transforms)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = Dataset(data=[{self._input_dataset_key: img_name}], transform=pre_transforms)
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )  # Should the batch_size be dynamic?

        with torch.no_grad():
            for d in dataloader:
                images = d[self._input_dataset_key].to(device)
                self._logger.info(f"Input of {type(images)} shape: {images.shape}")
                if self.inferer == InfererType.SLIDING_WINDOW:
                    d[self._pred_dataset_key] = sliding_window_inference(
                        inputs=images,
                        roi_size=self.roi_size,
                        overlap=self.overlap,
                        sw_batch_size=self.sw_batch_size,
                        predictor=self._model,
                        **self._filtered_swi_params,  # Additional sliding window arguments
                    )
                elif self.inferer == InfererType.SIMPLE:
                    # Instantiates the SimpleInferer and directly uses its __call__ function
                    d[self._pred_dataset_key] = simple_inference()(inputs=images, network=self._model)
                else:
                    raise ValueError(
                        f"Unknown inferer: {self.inferer!r}. Available options are "
                        f"{InfererType.SLIDING_WINDOW!r} and {InfererType.SIMPLE!r}."
                    )

                d = [post_transforms(i) for i in decollate_batch(d)]
                self._logger.info(f"Post transform length/batch size of output: {len(d)}")
                self._logger.info(
                    f"Post transform pixel spacings for {self._pred_dataset_key}: {d[0][self._pred_dataset_key].pixdim}"
                )
                out_ndarray = d[0][self._pred_dataset_key].cpu().numpy()  # Single output to numpy on CPU
                self._logger.info(
                    f"Post transform {self._pred_dataset_key} of {type(out_ndarray)} shape: {out_ndarray.shape}"
                )
                # Need to squeeze out the channel dim fist
                out_ndarray = np.squeeze(out_ndarray, 0)
                # NOTE: The domain Image object simply contains a Arraylike obj as image as of now.
                #       When the original DICOM series is converted by the Series to Volume operator,
                #       using pydicom pixel_array, the 2D ndarray of each slice has index order HW, and
                #       when all slices are stacked with depth as first axis, DHW. In the pre-transforms,
                #       the image gets transposed to WHD and used as such in the inference pipeline.
                #       So once post-transforms have completed, and the channel is squeezed out,
                #       the resultant ndarray for the prediction image needs to be transposed back, so the
                #       array index order is back to DHW, the same order as the in-memory input Image obj.
                out_ndarray = out_ndarray.T.astype(np.uint8)
                self._logger.info(
                    f"Output Seg image numpy array of type {type(out_ndarray)} shape: {out_ndarray.shape}"
                )
                self._logger.info(f"Output Seg image pixel max value: {np.amax(out_ndarray)}")

                return Image(out_ndarray, input_img_metadata)

    def pre_process(self, data: Any, *args, **kwargs) -> Union[Any, Image, Tuple[Any, ...], Dict[Any, Any]]:
        """Transforms input before being used for predicting on a model.

        This method must be overridden by a derived class.
        Expected return is monai.transforms.Compose.

        Args:
            data(monai.data.ImageReader): Reader used in LoadImage to load `monai.deploy.core.Image` as the input.

        Returns:
            monai.transforms.Compose encapsulating pre transforms

        Raises:
            NotImplementedError: When the subclass does not override this method.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def post_process(self, data: Any, *args, **kwargs) -> Union[Any, Image, Tuple[Any, ...], Dict[Any, Any]]:
        """Transforms the prediction results from the model(s).

        This method must be overridden by a derived class.
        Expected return is monai.transforms.Compose.

        Args:
            data(monai.transforms.Compose): The pre-processing transforms in a Compose object.

        Returns:
            monai.transforms.Compose encapsulating post-processing transforms.

        Raises:
            NotImplementedError: When the subclass does not override this method.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def predict(self, data: Any, *args, **kwargs) -> Union[Image, Any, Tuple[Any, ...], Dict[Any, Any]]:
        """Predicts results using the models(s) with input tensors.

        This method is currently not used in this class, instead monai.inferers.sliding_window_inference is used.

        Raises:
            NotImplementedError: When the subclass does not override this method.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class InMemImageReader(ImageReader):
    """Converts the App SDK Image object from memory.

    This is derived from MONAI ImageReader. Instead of reading image from file system, this
    class simply converts a in-memory SDK Image object to the expected formats from ImageReader.

    The loaded data array will be in C order, for example, a 3D image NumPy array index order
    will be `WHDC`. The actual data array loaded is to be the same as that from the
    MONAI ITKReader, which can also load DICOM series. Furthermore, all Readers need to return the
    array data the same way as the NibabelReader, i.e. a numpy array of index order WHDC with channel
    being the last dim if present. More details are in the get_data() function.


    """

    def __init__(self, input_image: Image, channel_dim: Optional[int] = None, **kwargs):
        super().__init__()
        self.input_image = input_image
        self.kwargs = kwargs
        self.channel_dim = channel_dim

    def verify_suffix(self, filename: Union[Sequence[str], str]) -> bool:
        return True

    def read(self, data: Union[Sequence[str], str], **kwargs) -> Union[Sequence[Any], Any]:
        # Really does not have anything to do. Simply return the Image object
        return self.input_image

    def get_data(self, input_image):
        """Extracts data array and meta data from loaded image and return them.

        This function returns two objects, first is numpy array of image data, second is dict of meta data.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        A single image is loaded with a single set of metadata as of now.

        The App SDK Image asnumpy() function is expected to return a numpy array of index order `DHW`.
        This is because in the DICOM series to volume operator pydicom Dataset pixel_array is used to
        to get per instance pixel numpy array, with index order of `HW`. When all instances are stacked,
        along the first axis, the Image numpy array's index order is `DHW`. ITK array_view_from_image
        and SimpleITK GetArrayViewFromImage also returns a numpy array with the index order of `DHW`.
        The channel would be the last dim/index if present. In the ITKReader get_data(), this numpy array
        is then transposed, and the channel axis moved to be last dim post transpose; this is to be
        consistent with the numpy returned from NibabelReader get_data().

        The NibabelReader loads NIfTI image and uses the get_fdata() function of the loaded image to get
        the numpy array, which has the index order in WHD with the channel being the last dim if present.

        Args:
            input_image (Image): an App SDK Image object.
        """

        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        for i in ensure_tuple(input_image):
            if not isinstance(i, Image):
                raise TypeError("Only object of Image type is supported.")

            # The Image asnumpy() returns NumPy array similar to ITK array_view_from_image
            # The array then needs to be transposed, as does in MONAI ITKReader, to align
            # with the output from Nibabel reader loading NIfTI files.
            data = i.asnumpy().T
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

        # Referring to the MONAI ITKReader, the spacing is simply a NumPy array from the ITK image
        # GetSpacing, in WHD.
        meta_dict["spacing"] = np.asarray(
            [
                img_meta_dict["row_pixel_spacing"],
                img_meta_dict["col_pixel_spacing"],
                img_meta_dict["depth_pixel_spacing"],
            ]
        )

        # Use define metadata kyes directly
        meta_dict[MetaKeys.ORIGINAL_AFFINE] = np.asarray(img_meta_dict.get("nifti_affine_transform", None))
        meta_dict[MetaKeys.AFFINE] = meta_dict[MetaKeys.ORIGINAL_AFFINE].copy()
        # Disabled setting the SPACE key below as it has changed to be set in the actual loader implementation.
        # meta_dict[MetaKeys.SPACE] = SpaceKeys.LPS  # not using SpaceKeys.RAS or affine_lps_to_ras
        # The spatial shape, again, referring to ITKReader, it is the WHD
        meta_dict[MetaKeys.SPATIAL_SHAPE] = np.asarray(img.asnumpy().T.shape)
        # Well, no channel as the image data shape is forced to the the same as spatial shape
        meta_dict[MetaKeys.ORIGINAL_CHANNEL_DIM] = "no_channel"

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
        affine_key, shape_key = MetaKeys.AFFINE, MetaKeys.SPATIAL_SHAPE
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
    if meta_dict.get(MetaKeys.ORIGINAL_CHANNEL_DIM, None) not in ("no_channel", None):
        channel_dim = int(meta_dict[MetaKeys.ORIGINAL_CHANNEL_DIM])
        return np.concatenate(image_list, axis=channel_dim)
    # stack at a new first dim as the channel dim, if `'original_channel_dim'` is unspecified
    meta_dict[MetaKeys.ORIGINAL_CHANNEL_DIM] = 0
    return np.stack(image_list, axis=0)
