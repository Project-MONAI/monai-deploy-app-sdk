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

import json
import os
import tempfile

# from types import NoneType
import zipfile
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from monai.deploy.utils.importutil import optional_import
from monai.transforms.io.dictionary import LoadImageD

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
ensure_tuple, _ = optional_import("monai.utils", name="ensure_tuple")
Compose_, _ = optional_import("monai.transforms", name="Compose")
MapTransform_, _ = optional_import("monai.transforms", name="MapTransform")
LoadImaged_, _ = optional_import("monai.transforms", name="LoadImaged")
ConfigParser_, _ = optional_import("monai.bundle", name="ConfigParser")
SimpleInferer, _ = optional_import("monai.inferers", name="SimpleInferer")
# Dynamic class is not handled so make it Any for now: https://github.com/python/mypy/issues/2477
Compose: Any = Compose_
MapTransform: Any = MapTransform_
LoadImaged: Any = LoadImaged_
ConfigParser: Any = ConfigParser_

simple_inference, _ = optional_import("monai.inferers", name="simple_inference")
sliding_window_inference, _ = optional_import("monai.inferers", name="sliding_window_inference")

import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, OutputContext

from .inference_operator import InferenceOperator

__all__ = ["MonaiBundleInferenceOperator", "InMemImageReader"]

# TODO: For now, assuming single input and single output, but need to change
@md.input("image", Image, IOType.IN_MEMORY)
@md.output("seg_image", Image, IOType.IN_MEMORY)
#@md.env(pip_packages=["monai>=0.9.0", "torch>=1.10.02", "numpy>=1.21"])
@md.env(pip_packages=["monai-weekly>=0.9.dev2221", "torch>=1.10.02", "numpy>=1.21"])
class MonaiBundleInferenceOperator(InferenceOperator):
    """This segmentation operator uses MONAI transforms and Sliding Window Inference.

    This operator preforms pre-transforms on a input image, inference
    using a given model, and post-transforms. The segmentation image is saved
    as a named Image object in memory.

    If specified in the post transforms, results may also be saved to disk.
    """

    DISALLOWED_TRANSFORMS = ["LoadImage", "SaveImage"]

    def __init__(
        self,
        model_name: Optional[str] = "",
        bundle_path: Optional[str] = None,
        preproc_name: Optional[str] = "preprocessing",
        postproc_name: Optional[str] = "postprocessing",
        inferer_name: Optional[str] = "inferer",
        pre_transforms: Optional[Compose] = None,
        post_transforms: Optional[Compose] = None,
        roi_size: Union[Sequence[int], int] = (
            96,
            96,
            96,
        ),
        overlap: float = 0.5,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self._executing = False
        self._lock = Lock()
        self._model_name = model_name if model_name else ""
        self._bundle_path = Path(bundle_path).expanduser().resolve() if bundle_path and len(bundle_path) > 0 else None
        self._preproc_name = preproc_name
        self._postproc_name = postproc_name
        self._inferer_name = inferer_name
        self._parser = None  # Delay init till execution context is set.
        self._pre_transform = pre_transforms
        self._post_transforms = post_transforms
        self._inferer = None

        # TODO MQ to clean up
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"
        self._input_image = None  # Image will come in when compute is called.
        self._reader: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, name: str):
        if not name or len(name) == 0:
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
            raise ValueError(f"Value must be a valid ConfigParser object.")

    ##

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

    def _get_bundle_config(self, bundle_path: Path) -> ConfigParser:
        """Get the MONAI configuration parser from the specified MONAI Bundle file path.

        Args:
            bundle_path (Path): Path of the MONAI Bundle

        Returns:
            ConfigParser: MONAI Bundle config parser
        """
        # The final path component, without its suffix, is expected to the model name
        name = bundle_path.stem
        parser = ConfigParser()

        print(f"bundle path: {bundle_path}")
        with tempfile.TemporaryDirectory() as td:
            archive = zipfile.ZipFile(str(bundle_path), "r")
            archive.extract(name + "/extra/metadata.json", td)
            archive.extract(name + "/extra/inference.json", td)

            parser.read_meta(f=f"{td}/{name}/extra/metadata.json")
            parser.read_config(f=f"{td}/{name}/extra/inference.json")

            parser.parse()

        return parser

    def _filter_compose(self, compose: Compose):
        """
        Remove transforms from the given Compose object which shouldn't be used in an Operator.
        """

        if not compose:
            return Compose([])  # Could just bounce the None input back.

        filtered = []
        for t in compose.transforms:
            tname = type(t).__name__
            if not any(dis in tname for dis in self.DISALLOWED_TRANSFORMS):
                filtered.append(t)

        return Compose(filtered)

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

        # If present, get the compliant model from context, else, from bundle path if given
        model = None
        if context.models:
            # `context.models.get(model_name)` returns a model instance if exists.
            # If model_name is not specified and only one model exists, it returns that model.
            model = context.models.get(self.model_name)
            if model:
                self.bundle_path = model.path
        if not model and self.bundle_path:
            print(f"Loading TorchScript model from: {self.bundle_path}")
            model = torch.jit.load(self.bundle_path, map_location=device)

        if not model:
            raise IOError("Cannot find model file.")

        # Load the ConfigParser
        self.parser = self._get_bundle_config(self.bundle_path)

        # Get the inferer
        if self._parser.get(self._inferer_name) is not None:
            self._inferer = self._parser.get_parsed_content(self._inferer_name)
        else:
            self._inferer = SimpleInferer()

        try:
            input_image = op_input.get()
            if not input_image:
                raise ValueError("Input is None.")

            input_img_metadata = input_image.metadata()
            # Need to give a name to the image as in-mem Image obj has no name.
            img_name = str(input_img_metadata.get("SeriesInstanceUID", "Img_in_context"))

            self._reader = InMemImageReader(input_image)  # For convering Image to MONAI expected format
            pre_transforms: Compose = self._pre_transform if self._pre_transform else self.pre_process(self._reader)
            post_transforms: Compose = (
                self._post_transforms if self._post_transforms else self.post_process(pre_transforms)
            )

            # TODO: From bundle config
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            dataset = Dataset(data=[{self._input_dataset_key: img_name}], transform=pre_transforms)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

            with torch.no_grad():
                for d in dataloader:
                    images = d[self._input_dataset_key].to(device)
                    d[self._pred_dataset_key] = self._inferer(inputs=images, network=model)
                    d = [post_transforms(i) for i in decollate_batch(d)]
                    out_ndarray = d[0][self._pred_dataset_key].cpu().numpy()
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
                    print(f"Output Seg image numpy array shaped: {out_ndarray.shape}")
                    print(f"Output Seg image pixel max value: {np.amax(out_ndarray)}")
                    out_image = Image(out_ndarray, input_img_metadata)
                    op_output.set(out_image)
        finally:
            # Reset state on completing this method execution.
            with self._lock:
                self._executing = False

    def pre_process(self, img_reader) -> Union[Any, Image, Compose]:
        """Transforms input before being used for predicting on a model."""

        if not self.parser:
            raise RuntimeError("ConfigParser object is None.")

        if self.parser.get(self._preproc_name) is not None:
            preproc = self.parser.get_parsed_content(self._preproc_name)
            self._pre_transform = self._filter_compose(preproc)
        else:
            self._pre_transform = Compose([])  # Could there be a scenario with no pre_processing?

        # Need to add the loadimage transform, single dataset key for now.
        # TODO: MQ to find a better solution, or use Compose callable directly instead of dataloader
        load_image_transform = LoadImaged(keys=self.input_dataset_key, reader=img_reader)
        self._pre_transform.transforms = (load_image_transform,) + self._pre_transform.transforms

        return self._pre_transform

    def post_process(self, pre_transforms: Compose, out_dir: str = "./infer_out") -> Union[Any, Image, Compose]:
        """Transforms the prediction results from the model(s)."""

        if self.parser.get(self._postproc_name) is not None:
            postproc = self.parser.get_parsed_content(self._postproc_name)
            self._post_transforms = self._filter_compose(postproc)
        else:
            self._post_transforms = Compose([])
        return self._post_transforms

    def predict(self, data: Any, *args, **kwargs) -> Union[Image, Any]:
        """Predicts results using the models(s) with input tensors.

        This method must be overridden by a derived class.

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
        the numpy array, which has the index order in WHD with the channel being the last dim_get_compose if present.

        Args:
            input_image (Image): an App SDK Image object.
        """

        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        for i in ensure_tuple(input_image):
            if not isinstance(i, Image):
                raise TypeError("Only object of Image type is supported.")

            # The Image asnumpy() returns NumPy array similar to ITK array_view_from_image
            # The array then needs to be transposed, as does in MONAI ITKReader, to align_get_compose
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
        meta_dict["original_affine"] = np.asarray(img_meta_dict.get("nifti_affine_transform", None))
        meta_dict["affine"] = meta_dict["original_affine"]
        # The spatial shape, again, referring to ITKReader, it is the WHD
        meta_dict["spatial_shape"] = np.asarray(img.asnumpy().T.shape)
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
