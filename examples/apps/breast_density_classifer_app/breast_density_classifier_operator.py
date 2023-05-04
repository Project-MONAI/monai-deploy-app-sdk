import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch

from monai.data import DataLoader, Dataset
from monai.deploy.core import AppContext, ConditionType, Fragment, Image, Operator, OperatorSpec
from monai.deploy.operators.monai_seg_inference_operator import InMemImageReader
from monai.transforms import (
    Activations,
    Compose,
    EnsureChannelFirst,
    EnsureType,
    LoadImage,
    NormalizeIntensity,
    RepeatChannel,
    Resize,
    SqueezeDim,
)


# @env(pip_packages=["monai~=1.1.0"])
class ClassifierOperator(Operator):
    """Performs breast density classification using a DL model with an image converted from a DICOM MG series.

    Named inputs:
        image: Image object for which to generate the classification.
        output_folder: Optional, the path to save the results JSON file, overridingthe the one set on __init__

    Named output:
        result_text: The classification results in text.
    """

    DEFAULT_OUTPUT_FOLDER = Path.cwd() / "classification_results"
    # For testing the app directly, the model should be at the following path.
    MODEL_LOCAL_PATH = Path(os.environ.get("HOLOSCAN_MODEL_PATH", Path.cwd() / "model/model.ts"))

    def __init__(
        self,
        frament: Fragment,
        *args,
        model_name: Optional[str] = "",
        app_context: AppContext,
        model_path: Path = MODEL_LOCAL_PATH,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
        **kwargs,
    ):
        """Creates an instance with the reference back to the containing application/fragment.

        fragment (Fragment): An instance of the Application class which is derived from Fragment.
        model_name (str, optional): Name of the model. Default to "" for single model app.
        model_path (Path): Path to the model file. Defaults to model/models.ts of current working dir.
        output_folder (Path, optional): output folder for saving the classification results JSON file.
        """

        # the names used for the model inference input and output
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"

        # The names used for the operator input and output
        self.input_name_image = "image"
        self.output_name_result = "result_text"

        # The name of the optional input port for passing data to override the output folder path.
        self.input_name_output_folder = "output_folder"

        # The output folder set on the object can be overriden at each compute by data in the optional named input
        self.output_folder = output_folder

        # Need the name when there are multiple models loaded
        self._model_name = model_name.strip() if isinstance(model_name, str) else ""
        # Need the path to load the models when they are not loaded in the execution context
        self.model_path = model_path

        # Use AppContect object for getting the loaded models
        self.app_context = app_context

        self.model = self._get_model(self.app_context, self.model_path, self._model_name)

        super().__init__(frament, *args, **kwargs)

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
            model = torch.jit.load(
                ClassifierOperator.MODEL_LOCAL_PATH,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

        return model

    def setup(self, spec: OperatorSpec):
        """Set up the operator named input and named output, both are in-memory objects."""

        spec.input(self.input_name_image)
        spec.input(self.input_name_output_folder).condition(ConditionType.NONE)  # Optional for overriding.
        spec.output(self.output_name_result).condition(ConditionType.NONE)  # Not forcing a downstream receiver.

    def _convert_dicom_metadata_datatype(self, metadata: Dict):
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

    def compute(self, op_input, op_output, context):
        input_image = op_input.receive(self.input_name_image)
        if not input_image:
            raise ValueError("Input image is not found.")
        if not isinstance(input_image, Image):
            raise ValueError(f"Input is not the required type: {type(Image)!r}")

        _reader = InMemImageReader(input_image)
        input_img_metadata = self._convert_dicom_metadata_datatype(input_image.metadata())
        img_name = str(input_img_metadata.get("SeriesInstanceUID", "Img_in_context"))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Need to get the model from context, when it is re-implemented, and for now, load it directly here.
        # model = context.models.get()
        model = torch.jit.load(self.model_path, map_location=device)

        pre_transforms = self.pre_process(_reader)
        post_transforms = self.post_process()

        dataset = Dataset(data=[{self._input_dataset_key: img_name}], transform=pre_transforms)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)

        with torch.no_grad():
            for d in dataloader:
                image = d[0].to(device)
                outputs = model(image)
                out = post_transforms(outputs).data.cpu().numpy()[0]
                print(out)

        result_dict = (
            "A " + ":" + str(out[0]) + " B " + ":" + str(out[1]) + " C " + ":" + str(out[2]) + " D " + ":" + str(out[3])
        )

        op_output.emit(result_dict, "result_text")

        # Get output folder, with value in optional input port overriding the obj attribute
        output_folder_on_compute = op_input.receive(self.input_name_output_folder) or self.output_folder
        Path.mkdir(output_folder_on_compute, parents=True, exist_ok=True)  # Let exception bubble up if raised.
        output_path = output_folder_on_compute / "output.json"
        with open(output_path, "w") as fp:
            json.dump(result_dict, fp)

    def pre_process(self, image_reader) -> Compose:
        return Compose(
            [
                LoadImage(reader=image_reader, image_only=True),
                EnsureChannelFirst(),
                SqueezeDim(dim=3),
                NormalizeIntensity(),
                Resize(spatial_size=(299, 299)),
                RepeatChannel(repeats=3),
                EnsureChannelFirst(),
            ]
        )

    def post_process(self) -> Compose:
        return Compose([EnsureType(), Activations(sigmoid=True)])
