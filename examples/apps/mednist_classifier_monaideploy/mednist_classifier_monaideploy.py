# Copyright 2021-2023 MONAI Consortium
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
import os
from pathlib import Path
from typing import Optional

import torch

from monai.deploy.conditions import CountCondition
from monai.deploy.core import AppContext, Application, ConditionType, Fragment, Image, Operator, OperatorSpec
from monai.deploy.operators.dicom_text_sr_writer_operator import DICOMTextSRWriterOperator, EquipmentInfo, ModelInfo
from monai.transforms import Compose, EnsureChannelFirst, EnsureType, ScaleIntensity

MEDNIST_CLASSES = ["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"]


# Decorator support is not available in this version of the SDK, to be re-introduced later
# @md.env(pip_packages=["pillow"])
class LoadPILOperator(Operator):
    """Load image from the given input (DataPath) and set numpy array to the output (Image)."""

    DEFAULT_INPUT_FOLDER = Path.cwd() / "input"
    DEFAULT_OUTPUT_NAME = "image"

    # For now, need to have the input folder as an instance attribute, set on init.
    # If dynamically changing the input folder, per compute, then use a (optional) input port to convey the
    # value of the input folder, which is then emitted by a upstream operator.
    def __init__(
        self,
        fragment: Fragment,
        *args,
        input_folder: Path = DEFAULT_INPUT_FOLDER,
        output_name: str = DEFAULT_OUTPUT_NAME,
        **kwargs,
    ):
        """Creates an loader object with the input folder and the output port name overrides as needed.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            input_folder (Path): Folder from which to load input file(s).
                                 Defaults to `input` in the current working directory.
            output_name (str): Name of the output port, which is an image object. Defaults to `image`.
        """

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.input_path = input_folder
        self.index = 0
        self.output_name_image = (
            output_name.strip() if output_name and len(output_name.strip()) > 0 else LoadPILOperator.DEFAULT_OUTPUT_NAME
        )

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Set up the named input and output port(s)"""
        spec.output(self.output_name_image)

    def compute(self, op_input, op_output, context):
        import numpy as np
        from PIL import Image as PILImage

        # Input path is stored in the object attribute, but could change to use a named port if need be.
        input_path = self.input_path
        if input_path.is_dir():
            input_path = next(self.input_path.glob("*.*"))  # take the first file

        image = PILImage.open(input_path)
        image = image.convert("L")  # convert to greyscale image
        image_arr = np.asarray(image)

        output_image = Image(image_arr)  # create Image domain object with a numpy array
        op_output.emit(output_image, self.output_name_image)  # cannot omit the name even if single output.


# @md.env(pip_packages=["monai"])
class MedNISTClassifierOperator(Operator):
    """Classifies the given image and returns the class name.

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
        app_context: AppContext,
        model_name: Optional[str] = "",
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
        self.app_context = app_context
        self.model = self._get_model(self.app_context, self.model_path, self._model_name)

        # This needs to be at the end of the constructor.
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
                MedNISTClassifierOperator.MODEL_LOCAL_PATH,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

        return model

    def setup(self, spec: OperatorSpec):
        """Set up the operator named input and named output, both are in-memory objects."""

        spec.input(self.input_name_image)
        spec.input(self.input_name_output_folder).condition(ConditionType.NONE)  # Optional for overriding.
        spec.output(self.output_name_result).condition(ConditionType.NONE)  # Not forcing a downstream receiver.

    @property
    def transform(self):
        return Compose([EnsureChannelFirst(channel_dim="no_channel"), ScaleIntensity(), EnsureType()])

    def compute(self, op_input, op_output, context):
        import json

        import torch

        img = op_input.receive(self.input_name_image).asnumpy()  # (64, 64), uint8. Input validation can be added.
        image_tensor = self.transform(img)  # (1, 64, 64), torch.float64
        image_tensor = image_tensor[None].float()  # (1, 1, 64, 64), torch.float32

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        _, output_classes = outputs.max(dim=1)

        result = MEDNIST_CLASSES[output_classes[0]]  # get the class name
        print(result)
        op_output.emit(result, self.output_name_result)

        # Get output folder, with value in optional input port overriding the obj attribute
        output_folder_on_compute = op_input.receive(self.input_name_output_folder) or self.output_folder
        Path.mkdir(output_folder_on_compute, parents=True, exist_ok=True)  # Let exception bubble up if raised.
        output_path = output_folder_on_compute / "output.json"
        with open(output_path, "w") as fp:
            json.dump(result, fp)


# Decorator support is not available in this version of the SDK, to be re-introduced later
# @md.resource(cpu=1, gpu=1, memory="1Gi")
# @md.env(pip_packages=["pydicom >= 2.3.0", "highdicom>=0.18.2"])  # because of the use of DICOM writer operator
class App(Application):
    """Application class for the MedNIST classifier."""

    def compose(self):
        # Use Commandline options over environment variables to init context.
        app_context: AppContext = Application.init_app_context(self.argv)
        app_input_path = Path(app_context.input_path)
        app_output_path = Path(app_context.output_path)
        model_path = Path(app_context.model_path)
        load_pil_op = LoadPILOperator(self, CountCondition(self, 1), input_folder=app_input_path, name="pil_loader_op")
        classifier_op = MedNISTClassifierOperator(
            self, app_context=app_context, output_folder=app_output_path, model_path=model_path, name="classifier_op"
        )

        my_model_info = ModelInfo("MONAI WG Trainer", "MEDNIST Classifier", "0.1", "1234")
        my_equipment = EquipmentInfo(manufacturer="MOANI Deploy App SDK", manufacturer_model="DICOM SR Writer")
        my_special_tags = {"SeriesDescription": "Not for clinical use. The result is for research use only."}
        dicom_sr_operator = DICOMTextSRWriterOperator(
            self,
            copy_tags=False,
            model_info=my_model_info,
            equipment_info=my_equipment,
            custom_tags=my_special_tags,
            output_folder=app_output_path,
        )

        self.add_flow(load_pil_op, classifier_op, {("image", "image")})
        self.add_flow(classifier_op, dicom_sr_operator, {("result_text", "text")})


if __name__ == "__main__":
    App().run()
