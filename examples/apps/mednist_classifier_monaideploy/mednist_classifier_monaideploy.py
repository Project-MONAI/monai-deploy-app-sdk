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

from typing import Text

import monai.deploy.core as md
from monai.deploy.core import (
    Application,
    DataPath,
    ExecutionContext,
    Image,
    InputContext,
    IOType,
    Operator,
    OutputContext,
)
from monai.deploy.operators.dicom_text_sr_writer_operator import DICOMTextSRWriterOperator, EquipmentInfo, ModelInfo
from monai.transforms import AddChannel, Compose, EnsureType, ScaleIntensity

MEDNIST_CLASSES = ["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"]


@md.input("image", DataPath, IOType.DISK)
@md.output("image", Image, IOType.IN_MEMORY)
@md.env(pip_packages=["pillow"])
class LoadPILOperator(Operator):
    """Load image from the given input (DataPath) and set numpy array to the output (Image)."""

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        import numpy as np
        from PIL import Image as PILImage

        input_path = op_input.get().path
        if input_path.is_dir():
            input_path = next(input_path.glob("*.*"))  # take the first file

        image = PILImage.open(input_path)
        image = image.convert("L")  # convert to greyscale image
        image_arr = np.asarray(image)

        output_image = Image(image_arr)  # create Image domain object with a numpy array
        op_output.set(output_image)


@md.input("image", Image, IOType.IN_MEMORY)
@md.output("result_text", Text, IOType.IN_MEMORY)
@md.env(pip_packages=["monai"])
class MedNISTClassifierOperator(Operator):
    """Classifies the given image and returns the class name."""

    @property
    def transform(self):
        return Compose([AddChannel(), ScaleIntensity(), EnsureType()])

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        import json

        import torch

        img = op_input.get().asnumpy()  # (64, 64), uint8
        image_tensor = self.transform(img)  # (1, 64, 64), torch.float64
        image_tensor = image_tensor[None].float()  # (1, 1, 64, 64), torch.float32

        # Comment below line if you want to do CPU inference
        image_tensor = image_tensor.cuda()

        model = context.models.get()  # get a TorchScriptModel object
        # Uncomment the following line if you want to do CPU inference
        # model.predictor = torch.jit.load(model.path, map_location="cpu").eval()

        with torch.no_grad():
            outputs = model(image_tensor)

        _, output_classes = outputs.max(dim=1)

        result = MEDNIST_CLASSES[output_classes[0]]  # get the class name
        print(result)
        op_output.set(result, "result_text")

        # Get output (folder) path and create the folder if not exists
        # The following gets the App context's output path, instead the operator's.
        output_folder = context.output.get().path
        output_folder.mkdir(parents=True, exist_ok=True)

        # Write result to "output.json"
        output_path = output_folder / "output.json"
        with open(output_path, "w") as fp:
            json.dump(result, fp)


@md.resource(cpu=1, gpu=1, memory="1Gi")
class App(Application):
    """Application class for the MedNIST classifier."""

    def compose(self):
        load_pil_op = LoadPILOperator()
        classifier_op = MedNISTClassifierOperator()

        my_model_info = ModelInfo("MONAI WG Trainer", "MEDNIST Classifier", "0.1", "xyz")
        my_equipment = EquipmentInfo(manufacturer="MOANI Deploy App SDK", manufacturer_model="DICOM SR Writer")
        my_special_tags = {"SeriesDescription": "Not for clinical use. The result is for research use only."}
        dicom_sr_operator = DICOMTextSRWriterOperator(
            copy_tags=False, model_info=my_model_info, equipment_info=my_equipment, custom_tags=my_special_tags
        )

        self.add_flow(load_pil_op, classifier_op)
        self.add_flow(classifier_op, dicom_sr_operator, {"result_text": "classification_result"})


if __name__ == "__main__":
    App(do_run=True)
