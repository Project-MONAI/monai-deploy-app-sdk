import json
from typing import Dict, Text

import torch

import monai.deploy.core as md
from monai.data import DataLoader, Dataset
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, Operator, OutputContext
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


@md.input("image", Image, IOType.IN_MEMORY)
@md.output("result_text", Text, IOType.IN_MEMORY)
class ClassifierOperator(Operator):
    def __init__(self):
        super().__init__()
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"

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

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        input_image = op_input.get("image")
        _reader = InMemImageReader(input_image)
        input_img_metadata = self._convert_dicom_metadata_datatype(input_image.metadata())
        img_name = str(input_img_metadata.get("SeriesInstanceUID", "Img_in_context"))

        output_path = context.output.get().path

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = context.models.get()

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
        result_dict_out = {"A": str(out[0]), "B": str(out[1]), "C": str(out[2]), "D": str(out[3])}
        output_folder = context.output.get().path
        output_folder.mkdir(parents=True, exist_ok=True)

        output_path = output_folder / "output.json"
        with open(output_path, "w") as fp:
            json.dump(result_dict, fp)

        op_output.set(result_dict, "result_text")

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
