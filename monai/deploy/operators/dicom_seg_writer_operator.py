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

import os
from pathlib import Path
from random import randint
from typing import TYPE_CHECKING, List, Union

import numpy as np

from monai.deploy.utils.importutil import optional_import
from monai.deploy.utils.version import get_sdk_semver

dcmread, _ = optional_import("pydicom", name="dcmread")
generate_uid, _ = optional_import("pydicom.uid", name="generate_uid")
ImplicitVRLittleEndian, _ = optional_import("pydicom.uid", name="ImplicitVRLittleEndian")
Dataset, _ = optional_import("pydicom.dataset", name="Dataset")
FileDataset, _ = optional_import("pydicom.dataset", name="FileDataset")
Sequence, _ = optional_import("pydicom.sequence", name="Sequence")
hd, _ = optional_import("highdicom")
sitk, _ = optional_import("SimpleITK")
codes, _ = optional_import("pydicom.sr.codedict", name="codes")
if TYPE_CHECKING:
    from highdicom.seg import SegmentDescription
else:
    SegmentDescription, _ = optional_import("highdicom.seg", name="SegmentDescription")

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, Image, InputContext, IOType, Operator, OutputContext
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries


@md.input("seg_image", Image, IOType.IN_MEMORY)
@md.input("study_selected_series_list", List[StudySelectedSeries], IOType.IN_MEMORY)
@md.output("dicom_seg_instance", DataPath, IOType.DISK)
@md.env(pip_packages=["pydicom >= 2.3.0", "highdicom >= 0.16.0"])
class DICOMSegmentationWriterOperator(Operator):
    """
    This operator writes out a DICOM Segmentation Part 10 file to disk
    """

    # Supported input image format, based on extension.
    SUPPORTED_EXTENSIONS = [".nii", ".nii.gz", ".mhd"]
    # DICOM instance file extension. Case insensitive in string comparison.
    DCM_EXTENSION = ".dcm"
    # Suffix to add to file name to indicate DICOM Seg dcm file.
    DICOMSEG_SUFFIX = "-DICOMSEG"

    def __init__(self, segment_descriptions: List[SegmentDescription], *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Instantiates the DICOM Seg Writer instance with optional list of segment label strings.

        Each unique, non-zero integer value in the segmentation image represents a segment that must be
        described by an item of the segment descriptions list with the corresponding segment number.
        Items in the list must be arranged starting at segment number 1 and increasing by 1.

        For example, in the CT Spleen Segmentation application, the whole image background has a value
        of 0, and the Spleen segment of value 1. This then only requires the caller to pass in a list
        containing a segment description, which is used as label for the Spleen in the DICOM Seg instance.

        Note: this interface is subject to change. It is planned that a new object will encapsulate the
        segment label information, including label value, name, description etc.

        Args:
            segment_descriptions: Object encapsulating the description of each segment present in the
            segmentation.
        """

        self._seg_descs = segment_descriptions

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        """Performs computation for this operator and handles I/O.

        For now, only a single segmentation image object or file is supported and the selected DICOM
        series for inference is required, because the DICOM Seg IOD needs to refer to original instance.
        When there are multiple selected series in the input, the first series' containing study will
        be used for retrieving DICOM Study module attributes, e.g. StudyInstanceUID.

        Raises:
            FileNotFoundError: When image object not in the input, and segmentation image file not found either.
            ValueError: Neither image object nor image file's folder is in the input, or no selected series.
        """

        # Gets the input, prepares the output folder, and then delegates the processing.
        study_selected_series_list = op_input.get("study_selected_series_list")
        if not study_selected_series_list or len(study_selected_series_list) < 1:
            raise ValueError("Missing input, list of 'StudySelectedSeries'.")
        for study_selected_series in study_selected_series_list:
            if not isinstance(study_selected_series, StudySelectedSeries):
                raise ValueError("Element in input is not expected type, 'StudySelectedSeries'.")

        seg_image = op_input.get("seg_image")
        # In case the Image object is not in the input, and input is the seg image file folder path.
        if not isinstance(seg_image, Image):
            if isinstance(seg_image, DataPath):
                seg_image, _ = self.select_input_file(seg_image.path)
            else:
                raise ValueError("Input 'seg_image' is not Image or DataPath.")

        output_dir = op_output.get().path
        output_dir.mkdir(parents=True, exist_ok=True)

        self.process_images(seg_image, study_selected_series_list, output_dir)

    def process_images(
        self, image: Union[Image, Path], study_selected_series_list: List[StudySelectedSeries], output_dir: Path
    ):
        """ """
        # Get the seg image in numpy, and if the image is passed in as object, need to fake a input path.
        seg_image_numpy = None
        input_path = "dicom_seg"

        if isinstance(image, Image):
            seg_image_numpy = image.asnumpy()
        elif isinstance(image, Path):
            input_path = str(image)  # It is expected that this is the image file path.
            seg_image_numpy = self._image_file_to_numpy(input_path)
        else:
            raise ValueError("'image' is not an Image object or a supported image file.")

        # The output DICOM Seg instance file name is based on the actual or made-up input image file name.
        output_filename = "{0}{1}{2}".format(
            os.path.splitext(os.path.basename(input_path))[0],
            DICOMSegmentationWriterOperator.DICOMSEG_SUFFIX,
            DICOMSegmentationWriterOperator.DCM_EXTENSION,
        )
        output_path = output_dir / output_filename
        # Pick DICOM Series that was used as input for getting the seg image.
        # For now, first one in the list.
        for study_selected_series in study_selected_series_list:
            if not isinstance(study_selected_series, StudySelectedSeries):
                raise ValueError("Element in input is not expected type, 'StudySelectedSeries'.")
            selected_series = study_selected_series.selected_series[0]
            dicom_series = selected_series.series
            self.create_dicom_seg(seg_image_numpy, dicom_series, output_path)
            break

    def create_dicom_seg(self, image: np.ndarray, dicom_series: DICOMSeries, file_path: Path):
        file_path.parent.absolute().mkdir(parents=True, exist_ok=True)

        dicom_dataset_list = [i.get_native_sop_instance() for i in dicom_series.get_sop_instances()]

        try:
            version_str = get_sdk_semver()  # SDK Version
        except Exception:
            version_str = "0.1"  # Fall back to the initial version

        seg = hd.seg.Segmentation(
            source_images=dicom_dataset_list,
            pixel_array=image,
            segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
            segment_descriptions=self._seg_descs,
            series_instance_uid=hd.UID(),
            series_number=random_with_n_digits(4),
            sop_instance_uid=hd.UID(),
            instance_number=1,
            manufacturer="The MONAI Consortium",
            manufacturer_model_name="MONAI Deploy App SDK",
            software_versions=version_str,
            device_serial_number="0000",
        )
        seg.save_as(file_path)

        try:
            # Test reading back
            _ = self._read_from_dcm(str(file_path))
        except Exception as ex:
            print("DICOMSeg creation failed. Error:\n{}".format(ex))
            raise

    def _read_from_dcm(self, file_path: str):
        """Read dcm file into pydicom Dataset

        Args:
            file_path (str): The path to dcm file
        """
        return dcmread(file_path)

    def select_input_file(self, input_folder, extensions=SUPPORTED_EXTENSIONS):
        """Select the input files based on supported extensions.

        Args:
            input_folder (string): the path of the folder containing the input file(s)
            extensions (array): the supported file formats identified by the extensions.

        Returns:
            file_path (string) : The path of the selected file
            ext (string): The extension of the selected file
        """

        def which_supported_ext(file_path, extensions):
            for ext in extensions:
                if file_path.casefold().endswith(ext.casefold()):
                    return ext
            return None

        if os.path.isdir(input_folder):
            for file_name in os.listdir(input_folder):
                file_path = os.path.join(input_folder, file_name)
                if os.path.isfile(file_path):
                    ext = which_supported_ext(file_path, extensions)
                    if ext:
                        return file_path, ext
            raise IOError("No supported input file found ({})".format(extensions))
        elif os.path.isfile(input_folder):
            ext = which_supported_ext(input_folder, extensions)
            if ext:
                return input_folder, ext
        else:
            raise FileNotFoundError("{} is not found.".format(input_folder))

    def _image_file_to_numpy(self, input_path: str):
        """Converts image file to numpy"""

        img = sitk.ReadImage(input_path)
        data_np = sitk.GetArrayFromImage(img)
        if data_np is None:
            raise RuntimeError("Failed to convert image file to numpy: {}".format(input_path))
        return data_np.astype(np.uint8)


def random_with_n_digits(n):
    assert isinstance(n, int), "Argument n must be a int."
    n = n if n >= 1 else 1
    range_start = 10 ** (n - 1)
    range_end = (10**n) - 1
    return randint(range_start, range_end)


def test():
    from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
    from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
    from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator

    current_file_dir = Path(__file__).parent.resolve()
    data_path = current_file_dir.joinpath("../../../examples/ai_spleen_seg_data/dcm")
    out_path = current_file_dir.joinpath("../../../examples/output_seg_op/dcm_seg_test.dcm")
    segment_descriptions = [
        hd.seg.SegmentDescription(
            segment_number=1,
            segment_label="Spleen",
            segmented_property_category=codes.SCT.Organ,
            segmented_property_type=codes.SCT.Spleen,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=hd.AlgorithmIdentificationSequence(
                name="Test algorithm",
                family=codes.DCM.ArtificialIntelligence,
                version="0.0.2",
            ),
        )
    ]

    loader = DICOMDataLoaderOperator()
    series_selector = DICOMSeriesSelectorOperator()
    dcm_to_volume_op = DICOMSeriesToVolumeOperator()
    seg_writer = DICOMSegmentationWriterOperator(segment_descriptions)

    # Testing with more granular functions
    study_list = loader.load_data_to_studies(data_path.absolute())
    series = study_list[0].get_all_series()[0]

    dcm_to_volume_op.prepare_series(series)
    voxels = dcm_to_volume_op.generate_voxel_data(series)
    metadata = dcm_to_volume_op.create_metadata(series)
    image = dcm_to_volume_op.create_volumetric_image(voxels, metadata)
    # Very crude thresholding
    image_numpy = (image.asnumpy() > 400).astype(np.uint8)

    seg_writer.create_dicom_seg(image_numpy, series, Path(out_path).absolute())

    # Testing with the main entry functions
    study_list = loader.load_data_to_studies(data_path.absolute())
    study_selected_series_list = series_selector.filter(None, study_list)
    image = dcm_to_volume_op.convert_to_image(study_selected_series_list)
    # Very crude thresholding
    image_numpy = (image.asnumpy() > 400).astype(np.uint8)
    image = Image(image_numpy)
    seg_writer.process_images(image, study_selected_series_list, out_path.parent.absolute())


if __name__ == "__main__":
    test()
