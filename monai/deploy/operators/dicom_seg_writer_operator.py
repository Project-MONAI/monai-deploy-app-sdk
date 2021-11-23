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

import copy
import datetime
import json
import logging
import os
from pathlib import Path
from random import randint
from typing import List, Optional, Union

import numpy as np

from monai.deploy.utils.importutil import optional_import
from monai.deploy.utils.version import get_sdk_semver

dcmread, _ = optional_import("pydicom", name="dcmread")
generate_uid, _ = optional_import("pydicom.uid", name="generate_uid")
ImplicitVRLittleEndian, _ = optional_import("pydicom.uid", name="ImplicitVRLittleEndian")
Dataset, _ = optional_import("pydicom.dataset", name="Dataset")
FileDataset, _ = optional_import("pydicom.dataset", name="FileDataset")
Sequence, _ = optional_import("pydicom.sequence", name="Sequence")
sitk, _ = optional_import("SimpleITK")

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, Image, InputContext, IOType, Operator, OutputContext
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries


@md.input("seg_image", Image, IOType.IN_MEMORY)
@md.input("study_selected_series_list", List[StudySelectedSeries], IOType.IN_MEMORY)
@md.output("dicom_seg_instance", DataPath, IOType.DISK)
@md.env(pip_packages=["pydicom >= 1.4.2", "SimpleITK >= 2.0.0"])
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

    def __init__(self, seg_labels: Optional[Union[List[str], str]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Instantiates the DICOM Seg Writer instance with optional list of segment label strings.

        A string can be used instead of a numerical value for a segment in the segmentation image.
        As of now, integer values are supported for segment mask, and it is further required that the named
        segment will start with 1 and increment sequentially if there are additional segments, while the
        background is of value 0. The caller needs to pass in a string list, whose length corresponds
        to the number of actual segments. The position index + 1 would be the corresponding segment's
        numerical value.

        For example, in the CT Spleen Segmentation application, the whole image background has a value
        of 0, and the Spleen segment of value 1. This then only requires the caller to pass in a list
        containing a single string, which is used as label for the Spleen in the DICOM Seg instance.

        Note: this interface is subject to change. It is planned that a new object will encapsulate the
        segment label information, including label value, name, description etc.

        Args:
            seg_labels: The string name for each segment
        """

        self._seg_labels = ["SegmentLabel-default"]
        if isinstance(seg_labels, str):
            self._seg_labels = [seg_labels]
        elif isinstance(seg_labels, list):
            self._seg_labels = []
            for label in seg_labels:
                if isinstance(label, str) or isinstance(label, int):
                    self._seg_labels.append(label)
                else:
                    raise ValueError(f"List of strings expected, but contains {label} of type {type(label)}.")

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
                input_path = op_input.get("segmentation_image").path
                seg_image, _ = self.select_input_file(input_path)
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

    def create_dicom_seg(self, image: Image, dicom_series: DICOMSeries, file_path: Path):
        file_path.parent.absolute().mkdir(parents=True, exist_ok=True)

        dicom_dataset_list = [i.get_native_sop_instance() for i in dicom_series.get_sop_instances()]
        # DICOM Seg creation
        self._seg_writer = DICOMSegWriter()
        try:
            self._seg_writer.write(image, dicom_dataset_list, str(file_path), self._seg_labels)
            # TODO: get a class to encapsulate the seg label information.

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
        """Select the inut files based on supported extensions.

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

    def _get_label_list(self, stringfied_list_of_labels: str = ""):
        """Parse the string to get the label list.

        If empty string is provided, a list of a single element is retured.

        Args:
            stringfied_list_of_labels (str): string representing the list of segmentation labels.

        Returns:
            list of label strings
        """

        # Use json.loads as a convenience method to convert string to list of strings
        assert isinstance(stringfied_list_of_labels, str), "Expected stringfied list pf labels."

        label_list = ["default-label"]  # Use this as default if empty string
        if stringfied_list_of_labels:
            label_list = json.loads(stringfied_list_of_labels)

        return label_list


class DICOMSegWriter(object):
    def __init__(self):
        """Class to write DICOM SEG with the segmentation image and DICOM dataset."""

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

    def write(self, seg_img, input_ds, outfile, seg_labels):
        """Write DICOM Segmentation object for the segmentation image

        Args:
            seg_img (numpy array): numpy array of the segmentation image.
            input_ds (list): list of Pydicom datasets of the original DICOM instances.
            outfile (str): path for the output DICOM instance file.
            seg_labels: list of labels for the segments
        """

        if seg_img is None:
            raise ValueError("Argument seg_img cannot be None.")
        if not isinstance(input_ds, list) or len(input_ds) < 1:
            raise ValueError("Argument input_ds must not be empty.")
        if not outfile:
            raise ValueError("Argument outfile must not be a valid string.")
        if not isinstance(seg_labels, list) or len(seg_labels) < 1:
            raise ValueError("Argument seg_labels must not be empty.")

        # Find out the number of DICOM instance datasets
        num_of_dcm_ds = len(input_ds)
        self._logger.info("Number of DICOM instance datasets in the list: {}".format(num_of_dcm_ds))

        # Find out the number of slices in the numpy array
        num_of_img_slices = seg_img.shape[0]
        self._logger.info("Number of slices in the numpy image: {}".format(num_of_img_slices))

        # Find out the labels
        self._logger.info("Labels of the segments: {}".format(seg_labels))

        # Find out the unique values in the seg image
        unique_elements = np.unique(seg_img, return_counts=False)
        self._logger.info("Unique values in seg image: {}".format(unique_elements))

        dcm_out = create_multiframe_metadata(outfile, input_ds[0])
        create_label_segments(dcm_out, seg_labels)
        set_pixel_meta(dcm_out, input_ds[0])
        segslice_from_mhd(dcm_out, seg_img, input_ds, len(seg_labels))

        self._logger.info("Saving output file {}".format(outfile))
        dcm_out.save_as(outfile, False)
        self._logger.info("File saved.")


# The following functions are mostly based on the implementation demo'ed at RSNA 2019.
# They can be further refactored and made into class methods, but work for now.


def safe_get(ds, key):
    """Safely gets the tag value if present from the Dataset and logs failure.

    The safe get method of dict works for str, but not the hex key. The added
    benefit of this funtion is that it logs the failure to get the keyed value.

    Args:
        ds (Dataset): pydicom Dataset
        key (hex | str): Hex code or string name for a key.
    """

    try:
        return ds[key].value
    except KeyError as e:
        logging.error("Failed to get value for key: {}".format(e))
    return ""


def random_with_n_digits(n):
    assert isinstance(n, int), "Argument n must be a int."
    n = n if n >= 1 else 1
    range_start = 10 ** (n - 1)
    range_end = (10 ** n) - 1
    return randint(range_start, range_end)


def create_multiframe_metadata(dicom_file, input_ds):
    """Creates the DICOM metadata for the multiframe object, e.g. SEG

    Args:
        dicom_file (str or object): The filename or the object type of the file-like the FileDataset was read from.
        input_ds (Dataset): pydicom dataset of original DICOM instance.

    Returns:
        FileDataset: The object with metadata assigned.
    """

    currentDateRaw = datetime.datetime.now()
    currentDate = currentDateRaw.strftime("%Y%m%d")
    currentTime = currentDateRaw.strftime("%H%M%S.%f")  # long format with micro seconds
    segmentationSeriesInstanceUID = generate_uid(prefix=None)
    segmentationSOPInstanceUID = generate_uid(prefix=None)

    # Populate required values for file meta information

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.66.4"
    file_meta.MediaStorageSOPInstanceUID = segmentationSOPInstanceUID
    file_meta.ImplementationClassUID = "1.2.840.10008.5.1.4.1.1.66.4"
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    # create dicom global metadata
    dicomOutput = FileDataset(dicom_file, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # It is important to understand the Types of DICOM attributes when getting from the original
    # dataset, and creating/setting them in the new dataset, .e.g Type 1 is mandatory, though
    # non-conformant instance may not have them, Type 2 present but maybe blank, and Type 3 may
    # be absent.

    # None of Patient module attributes are mandatory.
    # The following are Type 2, present though could be blank
    dicomOutput.PatientName = input_ds.get("PatientName", "")  # name is actual suppoted
    dicomOutput.add_new(0x00100020, "LO", safe_get(input_ds, 0x00100020))  # PatientID
    dicomOutput.add_new(0x00100030, "DA", safe_get(input_ds, 0x00100030))  # PatientBirthDate
    dicomOutput.add_new(0x00100040, "CS", safe_get(input_ds, 0x00100040))  # PatientSex
    dicomOutput.add_new(0x00104000, "LT", safe_get(input_ds, "0x00104000"))  # PatientComments

    # For Study module, copy original StudyInstanceUID and other Type 2 study attributes
    # Only Study Instance UID is Type 1, though still may be absent, so try to get
    dicomOutput.add_new(0x0020000D, "UI", safe_get(input_ds, 0x0020000D))  # StudyInstanceUID
    dicomOutput.add_new(0x00080020, "DA", input_ds.get("StudyDate", currentDate))  # StudyDate
    dicomOutput.add_new(0x00080030, "TM", input_ds.get("StudyTime", currentTime))  # StudyTime
    dicomOutput.add_new(0x00080090, "PN", safe_get(input_ds, 0x00080090))  # ReferringPhysicianName
    dicomOutput.add_new(0x00200010, "SH", safe_get(input_ds, 0x00200010))  # StudyID
    dicomOutput.add_new(0x00080050, "SH", safe_get(input_ds, 0x00080050))  # AccessionNumber

    # Series module with new attribute values, only Modality and SeriesInstanceUID are Type 1
    dicomOutput.add_new(0x00080060, "CS", "SEG")  # Modality
    dicomOutput.add_new(0x0020000E, "UI", segmentationSeriesInstanceUID)  # SeriesInstanceUID
    dicomOutput.add_new(0x00200011, "IS", random_with_n_digits(4))  # SeriesNumber (randomized)
    descr = "CAUTION: Research Use Only. MONAI Deploy App SDK generated DICOM SEG"
    if safe_get(input_ds, 0x0008103E):
        descr += " for " + safe_get(input_ds, 0x0008103E)
    dicomOutput.add_new(0x0008103E, "LO", descr)  # SeriesDescription
    dicomOutput.add_new(0x00080021, "DA", currentDate)  # SeriesDate
    dicomOutput.add_new(0x00080031, "TM", currentTime)  # SeriesTime

    # General Equipment module, only Manufacturer is Type 2, the rest Type 3
    dicomOutput.add_new(0x00181000, "LO", "0000")  # DeviceSerialNumber
    dicomOutput.add_new(0x00080070, "LO", "MONAI Deploy")  # Manufacturer
    dicomOutput.add_new(0x00081090, "LO", "App SDK")  # ManufacturerModelName
    try:
        version_str = get_sdk_semver()  # SDK Version
    except Exception:
        version_str = "0.1"  # Fall back to the initial version
    dicomOutput.add_new(0x00181020, "LO", version_str)  # SoftwareVersions

    # SOP common, only SOPClassUID and SOPInstanceUID are Type 1
    dicomOutput.add_new(0x00200013, "IS", 1)  # InstanceNumber
    dicomOutput.add_new(0x00080016, "UI", "1.2.840.10008.5.1.4.1.1.66.4")  # SOPClassUID, per DICOM.
    dicomOutput.add_new(0x00080018, "UI", segmentationSOPInstanceUID)  # SOPInstanceUID
    dicomOutput.add_new(0x00080012, "DA", currentDate)  # InstanceCreationDate
    dicomOutput.add_new(0x00080013, "TM", currentTime)  # InstanceCreationTime

    # General Image module.
    dicomOutput.add_new(0x00080008, "CS", ["DERIVED", "PRIMARY"])  # ImageType
    dicomOutput.add_new(0x00200020, "CS", "")  # PatientOrientation, forced empty
    # Set content date/time
    dicomOutput.ContentDate = currentDate
    dicomOutput.ContentTime = currentTime

    # Image Pixel
    dicomOutput.add_new(0x00280002, "US", 1)  # SamplesPerPixel
    dicomOutput.add_new(0x00280004, "CS", "MONOCHROME2")  # PhotometricInterpretation

    # Common Instance Reference module
    dicomOutput.add_new(0x00081115, "SQ", [Dataset()])  # ReferencedSeriesSequence
    # Set the referenced SeriesInstanceUID
    dicomOutput.get(0x00081115)[0].add_new(0x0020000E, "UI", safe_get(input_ds, 0x0020000E))

    # Multi-frame Dimension Module
    dimensionID = generate_uid(prefix=None)
    dimensionOragnizationSequence = Sequence()
    dimensionOragnizationSequenceDS = Dataset()
    dimensionOragnizationSequenceDS.add_new(0x00209164, "UI", dimensionID)  # DimensionOrganizationUID
    dimensionOragnizationSequence.append(dimensionOragnizationSequenceDS)
    dicomOutput.add_new(0x00209221, "SQ", dimensionOragnizationSequence)  # DimensionOrganizationSequence

    dimensionIndexSequence = Sequence()
    dimensionIndexSequenceDS = Dataset()
    dimensionIndexSequenceDS.add_new(0x00209164, "UI", dimensionID)  # DimensionOrganizationUID
    dimensionIndexSequenceDS.add_new(0x00209165, "AT", 0x00209153)  # DimensionIndexPointer
    dimensionIndexSequenceDS.add_new(0x00209167, "AT", 0x00209153)  # FunctionalGroupPointer
    dimensionIndexSequence.append(dimensionIndexSequenceDS)
    dicomOutput.add_new(0x00209222, "SQ", dimensionIndexSequence)  # DimensionIndexSequence

    return dicomOutput


def create_label_segments(dcm_output, seg_labels):
    """ "Creates the segments with the given labels"""

    def create_label_segment(label, name):
        """Creates segment labels"""
        segment = Dataset()
        segment.add_new(0x00620004, "US", int(label))  # SegmentNumber
        segment.add_new(0x00620005, "LO", name)  # SegmentLabel
        segment.add_new(0x00620009, "LO", "AI Organ Segmentation")  # SegmentAlgorithmName
        segment.SegmentAlgorithmType = "AUTOMATIC"  # SegmentAlgorithmType
        segment.add_new(0x0062000D, "US", [128, 174, 128])  # RecommendedDisplayCIELabValue
        # Create SegmentedPropertyCategoryCodeSequence
        segmentedPropertyCategoryCodeSequence = Sequence()
        segmentedPropertyCategoryCodeSequenceDS = Dataset()
        segmentedPropertyCategoryCodeSequenceDS.add_new(0x00080100, "SH", "T-D0050")  # CodeValue
        segmentedPropertyCategoryCodeSequenceDS.add_new(0x00080102, "SH", "SRT")  # CodingSchemeDesignator
        segmentedPropertyCategoryCodeSequenceDS.add_new(0x00080104, "LO", "Anatomical Structure")  # CodeMeaning
        segmentedPropertyCategoryCodeSequence.append(segmentedPropertyCategoryCodeSequenceDS)
        segment.SegmentedPropertyCategoryCodeSequence = segmentedPropertyCategoryCodeSequence
        # Create SegmentedPropertyTypeCodeSequence
        segmentedPropertyTypeCodeSequence = Sequence()
        segmentedPropertyTypeCodeSequenceDS = Dataset()
        segmentedPropertyTypeCodeSequenceDS.add_new(0x00080100, "SH", "T-D0050")  # CodeValue
        segmentedPropertyTypeCodeSequenceDS.add_new(0x00080102, "SH", "SRT")  # CodingSchemeDesignator
        segmentedPropertyTypeCodeSequenceDS.add_new(0x00080104, "LO", "Organ")  # CodeMeaning
        segmentedPropertyTypeCodeSequence.append(segmentedPropertyTypeCodeSequenceDS)
        segment.SegmentedPropertyTypeCodeSequence = segmentedPropertyTypeCodeSequence
        return segment

    segments = Sequence()
    # Assumes the label starts at 1 and increment sequentially.
    # TODO: This part needs to be more deteministic, e.g. with a dict.
    for lb, name in enumerate(seg_labels, 1):
        segment = create_label_segment(lb, name)
        segments.append(segment)
    dcm_output.add_new(0x00620002, "SQ", segments)  # SegmentSequence


def create_frame_meta(input_ds, label, ref_instances, dimIdxVal, instance_num):
    """Creates the metadata for the each frame"""

    sop_inst_uid = safe_get(input_ds, 0x00080018)  # SOPInstanceUID
    sourceInstanceSOPClass = safe_get(input_ds, 0x00080016)  # SOPClassUID

    # add frame to Referenced Image Sequence
    frame_ds = Dataset()
    referenceInstance = Dataset()
    referenceInstance.add_new(0x00081150, "UI", sourceInstanceSOPClass)  # ReferencedSOPClassUID
    referenceInstance.add_new(0x00081155, "UI", sop_inst_uid)  # ReferencedSOPInstanceUID

    ref_instances.append(referenceInstance)
    ############################
    # CREATE METADATA
    ############################
    # Create DerivationImageSequence within Per-frame Functional Groups sequence
    derivationImageSequence = Sequence()
    derivationImage = Dataset()
    # Create SourceImageSequence within DerivationImageSequence
    sourceImageSequence = Sequence()
    sourceImage = Dataset()
    # TODO if CT multi-frame
    # sourceImage.add_new(0x00081160, 'IS', inputFrameCounter + 1) # Referenced Frame Number
    sourceImage.add_new(0x00081150, "UI", sourceInstanceSOPClass)  # ReferencedSOPClassUID
    sourceImage.add_new(0x00081155, "UI", sop_inst_uid)  # ReferencedSOPInstanceUID
    # Create PurposeOfReferenceCodeSequence within SourceImageSequence
    purposeOfReferenceCodeSequence = Sequence()
    purposeOfReferenceCode = Dataset()
    purposeOfReferenceCode.add_new(0x00080100, "SH", "121322")  # CodeValue
    purposeOfReferenceCode.add_new(0x00080102, "SH", "DCM")  # CodingSchemeDesignator
    purposeOfReferenceCode.add_new(0x00080104, "LO", "Anatomical Stucture")  # CodeMeaning
    purposeOfReferenceCodeSequence.append(purposeOfReferenceCode)
    sourceImage.add_new(0x0040A170, "SQ", purposeOfReferenceCodeSequence)  # PurposeOfReferenceCodeSequence
    sourceImageSequence.append(sourceImage)  # AEH Beck commentout
    # Create DerivationCodeSequence within DerivationImageSequence
    derivationCodeSequence = Sequence()
    derivationCode = Dataset()
    derivationCode.add_new(0x00080100, "SH", "113076")  # CodeValue
    derivationCode.add_new(0x00080102, "SH", "DCM")  # CodingSchemeDesignator
    derivationCode.add_new(0x00080104, "LO", "Segmentation")  # CodeMeaning
    derivationCodeSequence.append(derivationCode)
    derivationImage.add_new(0x00089215, "SQ", derivationCodeSequence)  # DerivationCodeSequence
    derivationImage.add_new(0x00082112, "SQ", sourceImageSequence)  # SourceImageSequence
    derivationImageSequence.append(derivationImage)
    frame_ds.add_new(0x00089124, "SQ", derivationImageSequence)  # DerivationImageSequence
    # Create FrameContentSequence within Per-frame Functional Groups sequence
    frameContent = Sequence()
    dimensionIndexValues = Dataset()
    dimensionIndexValues.add_new(0x00209157, "UL", [dimIdxVal, instance_num])  # DimensionIndexValues
    frameContent.append(dimensionIndexValues)
    frame_ds.add_new(0x00209111, "SQ", frameContent)  # FrameContentSequence
    # Create PlanePositionSequence within Per-frame Functional Groups sequence
    planePositionSequence = Sequence()
    imagePositionPatient = Dataset()
    imagePositionPatient.add_new(0x00200032, "DS", safe_get(input_ds, 0x00200032))  # ImagePositionPatient
    planePositionSequence.append(imagePositionPatient)
    frame_ds.add_new(0x00209113, "SQ", planePositionSequence)  # PlanePositionSequence
    # Create PlaneOrientationSequence within Per-frame Functional Groups sequence
    planeOrientationSequence = Sequence()
    imageOrientationPatient = Dataset()
    imageOrientationPatient.add_new(0x00200037, "DS", safe_get(input_ds, 0x00200037))  # ImageOrientationPatient
    planeOrientationSequence.append(imageOrientationPatient)
    frame_ds.add_new(0x00209116, "SQ", planeOrientationSequence)  # PlaneOrientationSequence
    # Create SegmentIdentificationSequence within Per-frame Functional Groups sequence
    segmentIdentificationSequence = Sequence()
    referencedSegmentNumber = Dataset()
    # TODO lop over label and only get pixel with that value
    referencedSegmentNumber.add_new(0x0062000B, "US", label)  # ReferencedSegmentNumber, which label is this frame
    segmentIdentificationSequence.append(referencedSegmentNumber)
    frame_ds.add_new(0x0062000A, "SQ", segmentIdentificationSequence)  # SegmentIdentificationSequence
    return frame_ds


def set_pixel_meta(dicomOutput, input_ds):
    """Sets the pixel metadata in the DICOM object"""

    dicomOutput.Rows = input_ds.Rows
    dicomOutput.Columns = input_ds.Columns
    dicomOutput.BitsAllocated = 8  # add_new(0x00280100, 'US', 8) # Bits allocated
    dicomOutput.BitsStored = 1
    dicomOutput.HighBit = 0
    dicomOutput.PixelRepresentation = 0
    # dicomOutput.PixelRepresentation = input_ds.PixelRepresentation
    dicomOutput.SamplesPerPixel = 1
    dicomOutput.ImageType = "DERIVED\\PRIMARY"
    dicomOutput.ContentLabel = "SEGMENTATION"
    dicomOutput.ContentDescription = ""
    dicomOutput.ContentCreatorName = ""
    dicomOutput.LossyImageCompression = "00"
    dicomOutput.SegmentationType = "BINARY"
    dicomOutput.MaximumFractionalValue = 1
    dicomOutput.SharedFunctionalGroupsSequence = Sequence()
    dicomOutput.PixelPaddingValue = 0
    # Try to get the attributes from the original.
    # Even though they are Type 1 and 2, can still be absent
    dicomOutput.PixelSpacing = copy.deepcopy(input_ds.get("PixelSpacing", None))
    dicomOutput.SliceThickness = input_ds.get("SliceThickness", "")
    dicomOutput.RescaleSlope = 1
    dicomOutput.RescaleIntercept = 0
    # Set the transfer syntax
    dicomOutput.is_little_endian = False  # True
    dicomOutput.is_implicit_VR = False  # True


def segslice_from_mhd(dcm_output, seg_img, input_ds, num_labels):
    """Sets the pixel data from the input numpy image"""

    if np.amax(seg_img) == 0 and np.amin(seg_img) == 0:
        raise ValueError("Seg mask is not detected; all 0's.")

    # add frames
    out_frame_counter = 0
    out_frames = Sequence()

    out_pixels = None

    referenceInstances = Sequence()

    for img_slice in range(seg_img.shape[0]):

        dimIdxVal = 0

        for label in range(1, num_labels + 1):

            # Determine if frame gets output
            if np.count_nonzero(seg_img[img_slice, ...] == label) == 0:  # no label for this frame --> skip
                continue

            dimIdxVal += 1

            frame_meta = create_frame_meta(input_ds[img_slice], label, referenceInstances, dimIdxVal, img_slice)

            out_frames.append(frame_meta)
            logging.debug(
                "img slice {}, label {}, frame {}, img pos {}".format(
                    img_slice, label, out_frame_counter, safe_get(input_ds[img_slice], 0x00200032)
                )
            )
            seg_slice = np.zeros((1, seg_img.shape[1], seg_img.shape[2]), dtype=bool)

            seg_slice[np.expand_dims(seg_img[img_slice, ...] == label, 0)] = 1

            if out_pixels is None:
                out_pixels = seg_slice
            else:
                out_pixels = np.concatenate((out_pixels, seg_slice), axis=0)

            out_frame_counter = out_frame_counter + 1

    dcm_output.add_new(0x52009230, "SQ", out_frames)  # PerFrameFunctionalGroupsSequence
    dcm_output.NumberOfFrames = out_frame_counter
    dcm_output.PixelData = np.packbits(np.flip(np.reshape(out_pixels.astype(bool), (-1, 8)), 1)).tobytes()

    dcm_output.get(0x00081115)[0].add_new(0x0008114A, "SQ", referenceInstances)  # ReferencedInstanceSequence

    # Create shared Functional Groups sequence
    sharedFunctionalGroups = Sequence()
    sharedFunctionalGroupsDS = Dataset()

    planeOrientationSeq = Sequence()
    planeOrientationDS = Dataset()
    planeOrientationDS.add_new("0x00200037", "DS", safe_get(input_ds[0], 0x00200037))  # ImageOrientationPatient
    planeOrientationSeq.append(planeOrientationDS)
    sharedFunctionalGroupsDS.add_new("0x00209116", "SQ", planeOrientationSeq)  # PlaneOrientationSequence

    pixelMeasuresSequence = Sequence()
    pixelMeasuresDS = Dataset()
    pixelMeasuresDS.add_new("0x00280030", "DS", safe_get(input_ds[0], "0x00280030"))  # PixelSpacing
    if input_ds[0].get("SpacingBetweenSlices", ""):
        pixelMeasuresDS.add_new("0x00180088", "DS", input_ds[0].get("SpacingBetweenSlices", ""))  # SpacingBetweenSlices
    pixelMeasuresDS.add_new("0x00180050", "DS", safe_get(input_ds[0], "0x00180050"))  # SliceThickness
    pixelMeasuresSequence.append(pixelMeasuresDS)
    sharedFunctionalGroupsDS.add_new("0x00289110", "SQ", pixelMeasuresSequence)  # PixelMeasuresSequence

    sharedFunctionalGroups.append(sharedFunctionalGroupsDS)

    dcm_output.add_new(0x52009229, "SQ", sharedFunctionalGroups)  # SharedFunctionalGroupsSequence


# End DICOM Seg Writer temp


def test():
    from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
    from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
    from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator

    current_file_dir = Path(__file__).parent.resolve()
    data_path = current_file_dir.joinpath("../../../examples/ai_spleen_seg_data/dcm")
    out_path = current_file_dir.joinpath("../../../examples/output_seg_op/dcm_seg_test.dcm")

    loader = DICOMDataLoaderOperator()
    series_selector = DICOMSeriesSelectorOperator()
    dcm_to_volume_op = DICOMSeriesToVolumeOperator()
    seg_writer = DICOMSegmentationWriterOperator()

    # Testing with more granular functions
    study_list = loader.load_data_to_studies(data_path.absolute())
    series = study_list[0].get_all_series()[0]

    dcm_to_volume_op.prepare_series(series)
    voxels = dcm_to_volume_op.generate_voxel_data(series)
    metadata = dcm_to_volume_op.create_metadata(series)
    image = dcm_to_volume_op.create_volumetric_image(voxels, metadata)
    image_numpy = image.asnumpy()

    seg_writer.create_dicom_seg(image_numpy, series, Path(out_path).absolute())

    # Testing with the main entry functions
    study_list = loader.load_data_to_studies(data_path.absolute())
    study_selected_series_list = series_selector.filter(None, study_list)
    image = dcm_to_volume_op.convert_to_image(study_selected_series_list)
    seg_writer.process_images(image, study_selected_series_list, out_path.parent.absolute())


if __name__ == "__main__":
    test()
