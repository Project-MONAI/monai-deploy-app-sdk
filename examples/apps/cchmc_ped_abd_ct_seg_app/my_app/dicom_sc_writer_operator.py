# Copyright 2021-2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pydicom

from monai.deploy.core import Fragment, Operator, OperatorSpec
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries
from monai.deploy.operators.dicom_utils import EquipmentInfo, ModelInfo, write_common_modules
from monai.deploy.utils.importutil import optional_import
from monai.deploy.utils.version import get_sdk_semver

dcmread, _ = optional_import("pydicom", name="dcmread")
dcmwrite, _ = optional_import("pydicom.filewriter", name="dcmwrite")
generate_uid, _ = optional_import("pydicom.uid", name="generate_uid")
ImplicitVRLittleEndian, _ = optional_import("pydicom.uid", name="ImplicitVRLittleEndian")
ExplicitVRLittleEndian, _ = optional_import("pydicom.uid", name="ExplicitVRLittleEndian")
Dataset, _ = optional_import("pydicom.dataset", name="Dataset")
FileDataset, _ = optional_import("pydicom.dataset", name="FileDataset")
Sequence, _ = optional_import("pydicom.sequence", name="Sequence")


class DICOMSCWriterOperator(Operator):
    """Class to write a new DICOM Secondary Capture (DICOM SC) instance with source DICOM Series metadata included.

    Named inputs:
        input_overlay_image: Image object or numpy array of the secondary capture content (RGB).
        study_selected_series_list: DICOM Series for copying metadata from.

    Named output:
        None.

    File output:
        New, updated DICOM SC file (with source DICOM Series metadata) in the provided output folder.
    """

    # file extension for the generated DICOM Part 10 file
    DCM_EXTENSION = ".dcm"
    # the default output folder for saving the generated DICOM instance file
    # DEFAULT_OUTPUT_FOLDER = Path(os.path.join(os.path.dirname(__file__))) / "output"
    DEFAULT_OUTPUT_FOLDER = Path.cwd() / "output"

    def __init__(
        self,
        fragment: Fragment,
        *args,
        output_folder: Union[str, Path],
        model_info: ModelInfo,
        equipment_info: Optional[EquipmentInfo] = None,
        custom_tags: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Class to write a new DICOM Secondary Capture (DICOM SC) instance with source DICOM Series metadata.

        Args:
            output_folder (str or Path): The folder for saving the generated DICOM SC instance file.
            model_info (ModelInfo): Object encapsulating model creator, name, version and UID.
            equipment_info (EquipmentInfo, optional): Object encapsulating info for DICOM Equipment Module.
                                                      Defaults to None.
            custom_tags (Dict[str, str], optional): Dictionary for setting custom DICOM tags using Keywords and str values only.
                                                    Defaults to None.

        Raises:
            ValueError: If result cannot be found either in memory or from file.
        """

        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")

        # need to init the output folder until the execution context supports dynamic FS path
        # not trying to create the folder to avoid exception on init
        self.output_folder = Path(output_folder) if output_folder else DICOMSCWriterOperator.DEFAULT_OUTPUT_FOLDER
        
        # CHANGED (V3/V2): Input names adapted for direct image injection
        self.input_name_study_series = "study_selected_series_list"
        self.input_overlay_image = "input_overlay_image"

        # for copying DICOM attributes from a provided DICOMSeries
        # required input for write_common_modules; will always be True for this implementation
        self.copy_tags = True

        self.model_info = model_info if model_info else ModelInfo()
        self.equipment_info = equipment_info if equipment_info else EquipmentInfo()
        self.custom_tags = custom_tags
        
        self.modality_type = "OT"  # OT Modality for Secondary Capture
        self.sop_class_uid = (
            "1.2.840.10008.5.1.4.1.1.7.4"  # SOP Class UID for Multi-frame True Color Secondary Capture Image Storage
        )
        # custom OverlayImageLabeld post-processing transform creates an RBG overlay

        # equipment version may be different from contributing equipment version
        try:
            self.software_version_number = get_sdk_semver()  # SDK Version
        except Exception:
            self.software_version_number = ""
        self.operators_name = f"AI Algorithm {self.model_info.name}"

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Set up the named input(s), and output(s) if applicable.

        This operator does not have an output for the next operator, rather file output only.

        Args:
            spec (OperatorSpec): The Operator specification for inputs and outputs etc.
        """
        spec.input(self.input_overlay_image)
        spec.input(self.input_name_study_series)

    def compute(self, op_input, op_output, context):
        """Performs computation for this operator and handles I/O.

        For now, only a single result content is supported, which could be in memory or an accessible file.
        The DICOM Series used during inference is required (and copy_tags is hardcoded to True).

        When there are multiple selected series in the input, the first series' containing study will
        be used for retrieving DICOM Study module attributes, e.g. StudyInstanceUID.

        Raises:
            NotADirectoryError: When temporary DICOM SC path is not a directory.
            FileNotFoundError: When result object not in the input, and result file not found either.
            ValueError: Content object and file path not in the inputs, or no DICOM series provided.
            IOError: If the input content is blank.
        """
        # input_overlay_image
        overlay_image = op_input.receive(self.input_overlay_image)
        
        # Recieve input series
        study_selected_series_list = op_input.receive(self.input_name_study_series)
        if not study_selected_series_list or len(study_selected_series_list) < 1:
            raise ValueError("Missing input, list of 'StudySelectedSeries'.")

        # retrieve the DICOM Series used during inference in order to grab appropriate study/series level tags
        # this will be the 1st Series in study_selected_series_list
        dicom_series = None
        for study_selected_series in study_selected_series_list:
            if not isinstance(study_selected_series, StudySelectedSeries):
                raise ValueError(f"Element in input is not expected type, {StudySelectedSeries}.")
            selected_series = study_selected_series.selected_series[0]
            dicom_series = selected_series.series
            break

        # log basic DICOM metadata for the retrieved DICOM Series
        self._logger.debug(f"Dicom Series: {dicom_series}")

        # the output folder should come from the execution context when it is supported
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # write the new DICOM SC instance
        self.process_images(overlay_image, dicom_series, self.output_folder)

    
    def process_images(self, overlay_image, dicom_series: DICOMSeries, output_folder: Path):
        """Process the overlay image and write the DICOM SC instance.

        Args:
            overlay_image: The overlay image (temporary DICOM SC).
            dicom_series (DICOMSeries): DICOMSeries object encapsulating the original series.
            output_folder (Path): The folder for saving the generated DICOM SC instance file.
            
        Returns:
            None
        """
    
        # use write_common_modules to copy metadata from dicom_series
        # this will copy metadata and return an updated Dataset
        ds = write_common_modules(
            dicom_series,
            self.copy_tags,  # always True for this implementation
            self.modality_type,
            self.sop_class_uid,
            self.model_info,
            self.equipment_info,
        )

        # CHANGED (V3): Restored V1 Logic for Private Tags
        # add source SeriesInstanceUID as private DICOM tag
        # tag identifier: 0019,1001, label: CCHMC Private, VR: UI
        block = ds.private_block(0x0019, "CCHMC Private", create=True)
        block.add_new(0x01, "UI", f"{dicom_series._series_instance_uid}") # 0x01 is the offset (0x1001 → offset = 0x01)

        # Secondary Capture specific tags
        ds.ImageType = ["DERIVED", "SECONDARY"]
        
        # Convert overlay_image to numpy array if it's an Image object
        from monai.deploy.core import Image
        if isinstance(overlay_image, Image):
            image_numpy = overlay_image.asnumpy()
        elif isinstance(overlay_image, np.ndarray):
            image_numpy = overlay_image
        else:
            raise ValueError(f"Unsupported overlay_image type: {type(overlay_image)}")
        
        # Handle 3D RGB image (multi-frame)
        # Expected formats: (Slices, Channels, Height, Width), (Slices, Height, Width, Channels), or (Channels, Slices, Height, Width)
        if image_numpy.ndim == 4:
            # Handle (Channels, Slices, Height, Width) -> (Slices, Channels, Height, Width)
            if image_numpy.shape[0] == 3:
                image_numpy = np.transpose(image_numpy, (1, 0, 2, 3))
            # Check if channels are in position 1 or 3
            if image_numpy.shape[1] == 3:
                # Format: (Slices, 3, Height, Width) -> (Slices, Height, Width, 3)
                image_numpy = np.transpose(image_numpy, (0, 2, 3, 1))
            elif image_numpy.shape[3] != 3:
                raise ValueError(f"Expected 3 channels for RGB, got shape: {image_numpy.shape}")
            # Now in format: (Slices, Height, Width, 3)
            num_frames, rows, cols, samples_per_pixel = image_numpy.shape
        elif image_numpy.ndim == 3:
            # Single frame: (3, Height, Width) or (Height, Width, 3)
            if image_numpy.shape[0] == 3:
                image_numpy = np.transpose(image_numpy, (1, 2, 0))
            rows, cols, samples_per_pixel = image_numpy.shape
            num_frames = 1
            # Add frame dimension: (Height, Width, 3) -> (1, Height, Width, 3)
            image_numpy = image_numpy[np.newaxis, ...]
        else:
            raise ValueError(f"Unexpected image dimensions: {image_numpy.shape}")
        
        # Ensure uint8 data type for RGB
        if image_numpy.dtype != np.uint8:
            # Normalize to 0-255 range if needed
            if image_numpy.max() <= 1.0:
                image_numpy = (image_numpy * 255).astype(np.uint8)
            else:
                image_numpy = image_numpy.astype(np.uint8)
        
        # Set image-specific DICOM tags for RGB Secondary Capture
        ds.Rows = rows
        ds.Columns = cols
        ds.SamplesPerPixel = samples_per_pixel
        ds.PhotometricInterpretation = "RGB"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0  # unsigned
        ds.PlanarConfiguration = 0  # interleaved RGB (R1G1B1R2G2B2...)
        ds.NumberOfFrames = num_frames
        
        # Set the pixel data (flatten all frames)
        ds.PixelData = image_numpy.tobytes()
        
        # Generate unique SOP Instance UID for this instance
        sc_sop_instance_uid = generate_uid()
        ds.SOPInstanceUID = sc_sop_instance_uid
        
        # Add date and time stamps
        dt_now = datetime.datetime.now()
        ds.SeriesDate = dt_now.strftime("%Y%m%d")
        ds.SeriesTime = dt_now.strftime("%H%M%S")
        ds.ContentDate = dt_now.strftime("%Y%m%d")
        ds.ContentTime = dt_now.strftime("%H%M%S")
        ds.TimezoneOffsetFromUTC = (
            dt_now.astimezone().isoformat()[-6:].replace(":", "")
        )
        
        # Apply custom tags if provided
        if self.custom_tags:
            for tag_keyword, tag_value in self.custom_tags.items():
                try:
                    if hasattr(ds, tag_keyword):
                        setattr(ds, tag_keyword, tag_value)
                        self._logger.info(f"Custom tag {tag_keyword} set to: {tag_value}")
                    else:
                        self._logger.warning(f"Unknown DICOM tag keyword: {tag_keyword}")
                except Exception as ex:
                    self._logger.warning(f"Failed to set custom tag {tag_keyword}: {ex}")
        
        # Ensure required UIDs are set
        if not hasattr(ds, 'SeriesInstanceUID') or not ds.SeriesInstanceUID:
            ds.SeriesInstanceUID = generate_uid()
        
        # Set file meta information
        file_meta = Dataset()
        file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = self.sop_class_uid
        file_meta.MediaStorageSOPInstanceUID = sc_sop_instance_uid
        file_meta.ImplementationClassUID = generate_uid()
        
        # Create output path
        output_path = output_folder / f"{sc_sop_instance_uid}{DICOMSCWriterOperator.DCM_EXTENSION}"
        
        # Create FileDataset and save
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        
        # Save the DICOM file
        ds.save_as(output_path, write_like_original=False)
        
        self._logger.info(f"DICOM Secondary Capture saved to: {output_path}")
        self._logger.info(f"Number of frames: {num_frames}, Dimensions: {rows}x{cols}, Channels: {samples_per_pixel}")
        
        # Verify the file was created
        try:
            if output_path.exists():
                self._logger.info(f"File size: {output_path.stat().st_size} bytes")
            else:
                self._logger.error(f"File was not created: {output_path}")
        except Exception as ex:
            self._logger.warning(f"Could not verify output file: {ex}")


def test():
    from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
    from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
    
    current_file_dir = Path(__file__).parent.resolve()
    # Update data_path to point to a valid DICOM series folder on your system for testing metadata copy
    data_path = current_file_dir.joinpath("../../../inputs/livertumor_ct/dcm/1-CT_series_liver_tumor_from_nii014")
    out_path = Path("output_sc_op").absolute()
    
    # 1. Generate Synthetic RGB Image (Slices, Channels, H, W) -> (2, 3, 256, 256)
    # This simulates a 2-frame RGB overlay
    print("Generating synthetic RGB image...")
    dummy_image = np.random.randint(0, 255, size=(2, 3, 256, 256), dtype=np.uint8)
    
    # 2. Setup Operators
    fragment = Fragment()
    loader = DICOMDataLoaderOperator(fragment, name="loader_op")
    series_selector = DICOMSeriesSelectorOperator(fragment, name="selector_op")
    sc_writer = DICOMSCWriterOperator(
        fragment,
        output_folder=out_path,
        model_info=None,
        equipment_info=EquipmentInfo(),
        custom_tags={"SeriesDescription": "Secondary Capture from AI Algorithm"},
        name="sc_writer"
    )

    # 3. Load DICOM Series (if available)
    dicom_series = None
    try:
        print(f"Loading DICOM series from: {data_path}")
        study_list = loader.load_data_to_studies(Path(data_path).absolute())
        study_selected_series_list = series_selector.filter(None, study_list)
        
        if study_selected_series_list and len(study_selected_series_list) > 0:
            for study_selected_series in study_selected_series_list:
                for selected_series in study_selected_series.selected_series:
                    dicom_series = selected_series.series
                    break
            print("DICOM Series loaded successfully.")
        else:
            print("Warning: No DICOM series found. Test will fail if Series metadata is required.")
            # Create a dummy series object if needed for robust testing without files, 
            # but for now we assume files exist or we accept failure.
    except Exception as e:
        print(f"Skipping DICOM loading due to environment error: {e}")

    # 4. Run Writer Logic directly
    print(f"Writing Secondary Capture to {out_path}...")
    try:
        if dicom_series:
             sc_writer.process_images(dummy_image, dicom_series, out_path)
             print("Test Success: DICOM SC written.")
        else:
             print("Test Aborted: No valid input DICOM series available to copy metadata from.")
    except Exception as e:
        print(f"Test Failed during write: {e}")

if __name__ == "__main__":
    test()