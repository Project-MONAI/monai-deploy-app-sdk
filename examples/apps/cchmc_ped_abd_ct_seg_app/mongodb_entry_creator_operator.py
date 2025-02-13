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

import logging
from datetime import datetime
from typing import Any, Dict, Union

import pydicom
import pytz

from monai.deploy.core import Fragment, Operator, OperatorSpec
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries


class MongoDBEntryCreatorOperator(Operator):
    """Class to create a database entry for downstream MONAI Deploy Express MongoDB database writing.
    Provided text input and source DICOM Series DICOM tags are used to create the entry.

    Named inputs:
        text: text content to be included in the database entry.
        study_selected_series_list: DICOM series for copying metadata from.

    Named output:
        mongodb_database_entry: formatted MongoDB database entry. Downstream receiver MongoDBWriterOperator will write
        the entry to the MONAI Deploy Express MongoDB database.
    """

    def __init__(self, fragment: Fragment, *args, map_version: str, **kwargs):
        """Class to create a MONAI Deploy Express MongoDB database entry. Provided text input and
        source DICOM Series DICOM tags are used to create the entry.

        Args:
            map_version (str): version of the MAP.

        Raises:
            ValueError: If result cannot be found either in memory or from file.
        """

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

        self.map_version = map_version

        self.input_name_text = "text"
        self.input_name_dcm_series = "study_selected_series_list"

        self.output_name_db_entry = "mongodb_database_entry"

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Set up the named input(s), and output(s).

        Args:
            spec (OperatorSpec): The Operator specification for inputs and outputs etc.
        """

        spec.input(self.input_name_text)
        spec.input(self.input_name_dcm_series)

        spec.output(self.output_name_db_entry)

    def compute(self, op_input, op_output, context):
        """Performs computation for this operator and handles I/O.

        For now, only a single result content is supported, which could be in memory or an accessible file.
        The DICOM Series used during inference is required.

        When there are multiple selected series in the input, the first series' containing study will
        be used for retrieving DICOM Study module attributes, e.g. StudyInstanceUID.

        Raises:
            FileNotFoundError: When result object not in the input, and result file not found either.
            ValueError: Content object and file path not in the inputs, or no DICOM series provided.
            IOError: If the input content is blank.
        """

        # receive the result text and study selected series list
        result_text = str(op_input.receive(self.input_name_text)).strip()
        if not result_text:
            raise IOError("Input is read but blank.")

        study_selected_series_list = None
        try:
            study_selected_series_list = op_input.receive(self.input_name_dcm_series)
        except Exception:
            pass
        if not study_selected_series_list or len(study_selected_series_list) < 1:
            raise ValueError("Missing input, list of 'StudySelectedSeries'.")

        # retrieve the DICOM Series used during inference in order to grab appropriate Study/Series level tags
        # this will be the 1st Series in study_selected_series_list
        dicom_series = None
        for study_selected_series in study_selected_series_list:
            if not isinstance(study_selected_series, StudySelectedSeries):
                raise ValueError(f"Element in input is not expected type, {StudySelectedSeries}.")
            selected_series = study_selected_series.selected_series[0]
            dicom_series = selected_series.series
            break

        # create MongoDB entry
        mongodb_database_entry = self.create_entry(result_text, dicom_series, self.map_version)

        # emit MongoDB entry
        op_output.emit(mongodb_database_entry, self.output_name_db_entry)

    def create_entry(self, result_text: str, dicom_series: DICOMSeries, map_version: str):
        """Creates the MONAI Deploy Express MongoDB database entry.

        Args:
            result_text (str): text content to be included in the database entry.
            dicom_series (DICOMSeries): DICOMSeries object encapsulating the original series.
            map_version (str): version of the MAP.

        Returns:
            mongodb_database_entry: formatted MongoDB database entry.
        """

        if not result_text or not len(result_text.strip()):
            raise ValueError("Content is empty.")

        # get one of the SOP instance's native sop instance dataset
        # we will pull Study level (and some Series level) DICOM tags from this SOP instance
        # this same strategy is employed by write_common_modules
        orig_ds = dicom_series.get_sop_instances()[0].get_native_sop_instance()

        # # loop through dicom series tags; look for discrepancies from SOP instances
        # for sop_instance in dicom_series.get_sop_instances():
        #     # get the native SOP instance dataset
        #     dicom_image = sop_instance.get_native_sop_instance()

        #     # check if the tag is present in the dataset
        #     if hasattr(dicom_image, 'Exposure'):
        #         tag = dicom_image.Exposure
        #         print(f"Exposure: {tag}")
        #     else:
        #         print("Exposure tag not found in this SOP instance.")

        # DICOM TAG WRITING TO MONGODB
        # edge cases addressed by looking at DICOM tag Type, Value Representation (VR),
        # and Value Multiplicity (VM) specifically for the CT Image CIOD
        # https://dicom.innolitics.com/ciods/ct-image

        # define Tag Absent variable
        tag_absent = "Tag Absent"

        # STUDY AND SERIES LEVEL DICOM TAGS

        # AccessionNumber - Type: Required (2), VR: SH, VM: 1
        accession_number = orig_ds.AccessionNumber

        # StudyInstanceUID - Type: Required (1), VR: UI, VM: 1
        study_instance_uid = orig_ds.StudyInstanceUID

        # StudyDescription: Type: Optional (3), VR: LO, VM: 1
        # while Optional, only studies with this tag will be routed from Compass and MAP launched per workflow def
        study_description = orig_ds.get("StudyDescription", tag_absent)

        # SeriesInstanceUID: Type: Required (1), VR: UI, VM: 1
        series_instance_uid = dicom_series._series_instance_uid

        # SeriesDescription: Type: Optional (3), VR: LO, VM: 1
        series_description = orig_ds.get("SeriesDescription", tag_absent)

        # sop instances should always be available on the MONAI DICOM Series object
        series_sop_instances = len(dicom_series._sop_instances)

        # PATIENT DETAIL DICOM TAGS

        # PatientID - Type: Required (2), VR: LO, VM: 1
        patient_id = orig_ds.PatientID

        # PatientName - Type: Required (2), VR: PN, VM: 1
        # need to convert to str; pydicom can't encode PersonName object
        patient_name = str(orig_ds.PatientName)

        # PatientSex - Type: Required (2), VR: CS, VM: 1
        patient_sex = orig_ds.PatientSex

        # PatientBirthDate - Type: Required (2), VR: DA, VM: 1
        patient_birth_date = orig_ds.PatientBirthDate

        # PatientAge - Type: Optional (3), VR: AS, VM: 1
        patient_age = orig_ds.get("PatientAge", tag_absent)

        # EthnicGroup - Type: Optional (3), VR: SH, VM: 1
        ethnic_group = orig_ds.get("EthnicGroup", tag_absent)

        # SCAN ACQUISITION PARAMETER DICOM TAGS

        # on CCHMC test cases, the following tags had consistent values for all SOP instances

        # Manufacturer - Type: Required (2), VR: LO, VM: 1
        manufacturer = orig_ds.Manufacturer

        # ManufacturerModelName - Type: Optional (3), VR: LO, VM: 1
        manufacturer_model_name = orig_ds.get("ManufacturerModelName", tag_absent)

        # BodyPartExamined - Type: Optional (3), VR: CS, VM: 1
        body_part_examined = orig_ds.get("BodyPartExamined", tag_absent)

        # row and column pixel spacing are derived from PixelSpacing
        # PixelSpacing - Type: Required (1), VR: DS, VM: 2 (handled by MONAI)
        row_pixel_spacing = dicom_series._row_pixel_spacing
        column_pixel_spacing = dicom_series._col_pixel_spacing

        # per DICOMSeriesToVolumeOperator, depth pixel spacing will always be defined
        depth_pixel_spacing = dicom_series._depth_pixel_spacing

        # SliceThickness - Type: Required (2), VR: DS, VM: 1
        slice_thickness = orig_ds.SliceThickness

        # PixelRepresentation - Type: Required (1), VR: US, VM: 1
        pixel_representation = orig_ds.PixelRepresentation

        # BitsStored - Type: Required (1), VR: US, VM: 1
        bits_stored = orig_ds.BitsStored

        # WindowWidth - Type: Conditionally Required (1C), VR: DS, VM: 1-n
        window_width = orig_ds.get("WindowWidth", tag_absent)
        # for MultiValue case:
        if isinstance(window_width, pydicom.multival.MultiValue):
            # join multiple values into a single string separated by a |
            # convert DSfloat objects to strs to allow joining
            window_width = " | ".join([str(window) for window in window_width])

        # RevolutionTime - Type: Optional (3), VR: FD, VM: 1
        revolution_time = orig_ds.get("RevolutionTime", tag_absent)

        # FocalSpots - Type: Optional (3), VR: DS, VM: 1-n
        focal_spots = orig_ds.get("FocalSpots", tag_absent)
        # for MultiValue case:
        if isinstance(focal_spots, pydicom.multival.MultiValue):
            # join multiple values into a single string separated by a |
            # convert DSfloat objects to strs to allow joining
            focal_spots = " | ".join([str(spot) for spot in focal_spots])

        # SpiralPitchFactor - Type: Optional (3), VR: FD, VM: 1
        spiral_pitch_factor = orig_ds.get("SpiralPitchFactor", tag_absent)

        # ConvolutionKernel - Type: Optional (3), VR: SH, VM: 1-n
        convolution_kernel = orig_ds.get("ConvolutionKernel", tag_absent)
        # for MultiValue case:
        if isinstance(convolution_kernel, pydicom.multival.MultiValue):
            # join multiple values into a single string separated by a |
            convolution_kernel = " | ".join(convolution_kernel)

        # ReconstructionDiameter - Type: Optional (3), VR: DS, VM: 1
        reconstruction_diameter = orig_ds.get("ReconstructionDiameter", tag_absent)

        # KVP - Type: Required (2), VR: DS, VM: 1
        kvp = orig_ds.KVP

        # on CCHMC test cases, the following tags did NOT have consistent values for all SOP instances
        # as such, if the tag value exists, it will be averaged over all SOP instances

        # initialize an averaged values dictionary
        averaged_values: Dict[str, Union[float, str]] = {}

        # tags to check and average
        tags_to_average = {
            "XRayTubeCurrent": tag_absent,  # Type: Optional (3), VR: IS, VM: 1
            "Exposure": tag_absent,  # Type: Optional (3), VR: IS, VM: 1
            "CTDIvol": tag_absent,  # Type: Optional (3), VR: FD, VM: 1
        }

        # check which tags are present on the 1st SOP instance
        for tag, default_value in tags_to_average.items():
            # if the tag exists
            if tag in orig_ds:
                # loop through SOP instances, grab tag values
                values = []
                for sop_instance in dicom_series.get_sop_instances():
                    ds = sop_instance.get_native_sop_instance()
                    value = ds.get(tag, default_value)
                    # if tag is present on current SOP instance
                    if value != default_value:
                        # add tag value to values; convert to float for averaging
                        values.append(float(value))
                # compute the average if values were collected
                if values:
                    averaged_values[tag] = round(sum(values) / len(values), 3)
                else:
                    averaged_values[tag] = default_value
            else:
                # if the tag is absent in the first SOP instance, keep the default value
                averaged_values[tag] = default_value

        # parse result_text (i.e. predicted organ volumes) and format
        map_results = {}
        for line in result_text.split("\n"):
            if ":" in line:
                key, value = line.split(":")
                key = key.replace(" ", "")
                map_results[key] = value.strip()

        # create the MongoDB database entry
        mongodb_database_entry: Dict[str, Any] = {
            "Timestamp": datetime.now(pytz.UTC),  # timestamp in UTC
            "MAPVersion": map_version,
            "DICOMSeriesDetails": {
                "AccessionNumber": accession_number,
                "StudyInstanceUID": study_instance_uid,
                "StudyDescription": study_description,
                "SeriesInstanceUID": series_instance_uid,
                "SeriesDescription": series_description,
                "SeriesFileCount": series_sop_instances,
            },
            "PatientDetails": {
                "PatientID": patient_id,
                "PatientName": patient_name,
                "PatientSex": patient_sex,
                "PatientBirthDate": patient_birth_date,
                "PatientAge": patient_age,
                "EthnicGroup": ethnic_group,
            },
            "ScanAcquisitionDetails": {
                "Manufacturer": manufacturer,
                "ManufacturerModelName": manufacturer_model_name,
                "BodyPartExamined": body_part_examined,
                "RowPixelSpacing": row_pixel_spacing,
                "ColumnPixelSpacing": column_pixel_spacing,
                "DepthPixelSpacing": depth_pixel_spacing,
                "SliceThickness": slice_thickness,
                "PixelRepresentation": pixel_representation,
                "BitsStored": bits_stored,
                "WindowWidth": window_width,
                "RevolutionTime": revolution_time,
                "FocalSpots": focal_spots,
                "SpiralPitchFactor": spiral_pitch_factor,
                "ConvolutionKernel": convolution_kernel,
                "ReconstructionDiameter": reconstruction_diameter,
                "KVP": kvp,
            },
            "MAPResults": map_results,
        }

        # integrate averaged tags into MongoDB entry:
        mongodb_database_entry["ScanAcquisitionDetails"].update(averaged_values)

        return mongodb_database_entry
