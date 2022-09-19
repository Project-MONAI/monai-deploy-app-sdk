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

import glob
import os

import pydicom
from pydicom.uid import generate_uid
from datetime import datetime

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext
from app.inference import DetectionResultList


@md.input("original_dicom", DataPath, IOType.DISK)
@md.input("detection_predictions", DetectionResultList, IOType.IN_MEMORY)
@md.output("gsps_files", DataPath, IOType.DISK)
class GenerateGSPS(Operator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_gsps(self, image_ds: pydicom.Dataset, gsps_ds: pydicom.Dataset):

        gsps_ds.add_new(0x00080016, 'UI', '1.2.840.10008.5.1.4.1.1.11.1')  # SOPClassUID ==  GSPS
        gsps_ds.add_new(0x0020000D, 'UI', _studyInstanceUid)  # StudyInstanceUID
        gsps_ds.add_new(0x0020000E, 'UI', _seriesInstanceUID)  # SeriesInstanceUID
        gsps_ds.add_new(0x00080018, 'UI', generate_uid())  # SOP Instance UID

        gsps_ds.add_new(0x0010020, 'LO', _patientId)  # PatientId
        gsps_ds.add_new(0x00100010, 'PN', _patientName)  # PatientName
        gsps_ds.add_new(0x00100030, 'DA', _patientBirthDate)  # PatientBirthdate
        gsps_ds.add_new(0x00100040, 'CS', _patientSex)  # PatientSex

        gsps_ds.add_new(0x00080020, 'DA', _studyDateTime)  # StudyDate
        gsps_ds.add_new(0x00080030, 'TM', _studyDateTime)  # StudyTime
        gsps_ds.add_new(0x00080050, 'SH', _accessionNumber)  # AccessionNumber
        gsps_ds.add_new(0x00080090, 'PN', _referringPhysicianName)  # PhysicianName
        gsps_ds.add_new(0x00200010, 'SH', _studyID)  # StudyID
        gsps_ds.add_new(0x00200011, 'IS', _seriesNumber)  # SeriesNumber

        gsps_ds.add_new(0x00080060, 'CS', _modality)  # Modality
        gsps_ds.add_new(0x00080070, 'LO', _manufacturer)  # Manufacturer

        gsps_ds.add_new(0x00700082, 'DA', datetime.utcnow().strftime("%Y%m%d"))  # PresentationCreationDate
        gsps_ds.add_new(0x00700083, 'TM', datetime.utcnow().strftime("%Y%m%d"))  # PresentationCreationTime

        series: pydicom.Dataset = pydicom.Dataset()
        series.add_new(0x0020000E, 'UI', _seriesInstanceUID)  # SeriesInstanceUID
        series.add_new(0x00081140, 'SQ', image_ds)  # ReferencedImageSequence

        gsps_ds.add_new(0x00081115, 'SQ', series)  # ReferencedSeriesSequence

        # add geometric annotations from detector findings
        displayedArea: pydicom.Dataset = pydicom.Dataset()
        displayedArea.add_new(0x00700052, 'SL', "50\\50")  # DisplayedAreaTopLeftHandCorner
        displayedArea.add_new(0x00700053, 'SL', "100\\100")  # DisplayedAreaBottomRightHandCorner
        displayedArea.add_new(0x00700100, 'CS', "SCALE TO FIT")  # PresentationSizeMode

        gsps_ds.add_new(0x0070005A, 'SQ', displayedArea)

        gsps_ds.add_new(0x00282000, 'OB', b'\x00\x00\x00\x01')  # ICCProfile

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        input_path = op_input.get("original_dicom").path
        dcm_files = glob.glob(f"{os.path.sep}".join([f"{input_path}", "**", "*.dcm"]), recursive=True)

        for dcm_file in dcm_files:
            ds = pydicom.dcmread(dcm_file)
            series_uid = ds[0x0020000D].value

            self.upload_gsps(
                os.path.join(self.output_path, dcm_file),
                series_uid=series_uid,
            )
