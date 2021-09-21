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
import math

import numpy as np

import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, Operator, OutputContext
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator


@md.input("dicom_series", DICOMSeries, IOType.IN_MEMORY)
@md.output("image", Image, IOType.IN_MEMORY)
class DICOMSeriesToVolumeOperator(Operator):
    """This operator converts an instance of DICOMSeries into an Image object.

    The loaded Image Object can be used for further processing via other operators.
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        """Extracts the pixel data from a DICOM Series and other attributes to for an instance of Image object"""
        dicom_series = op_input.get()
        self.prepare_series(dicom_series)
        metadata = self.create_metadata(dicom_series)

        voxel_data = self.generate_voxel_data(dicom_series)
        image = self.create_volumetric_image(voxel_data, metadata)
        op_output.set(image, "image")

    def generate_voxel_data(self, series):
        """Applies rescale slope and rescale intercept to the pixels.

        Args:
            series: DICOM Series for which the pixel data needs to be extracted.

        Returns:
            A 3D numpy tensor rerepesenting the volumetric data.
        """
        slices = series.get_sop_instances()
        # Need to transpose the DICOM pixel_array and pack the slice as the last dim.
        # This is to have the same numpy ndarray as from Monai ImageReader (ITK, NiBabel etc).
        vol_data = np.stack([np.transpose(s.get_pixel_array()) for s in slices], axis=-1)
        vol_data = vol_data.astype(np.int16)
        intercept = slices[0][0x0028, 0x1052].value
        slope = slices[0][0x0028, 0x1053].value

        if slope != 1:
            vol_data = slope * vol_data.astype(np.float64)
            vol_data = vol_data.astype(np.int16)
        vol_data += np.int16(intercept)
        return np.array(vol_data, dtype=np.int16)

    def create_volumetric_image(self, vox_data, metadata):
        """Creates an instance of 3D image.

        Args:
            vox_data: A numpy array representing the volumetric data.
            metadata: DICOM attributes in a dictionary.

        Returns:
            An instance of Image object.
        """
        image = Image(vox_data, metadata)
        return image

    def prepare_series(self, series):
        """Computes the slice normal for each slice and then projects the first voxel of each
        slice on that slice normal.

        It computes the distance of that point from the origin of the aient coordinate system along the alice normal.
        It orders the slices in the series according to that distance.

        Args:
            series: An instance of DICOMSeries.
        """

        if len(series._sop_instances) <= 1:
            return

        slice_indices_to_be_removed = []
        depth_pixel_spacing = 0.0
        last_slice_normal = [0.0, 0.0, 0.0]

        for slice_index, slice in enumerate(series._sop_instances):
            distance = 0.0
            point = [0.0, 0.0, 0.0]
            slice_normal = [0.0, 0.0, 0.0]
            slice_position = None
            cosines = None

            try:
                image_orientation_patient_de = slice[0x0020, 0x0037]
                if image_orientation_patient_de is not None:
                    image_orientation_patient = image_orientation_patient_de.value
                    cosines = image_orientation_patient
            except KeyError:
                pass

            try:
                image_poisition_patient_de = slice[0x0020, 0x0032]
                if image_poisition_patient_de is not None:
                    image_poisition_patient = image_poisition_patient_de.value
                    slice_position = image_poisition_patient
            except KeyError:
                pass

            distance = 0.0

            if (cosines is not None) and (slice_position is not None):
                slice_normal[0] = cosines[1] * cosines[5] - cosines[2] * cosines[4]
                slice_normal[1] = cosines[2] * cosines[3] - cosines[0] * cosines[5]
                slice_normal[2] = cosines[0] * cosines[4] - cosines[1] * cosines[3]

                last_slice_normal = copy.deepcopy(slice_normal)

                i = 0
                while i < 3:
                    point[i] = slice_normal[i] * slice_position[i]
                    i += 1

                distance += point[0] + point[1] + point[2]

                series._sop_instances[slice_index].distance = distance
                series._sop_instances[slice_index].first_pixel_on_slice_normal = point
            else:
                print("going to removing slice ", slice_index)
                slice_indices_to_be_removed.append(slice_index)

        for sl_index, _ in enumerate(slice_indices_to_be_removed):
            del series._sop_instances[sl_index]

        series._sop_instances = sorted(series._sop_instances, key=lambda s: s.distance)
        series.depth_direction_cosine = copy.deepcopy(last_slice_normal)

        if len(series._sop_instances) > 1:
            p1 = series._sop_instances[0].first_pixel_on_slice_normal
            p2 = series._sop_instances[1].first_pixel_on_slice_normal
            depth_pixel_spacing = (
                (p1[0] - p2[0]) * (p1[0] - p2[0])
                + (p1[1] - p2[1]) * (p1[1] - p2[1])
                + (p1[2] - p2[2]) * (p1[2] - p2[2])
            )
            depth_pixel_spacing = math.sqrt(depth_pixel_spacing)
            series.depth_pixel_spacing = depth_pixel_spacing

        s_1 = series._sop_instances[0]
        s_n = series._sop_instances[-1]
        num_slices = len(series._sop_instances)
        self.compute_affine_transform(s_1, s_n, num_slices, series)

    def compute_affine_transform(self, s_1, s_n, n, series):
        """Computes the affine transform for this series. It does it in both DICOM Patient oriented
        coordinate system as well as the pne preferred by NIFTI standard. Accordingly, the two attributes
        dicom_affine_transform and nifti_affine_transform are stored in the series instance.

        The Image Orientation Patient contains two triplets, [rx ry rz cx cy cz], which encode
        direction cosines of the row and column of an image slice. The Image Position Patient of the first slice in
        a volume, [x1 y1 z1], is the x, y, z coordinates of the upper-left corner voxel of the slice. These two
        parameters define the location of the slice in PCS. To determine the location of a volume, the Image
        Position Patient of another slice is normally needed. In practice, we tend to use the position of the last
        slice in a volume, [xn yn zn]. The voxel size within the slice plane, [vr vc], is stored in object Pixel Spacing.

        Args:
            s_1: A first slice in the series.
            s_n: A last slice in the series.
            n: A number of slices in the series.
            series: An instance of DICOMSeries.
        """

        m1 = np.arange(1, 17, dtype=float).reshape(4, 4)
        m2 = np.arange(1, 17, dtype=float).reshape(4, 4)

        image_orientation_patient = None
        try:
            image_orientation_patient_de = s_1[0x0020, 0x0037]
            if image_orientation_patient_de is not None:
                image_orientation_patient = image_orientation_patient_de.value
        except KeyError:
            pass
        rx = image_orientation_patient[0]
        ry = image_orientation_patient[1]
        rz = image_orientation_patient[2]
        cx = image_orientation_patient[3]
        cy = image_orientation_patient[4]
        cz = image_orientation_patient[5]

        vr = 0.0
        vc = 0.0
        try:
            pixel_spacing_de = s_1[0x0028, 0x0030]
            if pixel_spacing_de is not None:
                vr = pixel_spacing_de.value[0]
                vc = pixel_spacing_de.value[1]
        except KeyError:
            pass

        x1 = 0.0
        y1 = 0.0
        z1 = 0.0

        xn = 0.0
        yn = 0.0
        zn = 0.0

        ip1 = None
        ip2 = None
        try:
            ip1_de = s_1[0x0020, 0x0032]
            ipn_de = s_n[0x0020, 0x0032]
            ip1 = ip1_de.value
            ipn = ipn_de.value

        except KeyError:
            pass

        x1 = ip1[0]
        y1 = ip1[1]
        z1 = ip1[2]

        xn = ipn[0]
        yn = ipn[1]
        zn = ipn[2]

        m1[0, 0] = rx * vr
        m1[0, 1] = cx * vc
        m1[0, 2] = (xn - x1) / (n - 1)
        m1[0, 3] = x1

        m1[1, 0] = ry * vr
        m1[1, 1] = cy * vc
        m1[1, 2] = (yn - y1) / (n - 1)
        m1[1, 3] = y1

        m1[2, 0] = rz * vr
        m1[2, 1] = cz * vc
        m1[2, 2] = (zn - z1) / (n - 1)
        m1[2, 3] = z1

        m1[3, 0] = 0
        m1[3, 1] = 0
        m1[3, 2] = 0
        m1[3, 3] = 1

        series.dicom_affine_transform = m1

        m2[0, 0] = -rx * vr
        m2[0, 1] = -cx * vc
        m2[0, 2] = -(xn - x1) / (n - 1)
        m2[0, 3] = -x1

        m2[1, 0] = -ry * vr
        m2[1, 1] = -cy * vc
        m2[1, 2] = -(yn - y1) / (n - 1)
        m2[1, 3] = -y1

        m2[2, 0] = rz * vr
        m2[2, 1] = cz * vc
        m2[2, 2] = (zn - z1) / (n - 1)
        m2[2, 3] = z1

        m2[3, 0] = 0
        m2[3, 1] = 0
        m2[3, 2] = 0
        m2[3, 3] = 1

        series.nifti_affine_transform = m2

    def create_metadata(self, series):
        """Collects all relevant metadata from the DICOM Series and creates a dictionary.

        Args:
            series: An instance of DICOMSeries.

        Returns:
            An instance of a dictionary containing metadata for the volumetric image.
        """
        metadata = {}
        metadata["series_instance_uid"] = series.get_series_instance_uid()

        if series.series_date is not None:
            metadata["series_date"] = series.series_date

        if series.series_time is not None:
            metadata["series_time"] = series.series_time

        if series.modality is not None:
            metadata["modality"] = series.modality

        if series.series_description is not None:
            metadata["series_description"] = series.series_description

        if series.row_pixel_spacing is not None:
            metadata["row_pixel_spacing"] = series.row_pixel_spacing

        if series.col_pixel_spacing is not None:
            metadata["col_pixel_spacing"] = series.col_pixel_spacing

        if series.depth_pixel_spacing is not None:
            metadata["depth_pixel_spacing"] = series.depth_pixel_spacing

        if series.row_direction_cosine is not None:
            metadata["row_direction_cosine"] = series.row_direction_cosine

        if series.col_direction_cosine is not None:
            metadata["col_direction_cosine"] = series.col_direction_cosine

        if series.depth_direction_cosine is not None:
            metadata["depth_direction_cosine"] = series.depth_direction_cosine

        if series.dicom_affine_transform is not None:
            metadata["dicom_affine_transform"] = series.dicom_affine_transform

        if series.nifti_affine_transform is not None:
            metadata["nifti_affine_transform"] = series.nifti_affine_transform

        return metadata


def main():
    op = DICOMSeriesToVolumeOperator()
    # data_path = "/home/rahul/medical-images/mixed-data/"
    # data_path = "/home/rahul/medical-images/lung-ct-2/"
    data_path = "/home/rahul/medical-images/spleen-ct/"
    files = []
    loader = DICOMDataLoaderOperator()
    loader._list_files(data_path, files)
    study_list = loader._load_data(files)

    series = study_list[0].get_all_series()[0]
    op.prepare_series(series)
    voxels = op.generate_voxel_data(series)
    metadata = op.create_metadata(series)
    image = op.create_volumetric_image(voxels, metadata)

    print(series)
    print(metadata.keys())


if __name__ == "__main__":
    main()
