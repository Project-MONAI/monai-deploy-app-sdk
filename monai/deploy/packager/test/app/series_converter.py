# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import logging
import os
from urllib.parse import urlparse

import SimpleITK as sitk
import simplejson as json

from metadata_parser import DicomMetadata, DICOMParser, DicomSeries, DicomStudy, SOPInstance


class SeriesConverter:
    """This class transforms certain types of content of a DICOM sereis.

    For CT and MR modality type, the content of the instances in a series
    will be converted into volume image, in mhd, nii or nii.gz format.
    Radiography image for CR, DX, and IO, can also be converted to mhd, nii, nii.gz
    as well as PNG image.
    Images for other modality types will try to be converted and allow to fail.
    """

    MODALITIES_FOR_IMAGE_CONVERSION = ['CT', 'MR', 'CR', 'DX', 'IO']
    SUPPORTED_IMAGE_FORMAT = ['mhd', 'nii', 'nii.gz', 'png']

    def __init__(self):

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

    def series_to_image(self,
                        input_dir,
                        output_dir,
                        output_format='nii',
                        metadata=None,
                        selected_series_uids=None):
        """Creates a volume image file for applicable dicom series

        Searches the input folder for dicom series. With the images in each series
        writes an image file to the output folder.

        Args:
            input_dir (str): An existing folder containing dicom files
            output_dir (str): The folder where volume image files will be written
            output_format(str): The requested output file format, default is mhd.
            metadata (DicomMetadata): Input DICOM study metadata obhect
            selected_series_uids (list): convert image for these series instance UIDs
        """

        def _is_selected(series_uid):
            return True if selected_series_uids is None \
                else series_uid.casefold() in selected_series_uids

        self._logger.info('Converting series in {} to image files of type {} in folder {}.'.format(
            input_dir, output_format, output_dir))

        output_format = output_format.strip('.')
        if not output_format.casefold() in [x.casefold() for x in SeriesConverter.SUPPORTED_IMAGE_FORMAT]:
            raise ValueError('Output image format "{}" not supported'.format(output_format))

        # Input directory must already exists.
        SeriesConverter.ensure_dir_exists(input_dir, False)

        # The output directory can be created if it does not exist yet.
        SeriesConverter.ensure_dir_exists(output_dir, True)

        series_to_content_path = {}  # series_instance_UID : converted_content path
        # Rely on the metadata for series information to decide what to process
        if metadata and isinstance(metadata, DicomMetadata) and len(metadata.studies) > 0:
            for study in metadata.studies:
                for series in study.series:
                    try:
                        if not (_is_selected(series.SeriesInstanceUID)):
                            self._logger.info('Not selected for conversion, UID: {}'.format(series.SeriesInstanceUID))
                            continue

                        instance_paths = self.resolve_paths(series)
                        file_path = self.write_image(
                            series.SeriesInstanceUID,
                            instance_paths,
                            output_dir,
                            output_format)
                        series_to_content_path[series.SeriesInstanceUID] = file_path
                    except Exception:
                        if series.Modality.casefold() in \
                                [x.casefold() for x in SeriesConverter.MODALITIES_FOR_IMAGE_CONVERSION]:
                            self._logger.exception('Failed to convert {} series, UID {}'.format(
                                series.Modality, series.SeriesInstanceUID))
                            raise
                        # OK to have failed to convert instances of other Modality types.
                        self._logger.info('{} series, UID {}, not supported for image conversion.'.format(
                            series.Modality, series.SeriesInstanceUID))
        else:
            # This is a fallback when DicomMetadata is not available.
            # Find all series in the input dir and try to convert pixel for the given format.
            series_instance_dict = DICOMParser.find_series_and_instances(input_dir)
            if series_instance_dict and len(series_instance_dict.keys()) > 0:
                for s_uid, paths in series_instance_dict.items():
                    try:
                        file_path = self.write_image(
                            s_uid,
                            paths,
                            output_dir,
                            output_format)
                        series_to_content_path[s_uid] = file_path
                    except Exception:
                        self._logger.info('Series, UID {}, not supported for image conversion.'.format(s_uid))
            else:
                logging.warn('No DICOM series found.')

        self._logger.info('Created {} volume files:\n{}'.format(
            len(series_to_content_path),
            series_to_content_path))

        return series_to_content_path

    def write_image(self, series_id, instance_paths, output_dir, out_format='mhd'):
        """Writes a 3D image file with instances/2D images in the series

        This function uses SimpleITK to read in instances in a DICOM
        series and create a 3D image file, of the supported format.
        The file is saved to the output folder using the series uid as its name.

        Args:
            series_uid (str): The instance UID of the series.
            instance_paths (str): File paths of the DICOM instances of the series.
            output_dir (str): The folder where to save the image file.
            out_format (str): Format of the ocnverted image, extension only sans '.'

        Returns:
            The relative file path in the folder 'output_dir'
        """

        out_format = out_format.strip('.')
        if not out_format.casefold() in [x.casefold() for x in SeriesConverter.SUPPORTED_IMAGE_FORMAT]:
            raise ValueError('Output image format \"{}\" not supported'.format(out_format))

        self._logger.info('Series Instance UID {} with image file paths:\n{}'.format(series_id, instance_paths))

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(instance_paths)
        image = reader.Execute()

        # Save the image to a file in the output dir, i.e. <series_id>.mhd.
        # Image file format used is configurable at runtime
        image_file_name = '{0}.{1}'.format(series_id, out_format)
        image_path = os.path.join(output_dir, image_file_name)
        logging.info('Saving {} of size {}.'.format(image_path, image.GetSize()))
        sitk.WriteImage(image, image_path)
        self._logger.info('Image saved successfully in file {}'.format(image_path))

        # Return file path sans the output_dir
        return image_file_name

    def resolve_paths(self, series):
        """Resolve to local paths of the series's DICOM instance files
        """

        paths = []
        for sop_instance in series.sop_instances:
            # Make data locally available
            # Currently only local files are used.
            parsed = urlparse(sop_instance.RetrieveURL)
            local_path = parsed.path
            paths.append(local_path)

        self._logger.info("Resolved instance file paths for series instance UID {}:\n{}".format(
            series.SeriesInstanceUID,
            paths))

        return paths

    @staticmethod
    def ensure_dir_exists(dir, to_create=False):
        """ Check dir exists, and if not, try to create it if requested.

        Args:
            dir (str): Path of the directory
            to_create (bool): If true, create the directory if not found.
        """

        if not os.path.exists(dir):
            if not to_create:
                msg = 'Directory "{}" does not exist.'.format(dir)
                logging.error(msg)
                raise IOError(msg)
            else:
                logging.warning('Creating directory "{}".'.format(dir))
                try:
                    os.makedirs(dir)
                except Exception as ex:
                    logging.error('Failed to create directory due to: "{}"'.format(ex))
                    raise
