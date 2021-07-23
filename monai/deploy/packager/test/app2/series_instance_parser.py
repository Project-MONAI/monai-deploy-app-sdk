# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import pydicom
import logging
from pathlib import Path
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
import SimpleITK as sitk

class SeriesInstanceParser(object):
    """Parses DICOM instance(s) in Part 10 format to capture DICOM datasets.
    """

    # DICOM instance file extension. Case insentiive in string comaprision.
    DCM_EXTENSION = '.dcm'

    def __init__(self):
        """Initializes the object.
        """

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

    def parse(self, path_root, series_uid=None):
        """Parses all dicom files under the path to get a list of Datasets.
        
        Args:
            path_root (str): Path to a folder containing DICOM files or a single file.
            series_uid (str): Parse only this series if instance UID is given, otherwise all.

        Returns:
            list of PyDicom Datasets
        """
        
        if path_root is None:
            raise ValueError('Argument cannot be None.')
        if not os.path.isdir(path_root) and not os.path.isfile(path_root):
            raise IOError('The path_root is not a folder or file.')

        dcm_file_paths = self._list_dcm_files_sorted(path_root, series_uid)

        ds_list = []
        for path in dcm_file_paths:
            ds_list.append(self._parse_dicom_file(path))

        return ds_list

    def _list_dcm_files_unsorted(self, path_root):
        """Lists all dcm files in the path, unsorted.

        Args:
            path_root (str): Path to the folder or the file

        Return:
            list of dcm file path strings
        """

        if not os.path.exists(path_root):
            raise IOError('File or folder not found: {}'.format(path_root))

        dcm_files = []
        if os.path.isdir(path_root):
            for matched_file in Path(path_root).rglob('*'):
                # Need to get the file name in str
                f_name = str(matched_file)
                if self._is_dcm_file(f_name):
                    dcm_files.append(f_name)
                else:
                    self._logger.warn('Ignoring non dcm file: {}'.format(f_name))
        elif os.path.isfile(path_root):
            if self._is_dcm_file(path_root):
                dcm_files.append(path_root)
            else:
                self._logger.warn('Ignoring non dcm file: {}'.format(path_root))

        return dcm_files

    def _is_dcm_file(self, file_name):
        """Determines if a file is dcm, based on the extension only."""

        if isinstance(file_name, str):
            return os.path.isfile(file_name) and file_name.casefold().endswith(SeriesInstanceParser.DCM_EXTENSION.casefold())

        #All other cases should be false
        return False

    def _list_dcm_files_sorted(self, path_root, series_uid=None):
        """Lists all dcm files in the path, sorted with multiple strateies.

        The underlying library first uses DICOM metadata Image Position Patient and Image Orientation Patient,
        if fails, then DICOM instance number, if fails, then file name lexicographically.

        The instance files for the same series must be in the same folder.

        Args:
            path_root (str): Path to the folder or the file
            series_uid (str): Parse only this series if instance UID is given, otherwise all.

        Return:
            list of dcm file path strings
        """

        # Finds all the folders containing dcm files, and in each folder, find the series IDs and then dcm files.
        # It is assumed that all instances for a specific series are in the same folder.

        unsorted_dcm_files = self._list_dcm_files_unsorted(path_root)
        if len(unsorted_dcm_files) <= 1:
            return unsorted_dcm_files

        dcm_folders = set()
        for file_name in unsorted_dcm_files:
            dcm_folders.add(os.path.dirname(str(file_name)))

        if len(dcm_folders) > 1:
            self._logger.warn("More than one folder contains dcm files:{}".format(dcm_folders))

        dcm_files = []
        found_specific_series = False

        for folder in dcm_folders:
            self._logger.info("Finding series in the given directory, {}.".format(folder))
            series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder)
            if not series_IDs:
                self._logger.warn("No series found in the given directory, {}.".format(folder))
                continue

            for series_ID in series_IDs:
                found_specific_series = (series_uid.casefold() == series_ID.casefold()) \
                    if series_uid else False
                self._logger.info("Getting sorted dcm files for series ID: {}".format(series_ID))
                series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder, series_ID)
                dcm_files.extend(series_file_names)
                if found_specific_series:
                    break

            if found_specific_series:
                break

        return dcm_files

    def _parse_dicom_file(self, file_path):
        """Parse a single DICOM Part 10 file

        Args:
            file_path (str): path of the dcm file
        
        Returns:
            PyDicom Dataset
        """

        self._logger.info('Parsing file: {}.'.format(file_path))

        if file_path is None:
            raise ValueError('Argument is none or empty.')
        if not os.path.isfile(file_path):
            raise IOError('File expected but not found: {}'.format(file_path))
        
        return dcmread(file_path)