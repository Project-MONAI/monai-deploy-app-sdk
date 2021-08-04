# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pydicom
import SimpleITK as sitk
import simplejson as json
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence


class DICOMParser(object):
    """Parses DICOM instance(s) in Part 10 format to capture DICOM metadata.

    Attributes:
        meta_data (DicomMetadata): Metadata of DICOM study, seriese, and instances.
    """

    # DICOM instance file extension. Case insentiive in string comaprision.
    DCM_EXTENSION = '.dcm'

    def __init__(self):
        """Initializes the object with a path to a folder of or a single DICOM file.
        """

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.meta_data = None

    def parse(self, path_root):
        """Parses all dicom files to retrieve metadata.

        This method clears any previous metadata if any, and then parses the DICOM instances
        to capture the metadata.

        Args:
            path_root (str): Path to a folder containing DICOM files or a single file.
        """

        if path_root is None:
            raise ValueError('Argument cannot be None.')
        print(f"Metadata parser : {path_root}")
        if not os.path.isdir(path_root) and not os.path.isfile(path_root):
            raise IOError('The path_root is not a folder or file.')

        # Clear metadata first if any, by creating a new instance of metadata
        self.meta_data = DicomMetadata()

        series_instance_dict = DICOMParser.find_series_and_instances(path_root)

        if not series_instance_dict or len(series_instance_dict.keys()) < 1:
            self._logger.warn('No DICOM series found.')
            return

        for series_uid, file_paths in series_instance_dict.items():
            self._logger.info("Parsing Series UID: {} with instance files:\n{}".format(series_uid, file_paths))
            for file_path in file_paths:
                self._parse_dicom_file(file_path)

        self._show_stats()

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
            return os.path.isfile(file_name) and file_name.casefold().endswith(DICOMParser.DCM_EXTENSION.casefold())

        # All other cases should be false
        return False

    def _list_dcm_files_sorted(self, path_root):
        """Lists all dcm files in the path, sorted with multiple strateies.

        The underlying library first uses DICOM metadata Image Position Patient and Image Orientation Patient,
        if fails, then DICOM instance number, if fails, then file name lexicographically.

        The instance files for the same series must be in the same folder.

        Args:
            path_root (str): Path to the folder or the file

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
        for folder in dcm_folders:
            self._logger.info("Finding series in the given directory, {}.".format(folder))
            series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder)
            if not series_IDs:
                self._logger.warn("No series found in the given directory, {}.".format(folder))
            else:
                for series_ID in series_IDs:
                    self._logger.info("Getting sorted dcm files for series ID: {}".format(series_ID))
                    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder, series_ID)
                    dcm_files.extend(series_file_names)

        return dcm_files

    @staticmethod
    def find_series_and_instances(input_dir):
        """Find series and sorted instance paths for each series.

        Find each series in the input_dir, and the paths of its instances.
        Results are stored in a dictionary, with key being the series instance UID
        and the value being the list of instance paths.

        The oridering of instance files for a series is based on the following strategy:
        Read all images in the directory (assuming there is only one study/serie)
        1. Extract Image Orientation & Image Position from DICOM images, and
           then calculate the ordering based on the 3D coordinate of the slice
        2. If for some reason this information is not found or failed,
           another strategy is used: the ordering is based on 'Instance Number'
        3. If it also failed, then the filenames are ordered by lexicographical order.

        Args:
            input_dir (str): the directory to search for series and istances.

        Returns:
            dict: Series instance UID is the key, and paths of instances the value
        """

        series_id_to_instance_paths = {}

        series_uids = DICOMParser.find_series_uids(input_dir)
        if not series_uids:
            logging.warn('No series found.'.format(series_uids))
            return series_id_to_instance_paths
        logging.info('Have found the following series uids:\n{}'.format(series_uids))

        for series_id in series_uids:
            reader = sitk.ImageSeriesReader()
            image_paths = reader.GetGDCMSeriesFileNames(
                directory=input_dir,
                seriesID=series_id,
                useSeriesDetails=False,
                recursive=True)

            if image_paths and len(image_paths) > 0:
                logging.info('Found instances paths for series instance UID {}:\n{}'.format(
                    series_id, image_paths))
                series_id_to_instance_paths[series_id] = image_paths
            else:
                logging.error('No instance file found for series instance UID: {}'.format(series_id))

        return series_id_to_instance_paths

    @staticmethod
    def find_series_uids(input_dir, series_uids=None):
        """Find unique series instance UIDs from the DICOM instances under a directory

        Recursively search current and all sub directories to find DICOM series instance
        UIDs in the DICOM files. Any UID found is added a set of unique values.

        Args:
            input_dir: The directory to search for series instance UIDs.
            series_uids (list): Series instance UID already found.

        Returns:
            A set of unique series instance UIDs.
        """

        if not os.path.isdir(input_dir):
            logging.warn('No op as input_dir "{}" is not a directory.'.format(input_dir))
            return series_uids

        logging.info('Search for series in the directory "{}".'.format(input_dir))

        series_uids = series_uids if isinstance(series_uids, set) else set()
        reader = sitk.ImageSeriesReader()
        # Find series in current dir
        for series_uid in reader.GetGDCMSeriesIDs(input_dir):
            logging.info('In directory "{}" found a series uids:\n{}'.format(input_dir, series_uid))
            series_uids.add(series_uid)

        # Get all subdirs and then recurse
        for sub_folder in [f.path for f in os.scandir(input_dir) if f.is_dir()]:
            DICOMParser.find_series_uids(sub_folder, series_uids)

        return series_uids

    @property
    def MetadataInJSON(self):
        """Return metadata in JSON format
        """

        return self.meta_data.to_json()

    def _show_stats(self):
        """Show the stats of the parsed DICOM studies """

        if self.meta_data is None or len(self.meta_data.studies) < 1:
            self._logger.info('There is no metadata yet.')

        self._logger.info('Metadata for {} studies.'.format(len(self.meta_data.studies)))
        for study in self.meta_data.studies:
            self._logger.info('Study instance UID: {}\nTotal # of series: {}'.format(
                study.StudyInstanceUID, len(study.series)))
            for series in study.series:
                self._logger.info('Series instance UID: {}\nTotal # of instances: {}'.format(
                    series.SeriesInstanceUID, len(series.sop_instances)))
                for instance in series.sop_instances:
                    self._logger.info('Instance Num: {},\tUID: {}'.format(
                        instance.InstanceNumber, instance.SOPInstanceUID))

    def _parse_dicom_file(self, file_path):
        """Parse a single DICOM Part 10 file

        Args:
            file_path (str): path of the dcm file
        """

        self._logger.info('Parsing file: {}.'.format(file_path))

        if file_path is None:
            raise ValueError('Argument is none or empty.')
        if not os.path.isfile(file_path):
            raise IOError('File expected but not found: {}'.format(file_path))

        ds = InstanceDataset().from_file(file_path)

        self.meta_data.process_sop_instance(ds)
        self._logger.info('Parsed file: {}.'.format(file_path))
        self._logger.debug('Dataset:\n{}'.format(ds))


class InstanceDataset(object):
    """Class encapsulating the instance Dataset and its URL

    This class currently supports only instance files on local file system.
    It can extended to access remote instances via a number of protocols, e.g.
    CIFS, HTTP, etc.

    Attribute:
        dataset (Dataset): Instance metada in pydicom Dataset
        url (string): URL of the instance file if known, e.g. file:///dcm/study1/series1/1234.dcm.
    """

    def __init__(self):
        """Init an instance"""

        self.dataset = None
        self.url = None

    def from_file(self, file_path):
        """Load from instance file

        Args:
            file_path (string): Path to the DICOM instance file.
        """

        if not file_path:
            raise ValueError('Argument is None or empty.')
        if not os.path.isfile(file_path):
            raise IOError('File expected but not found: {}'.format(file_path))

        self.dataset = dcmread(file_path)
        # Create the file URL with the local path.
        self.url = 'file://{}'.format(file_path)
        setattr(self.dataset, 'RetrieveURL', self.url)

        return self


class DicomMetadata(object):
    """Class encapsulating DICOM metadata for studies, series, and instances

    Attributes:
        studies (dict): Dictionary of {study instanct UID : DicomStudy} 
    """

    def __init__(self):
        """ Class encapsulating parsed DICOM study metadata.

        Attribute:
            studies (list): List of already parsed DicomStudy
            study_dictionary (dict): Dictionary {study inatance UID : DicomStudy}
        """

        self.studies = []

    @property
    def study_dictionary(self):
        """ Dictionary of parsed DicomStudy instances with study instance UID being the key."""

        study_dict = {}
        for study in self.studies:
            study_dict[study.StudyInstanceUID] = study

        return study_dict

    def to_json(self):
        """Dump metadata in JSON string"""

        return json.dumps(self, default=lambda x: x.__dict__, indent=4)

    def process_sop_instance(self, instance_ds):
        """Parse a DICOM SOP instance and create or add meta for the study hierarchy.

        Args:
            instance_ds (InstanceDataset): SOP instance Dataset, pydicom type, and URL.
        """

        if not isinstance(instance_ds, InstanceDataset):
            raise ValueError('Argument is not an InstanceDataset.')

        ds = instance_ds.dataset
        if not ds:
            return

        # If study does not exist, create study instance (which creates its own series),
        # else, delegate processing to the study instance.
        study_dict = self.study_dictionary
        if ds.StudyInstanceUID in study_dict.keys():
            logging.info('Instance belongs to an existing study, instance UID: {}'.format(
                         ds.StudyInstanceUID))
            study_dict[ds.StudyInstanceUID].add_series(instance_ds)
        else:
            logging.info('Instance belongs to a new study, instance UID: {}'.format(
                ds.StudyInstanceUID))
            self.studies.append(DicomStudy(instance_ds))


class DicomStudy(object):
    """Object encapsulating a DICOM study metadata
    """

    def __init__(self, instance_ds, study_only=False):
        """Initialize a instance with pydicom Dataset object

        Args:
            instance_ds (InstanceDataset): SOP instance Dataset, pydicom type, and URL
            study_only (bool): Create object with study level attributes without series
        """

        if not isinstance(instance_ds, InstanceDataset):
            raise ValueError('Argument is not an InstanceDataset.')

        ds = instance_ds.dataset
        if not ds:
            return

        # The following Type 1 attributes are required to be in the SOP Instance
        # and shall have a valid value.
        self.SOPClassUID = ds.SOPClassUID
        self.StudyInstanceUID = ds.StudyInstanceUID

        # The following Type 2 attributes are required to be in the SOP Instance
        # but may contain the value of "unknown", or a zero length value.
        # But still use try-get for the non-critical ones to avoid exception.
        self.StudyDate = ds.get('StudyDate', '')
        self.StudyTime = ds.get('StudyTime', '')
        self.StudyID = ds.get('StudyID', '')
        self.AccessionNumber = ds.get('AccessionNumber', '')
        self.PatientName = str(ds.get('PatientName', ''))
        self.PatientID = ds.get('PatientID', '')
        self.PatientSex = ds.get('PatientSex', '')
        self.PatientBirthDate = ds.get('PatientBirthDate', '')

        # The following Type 3 attributes are optional.
        # May or may not be included and could be zero length.
        # Use try-get to avoid exception.
        self.StudyDescription = ds.get('StudyDescription', '')
        self.ReferringPhysicianName = str(ds.get('ReferringPhysicianName', ''))

        self.series = []  # List of series.

        if not study_only:
            # Create child series.
            # Series instance UID is Type 1, so it is safe to access it.
            self.series.append(DicomSeries(instance_ds, self))

    @property
    def series_dictionary(self):
        """Dictionary of DicomSeries with series instance UID being the key."""
        series_dict = {}
        for series in self.series:
            series_dict[series.SeriesInstanceUID] = series
        return series_dict

    def add_series(self, instance_ds):
        """Add a series to this study

        Args:
            instance_ds (InstanceDataset): SOP instance Dataset, pydicom type, and URL.
        """

        if not isinstance(instance_ds, InstanceDataset):
            raise ValueError('Argument is not an InstanceDataset.')

        ds = instance_ds.dataset
        if not ds:
            return

        # Check the study instance UID match
        if not self.StudyInstanceUID == ds.StudyInstanceUID:
            raise ValueError('Series does not belong to this study, instance UID:{}'.format(
                             self.StudyInstanceUID))

        # Create a new series if it does not exist, otherwise call the series to add instance.
        if ds.SeriesInstanceUID in self.series_dictionary.keys():
            logging.info('Instance belongs to an existing series, instance UID: {}'.format(
                         ds.SeriesInstanceUID))
            self.series_dictionary[ds.SeriesInstanceUID].add_instance(instance_ds)
        else:
            logging.info('Instance belongs to a new series, intance UID: {}'.format(
                         ds.SeriesInstanceUID))
            self.series.append(DicomSeries(instance_ds, self))


class DicomSeries(object):
    """ Object encapsulating a DICOM Series metadata."""

    def __init__(self, instance_ds, parent_study=None, series_only=False):
        """Init object with PyDicom Dataset object

        Args:
            instance_ds (InstanceDataset): SOP instance Dataset, pydicom type, and URL
            parent_study (DicomStudy): Parent study of the series
            series_only (bool): Create object without contained sop instances.
        """

        if not isinstance(instance_ds, InstanceDataset):
            raise ValueError('Argument is not an InstanceDataset.')

        ds = instance_ds.dataset
        if not ds:
            return

        # Duplicated from Study level for easy access
        if parent_study is None or not isinstance(parent_study, DicomStudy):
            parent_study = DicomStudy(instance_ds, study_only=True)

        self.StudyInstanceUID = parent_study.StudyInstanceUID
        self.StudyDate = parent_study.StudyDate
        self.StudyTime = parent_study.StudyTime
        self.StudyID = parent_study.StudyID
        self.StudyDescription = parent_study.StudyDescription
        self.AccessionNumber = parent_study.AccessionNumber
        self.PatientName = parent_study.PatientName
        self.PatientID = parent_study.PatientID
        self.PatientSex = parent_study.PatientSex
        self.PatientBirthDate = parent_study.PatientBirthDate
        self.ReferringPhysicianName = parent_study.ReferringPhysicianName
        self.SOPClassUID = parent_study.SOPClassUID

        # The following Type 1 attributes are required to be in the SOP Instance
        # and shall have a valid value.
        # But still use try-get for the non-critical ones to avoid exception.
        self.SeriesInstanceUID = ds.SeriesInstanceUID
        self.Modality = ds.Modality
        # This following are instance/image attributes, but keeping them here is OK.
        self.ImageType = list((ds.get('ImageType', '')))
        self.PixelSpacing = list((ds.get('PixelSpacing', '')))
        self.SliceThickness = ds.get('SliceThickness', '')
        self.ImageOrientationPatient = list(ds.get('ImageOrientationPatient', ''))
        self.ImagePositionPatient = list(ds.get('ImagePositionPatient', ''))

        # The following Type 2 attributes are required to be in the SOP Instance
        # but may contain the value of "unknown", or a zero length value.
        # But still use try-get for the non-critical ones to avoid exception.
        self.SeriesNumber = ds.get('SeriesNumber', '')

        # The following Type 3 attributes are optional.
        # May or may not be included and could be zero length.
        # Use try-get to avoid exception.
        self.SeriesDescription = ds.get('SeriesDescription', '')
        self.SeriesDate = ds.get('SeriesDate', '')
        self.SeriesTime = ds.get('SeriesTime', '')
        self.FrameOfReferenceUID = ds.get('FrameOfReferenceUID', '')
        self.PatientPosition = ds.get('PatientPosition', '')
        self.sop_instances = []  # List of instances. Use list for easier serialization. # Dict sop_instance_uid:SOPInstance

        if not series_only:
            if ds.SOPInstanceUID in self.sop_instance_dictionary.keys():
                # The instance has already been processed unexpected.
                # Log a warning and go to overwrite it
                logging.warn('Reprocess SOP instance UID: {}'.format(ds.SOPInstanceUID))
            else:
                logging.info('Processing SOP instance UID: {}'.format(ds.SOPInstanceUID))

            self.sop_instances.append(SOPInstance(instance_ds, self))

    @property
    def sop_instance_dictionary(self):
        """Dictionary of parsed SOPInstances with SOP instance UID being the key."""

        return self._get_sop_instance_dict()

    def add_instance(self, instance_ds):
        """ Add a DICOM instance to this series.

        Args:
            instance_ds (InstanceDataset): SOP instance Dataset, pydicom type, and URL
        """

        if not isinstance(instance_ds, InstanceDataset):
            raise ValueError('Argument is not a InstanceDataset.')

        ds = instance_ds.dataset
        if not ds:
            return

        # Ensure the instance belongs to this series
        if not ds.SeriesInstanceUID == self.SeriesInstanceUID:
            raise ValueError('SOP instance does not belong to this series, instance UID: {}'.format(
                             self.SeriesInstanceUID))

        if ds.SOPInstanceUID in self.sop_instance_dictionary:
            logging.info(
                'Reparsing metadata of an existing SOP instance: {}'.format(ds.SOPInstanceUID))
            self.sop_instance_dictionary[ds.SOPInstanceUID] = SOPInstance(instance_ds, self)
        else:
            self.sop_instances.append(SOPInstance(instance_ds, self))

    def _get_sop_instance_dict(self):
        """ Return a dictionary of instances with instance UID as key"""

        sop_instance_dict = {}
        for sop_instance in self.sop_instances:
            sop_instance_dict[sop_instance.SOPInstanceUID] = sop_instance
        return sop_instance_dict


class SOPInstance(object):
    """Internal object encapsulating a SOP instance.
    """

    def __init__(self, instance_ds, parent_series=None):
        """Init object with PyDicom Dataset object

        Args:
            instance_ds (InstanceDataset): SOP instance Dataset, pydicom type, and URL
            parent_series (DicomSeries): Parent series of the instance
        """

        if not isinstance(instance_ds, InstanceDataset):
            raise ValueError('Argument is not an InstanceDataset.')

        ds = instance_ds.dataset
        if not ds:
            return

        if parent_series is None or not isinstance(parent_series, DicomSeries):
            parent_series = DicomSeries(instance_ds, series_only=True)  # Creating this for parent attributes

        # Save the URL for the source instance file, e.g. url "file:///dcm/123.dcm"
        self.RetrieveURL = ds.RetrieveURL  # tag "00081190"

        # The following Type 1 attributes are required to be in the SOP Instance
        # and shall have a valid value.
        # But still use try-get for the non-critical ones to avoid exception.
        self.SOPClassUID = ds.SOPClassUID
        self.SOPInstanceUID = ds.SOPInstanceUID
        self.PixelSpacing = parent_series.PixelSpacing
        self.SliceThickness = parent_series.SliceThickness
        self.ImageOrientationPatient = parent_series.ImageOrientationPatient
        self.ImagePositionPatient = parent_series.ImagePositionPatient

        # The following Type 3 attributes are optional.
        # May or may not be included and could be zero length.
        # Use try-get to avoid exception.
        self.InstanceCreationDate = ds.get('InstanceCreationDate', '')
        self.InstanceCreationTime = ds.get('InstanceCreationTime', '')
        self.ContentDate = ds.get('ContentDate', '')
        self.ContentTime = ds.get('ContentTime', '')
        self.InstanceNumber = ds.get('InstanceNumber', '')
