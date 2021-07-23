# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import logging
import numbers
import os
import re

import simplejson as json


class SeriesSelector:
    """This class selects series using DICOM metadata matching rules.   
    """

    def __init__(self, meta_data_json):

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._meta_data = meta_data_json

    def select(self, rules_json):
        """ Selects the series with the given matching rules.

        If rules object is None, all series will be returned with series instance UID
        as both the key and value for each dictionary value.

        Simplistic matching is used for demonstration:
            Number: exactly matches
            String: matches case insensitive, if fails, trys RegEx search
            String array matches as subset, case insensitive

        Args:
            rules_json (object): JSON object containing the matching rules

        Returns:
            dict: Dictionay of selection name (key) and the series instance UID (value).
        """

        matched_series = {}

        if not rules_json:
            self._logger.warning('No selection rules given; select all series.')
            series_uids = self._find_all_series()
            for uid in series_uids:
                matched_series[uid] = uid
            return matched_series

        selections = rules_json.get('selections', None)
        if not selections:
            raise ValueError('Expected attribute "selections" not found.')

        for selection in selections:
            # Skip if no selection conditions are provided.
            conditions = selection.get('conditions', None)
            if not conditions:
                continue

            # Get the selection name, if blank, will use series instance UID in lieu
            selection_name = selection.get('name', '').strip()
            self._logger.info('Finding series matching Selection Name: "{}"'.format(selection_name))

            # Select the first series that matched
            series_uids = self._find_series(conditions, False)
            if series_uids and len(series_uids) > 0:
                selection_name = selection_name if selection_name.strip() else \
                    series_uids[0].strip()
                matched_series[selection_name] = series_uids[0]

        self._logger.info('Found {} matching series:\n {}'.format(
            len(matched_series.keys()),
            matched_series.values()))

        return matched_series

    def _find_all_series(self):
        """Select all series and list of series instance UID's

        Returns:
            list: list of series instance UID's
        """

        series_uids = []
        for study in self._meta_data['studies']:
            self._logger.info('Searching study, instance UID: {}\nTotal # of series: {}'.format(
                study['StudyInstanceUID'], len(study['series'])))
            for series in study['series']:
                s_iuid = series.get('SeriesInstanceUID', None)
                self._logger.info('Working on series, instance UID: {}'.format(s_iuid))
                if s_iuid:
                    series_uids.append(s_iuid)
        return series_uids

    def _find_series(self, attributes, find_all=False):
        """ Finds series whose attributes match the given attributes.

        Args:
            attributes (dict): Dictionary of attributes for matching
            find_all (bool): Find all series that match; default is False, 

        Returns:
            List of Series instance UID. At most 1 if find_all is False.
        """
        assert isinstance(attributes, dict), '"attributes" must be a dict.'

        found_series = []
        for study in self._meta_data['studies']:
            self._logger.info('Searching study, instance UID: {}\nTotal # of series: {}'.format(
                study['StudyInstanceUID'], len(study['series'])))
            for series in study['series']:
                self._logger.info('Working on series, instance UID: {}'.format(series.get('SeriesInstanceUID', None)))

                matched = True
                # Simple matching on attribute value
                for key, value in attributes.items():
                    self._logger.info('On attribute: "{}" to match value: "{}"'.format(key, value))
                    # Ignore None
                    if not value:
                        continue
                    # Try getting the key/value from metadata
                    meta_data = series.get(key, None)
                    self._logger.info('Series attribute value: "{}"'.format(meta_data))
                    if not meta_data:
                        matched = False
                    elif isinstance(meta_data, numbers.Number):
                        matched = (value == meta_data)
                    elif isinstance(meta_data, str):
                        matched = meta_data.casefold() == (value.casefold())
                        if not matched:
                            # For str, also try RegEx search to check for a match anywhere in the string
                            # unless the user constrains it in the expression.
                            self._logger.info('Series attribute srting value did not match. Switching to use regEx.')
                            if re.search(value, meta_data, re.IGNORECASE):
                                matched = True
                    elif isinstance(meta_data, list):
                        meta_data_set = set(str(element).lower() for element in meta_data)
                        if isinstance(value, list):
                            value_set = set(str(element).lower() for element in value)
                            matched = all(val in meta_data_set for val in value_set)
                        elif isinstance(value, (str, numbers.Number)):
                            matched = str(value).lower() in meta_data_set
                    else:
                        raise NotImplementedError('Not support for matching on this type: {}'.format(type(value)))

                    if not matched:
                        self._logger.info("This series does not match selection conditions.")
                        break

                if matched:
                    instance_uid = str(series['SeriesInstanceUID'])
                    self._logger.info('Found a matching series, instance UID: {}'.format(
                        instance_uid))
                    found_series.append(instance_uid)

                    if not find_all:
                        return found_series

        return found_series
