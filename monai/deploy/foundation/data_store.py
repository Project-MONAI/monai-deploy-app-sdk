# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class DataStore:

    """This class represents a Data Store
    that cachces inputs & outputs from all operators
    at run-time. Currently it is memory based. A more
    efficient representation of Data store will be
    be implemented in future. This will enable storing
    an entire snapshot of an application run so that it
    can be recreated later for debugging purpose.
    """

    _instance = None

    @staticmethod
    def get_instance():
        """static methid to get a handle to the singleton
        instance of the class
        """
        if DataStore._instance is None:
            DataStore()
        return DataStore._instance

    def __init__(self):
        """Almost private constructor"""
        if DataStore._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            DataStore._instance = self
            DataStore._instance.storage = {}

    def store(self, key, val):
        """Provides simple dictionary based storage of attributes
        Args:
          key: key for the data element to be stored
          val: value for the corresponding key
        """
        self.storage[key] = val

    def retrieve(self, key):
        """Provides simple dictionary based storage of attributes
        Args:
          key: key for the data element to be retrieved

        Returns:
          value for the specified key
        """
        return self.storage[key]
