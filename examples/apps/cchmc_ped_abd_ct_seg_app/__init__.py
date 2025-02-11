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

# __init__.py is used to initialize a Python package
# ensures that the directory __init__.py resides in is included at the start of the sys.path
# this is useful when you want to import modules from this directory, even if it’s not the
# directory where your Python script is running.

# give access to operating system and Python interpreter
import os
import sys

# grab absolute path of directory containing __init__.py
_current_dir = os.path.abspath(os.path.dirname(__file__))

# if sys.path is not the same as the directory containing the __init__.py file
if sys.path and os.path.abspath(sys.path[0]) != _current_dir:
    # insert directory containing __init__.py file at the beginning of sys.path
    sys.path.insert(0, _current_dir)
# delete variable
del _current_dir
