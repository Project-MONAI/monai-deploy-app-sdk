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

import site
import sys

from setuptools import find_namespace_packages, setup

import versioneer

# Workaround for editable installs with system's Python venv.
#   error: can't create or remove files in install directory
# (https://github.com/pypa/pip/issues/7953#issuecomment-645133255)
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_namespace_packages(include=["monai.*"]),
    include_package_data=True,
    zip_safe=False,
    # The following entry_points are for reference only as Holoscan sets them up
    # entry_points={
    #     "console_scripts": [
    #         "holoscan = holoscan.cli.__main__:main",
    #         "monai-deploy = holoscan.cli.__main__:main",
    #     ]
    # },
)
