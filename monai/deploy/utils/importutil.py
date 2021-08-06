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


def get_docstring(cls: type) -> str:
    """Get docstring of a class.

    Tries to get docstring from class itself, from its __doc__.
    It trims the preceeding whitespace from docstring.
    If __doc__ is not available, it returns empty string.

    Args:
        cls (type): class to get docstring from.

    Returns:
        A docstring of the class.
    """
    doc = cls.__doc__
    if doc is None:
        return ""
    # Trim white-space for each line in the string
    return "\n".join([line.strip() for line in doc.split("\n")])
