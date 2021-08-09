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

import inspect
import runpy
import sys
from pathlib import Path
from typing import Any, Union


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


def is_application(cls: Any) -> bool:
    """Check if the given type is a subclass of Application class."""
    if hasattr(cls, "_class_id") and cls._class_id == "monai.application":
        if inspect.isclass(cls) and hasattr(cls, "__abstractmethods__") and len(cls.__abstractmethods__) != 0:
            return False
        return True
    return False


def get_application(path: Union[str, Path]):
    """Get application object from path."""

    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    # Setup PYTHONPATH if the target path is a file
    if path.is_file() and sys.path[0] != str(path.parent):
        sys.path.insert(0, str(path.parent))

    # Execute the module with runpy (`run_name` would be '<run_path>' by default.)
    vars = runpy.run_path(str(path))

    # Get the Application class from the module and return an instance of it
    for var in vars.keys():
        if not var.startswith("_"):  # skip private variables
            app_cls = vars[var]
            if is_application(app_cls):
                # Create Application object with the application path
                app_obj = app_cls(do_run=False, path=path)
                return app_obj
    return None


def get_class_file_path(cls):
    """Get the file path of a class."""
    return Path(inspect.getfile(cls))
