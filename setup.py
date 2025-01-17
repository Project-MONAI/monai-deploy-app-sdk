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


import atexit
import site
import sys

from setuptools import find_namespace_packages, setup
from setuptools.command.install import install

import versioneer

# Workaround for editable installs with system's Python venv.
#   error: can't create or remove files in install directory
# (https://github.com/pypa/pip/issues/7953#issuecomment-645133255)
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]


class PostInstallCommand(install):
    """Contains post install actions."""

    def __init__(self, *args, **kwargs):
        super(PostInstallCommand, self).__init__(*args, **kwargs)
        atexit.register(PostInstallCommand.patch_holoscan)

    @staticmethod
    def patch_holoscan():
        """Patch Holoscan for its known issue of missing one import."""

        import importlib.util
        from pathlib import Path

        def needed_to_patch():
            from importlib.metadata import version

            try:
                version = version("holoscan")
                # This issue exists in the following versions
                if "2.7" in version or "2.8" in version:
                    print("Need to patch holoscan v2.7 and 2.8.")
                    return True
            except Exception:
                pass

            return False

        if not needed_to_patch():
            return

        print("Patching holoscan as needed...")
        spec = importlib.util.find_spec("holoscan")
        if spec:
            # holoscan core misses one class in its import in __init__.py
            module_to_add = "        MultiMessageConditionInfo,"
            module_path = Path(str(spec.origin)).parent.joinpath("core/__init__.py")
            print(f"Patching file {module_path}")
            if module_path.exists():
                lines_r = []
                existed = False
                with module_path.open("r") as f_to_patch:
                    in_block = False
                    for line_r in f_to_patch.readlines():
                        if "from ._core import (\n" in line_r:
                            in_block = True
                        elif in_block and module_to_add.strip() in line_r:
                            existed = True
                            break
                        elif in_block and ")\n" in line_r:
                            # Need to add the missing class.
                            line_r = f"{module_to_add}\n{line_r}"
                            in_block = False
                            print("Added missing module in holoscan.")

                        lines_r.append(line_r)

                if not existed:
                    with module_path.open("w") as f_w:
                        f_w.writelines(lines_r)
                print("Completed patching holoscan.")


setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass({"install": PostInstallCommand}),
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
