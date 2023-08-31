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

from contextlib import contextmanager

import pytest

# from unittest.mock import patch

# from pytest_lazyfixture import lazy_fixture


class ContainsString(str):
    def __eq__(self, other):
        return self in other


class DoesntContainsString(str):
    def __eq__(self, other):
        return self not in other


@contextmanager
def not_raises(exception):
    try:
        yield
    except exception as err:
        raise pytest.fail(f"DID RAISE {exception}") from err


# @pytest.mark.parametrize("cmd, expected_returncode", [("my correct test command", 0), ("my errored test command", 125)])
# @patch("subprocess.Popen")
# def test_run_cmd(mock_popen, cmd, expected_returncode):
#     from monai.deploy.runner import utils

#     mock_popen.return_value.wait.return_value = expected_returncode

#     actual_returncode = utils.run_cmd(cmd)

#     assert actual_returncode == expected_returncode


# @pytest.mark.parametrize("image_name", [lazy_fixture("sample_map_name")])
# @pytest.mark.parametrize(
#     "docker_images_output, image_present, image_pulled",
#     [(lazy_fixture("sample_map_name"), True, 0), ("", False, 0), ("", False, 1)],
# )
# @patch("subprocess.check_output")
# @patch("monai.deploy.runner.utils.run_cmd")
# def test_verify_image(mock_run_cmd, mock_check_output, image_name, docker_images_output, image_present, image_pulled):
#     from monai.deploy.runner import utils

#     mock_run_cmd.return_value = image_pulled
#     mock_check_output.return_value = docker_images_output

#     actual_response = utils.verify_image(image_name)

#     assert actual_response == image_present or (image_pulled == 0)

#     if not image_present:
#         mock_run_cmd.assert_called_once_with(ContainsString("docker pull"))
