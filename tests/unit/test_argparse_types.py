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

import getpass
from argparse import ArgumentTypeError
from contextlib import contextmanager
from pathlib import Path, PosixPath

import pytest
from pytest_lazyfixture import lazy_fixture


@pytest.mark.parametrize("expected_dir_path", [lazy_fixture("tmp_path"), lazy_fixture("non_existent_file_path")])
def test_valid_dir_path_valid_args(expected_dir_path):
    from monai.deploy.utils.argparse_types import valid_dir_path

    actual_dir_path = valid_dir_path(str(expected_dir_path))

    assert type(actual_dir_path) == PosixPath
    assert actual_dir_path == expected_dir_path
    assert expected_dir_path.exists() is True
    assert actual_dir_path.is_dir() is True
    assert actual_dir_path.is_absolute() is True
    assert actual_dir_path.owner() == getpass.getuser()


@pytest.mark.parametrize("expected_dir_path", [lazy_fixture("tmp_path"), lazy_fixture("non_existent_file_path")])
def test_valid_dir_path_valid_relative_path(expected_dir_path):
    from monai.deploy.utils.argparse_types import valid_dir_path

    @contextmanager
    def working_directory(path):
        import os

        prev_cwd = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(prev_cwd)

    root_dir = expected_dir_path.root
    with working_directory(root_dir) as _:
        relative_dir_path = expected_dir_path.relative_to(root_dir)
        actual_dir_path = valid_dir_path(str(relative_dir_path))

        assert type(actual_dir_path) == PosixPath
        assert actual_dir_path == expected_dir_path
        assert expected_dir_path.exists() is True
        assert actual_dir_path.is_dir() is True
        assert actual_dir_path.is_absolute() is True
        assert actual_dir_path.owner() == getpass.getuser()


def test_valid_dir_path_invalid_args(faux_file):
    from monai.deploy.utils.argparse_types import valid_dir_path

    expected_invalid_dir_path = faux_file
    assert expected_invalid_dir_path.exists() is True
    assert expected_invalid_dir_path.is_dir() is False

    with pytest.raises(ArgumentTypeError) as wrapped_error:
        valid_dir_path(str(expected_invalid_dir_path))
    assert wrapped_error.type == ArgumentTypeError

    assert expected_invalid_dir_path.exists() is True
    assert expected_invalid_dir_path.is_dir() is False


def test_valid_existing_dir_path_valid_args(tmp_path):
    from monai.deploy.utils.argparse_types import valid_existing_dir_path

    expected_dir_path = tmp_path
    assert expected_dir_path.exists() is True
    assert expected_dir_path.is_dir() is True
    actual_dir_path = valid_existing_dir_path(str(expected_dir_path))

    assert type(actual_dir_path) == PosixPath
    assert actual_dir_path == expected_dir_path
    assert expected_dir_path.exists() is True
    assert actual_dir_path.is_dir() is True
    assert actual_dir_path.owner() == getpass.getuser()


@pytest.mark.parametrize("input_path", [lazy_fixture("non_existent_file_path"), lazy_fixture("faux_file")])
def test_valid_existing_dir_path_invalid_args(input_path):
    from monai.deploy.utils.argparse_types import valid_existing_dir_path

    assert input_path.is_dir() is False
    with pytest.raises(ArgumentTypeError) as wrapped_error:
        valid_existing_dir_path(str(input_path))
    assert wrapped_error.type == ArgumentTypeError

    assert input_path.is_dir() is False


@pytest.mark.parametrize("expected_input_path", [lazy_fixture("faux_file"), lazy_fixture("faux_folder")])
def test_valid_existing_path_valid_args(expected_input_path):
    from monai.deploy.utils.argparse_types import valid_existing_path

    assert expected_input_path.exists() is True
    actual_input_path = valid_existing_path(str(expected_input_path))

    assert type(actual_input_path) == PosixPath
    assert actual_input_path == expected_input_path
    assert actual_input_path.exists()


@pytest.mark.parametrize("input_path", [lazy_fixture("non_existent_file_path")])
def test_valid_existing_path_invalid_args(input_path):
    from monai.deploy.utils.argparse_types import valid_existing_path

    assert input_path.exists() is False
    with pytest.raises(ArgumentTypeError) as wrapped_error:
        valid_existing_path(str(input_path))
    assert wrapped_error.type == ArgumentTypeError

    assert input_path.exists() is False
