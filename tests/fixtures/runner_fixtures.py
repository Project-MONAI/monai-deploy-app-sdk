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

import json
import shutil

import pytest


@pytest.fixture(scope='session')
def mock_manifest_export_dir(tmp_path_factory, faux_app_manifest, faux_pkg_manifest):
    dataset_path = tmp_path_factory.mktemp('manifest_export_dir')
    with open((dataset_path / 'app.json'), 'w') as f:
        json.dump(faux_app_manifest, f)
    with open((dataset_path / 'pkg.json'), 'w') as f:
        json.dump(faux_pkg_manifest, f)
    yield str(dataset_path)

    # Cleanup
    shutil.rmtree(dataset_path)


@pytest.fixture(scope='session')
def sample_map_name():
    yield 'test/map/image/name:tag'

@pytest.fixture(scope='session')
def faux_app_manifest():
    app_manifest = json.loads("""{
        "command": "/usr/bin/python3 -u /opt/monai/app/main.py",
        "input": {
            "path": "input",
            "formats": [
            {
                "data": "image",
                "format": "dicom",
                "protocols": [],
                "region": "spleen",
                "series-count": 1,
                "slice-per-file": 1
            }
            ]
        },
        "output": {
            "path": "output",
            "format": {
            "data": "segmentation-image",
            "format": "nifti",
            "series-count": 1,
            "slice-per-file": "*"
            }
        },
        "timeout": 600,
        "working-directory": "/var/monai"
        }""")
    yield app_manifest

@pytest.fixture(scope='session')
def faux_pkg_manifest():
    pkg_manifest = json.loads("""{
        "sdk-version": "0.0",
        "models": [
            {
            "name": "spleen-segmentation",
            "path": "/var/opt/monai/models/spleen_model/data.ts"
            }
        ],
        "resource": {
            "cpu": 1,
            "gpu": 1,
            "memory": "4Gi"
        }
        }
    """)
    yield pkg_manifest

@pytest.fixture(scope='session')
def faux_app_manifest_with_absolute_path():
    """App manifest with absolute input and output paths"""
    app_manifest = json.loads("""{
        "command": "/usr/bin/python3 -u /opt/monai/app/main.py",
        "input": {
            "path": "/input",
            "formats": [
            {
                "data": "image",
                "format": "dicom",
                "protocols": [],
                "region": "spleen",
                "series-count": 1,
                "slice-per-file": 1
            }
            ]
        },
        "output": {
            "path": "/output",
            "format": {
            "data": "segmentation-image",
            "format": "nifti",
            "series-count": 1,
            "slice-per-file": "*"
            }
        },
        "timeout": 600,
        "working-directory": "/var/monai"
        }""")
    yield app_manifest


@pytest.fixture(scope='session')
def faux_input_file(tmp_path_factory):
    input_dir = tmp_path_factory.mktemp('input')
    input_file = input_dir / 'input.jpg'
    input_file.touch()
    yield input_file


@pytest.fixture(scope='session')
def faux_input_folder(tmp_path_factory):
    input_dir = tmp_path_factory.mktemp('input')
    input_folder = input_dir
    yield input_folder
