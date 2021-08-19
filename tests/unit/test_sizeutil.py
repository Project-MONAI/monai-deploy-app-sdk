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

import pytest

from monai.deploy.utils.sizeutil import convert_bytes, get_bytes


def test_get_bytes():
    assert get_bytes(1024) == 1024
    assert get_bytes("1024") == 1024
    with pytest.raises(ValueError):
        get_bytes(-1)
    with pytest.raises(TypeError):
        get_bytes(0.3)
    with pytest.raises(ValueError):
        get_bytes("2.3 unknownlongunit")
    with pytest.raises(ValueError):
        get_bytes("2.3jb")
    assert get_bytes("2kb") == 2 * 1000
    assert get_bytes(" 2 KiB  ") == 2 * 1024
    with pytest.raises(ValueError):
        get_bytes("-2.3Gb")


def test_convert_bytes():
    with pytest.raises(ValueError):
        convert_bytes(-1)
    with pytest.raises(TypeError):
        convert_bytes(0.3)
    with pytest.raises(ValueError):
        convert_bytes(1024, "unknownunit")
    assert convert_bytes(1024 * 1024) == "1Mi"
    assert convert_bytes(1024, "b") == 1024
    assert convert_bytes(1024 * 1024 * 1024, "Mi") == "1024Mi"
    assert convert_bytes(1024 * 1024, "kib") == "1024kib"
    assert convert_bytes(int(1024 * 0.211), "kib") == "0.2kib"
