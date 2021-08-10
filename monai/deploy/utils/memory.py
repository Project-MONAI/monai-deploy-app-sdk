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

import re

prefix_bytes = {"kib": 2 ** 10,
                "mib": 2 ** 20,
                "gib": 2 ** 30,
                "tib": 2 ** 40,
                "pib": 2 ** 50,
                "eib": 2 ** 60,
                "zib": 2 ** 70,
                "yib": 2 ** 80,
                "kb": 10 ** 3,
                "mb": 10 ** 6,
                "gb": 10 ** 9,
                "tb": 10 ** 12,
                "pb": 10 ** 15,
                "eb": 10 ** 18,
                "zb": 10 ** 21,
                "yb": 10 ** 24,
                "b": 1}

def get_bytes(size: str) -> int:
    """Converts decimal and binary byte multiples to bytes

    Args:
        size (str): String representing memory size to be converted
        (eg. "5 YB")

    Returns:
        int: number of total bytes reresented by input string
    """
    parsed_size = float(re.findall('\d*\.?\d+', size)[0])
    parsed_prefix = re.findall('[a-z]+', size.lower())[0]
    return int(parsed_size * prefix_bytes[parsed_prefix])

def convert_bytes(bytes: float, prefix: str) -> str:
    """Converts number of bytes to equivalent binary or 
    decimal representation

    Args:
        bytes (float): Number of total bytes to convert
        prefix (str): target binary or decimal multiple prefix

    Returns:
        str: string represented converted number of bytes with desired prefix
    """
    prefix_lowered = prefix.lower()
    return str(bytes / prefix_bytes[prefix_lowered]) + prefix
