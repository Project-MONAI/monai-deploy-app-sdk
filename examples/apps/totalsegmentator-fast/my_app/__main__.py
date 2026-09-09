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

# __main__.py allows `python my_app` to resolve to this package.

import logging

try:  # package-style import (my_app.*)
    from my_app.app import TotalSegFastApp
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from app import TotalSegFastApp

if __name__ == "__main__":
    logging.info(f"Begin {__name__}")
    TotalSegFastApp().run()
    logging.info(f"End {__name__}")
