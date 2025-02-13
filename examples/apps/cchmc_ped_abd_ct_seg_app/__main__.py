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

# __main__.py is needed for MONAI Application Packager to detect the main app code (app.py) when
# app.py is executed in the application folder path
# e.g., python my_app

import logging

# import AIAbdomenSegApp class from app.py
from app import AIAbdomenSegApp

# if __main__.py is being run directly
if __name__ == "__main__":
    logging.info(f"Begin {__name__}")
    # create and run an instance of AIAbdomenSegApp
    AIAbdomenSegApp().run()
    logging.info(f"End {__name__}")
