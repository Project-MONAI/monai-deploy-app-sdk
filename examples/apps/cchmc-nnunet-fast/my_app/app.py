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

import logging
from pathlib import Path

from monai.deploy.core import Application


class CCHMCNNUnetFastApp(Application):
    """Fast-track nnU-Net segmentation app for CCHMC models.

    This application loads DICOM input, performs inference using an nnU-Net
    model, and writes segmentation outputs (DICOM SEG, SR, and Secondary Capture).

    The full pipeline (operators, flows, and output writers) will be implemented
    in Phase 1. This skeleton provides the app shell and compose() hook.
    """

    def __init__(self, *args, **kwargs):
        """Creates an application instance."""
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        """Entry point — delegates to the base Application runner."""
        self._logger.info(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._logger.info(f"End {self.run.__name__}")

    def compose(self):
        """Creates the app-specific operators and chains them into a processing DAG.

        TODO (Phase 1): Instantiate operators (DICOM loader, series selector,
        volume converter, nnU-Net inference, output writers) and connect them
        via self.add_flow().
        """
        logging.info(f"Begin {self.compose.__name__}")

        # Initialize app context from command-line / env
        app_context = Application.init_app_context(self.argv)
        _app_input_path = Path(app_context.input_path)
        _app_output_path = Path(app_context.output_path)
        _model_path = Path(app_context.model_path)

        # --- Placeholder: pipeline operators go here ---
        # study_loader_op = DICOMDataLoaderOperator(...)
        # series_selector_op = ...
        # series_to_vol_op = ...
        # inference_op = ...
        # dicom_seg_writer = ...
        #
        # self.add_flow(study_loader_op, series_selector_op, {...})
        # ...

        logging.info(f"End {self.compose.__name__}")


if __name__ == "__main__":
    logging.info(f"Begin {__name__}")
    CCHMCNNUnetFastApp().run()
    logging.info(f"End {__name__}")
