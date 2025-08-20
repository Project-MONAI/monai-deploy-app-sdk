# Copyright 2024 MONAI Consortium
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

import numpy as np

from monai.deploy.core import ConditionType, Fragment, Operator, OperatorSpec
from monai.deploy.core.domain import Image
from monai.deploy.utils.importutil import optional_import

nibabel, _ = optional_import("nibabel")


class NiftiWriter(Operator):
    """
    This operator writes segmentation results to NIfTI files.

    Named input:
        image: Image data to save (Image object or numpy array)
        filename: Optional filename to use for saving

    Named output:
        None
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        output_folder: Path,
        output_postfix: str = "seg",
        output_extension: str = ".nii.gz",
        **kwargs,
    ) -> None:
        """Creates an instance of the NIfTI writer.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            output_folder (Path): Path to output folder.
            output_postfix (str): Postfix to add to output filenames. Defaults to "seg".
            output_extension (str): File extension for output files. Defaults to ".nii.gz".
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.output_folder = Path(output_folder)
        self.output_postfix = output_postfix
        self.output_extension = output_extension

        # Input names
        self.input_name_image = "image"
        self.input_name_filename = "filename"

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_image)
        spec.input(self.input_name_filename).condition(ConditionType.NONE)  # Optional

    def compute(self, op_input, op_output, context):
        """Save the image to a NIfTI file."""

        # Get inputs
        image = op_input.receive(self.input_name_image)

        # Try to get filename
        filename = None
        try:
            filename = op_input.receive(self.input_name_filename)
        except Exception:
            pass

        if image is None:
            return

        # Get the image array
        if isinstance(image, Image):
            image_array = image.asnumpy() if hasattr(image, "asnumpy") else np.array(image)
            # Try to get metadata
            metadata = (
                image.metadata() if callable(image.metadata) else image.metadata if hasattr(image, "metadata") else {}
            )
        else:
            image_array = np.array(image)
            metadata = {}

        # Remove batch dimension if present
        if image_array.ndim == 4 and image_array.shape[0] == 1:
            image_array = image_array[0]

        # Remove channel dimension if it's 1
        if image_array.ndim == 4 and image_array.shape[-1] == 1:
            image_array = image_array[..., 0]

        # Use filename or generate one
        if not filename:
            filename = "output"

        # Create output path
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        # Handle template variables in output_postfix (e.g., "@output_postfix")
        if self.output_postfix and self.output_postfix.startswith("@"):
            # Default to "trans" for template variables
            actual_postfix = "trans"
        else:
            actual_postfix = self.output_postfix

        if actual_postfix:
            output_filename = f"{filename}_{actual_postfix}{self.output_extension}"
        else:
            output_filename = f"{filename}{self.output_extension}"

        output_path = self.output_folder / output_filename

        # Get affine matrix from metadata if available
        affine = np.eye(4)
        if isinstance(metadata, dict) and "affine" in metadata:
            affine = np.array(metadata["affine"])

        # Transpose from (N, H, W) to (H, W, N) for NIfTI format
        if image_array.ndim == 3:
            image_array = np.transpose(image_array, [1, 2, 0])

        # Save as NIfTI
        nifti_img = nibabel.Nifti1Image(image_array.astype(np.float32), affine)
        nibabel.save(nifti_img, str(output_path))

        self._logger.info(f"Saved segmentation to: {output_path}")
