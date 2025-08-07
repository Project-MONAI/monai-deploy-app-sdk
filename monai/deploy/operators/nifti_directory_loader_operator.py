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
from typing import List

import numpy as np

from monai.deploy.core import ConditionType, Fragment, Operator, OperatorSpec
from monai.deploy.utils.importutil import optional_import

SimpleITK, _ = optional_import("SimpleITK")


class NiftiDirectoryLoader(Operator):
    """
    This operator reads all NIfTI files from a directory and emits them one by one.
    Each call to compute() processes the next file in the directory.
    
    Named input:
        None
        
    Named output:
        image: A Numpy array object for the current NIfTI file
        filename: The filename (stem) of the current file being processed
    """
    
    def __init__(self, fragment: Fragment, *args, input_folder: Path, **kwargs) -> None:
        """Creates an instance that loads all NIfTI files from a directory.
        
        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            input_folder (Path): The directory Path to read NIfTI files from.
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.input_folder = Path(input_folder)
        
        if not self.input_folder.is_dir():
            raise ValueError(f"Input folder {self.input_folder} is not a directory")
            
        # Find all NIfTI files in the directory
        self.nifti_files = self._find_nifti_files()
        if not self.nifti_files:
            raise ValueError(f"No NIfTI files found in {self.input_folder}")
            
        self._logger.info(f"Found {len(self.nifti_files)} NIfTI files to process")
        
        # Track current file index
        self._current_index = 0
        
        # Output names
        self.output_name_image = "image"
        self.output_name_filename = "filename"
        
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    
    def _find_nifti_files(self) -> List[Path]:
        """Find all NIfTI files in the input directory."""
        nifti_files = []
        # Check for both .nii.gz and .nii files
        for pattern in ["*.nii.gz", "*.nii"]:
            for file in self.input_folder.glob(pattern):
                # Skip hidden files (starting with .)
                if not file.name.startswith('.'):
                    nifti_files.append(file)
        # Sort for consistent ordering
        return sorted(nifti_files)
    
    def setup(self, spec: OperatorSpec):
        spec.output(self.output_name_image).condition(ConditionType.NONE)
        spec.output(self.output_name_filename).condition(ConditionType.NONE)
    
    def compute(self, op_input, op_output, context):
        """Emits one file per call. The framework will call this repeatedly."""
        
        # Check if we have more files to process
        if self._current_index < len(self.nifti_files):
            file_path = self.nifti_files[self._current_index]
            self._logger.info(
                f"Processing file {self._current_index + 1}/{len(self.nifti_files)}: {file_path.name}"
            )
            
            try:
                # Load the NIfTI file
                image_np = self._load_nifti(file_path)
            except Exception as e:
                self._logger.error(f"Failed to load NIfTI file {file_path}: {e}")
                # Skip to next file instead of stopping execution
                self._current_index += 1
                return
            
            # Emit the image and filename
            op_output.emit(image_np, self.output_name_image)
            # Use pathlib's stem method for cleaner extension removal
            filename = file_path.stem
            if filename.endswith('.nii'):  # Handle .nii.gz case where stem is 'filename.nii'
                filename = filename[:-4]
            op_output.emit(filename, self.output_name_filename)
            
            # Move to next file for the next compute() call
            self._current_index += 1
        else:
            # No more files to process
            self._logger.info("All NIfTI files have been processed")
            # Return False to indicate we're done
            self.fragment.stop_execution()
    
    def _load_nifti(self, nifti_path: Path) -> np.ndarray:
        """Load a NIfTI file and return as numpy array."""
        image_reader = SimpleITK.ImageFileReader()
        image_reader.SetFileName(str(nifti_path))
        image = image_reader.Execute()
        # Transpose to match expected orientation
        image_np = np.transpose(SimpleITK.GetArrayFromImage(image), [2, 1, 0])
        return image_np 