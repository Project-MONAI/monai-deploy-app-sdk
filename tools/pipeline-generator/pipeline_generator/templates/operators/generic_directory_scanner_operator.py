# Copyright 2025 MONAI Consortium
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
from typing import List, Union

from monai.deploy.core import ConditionType, Fragment, Operator, OperatorSpec


class GenericDirectoryScanner(Operator):
    """Scan a directory for files matching specified extensions and emit file paths one by one.

    This operator provides a generic way to iterate through files in a directory,
    emitting one file path at a time. It can be chained with file-specific loaders
    to create flexible data loading pipelines.

    Named Outputs:
        file_path: Path to the current file being processed
        filename: Name of the current file (without extension)
        file_index: Current file index (0-based)
        total_files: Total number of files found
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        input_folder: Union[str, Path],
        file_extensions: List[str],
        recursive: bool = True,
        case_sensitive: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the GenericDirectoryScanner.

        Args:
            fragment: An instance of the Application class
            input_folder: Path to folder containing files to scan
            file_extensions: List of file extensions to scan for (e.g., ['.jpg', '.png'])
            recursive: If True, scan subdirectories recursively
            case_sensitive: If True, perform case-sensitive extension matching
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._input_folder = Path(input_folder)
        self._file_extensions = [ext if ext.startswith(".") else f".{ext}" for ext in file_extensions]
        self._recursive = bool(recursive)
        self._case_sensitive = bool(case_sensitive)

        # State tracking
        self._files: List[Path] = []
        self._current_index = 0

        super().__init__(fragment, *args, **kwargs)

    def _find_files(self) -> List[Path]:
        """Find all files matching the specified extensions."""
        files: List[Path] = []

        # Normalize extensions for comparison
        if not self._case_sensitive:
            extensions = [ext.lower() for ext in self._file_extensions]
        else:
            extensions = self._file_extensions

        # Choose search method based on recursive flag
        if self._recursive:
            search_pattern = "**/*"
            search_method = self._input_folder.rglob
        else:
            search_pattern = "*"
            search_method = self._input_folder.glob

        # Find all files and filter by extension
        for file_path in search_method(search_pattern):
            if file_path.is_file():
                # Skip hidden files (starting with .) to avoid macOS metadata files like ._file.nii.gz
                if file_path.name.startswith("."):
                    continue

                # Handle compound extensions like .nii.gz by checking if filename ends with any extension
                filename = file_path.name
                if not self._case_sensitive:
                    filename = filename.lower()

                # Check if filename ends with any of the specified extensions
                for ext in extensions:
                    if filename.endswith(ext):
                        files.append(file_path)
                        break  # Only add once even if multiple extensions match

        # Sort files for consistent ordering
        files.sort()
        return files

    def setup(self, spec: OperatorSpec):
        """Define the operator outputs."""
        spec.output("file_path")
        spec.output("filename")
        spec.output("file_index").condition(ConditionType.NONE)
        spec.output("total_files").condition(ConditionType.NONE)

        # Pre-initialize the files list
        if not self._input_folder.is_dir():
            raise ValueError(f"Input folder {self._input_folder} is not a directory")

        self._files = self._find_files()
        self._current_index = 0

        if not self._files:
            self._logger.warning(f"No files found in {self._input_folder} with extensions {self._file_extensions}")
        else:
            self._logger.info(f"Found {len(self._files)} files to process with extensions {self._file_extensions}")

    def compute(self, op_input, op_output, context):
        """Emit the next file path."""

        # Check if we have more files to process
        if self._current_index >= len(self._files):
            # No more files to process
            self._logger.info("All files have been processed")
            self.fragment.stop_execution()
            return

        # Get the current file path
        file_path = self._files[self._current_index]

        try:
            # Emit file information
            op_output.emit(str(file_path), "file_path")
            op_output.emit(file_path.stem, "filename")
            op_output.emit(self._current_index, "file_index")
            op_output.emit(len(self._files), "total_files")

            self._logger.info(f"Emitted file: {file_path.name} ({self._current_index + 1}/{len(self._files)})")

        except Exception as e:
            self._logger.error(f"Failed to process file {file_path}: {e}")

        # Move to the next file
        self._current_index += 1


def test():
    """Test the GenericDirectoryScanner operator."""
    import tempfile

    # Create a temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files with different extensions
        test_files = ["test1.jpg", "test2.png", "test3.nii", "test4.nii.gz", "test5.txt", "test6.jpeg"]

        for filename in test_files:
            (temp_path / filename).touch()

        # Create a subdirectory with more files
        sub_dir = temp_path / "subdir"
        sub_dir.mkdir()
        (sub_dir / "sub_test.jpg").touch()
        (sub_dir / "sub_test.nii").touch()

        # Test the operator with image extensions
        fragment = Fragment()
        scanner = GenericDirectoryScanner(
            fragment, input_folder=temp_path, file_extensions=[".jpg", ".jpeg", ".png"], recursive=True
        )

        # Simulate setup
        from monai.deploy.core import OperatorSpec

        spec = OperatorSpec()
        scanner.setup(spec)

        print(f"Found {len(scanner._files)} image files")

        # Simulate compute calls
        class MockOutput:
            def emit(self, data, name):
                print(f"Emitted {name}: {data}")

        mock_output = MockOutput()

        # Process a few files
        for i in range(min(3, len(scanner._files))):
            print(f"\n--- Processing file {i + 1} ---")
            scanner.compute(None, mock_output, None)


if __name__ == "__main__":
    test()
