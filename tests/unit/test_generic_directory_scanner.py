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

"""Unit tests for GenericDirectoryScanner operator."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from monai.deploy.core import Fragment
from monai.deploy.operators.generic_directory_scanner_operator import GenericDirectoryScanner


class TestGenericDirectoryScanner(unittest.TestCase):
    """Test cases for GenericDirectoryScanner operator."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Create a mock fragment
        self.fragment = Mock(spec=Fragment)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir)

    def _create_test_files(self, file_list):
        """Helper to create test files."""
        created_files = []
        for file_name in file_list:
            file_path = self.test_path / file_name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("test content")
            created_files.append(file_path)
        return created_files

    def test_compound_extension_detection(self):
        """Test that compound extensions like .nii.gz are properly detected."""
        # This is the main bug we fixed - ensure .nii.gz files are found
        test_files = [
            "scan1.nii.gz",
            "scan2.nii.gz", 
            "scan3.nii",
            "other.txt"
        ]
        self._create_test_files(test_files)

        scanner = GenericDirectoryScanner(
            self.fragment,
            input_folder=str(self.test_path),
            file_extensions=['.nii', '.nii.gz'],
            name="test_scanner"
        )

        found_files = scanner._find_files()
        found_names = [f.name for f in found_files]

        # Should find all .nii and .nii.gz files
        self.assertIn("scan1.nii.gz", found_names)
        self.assertIn("scan2.nii.gz", found_names)
        self.assertIn("scan3.nii", found_names)
        self.assertNotIn("other.txt", found_names)
        self.assertEqual(len(found_files), 3)

    def test_hidden_file_filtering(self):
        """Test that hidden files (starting with .) are filtered out."""
        # This covers the macOS metadata file issue we encountered
        test_files = [
            "scan1.nii.gz",
            "._scan1.nii.gz",  # macOS metadata file
            ".hidden_scan.nii.gz",  # hidden file
            "normal_scan.nii"
        ]
        self._create_test_files(test_files)

        scanner = GenericDirectoryScanner(
            self.fragment,
            input_folder=str(self.test_path),
            file_extensions=['.nii', '.nii.gz'],
            name="test_scanner"
        )

        found_files = scanner._find_files()
        found_names = [f.name for f in found_files]

        # Should only find non-hidden files
        self.assertIn("scan1.nii.gz", found_names)
        self.assertIn("normal_scan.nii", found_names)
        self.assertNotIn("._scan1.nii.gz", found_names)
        self.assertNotIn(".hidden_scan.nii.gz", found_names)
        self.assertEqual(len(found_files), 2)

    def test_case_sensitivity(self):
        """Test case sensitive vs case insensitive file matching."""
        test_files = [
            "scan1.NII.GZ",
            "scan2.nii.gz",
            "scan3.Nii.Gz"
        ]
        self._create_test_files(test_files)

        # Test case sensitive (default)
        scanner_sensitive = GenericDirectoryScanner(
            self.fragment,
            input_folder=str(self.test_path),
            file_extensions=['.nii.gz'],
            case_sensitive=True,
            name="test_scanner_sensitive"
        )

        found_files_sensitive = scanner_sensitive._find_files()
        self.assertEqual(len(found_files_sensitive), 1)  # Only scan2.nii.gz

        # Test case insensitive
        scanner_insensitive = GenericDirectoryScanner(
            self.fragment,
            input_folder=str(self.test_path),
            file_extensions=['.nii.gz'],
            case_sensitive=False,
            name="test_scanner_insensitive"
        )

        found_files_insensitive = scanner_insensitive._find_files()
        self.assertEqual(len(found_files_insensitive), 3)  # All three files

    def test_recursive_vs_non_recursive(self):
        """Test recursive vs non-recursive directory scanning."""
        # Create files in subdirectories
        test_files = [
            "root_scan.nii.gz",
            "subdir1/sub_scan1.nii.gz",
            "subdir1/subdir2/deep_scan.nii.gz"
        ]
        self._create_test_files(test_files)

        # Test non-recursive (default)
        scanner_non_recursive = GenericDirectoryScanner(
            self.fragment,
            input_folder=str(self.test_path),
            file_extensions=['.nii.gz'],
            recursive=False,
            name="test_scanner_non_recursive"
        )

        found_files_non_recursive = scanner_non_recursive._find_files()
        found_names_non_recursive = [f.name for f in found_files_non_recursive]
        self.assertIn("root_scan.nii.gz", found_names_non_recursive)
        self.assertNotIn("sub_scan1.nii.gz", found_names_non_recursive)
        self.assertNotIn("deep_scan.nii.gz", found_names_non_recursive)
        self.assertEqual(len(found_files_non_recursive), 1)

        # Test recursive
        scanner_recursive = GenericDirectoryScanner(
            self.fragment,
            input_folder=str(self.test_path),
            file_extensions=['.nii.gz'],
            recursive=True,
            name="test_scanner_recursive"
        )

        found_files_recursive = scanner_recursive._find_files()
        found_names_recursive = [f.name for f in found_files_recursive]
        self.assertIn("root_scan.nii.gz", found_names_recursive)
        self.assertIn("sub_scan1.nii.gz", found_names_recursive)
        self.assertIn("deep_scan.nii.gz", found_names_recursive)
        self.assertEqual(len(found_files_recursive), 3)

    def test_multiple_extensions(self):
        """Test scanning for multiple file extensions."""
        test_files = [
            "image1.jpg",
            "image2.png",
            "scan1.nii.gz",
            "scan2.nii",
            "doc.txt",
            "data.json"
        ]
        self._create_test_files(test_files)

        scanner = GenericDirectoryScanner(
            self.fragment,
            input_folder=str(self.test_path),
            file_extensions=['.jpg', '.png', '.nii', '.nii.gz'],
            name="test_scanner_multi"
        )

        found_files = scanner._find_files()
        found_names = [f.name for f in found_files]

        # Should find all image and NIfTI files
        self.assertIn("image1.jpg", found_names)
        self.assertIn("image2.png", found_names)
        self.assertIn("scan1.nii.gz", found_names)
        self.assertIn("scan2.nii", found_names)
        self.assertNotIn("doc.txt", found_names)
        self.assertNotIn("data.json", found_names)
        self.assertEqual(len(found_files), 4)

    def test_no_files_found(self):
        """Test behavior when no matching files are found."""
        # Create files that don't match the extensions
        test_files = ["doc.txt", "data.json", "image.bmp"]
        self._create_test_files(test_files)

        scanner = GenericDirectoryScanner(
            self.fragment,
            input_folder=str(self.test_path),
            file_extensions=['.nii', '.nii.gz'],
            name="test_scanner_empty"
        )

        found_files = scanner._find_files()
        self.assertEqual(len(found_files), 0)

    def test_file_sorting(self):
        """Test that files are returned in sorted order."""
        test_files = [
            "z_scan.nii.gz",
            "a_scan.nii.gz",
            "m_scan.nii.gz"
        ]
        self._create_test_files(test_files)

        scanner = GenericDirectoryScanner(
            self.fragment,
            input_folder=str(self.test_path),
            file_extensions=['.nii.gz'],
            name="test_scanner_sorted"
        )

        found_files = scanner._find_files()
        found_names = [f.name for f in found_files]

        # Should be sorted alphabetically
        expected_order = ["a_scan.nii.gz", "m_scan.nii.gz", "z_scan.nii.gz"]
        self.assertEqual(found_names, expected_order)

    def test_edge_case_extensions(self):
        """Test edge cases with extensions."""
        test_files = [
            "file.nii.gz.backup",  # Extension after compound extension
            "file.nii.gz",         # Correct compound extension
            "file.gz",             # Only second part of compound
            "file.nii",            # Only first part of compound
            "file.nii.tar.gz",     # Different compound extension
        ]
        self._create_test_files(test_files)

        scanner = GenericDirectoryScanner(
            self.fragment,
            input_folder=str(self.test_path),
            file_extensions=['.nii.gz'],
            name="test_scanner_edge"
        )

        found_files = scanner._find_files()
        found_names = [f.name for f in found_files]

        # Should only find exact matches
        self.assertIn("file.nii.gz", found_names)
        self.assertNotIn("file.nii.gz.backup", found_names)
        self.assertNotIn("file.gz", found_names)
        self.assertNotIn("file.nii", found_names)
        self.assertNotIn("file.nii.tar.gz", found_names)
        self.assertEqual(len(found_files), 1)

    def test_empty_directory(self):
        """Test behavior with empty directory."""
        scanner = GenericDirectoryScanner(
            self.fragment,
            input_folder=str(self.test_path),
            file_extensions=['.nii.gz'],
            name="test_scanner_empty_dir"
        )

        found_files = scanner._find_files()
        self.assertEqual(len(found_files), 0)

    def test_nonexistent_directory(self):
        """Test behavior with nonexistent directory."""
        nonexistent_path = self.test_path / "nonexistent"
        
        scanner = GenericDirectoryScanner(
            self.fragment,
            input_folder=str(nonexistent_path),
            file_extensions=['.nii.gz'],
            name="test_scanner_nonexistent"
        )

        # Should handle gracefully and return empty list
        found_files = scanner._find_files()
        self.assertEqual(len(found_files), 0)

    def test_init_parameters(self):
        """Test that initialization parameters are stored correctly."""
        scanner = GenericDirectoryScanner(
            self.fragment,
            input_folder=str(self.test_path),
            file_extensions=['.nii', '.nii.gz'],
            recursive=True,
            case_sensitive=False,
            name="test_scanner_init"
        )

        self.assertEqual(scanner._input_folder, Path(self.test_path))
        self.assertEqual(scanner._file_extensions, ['.nii', '.nii.gz'])
        self.assertTrue(scanner._recursive)
        self.assertFalse(scanner._case_sensitive)

    def test_compound_extension_with_hidden_files(self):
        """Test compound extension detection with hidden file filtering.
        
        This test covers the scenario where compound extensions like .nii.gz
        were not being detected due to using file_path.suffix instead of 
        checking filename.endswith(), and ensures hidden files are filtered out.
        """
        # Create test files with compound extensions and hidden files
        test_files = [
            "file_1.nii.gz",
            "file_11.nii.gz",
            "file_15.nii.gz",
            "file_23.nii.gz",
            "._file_1.nii.gz",      # macOS metadata file (hidden)
            "._file_11.nii.gz",     # Another metadata file (hidden)
            "some_other_file.txt"   # Non-matching file
        ]
        self._create_test_files(test_files)

        scanner = GenericDirectoryScanner(
            self.fragment,
            input_folder=str(self.test_path),
            file_extensions=['.nii', '.nii.gz'],
            recursive=True,
            name="compound_scanner"
        )

        found_files = scanner._find_files()
        found_names = [f.name for f in found_files]

        # Before the fix: This would return 0 files due to suffix-only matching
        # After the fix: Should find all 4 .nii.gz files, excluding hidden ones
        expected_files = [
            "file_1.nii.gz",
            "file_11.nii.gz", 
            "file_15.nii.gz",
            "file_23.nii.gz"
        ]

        for expected in expected_files:
            self.assertIn(expected, found_names, 
                         f"Failed to find {expected} - compound extension bug not fixed!")

        # Should NOT find hidden files or non-matching files
        self.assertNotIn("._file_1.nii.gz", found_names, 
                        "Hidden file should be filtered out")
        self.assertNotIn("._file_11.nii.gz", found_names,
                        "Hidden file should be filtered out")
        self.assertNotIn("some_other_file.txt", found_names,
                        "Non-matching file should not be found")

        self.assertEqual(len(found_files), 4, 
                        f"Expected 4 files, found {len(found_files)}: {found_names}")

    def test_regression_compound_vs_simple_extensions(self):
        """Test edge case where simple extension is subset of compound extension."""
        # This tests a potential regression where .gz files might be picked up 
        # when looking for .nii.gz
        test_files = [
            "archive.tar.gz",      # Should NOT match .nii.gz
            "data.gz",             # Should NOT match .nii.gz  
            "scan.nii.gz",         # Should match .nii.gz
            "backup.nii.gz.old",   # Should NOT match .nii.gz
            "scan.nii",            # Should match .nii
        ]
        self._create_test_files(test_files)

        scanner = GenericDirectoryScanner(
            self.fragment,
            input_folder=str(self.test_path),
            file_extensions=['.nii', '.nii.gz'],
            name="regression_scanner"
        )

        found_files = scanner._find_files()
        found_names = [f.name for f in found_files]

        # Should only match exact extensions
        self.assertIn("scan.nii.gz", found_names)
        self.assertIn("scan.nii", found_names)
        self.assertNotIn("archive.tar.gz", found_names)
        self.assertNotIn("data.gz", found_names)
        self.assertNotIn("backup.nii.gz.old", found_names)
        
        self.assertEqual(len(found_files), 2,
                        f"Expected 2 files, found {len(found_files)}: {found_names}")


if __name__ == '__main__':
    unittest.main()
