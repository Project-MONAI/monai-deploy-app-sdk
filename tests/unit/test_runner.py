# # Copyright 2021 MONAI Consortium
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #     http://www.apache.org/licenses/LICENSE-2.0
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import argparse
# from contextlib import contextmanager
# from pathlib import Path
# from unittest import TestCase
# from unittest.mock import patch

# import pytest
# from pytest_lazyfixture import lazy_fixture


# class ContainsString(str):
#     def __eq__(self, other):
#         return self in other


# class DoesntContainsString(str):
#     def __eq__(self, other):
#         return self not in other


# @contextmanager
# def not_raises(exception):
#     try:
#         yield
#     except exception as err:
#         raise pytest.fail(f"DID RAISE {exception}") from err


# @pytest.mark.parametrize("return_value", [0, 125])
# @patch("monai.deploy.runner.runner.run_cmd")
# @patch("tempfile.TemporaryDirectory")
# def test_fetch_map_manifest(
#     tempdir, mock_run_cmd, return_value, sample_map_name, faux_app_manifest, faux_pkg_manifest, mock_manifest_export_dir
# ):
#     from monai.deploy.runner import runner

#     tempdir.return_value.__enter__.return_value = mock_manifest_export_dir
#     mock_run_cmd.return_value = return_value

#     expected_app_manifest = {}
#     expected_pkg_manifest = {}
#     if return_value == 0:
#         expected_app_manifest = faux_app_manifest
#         expected_pkg_manifest = faux_pkg_manifest

#     actual_app_manifest, actual_pkg_manifest, returncode = runner.fetch_map_manifest(sample_map_name)

#     assert returncode == return_value
#     TestCase().assertDictEqual(actual_app_manifest, expected_app_manifest)
#     TestCase().assertDictEqual(actual_pkg_manifest, expected_pkg_manifest)
#     mock_run_cmd.assert_called_once_with(ContainsString(sample_map_name))
#     mock_run_cmd.assert_called_once_with(ContainsString(mock_manifest_export_dir))


# @pytest.mark.parametrize(
#     "return_value, input_path, output_path, quiet",
#     [
#         (0, lazy_fixture("faux_folder"), Path("output/"), False),
#         (0, lazy_fixture("faux_folder"), Path("output/"), True),
#         (125, lazy_fixture("faux_folder"), Path("output/"), False),
#     ],
# )
# @patch("monai.deploy.runner.runner.run_cmd")
# def test_run_app_without_gpu_request(
#     mock_run_cmd, return_value, input_path, output_path, quiet, sample_map_name, faux_app_manifest, faux_pkg_manifest
# ):
#     from monai.deploy.runner import runner

#     mock_run_cmd.return_value = return_value
#     app_manifest = faux_app_manifest
#     expected_container_input = Path(app_manifest["input"]["path"])
#     expected_container_output = Path(app_manifest["output"]["path"])
#     expected_container_input /= app_manifest["working-directory"]
#     expected_container_output /= app_manifest["working-directory"]

#     returncode = runner.run_app(sample_map_name, input_path, output_path, app_manifest, faux_pkg_manifest, quiet)

#     assert returncode == return_value
#     mock_run_cmd.assert_called_once_with(ContainsString("docker run"))
#     mock_run_cmd.assert_called_once_with(ContainsString(sample_map_name))
#     mock_run_cmd.assert_called_once_with(ContainsString(input_path))
#     mock_run_cmd.assert_called_once_with(ContainsString(expected_container_input))
#     mock_run_cmd.assert_called_once_with(ContainsString(output_path))
#     mock_run_cmd.assert_called_once_with(ContainsString(expected_container_output))
#     mock_run_cmd.assert_called_once_with(ContainsString("STDERR"))
#     if quiet:
#         mock_run_cmd.assert_called_once_with(DoesntContainsString("STDOUT"))
#     else:
#         mock_run_cmd.assert_called_once_with(ContainsString("STDOUT"))


# @pytest.mark.parametrize(
#     "return_value, input_path, output_path, quiet",
#     [
#         (0, lazy_fixture("faux_folder"), Path("output/"), False),
#         (0, lazy_fixture("faux_folder"), Path("output/"), True),
#         (125, lazy_fixture("faux_folder"), Path("output/"), False),
#     ],
# )
# @patch("monai.deploy.runner.runner.run_cmd")
# def test_run_app_with_gpu_request(
#     mock_run_cmd,
#     return_value,
#     input_path,
#     output_path,
#     quiet,
#     sample_map_name,
#     faux_app_manifest,
#     faux_pkg_manifest_with_gpu,
# ):
#     from monai.deploy.runner import runner

#     mock_run_cmd.return_value = return_value
#     app_manifest = faux_app_manifest
#     expected_container_input = Path(app_manifest["input"]["path"])
#     expected_container_output = Path(app_manifest["output"]["path"])
#     expected_container_input /= app_manifest["working-directory"]
#     expected_container_output /= app_manifest["working-directory"]

#     returncode = runner.run_app(
#         sample_map_name, input_path, output_path, app_manifest, faux_pkg_manifest_with_gpu, quiet
#     )

#     assert returncode == return_value
#     mock_run_cmd.assert_called_once_with(ContainsString("nvidia-docker run"))
#     mock_run_cmd.assert_called_once_with(ContainsString(sample_map_name))
#     mock_run_cmd.assert_called_once_with(ContainsString(input_path))
#     mock_run_cmd.assert_called_once_with(ContainsString(expected_container_input))
#     mock_run_cmd.assert_called_once_with(ContainsString(output_path))
#     mock_run_cmd.assert_called_once_with(ContainsString(expected_container_output))
#     mock_run_cmd.assert_called_once_with(ContainsString("STDERR"))
#     if quiet:
#         mock_run_cmd.assert_called_once_with(DoesntContainsString("STDOUT"))
#     else:
#         mock_run_cmd.assert_called_once_with(ContainsString("STDOUT"))


# @pytest.mark.parametrize(
#     "return_value, input_path, output_path, quiet",
#     [
#         (0, lazy_fixture("faux_folder_with_space"), Path("output with space/"), False),
#         (0, lazy_fixture("faux_folder_with_space"), Path("output with space/"), True),
#         (125, lazy_fixture("faux_folder_with_space"), Path("output with space/"), False),
#     ],
# )
# @patch("monai.deploy.runner.runner.run_cmd")
# def test_run_app_for_input_output_path_with_space(
#     mock_run_cmd, return_value, input_path, output_path, quiet, sample_map_name, faux_app_manifest, faux_pkg_manifest
# ):
#     from monai.deploy.runner import runner

#     mock_run_cmd.return_value = return_value
#     app_manifest = faux_app_manifest
#     expected_container_input = Path(app_manifest["input"]["path"])
#     expected_container_output = Path(app_manifest["output"]["path"])
#     expected_container_input /= app_manifest["working-directory"]
#     expected_container_output /= app_manifest["working-directory"]

#     returncode = runner.run_app(sample_map_name, input_path, output_path, app_manifest, faux_pkg_manifest, quiet)
#     input_path_with_quotes = '"{}"'.format(input_path.absolute())  # To quiet flake8 complaint
#     output_path_with_quotes = '"{}"'.format(output_path.absolute())

#     assert returncode == return_value
#     mock_run_cmd.assert_called_once_with(ContainsString(sample_map_name))
#     mock_run_cmd.assert_called_once_with(ContainsString(input_path_with_quotes))
#     mock_run_cmd.assert_called_once_with(ContainsString(expected_container_input))
#     mock_run_cmd.assert_called_once_with(ContainsString(output_path_with_quotes))
#     mock_run_cmd.assert_called_once_with(ContainsString(expected_container_output))
#     mock_run_cmd.assert_called_once_with(ContainsString("STDERR"))
#     if quiet:
#         mock_run_cmd.assert_called_once_with(DoesntContainsString("STDOUT"))
#     else:
#         mock_run_cmd.assert_called_once_with(ContainsString("STDOUT"))


# @pytest.mark.parametrize(
#     "return_value, input_path, output_path, quiet",
#     [
#         (0, lazy_fixture("faux_folder"), Path("output/"), False),
#         (0, lazy_fixture("faux_folder"), Path("output/"), True),
#         (125, lazy_fixture("faux_folder"), Path("output/"), False),
#     ],
# )
# @patch("monai.deploy.runner.runner.run_cmd")
# def test_run_app_for_absolute_paths_in_app_manifest(
#     mock_run_cmd,
#     return_value,
#     input_path,
#     output_path,
#     quiet,
#     sample_map_name,
#     faux_app_manifest_with_absolute_path,
#     faux_pkg_manifest,
# ):
#     from monai.deploy.runner import runner

#     mock_run_cmd.return_value = return_value
#     app_manifest = faux_app_manifest_with_absolute_path
#     expected_container_input = Path(app_manifest["input"]["path"])
#     expected_container_output = Path(app_manifest["output"]["path"])

#     returncode = runner.run_app(sample_map_name, input_path, output_path, app_manifest, faux_pkg_manifest, quiet)

#     assert returncode == return_value
#     mock_run_cmd.assert_called_once_with(ContainsString(sample_map_name))
#     mock_run_cmd.assert_called_once_with(ContainsString(input_path))
#     mock_run_cmd.assert_called_once_with(ContainsString(expected_container_input))
#     mock_run_cmd.assert_called_once_with(DoesntContainsString(app_manifest["working-directory"]))
#     mock_run_cmd.assert_called_once_with(ContainsString(output_path))
#     mock_run_cmd.assert_called_once_with(ContainsString(expected_container_output))
#     mock_run_cmd.assert_called_once_with(DoesntContainsString(app_manifest["working-directory"]))
#     mock_run_cmd.assert_called_once_with(ContainsString("STDERR"))
#     if quiet:
#         mock_run_cmd.assert_called_once_with(DoesntContainsString("STDOUT"))
#     else:
#         mock_run_cmd.assert_called_once_with(ContainsString("STDOUT"))


# @pytest.mark.parametrize(
#     "which_return, verify_image_return, expected_return_value",
#     [(True, True, True), (False, True, False), (True, False, False), (False, False, False)],
# )
# @patch("shutil.which")
# @patch("monai.deploy.runner.runner.verify_image")
# def test_dependency_verification(
#     mock_verify_image, mock_which, which_return, verify_image_return, expected_return_value, sample_map_name
# ):
#     from monai.deploy.runner import runner

#     mock_which.return_value = which_return
#     mock_verify_image.return_value = verify_image_return

#     actual_return_value = runner.dependency_verification(sample_map_name)
#     if which_return:
#         mock_verify_image.assert_called_once_with(sample_map_name)
#     assert expected_return_value == actual_return_value


# @pytest.mark.parametrize(
#     "dependency_verification_return, fetch_map_manifest_return, run_app_return",
#     [(True, (lazy_fixture("faux_app_manifest"), lazy_fixture("faux_pkg_manifest"), 0), 0)],
# )
# @pytest.mark.parametrize(
#     "parsed_args",
#     [argparse.Namespace(map=lazy_fixture("sample_map_name"), input="input", output="output", quiet=False)],
# )
# @patch("monai.deploy.runner.runner.run_app")
# @patch("monai.deploy.runner.runner.pkg_specific_dependency_verification")
# @patch("monai.deploy.runner.runner.fetch_map_manifest")
# @patch("monai.deploy.runner.runner.dependency_verification")
# def test_main(
#     mock_dependency_verification,
#     mock_fetch_map_manifest,
#     mock_pkg_specific_dependency_verification,
#     mock_run_app,
#     dependency_verification_return,
#     fetch_map_manifest_return,
#     run_app_return,
#     parsed_args,
# ):
#     from monai.deploy.runner import runner

#     mock_dependency_verification.return_value = dependency_verification_return
#     mock_fetch_map_manifest.return_value = fetch_map_manifest_return
#     mock_pkg_specific_dependency_verification.return_value = True
#     mock_run_app.return_value = run_app_return

#     with not_raises(SystemExit) as _:
#         runner.main(parsed_args)


# @pytest.mark.parametrize(
#     "dependency_verification_return, fetch_map_manifest_return, pkg_specific_dependency_verification_return, run_app_return",
#     [
#         (True, (lazy_fixture("faux_app_manifest"), lazy_fixture("faux_pkg_manifest"), 0), False, 0),
#         (True, (lazy_fixture("faux_app_manifest"), lazy_fixture("faux_pkg_manifest"), 0), True, 125),
#         (True, ({}, {}, 125), True, 0),
#         (False, ({}, {}, 125), True, 125),
#         (False, (lazy_fixture("faux_app_manifest"), lazy_fixture("faux_pkg_manifest"), 0), True, 0),
#         (False, (lazy_fixture("faux_app_manifest"), lazy_fixture("faux_pkg_manifest"), 0), True, 125),
#     ],
# )
# @pytest.mark.parametrize(
#     "parsed_args",
#     [argparse.Namespace(map=lazy_fixture("sample_map_name"), input="input", output="output", quiet=False)],
# )
# @patch("monai.deploy.runner.runner.run_app")
# @patch("monai.deploy.runner.runner.pkg_specific_dependency_verification")
# @patch("monai.deploy.runner.runner.fetch_map_manifest")
# @patch("monai.deploy.runner.runner.dependency_verification")
# def test_main_error_conditions(
#     mock_dependency_verification,
#     mock_fetch_map_manifest,
#     mock_pkg_specific_dependency_verification,
#     mock_run_app,
#     dependency_verification_return,
#     fetch_map_manifest_return,
#     pkg_specific_dependency_verification_return,
#     run_app_return,
#     parsed_args,
# ):
#     from monai.deploy.runner import runner

#     mock_dependency_verification.return_value = dependency_verification_return
#     mock_fetch_map_manifest.return_value = fetch_map_manifest_return
#     mock_pkg_specific_dependency_verification.return_value = pkg_specific_dependency_verification_return
#     mock_run_app.return_value = run_app_return

#     with pytest.raises(SystemExit) as wrapped_error:
#         runner.main(parsed_args)
#     assert wrapped_error.type == SystemExit
