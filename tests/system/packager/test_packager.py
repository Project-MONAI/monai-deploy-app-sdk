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

# import os
# import subprocess

# from monai.deploy.cli.main import main as monai_deploy


# def test_packager():
#     test_map_tag = "monaitest:latest"
#     test_app_path_rel = "examples/apps/simple_imaging_app/"
#     test_app_path = os.path.abspath(test_app_path_rel)
#     args = ["monai-deploy", "package", "-t", test_map_tag, test_app_path]
#     monai_deploy(args)

#     # Delete MONAI application package image
#     docker_rmi_cmd = ["docker", "rmi", "-f", test_map_tag]
#     subprocess.Popen(docker_rmi_cmd)
