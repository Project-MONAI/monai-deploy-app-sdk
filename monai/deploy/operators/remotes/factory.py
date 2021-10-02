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

# from typing import Dict, Optional

# from monai.deploy.exceptions import UnknownTypeError

# from .docker_operator import RemoteDockerOperator, RemoteOperator


# class RemoteOperatorFactory:
#     """RemoteOperatorFactory is an abstract class that provides a way to create a RemoteOperator object."""

#     NAMES = ["docker"]
#     DEFAULT = "docker"

#     @staticmethod
#     def create(remote_type: str, op_params: Optional[Dict] = None) -> RemoteOperator:
#         """Creates a remote operator object.

#         Args:
#             remote_type (str): A type of the operator.
#             op_params (Dict): A dictionary of parameters of the operator.

#         Returns:
#             RemoteOperator: A remote operator object.
#         """

#         op_params = op_params or {}

#         if remote_type == "docker":
#             return RemoteDockerOperator(**op_params)
#         else:
#             raise UnknownTypeError(f"Unknown remote type: {remote_type}")
