#!/usr/bin/env python3
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert nnUNet checkpoints to MONAI bundle format.
This script follows the logic in the conversion notebook but imports from local apps.nnunet_bundle.
"""

import argparse
import os
import sys

# Add the current directory to the path to find the local module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try importing from local apps.nnunet_bundle instead of from MONAI
try:
    from my_app.nnunet_bundle import convert_best_nnunet_to_monai_bundle
except ImportError:
    # If local import fails, try to find the module in alternate locations
    try:
        from monai.apps.nnunet_bundle import convert_best_nnunet_to_monai_bundle
    except ImportError:
        print(
            "Error: Could not import convert_best_nnunet_to_monai_bundle from my_app.nnunet_bundle or apps.nnunet_bundle"
        )
        print("Please ensure that nnunet_bundle.py is properly installed in your project.")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert nnUNet checkpoints to MONAI bundle format.")
    parser.add_argument(
        "--dataset_name_or_id", type=str, required=True, help="The name or ID of the dataset to convert."
    )
    parser.add_argument(
        "--MAP_root",
        type=str,
        default=os.getcwd(),
        help="The root directory where the Medical Application Package (MAP) will be created. Defaults to current directory.",
    )
    # parser.add_argument("--nnUNet_raw", type=str, required=False, default=None,
    #                     help="Path to nnUNet raw data directory.")
    # parser.add_argument("--nnUNet_preprocessed", type=str, required=False, default=None,
    #                     help="Path to nnUNet preprocessed data directory.")
    parser.add_argument(
        "--nnUNet_results",
        type=str,
        required=False,
        default=None,
        help="Path to nnUNet results directory with trained models.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create the nnUNet config dictionary
    nnunet_config = {
        "dataset_name_or_id": args.dataset_name_or_id,
    }

    # Create the MAP root directory
    MAP_root = args.MAP_root
    os.makedirs(MAP_root, exist_ok=True)

    # Set nnUNet environment variables if provided
    if args.nnUNet_results:
        os.environ["nnUNet_results"] = args.nnUNet_results
        print(f"Set nnUNet_results to: {args.nnUNet_results}")

    # Check if required environment variables are set
    required_env_vars = ["nnUNet_results"]
    missing_vars = [var for var in required_env_vars if var not in os.environ]

    if missing_vars:
        print(f"Error: The following required nnUNet environment variables are not set: {', '.join(missing_vars)}")
        print("Please provide them as arguments or set them in your environment before running this script.")
        sys.exit(1)

    print(f"Converting nnUNet checkpoints for dataset {nnunet_config['dataset_name_or_id']} to MONAI bundle format...")
    print(f"MAP will be created at: {MAP_root}")
    print(f"  nnUNet_results: {os.environ.get('nnUNet_results')}")

    # Convert the nnUNet checkpoints to MONAI bundle format
    try:
        convert_best_nnunet_to_monai_bundle(nnunet_config, MAP_root)
        print(f"Successfully converted nnUNet checkpoints to MONAI bundle at: {MAP_root}/models")
    except Exception as e:
        print(f"Error converting nnUNet checkpoints: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
