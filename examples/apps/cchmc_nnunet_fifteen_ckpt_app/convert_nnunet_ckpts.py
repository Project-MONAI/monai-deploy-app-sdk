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
from pathlib import Path

# Add the current directory to the path to find the local module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def _import_converter():
    """Deferred import so nnunetv2 is loaded AFTER nnUNet_results env var is set,
    preventing nnunetv2.paths from caching a None value for nnUNet_results."""
    try:
        from my_app.nnunet_bundle import convert_best_nnunet_to_monai_bundle

        return convert_best_nnunet_to_monai_bundle
    except ImportError:
        pass
    try:
        from monai.apps.nnunet_bundle import convert_best_nnunet_to_monai_bundle

        return convert_best_nnunet_to_monai_bundle
    except ImportError:
        pass
    print("Error: Could not import convert_best_nnunet_to_monai_bundle from my_app.nnunet_bundle or apps.nnunet_bundle")
    print("Please ensure that nnunet_bundle.py is properly installed in your project.")
    sys.exit(1)


def _validated_map_root(value: str) -> Path:
    """Resolve the MAP output path and keep it within the current directory."""
    allowed_root = Path.cwd().resolve()
    map_root = Path(value).expanduser()
    if not map_root.is_absolute():
        map_root = allowed_root / map_root
    map_root = map_root.resolve(strict=False)

    try:
        map_root.relative_to(allowed_root)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"MAP_root must be inside the current directory ({allowed_root})."
        ) from exc

    if map_root.exists() and not map_root.is_dir():
        raise argparse.ArgumentTypeError(f"MAP_root is not a directory: {map_root}")

    return map_root


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

    parser.add_argument(
        "--nnUNet_results",
        type=str,
        required=False,
        default=None,
        help="Path to nnUNet results directory with trained models.",
    )
    parser.add_argument(
        "--checkpoint_type",
        type=str,
        default="final",
        choices=["final", "best", "both"],
        help="Which nnUNet checkpoint(s) to convert: 'final' (default) saves checkpoint_final.pth weights as "
        "final_model.pt; 'best' saves checkpoint_best.pth weights as best_model.pt; "
        "'both' saves checkpoint_final.pth as final_model.pt and checkpoint_best.pth as best_model.pt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create the nnUNet config dictionary
    nnunet_config = {
        "dataset_name_or_id": args.dataset_name_or_id,
    }

    # Create the MAP root directory
    map_root = _validated_map_root(args.MAP_root)
    os.makedirs(map_root, exist_ok=True)

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
    print(f"MAP will be created at: {map_root}")
    print(f"  nnUNet_results: {os.environ.get('nnUNet_results')}")

    # Import AFTER env vars are set so nnunetv2.paths caches the correct nnUNet_results value
    convert_best_nnunet_to_monai_bundle = _import_converter()

    # Convert the nnUNet checkpoints to MONAI bundle format
    try:
        convert_best_nnunet_to_monai_bundle(nnunet_config, map_root, checkpoint_type=args.checkpoint_type)
        print(f"Successfully converted nnUNet checkpoints to MONAI bundle at: {map_root}/models")
    except Exception as e:
        print(f"Error converting nnUNet checkpoints: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
