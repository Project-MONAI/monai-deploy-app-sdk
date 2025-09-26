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

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from monai.deploy.core import ConditionType, Fragment, Operator, OperatorSpec


class JSONResultsWriter(Operator):
    """Write classification or prediction results to JSON files.

    This operator handles various types of model outputs (dictionaries, tensors, numpy arrays)
    and saves them as JSON files with proper formatting.

    Named Inputs:
        pred: Prediction results (dict, tensor, or numpy array)
        filename: Optional filename for the output (without extension)

    File Output:
        JSON files saved in the specified output folder
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        output_folder: Union[str, Path],
        result_key: str = "pred",
        **kwargs,
    ) -> None:
        """Initialize the JSONResultsWriter.

        Args:
            fragment: An instance of the Application class
            output_folder: Path to folder for saving JSON results
            result_key: Key to extract from prediction dict if applicable (default: "pred")
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.result_key = result_key

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Define the operator inputs."""
        spec.input("pred")
        spec.input("filename").condition(ConditionType.NONE)  # Optional input

    def compute(self, op_input, op_output, context):
        """Process and save prediction results as JSON."""
        pred = op_input.receive("pred")
        if pred is None:
            self._logger.warning("No prediction received")
            return

        # Try to get filename
        filename = None
        try:
            filename = op_input.receive("filename")
        except Exception:
            pass

        if not filename:
            # Generate a default filename
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"result_{timestamp}"

        # Process the prediction data
        result_data = self._process_prediction(pred, filename)

        # Save as JSON
        output_file = self.output_folder / f"{filename}_result.json"
        with open(output_file, "w") as f:
            json.dump(result_data, f, indent=2)

        self._logger.info(f"Saved results to {output_file}")

        # Print summary if it's a classification result
        if "probabilities" in result_data:
            self._print_classification_summary(result_data)

    def _process_prediction(self, pred: Any, filename: str) -> Dict[str, Any]:
        """Process various prediction formats into a JSON-serializable dictionary."""
        result: Dict[str, Any] = {"filename": filename}

        # Handle dictionary predictions (e.g., from MonaiBundleInferenceOperator)
        if isinstance(pred, dict):
            if self.result_key in pred:
                pred_data = pred[self.result_key]
            else:
                # Use the entire dict if our key isn't found
                pred_data = pred
        else:
            pred_data = pred

        # Convert to numpy if it's a tensor
        if hasattr(pred_data, "cpu"):  # PyTorch tensor
            pred_data = pred_data.cpu().numpy()
        elif hasattr(pred_data, "asnumpy"):  # MONAI MetaTensor
            pred_data = pred_data.asnumpy()

        # Handle different prediction types
        if isinstance(pred_data, np.ndarray):
            if pred_data.ndim == 1:  # 1D array (e.g., classification probabilities)
                # Assume classification with probabilities
                if len(pred_data) == 4:  # Breast density classification
                    result["probabilities"] = {
                        "A": float(pred_data[0]),
                        "B": float(pred_data[1]),
                        "C": float(pred_data[2]),
                        "D": float(pred_data[3]),
                    }
                else:
                    # Generic classification
                    result["probabilities"] = {f"class_{i}": float(pred_data[i]) for i in range(len(pred_data))}

                # Add predicted class
                max_idx = int(np.argmax(pred_data))
                result["predicted_class"] = list(result["probabilities"].keys())[max_idx]
                result["confidence"] = float(pred_data[max_idx])

            elif pred_data.ndim == 2:  # 2D array (batch of predictions)
                # Take the first item if it's a batch
                if pred_data.shape[0] == 1:
                    return self._process_prediction(pred_data[0], filename)
                else:
                    # Multiple predictions
                    result["predictions"] = pred_data.tolist()

            else:
                # Other array shapes - just convert to list
                result["data"] = pred_data.tolist()
                result["shape"] = list(pred_data.shape)

        elif isinstance(pred_data, (list, tuple)):
            result["predictions"] = list(pred_data)

        elif isinstance(pred_data, dict):
            # Already a dict, merge it
            result.update(pred_data)

        else:
            # Try to convert to string
            result["prediction"] = str(pred_data)

        return result

    def _print_classification_summary(self, result: Dict[str, Any]):
        """Print a summary of classification results."""
        print(f"\nClassification results for {result['filename']}:")
        probs = result.get("probabilities", {})
        for class_name, prob in probs.items():
            print(f"  {class_name}: {prob:.4f}")
        if "predicted_class" in result:
            print(f"  Predicted: {result['predicted_class']} (confidence: {result['confidence']:.4f})")


def test():
    """Test the JSONResultsWriter operator."""
    import tempfile

    import numpy as np

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test the operator
        fragment = Fragment()
        writer = JSONResultsWriter(fragment, output_folder=temp_path)

        # Simulate setup
        from monai.deploy.core import OperatorSpec

        spec = OperatorSpec()
        writer.setup(spec)

        # Test cases
        class MockInput:
            def __init__(self, pred, filename=None):
                self.pred = pred
                self.filename = filename

            def receive(self, name):
                if name == "pred":
                    return self.pred
                elif name == "filename":
                    if self.filename:
                        return self.filename
                    raise Exception("No filename")

        # Test 1: Classification probabilities
        print("Test 1: Classification probabilities")
        pred1 = {"pred": np.array([0.1, 0.7, 0.15, 0.05])}
        mock_input1 = MockInput(pred1, "test_image_1")
        writer.compute(mock_input1, None, None)

        # Test 2: Direct numpy array
        print("\nTest 2: Direct numpy array")
        pred2 = np.array([0.9, 0.05, 0.03, 0.02])
        mock_input2 = MockInput(pred2, "test_image_2")
        writer.compute(mock_input2, None, None)

        # Test 3: No filename provided
        print("\nTest 3: No filename provided")
        pred3 = {"classification": [0.2, 0.8]}
        mock_input3 = MockInput(pred3)
        writer.compute(mock_input3, None, None)

        # List generated files
        print("\nGenerated files:")
        for json_file in temp_path.glob("*.json"):
            print(f"  {json_file.name}")
            with open(json_file) as f:
                print(f"    Content: {json.load(f)}")


if __name__ == "__main__":
    test()
