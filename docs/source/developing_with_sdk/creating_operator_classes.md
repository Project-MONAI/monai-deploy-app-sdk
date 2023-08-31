# Creating Operator classes

## Operator Class

Each Operator class inherits [Operator](/modules/_autosummary/monai.deploy.core.Operator) class.

```{code-block} python
---
lineno-start: 1
caption: |
    An Operator class definition example
---
from pathlib import Path
from monai.deploy.core import (ExecutionContext, Image, InputContext,
                               Operator, OutputContext, OperatorSpec)


class MyOperator(Operator):
    """Sample Operator implementation."""

    def setup(self, spec: OperatorSpec):
        spec.input("image_path")
        spec.output("image")
        spec.output("metadata")

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        from skimage import filters, io

        # Get input image
        image_path = Path(op_input.receive("image_path"))  # omitting validation for brevity
        if image_path.is_dir():
            image_path = next(image_path.glob("*.*"))  # take the first file
        input_image = io.imread(image_path)

        # Get mask image
        mask = op_input.get("mask").asnumpy()

        # Apply filter
        output_image = filters.sobel(input_image, mask)

        # Prepare output metadata
        metadata = {"shape": input_image.shape, "dtype": input_image.dtype}

        # Set output
        op_output.emit(Image(output_image), "image")
        op_output.emit(metadata, "metadata")
```

### setup() method

In prior releases, the input and output properties of the operator are specified by using `@input` and `@output` decorators, but starting with release v0.6, the **<a href="../modules/_autosummary/monai.deploy.core.Operator.html#monai.deploy.core.Operator.setup">setup()</a>** method is used.

### compute() method

`compute()` method in Operator class is an abstract method that needs to be implemented by the Operator developer.

Please check the description of **<a href="../modules/_autosummary/monai.deploy.core.Operator.html#monai.deploy.core.Operator.compute">compute()</a>** method to find a way to access

1. Operator's input/output
2. Application's input/output
3. [Model](/modules/_autosummary/monai.deploy.core.models.Model)'s name/path/predictor

Note that, if the operator is a leaf operator in the workflow graph and the operator output is file(s) written to the file system, the path needs to be set using the operator's constructor or a named input for path object or string. The leaf operator can also have in-memory output(s) without requiring other operator(s) as receiver, if the output is configured correctly like in the [GaussionOperator in Simple Image Processing App](https://github.com/Project-MONAI/monai-deploy-app-sdk/blob/main/examples/apps/simple_imaging_app/gaussian_operator.py)

## Package dependencies

The dependencies of the operator need to be captured in a "[requirements.txt](https://pip.pypa.io/en/stable/cli/pip_install/#example-requirements-file)", instead of using the `@env` decorator as in earlier releases. The aggregated requirement definitions for an application are then store in a consolidated "[requirements.txt](https://pip.pypa.io/en/stable/cli/pip_install/#example-requirements-file)" file, to be installed at [packaging time](/developing_with_sdk/packaging_app).

## Creating a Reusable Operator

You can create a common Operator class so that other Operator classes can just inherit the common Operator and implement only part of the compute() method to handle specific cases.

Please refer to the following examples:

- [MedianOperator in Simple Image Processing App](https://github.com/Project-MONAI/monai-deploy-app-sdk/blob/main/examples/apps/simple_imaging_app/median_operator.py)
- <a href="../_modules/monai/deploy/operators/monai_seg_inference_operator.html#MonaiSegInferenceOperator">MonaiSegInferenceOperator</a> that inherits <a href="../_modules/monai/deploy/operators/inference_operator.html#InferenceOperator">InferenceOperator</a>
