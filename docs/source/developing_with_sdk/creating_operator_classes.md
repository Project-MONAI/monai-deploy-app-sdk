# Creating Operator classes

## Operator Class

Each Operator class inherits [Operator](/modules/_autosummary/monai.deploy.core.Operator) class.

```{code-block} python
---
lineno-start: 1
caption: |
    An Operator class definition example
---
from typing import Any, Dict

import monai.deploy.core as md  # 'md' stands for MONAI Deploy (or can use 'core' instead)
from monai.deploy.core import (DataPath, ExecutionContext, Image, InputContext,
                               IOType, Operator, OutputContext)


@md.input("image", DataPath, IOType.DISK)
@md.input("mask", Image, IOType.IN_MEMORY)
@md.output("image", Image, IOType.IN_MEMORY)
@md.output("metadata", Dict[str, Any], IOType.IN_MEMORY)
@md.env(pip_packages=["scikit-image>=0.17.2"])
class MyOperator(Operator):
    """Sample Operator implementation."""

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        from skimage import filters, io

        # Get input image
        image_path = op_input.get("image").path
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
        op_output.set(Image(output_image), "image")
        op_output.set(metadata, "metadata")
```

### Decorators

The input and output properties of the operator are specified by using [@input](/modules/_autosummary/monai.deploy.core.input) and [@output](/modules/_autosummary/monai.deploy.core.output) decorators.

[@input](/modules/_autosummary/monai.deploy.core.input) and [@output](/modules/_autosummary/monai.deploy.core.output) decorators accept (**`<Label>`**, [`<Data Type>`](/modules/domain_objects), [`<Storage Type>`](/modules/_autosummary/monai.deploy.core.IOType)) as parameters.

If no `@input` or `@output` decorator is specified, the following properties are used by default:

```python
@md.input("", DataPath, IOType.DISK)  # if no @input decorator is specified.
@md.output("", DataPath, IOType.DISK)  # if no @output decorator is specified.
```

[@env](/modules/_autosummary/monai.deploy.core.env) accepts `pip_packages` parameter as a string that is a path to requirements.txt file or a list of packages to install. If `pip_packages` is specified, the definition will be aggregated with the package dependency list of other operators and the application. The aggregated requirement definitions are stored as a "[requirements.txt](https://pip.pypa.io/en/stable/cli/pip_install/#example-requirements-file)" file and it would be installed in [packaging time](/developing_with_sdk/executing_packaged_app_locally).

### compute() method

`compute()` method in Operator class is an abstract method that needs to be implemented by the Operator developer.

Please check the description of **<a href="../modules/_autosummary/monai.deploy.core.Operator.html#monai.deploy.core.Operator.compute">compute()</a>** method to find a way to access

1. Operator's input/output
2. Application's input/output
3. [Model](/modules/_autosummary/monai.deploy.core.models.Model)'s name/path/predictor

Note that, if the operator is a leaf operator in the workflow graph and the operator output's `(<data type>, <storage type>) == (DataPath, DISK)`, you cannot call `op_output.set()` method.
Instead, you can use the destination path available by `op_output.get().path` to store output data and the
following logic is expected:

```python
output_folder = op_output.get().path                 # get the output folder path
output_path = output_folder / "final_output.png"     # get the output file path
imsave(output_path, data_out)                        # save the output data
```

## Creating a Reusable Operator

You can create a common Operator class so that other Operator classes can just inherit the common Operator and implement only part of the compute() method to handle specific cases.

Please refer to the following examples:

- [MedianOperator in Simple Image Processing App](https://github.com/Project-MONAI/monai-deploy-app-sdk/blob/main/examples/apps/simple_imaging_app/median_operator.py)
- <a href="../_modules/monai/deploy/operators/monai_seg_inference_operator.html#MonaiSegInferenceOperator">MonaiSegInferenceOperator</a> that inherits <a href="../_modules/monai/deploy/operators/inference_operator.html#InferenceOperator">InferenceOperator</a>
