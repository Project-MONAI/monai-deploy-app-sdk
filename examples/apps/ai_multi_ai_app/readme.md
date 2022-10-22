# About the Multi-Model/Multi-AI Example

This example demonstrates how to create a multi-model/multi-AI application.

## The important steps
- Place the model TorchScripts in a defined folder structure, see below for details
- Pass the model name to the inference operator instance in the app
- Connect the input to and output from the inference operators, as required by the app

## Required model folder structure:
- The model TorchScripts, be it MONAI Bundle compliant or not, must be placed in a parent folder, whose path is used as the path to the model(s) on app execution
- Each TorchScript file needs to be in a sub-folder, whose name is the model name

An example is shown below, where the `parent_foler` name can be the app's own choosing, and the sub-folder names become model names, `pancreas_ct_dints` and `spleen_model`, respectively.
```
<parent_fodler>
├── pancreas_ct_dints
│   └── model.ts
└── spleen_ct
    └── model.ts
```

## Note:
- The TorchScript files of MONAI Bundles can be downloaded from MONAI Model Zoo, at https://github.com/Project-MONAI/model-zoo/tree/dev/models
- The input DICOM instances are from a DICOM Series of CT Abdomen study, similar to the ones used in the Spleen Segmentation example
- This example is purely for technical demonstration, not for clinical use.