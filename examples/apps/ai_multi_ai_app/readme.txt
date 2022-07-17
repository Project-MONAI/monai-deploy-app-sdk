This example is used to demonstrate how to create a multi-model/multi-AI application.

The PyTorch models, be it MONAI Bundle compliant TorchScript or simply TorchScript must
be placed in a parent folder, whose path is used as the path to the model when using App SDK
CLI commands, e.g.
    <your_models_folder>
    ├── pancreas_ct_dints
    │   └── model.ts
    └── spleen_model
        └── model.ts

Note that the name of a subfolder, which contains the TorchScript file, is used as the
model name within the app on retrieving the model's network and path etc.
