# CCHMC nnU-Net Fast App

A streamlined MONAI Deploy Application Package (MAP) for running nnU-Net
segmentation models from CCHMC, optimized for reduced latency compared to
the multi-checkpoint ensemble baseline.

## Overview

This app follows the standard MONAI Deploy App SDK conventions:

- `my_app/app.py` — Application entry point and DAG composition
- `my_app/operators/` — Custom operators (inference, post-processing, etc.)
- `my_app/config/` — Model configuration and metadata files

## Status

**Scaffolded — Phase 0.** The directory structure and app skeleton are in
place. The full inference pipeline will be implemented in Phase 1.

## Dependencies

- Python >= 3.10
- `monai-deploy-app-sdk`
- `nnunetv2` (vendored, editable install)

## Running the app (once implemented)

```bash
python my_app -i <input_folder> -o <output_folder> -m <model_path>
```

## Building a MAP

```bash
monai-deploy package my_app -m <model_path> -c my_app/app.yaml -t <tag>
```
