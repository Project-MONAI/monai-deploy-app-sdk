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

## Environment Setup

The cu13 stack is installed into the uv-managed venv (`/tmp/monai-env/.venv`)
from the repo `requirements.txt` plus this app's `pyproject.toml`:

| Package | Pin | Notes |
|---------|-----|-------|
| `holoscan-cu13` | `>=4.0.0,<4.3.0` | Bundles the GXF 4.x runtime (see below) |
| `holoscan-cli` | `>=4.0.0,<4.3.0` | `holoscan` / `monai-deploy` CLIs |
| `cupy-cuda13x` | `>=13.6.0` | `cupy-cuda12x` must NOT coexist |
| `rmm-cu13` | `>=25.10.0` | GPU pool allocator |
| `torch` | cu130 build | CUDA 13 runtime |
| `pydicom` | `>=3.0.0` | SDK code uses the pydicom 3.x API |
| `highdicom` | `>=0.24.0` | 0.22.x is incompatible with pydicom 3.x |

### GXF 4.x runtime libraries (no separate install needed)

The native GXF runtime that `holoscan.flow_graphs` links against
(`libgxf_rmm.so`, `libgxf_std.so`, `libgxf_app.so`, `libgxf_core.so`,
`libgxf_cuda.so`, `libgxf_serialization.so`, `libgxf_ucx.so`, …) is
**bundled inside the `holoscan-cu13` wheel** under `holoscan/lib/` — there
is no separate package to install, and this step needs no GPU/driver.

If you see `ImportError: libgxf_*.so: cannot open shared object file` the
wheel installation is incomplete/corrupted (files listed in the wheel's
`RECORD` are missing on disk), not a missing dependency. Repair:

```bash
uv pip install --python /tmp/monai-env/.venv/bin/python \
    --force-reinstall --no-deps "holoscan-cu13==4.2.0"
```

Verify the repair (0 missing = healthy):

```bash
V=/tmp/monai-env/.venv/lib/python3.10/site-packages
awk -F',' 'NR>1 {print $1}' $V/holoscan_cu13-4.2.0.dist-info/RECORD | \
while read f; do [ -z "$f" ] && continue; [ -f "$V/$f" ] || echo "MISSING: $f"; done | wc -l
```

### GPU driver requirement

A **CUDA 13-capable NVIDIA driver** (R580 series or newer) is required to
actually run GPU code — importing the packages does not need it, but any
torch/CuPy/RMM device operation does. With an older driver (e.g. R570 /
CUDA 12.8) you will see `cudaErrorInsufficientDriver` and the RMM smoke
test (`bash .planning/scripts/test_rmm.py`) exits SKIP.

### Runtime notes

- Holoscan 4.x recommends a 32 MB thread stack: `ulimit -s 32768`
  (or `--ulimit stack=33554432` in Docker), otherwise expect a stack-size
  `RuntimeWarning` at import.

### Quick environment check

```bash
bash .planning/scripts/validate_venv.sh        # package presence (pip level)
/tmp/monai-env/.venv/bin/python -c "           # import-level (catches the
  import monai.deploy.core, holoscan.flow_graphs; print('stack OK')"  # corruption mode above)
/tmp/monai-env/.venv/bin/python .planning/scripts/test_rmm.py  # GPU level
```

## Running the app (once implemented)

```bash
python my_app -i <input_folder> -o <output_folder> -m <model_path>
```

## Building a MAP

```bash
monai-deploy package my_app -m <model_path> -c my_app/app.yaml -t <tag>
```
