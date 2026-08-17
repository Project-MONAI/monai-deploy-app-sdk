#!/usr/bin/env python3
"""
Demo target for the Nsight profiling harness (task 0.6).

Simulates the three pipeline stages of the segmentation app (preprocess /
inference / postprocess) with representative GPU work, wrapped in NVTX ranges,
and brackets them with the CUDA profiler API so that
``nsight_profile.sh --capture-range=cudaProfilerApi`` captures exactly this
region. Running this through the harness proves the end-to-end trace path
(harness -> NVTX -> .nsys-rep) without needing the full app instrumented.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvtx_markers import push_range, range_colors  # noqa: E402

import torch  # noqa: E402


def main():
    assert torch.cuda.is_available(), "GPU required"
    dev = torch.device("cuda")
    torch.cuda.init()
    print(f"[demo] device: {torch.cuda.get_device_name(0)}")

    # ~62-slice-ish volume, 432x432 (matches the airway corpus scale)
    volume = torch.randn(1, 256, 256, 256, device=dev, dtype=torch.float32)

    torch.cuda.synchronize()
    torch.cuda.profiler.start()

    with push_range("preprocess: transpose/crop/normalize", range_colors.preprocess):
        vol = volume.transpose(1, 2)
        vol = (vol - vol.mean()) / (vol.std() + 1e-8)

    with push_range("inference: sliding-window ensemble (x3)", range_colors.inference):
        x = torch.randn(4096, 4096, device=dev, dtype=torch.float32)
        w = torch.randn(4096, 4096, device=dev, dtype=torch.float32)
        for i in range(3):
            torch.cuda.nvtx.range_push(f"  tile ensemble member {i}")
            out = x @ w
            _ = out.sum()
            torch.cuda.nvtx.range_pop()

    with push_range("postprocess: argmax/connected-components", range_colors.postprocess):
        seg = (vol > 0).to(torch.uint8)
        seg = seg.flip(1)  # simulate revert-transform

    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"[demo] done; seg positives: {int(seg.sum().item())}")


if __name__ == "__main__":
    main()
