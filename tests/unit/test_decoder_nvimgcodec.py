import time
from pathlib import Path

import numpy as np
import pytest
from pydicom import dcmread
from pydicom.data import get_testdata_files
from monai.deploy.operators.decoder_nvimgcodec import (
    SUPPORTED_TRANSFER_SYNTAXES,
    _is_nvimgcodec_available,
    register_as_decoder_plugin,
    unregister_as_decoder_plugin,
)

from pydicom.uid import (
    JPEG2000TransferSyntaxes,
    JPEGTransferSyntaxes,
)

def get_test_dicoms():
    for path in get_testdata_files("*.dcm"):
        try:
            dataset = dcmread(path, stop_before_pixels=True, force=True)
            transfer_syntax = dataset.file_meta.TransferSyntaxUID
            if transfer_syntax not in JPEG2000TransferSyntaxes and transfer_syntax not in JPEGTransferSyntaxes:
                continue
            yield path
        except Exception as e:
            continue


@pytest.mark.skipif(
    (not _is_nvimgcodec_available()),
    reason="NVIDIA nvimgcodec must be available",
)
@pytest.mark.parametrize("path", list(get_test_dicoms()))
def test_nvimgcodec_decoder_matches_default(path: str) -> None:
    """Ensure NVIDIA nvimgcodec decoder matches default decoding for supported syntaxes."""
    baseline_total = 0.0
    nvimgcodec_total = 0.0
    rtol = 0.01
    atol = 1.0
    
    baseline_pixels: np.ndarray = np.array([])
    nv_pixels: np.ndarray = np.array([])

    dataset = dcmread(path, stop_before_pixels=True, force=True)
    transfer_syntax = dataset.file_meta.TransferSyntaxUID

    # Baseline (default pydicom) decode
    default_decoder_errored = False
    nvimgcodec_decoder_errored = False
    try:
        ds_default = dcmread(path, force=True)
        start = time.perf_counter()
        baseline_pixels = ds_default.pixel_array
        baseline_total += time.perf_counter() - start
        print(f"Default decoder total time: {baseline_total:.4f}s")
    except Exception as e:
        print(f"default decoder errored: {e}")
        default_decoder_errored = True

    # Register the nvimgcodec decoder plugin and unregister it after each use.
    register_as_decoder_plugin()
    try:
        ds_custom = dcmread(path, force=True)
        start = time.perf_counter()
        nv_pixels = ds_custom.pixel_array
        nvimgcodec_total += time.perf_counter() - start
        print(f"nvimgcodec decoder total time: {nvimgcodec_total:.4f}s")
    except Exception as e:
        print(f"nvimgcodec decoder errored: {e}")
        nvimgcodec_decoder_errored = True
    finally:
        unregister_as_decoder_plugin()

    if default_decoder_errored and nvimgcodec_decoder_errored:
        return
    elif nvimgcodec_decoder_errored and not default_decoder_errored:
        assert False, f"nvimgcodec decoder errored: {nvimgcodec_decoder_errored} but default decoder succeeded"

    assert baseline_pixels.shape == nv_pixels.shape, f"Shape mismatch for {Path(path).name}"
    assert baseline_pixels.dtype == nv_pixels.dtype, f"Dtype mismatch for {Path(path).name}"
    np.testing.assert_allclose(baseline_pixels, nv_pixels, rtol=rtol, atol=atol)


if __name__ == "__main__":
    for path in get_test_dicoms():
        test_nvimgcodec_decoder_matches_default(path)
