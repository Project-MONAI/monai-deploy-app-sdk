import time
from pathlib import Path
from typing import Any

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


def get_test_dicoms(folder_path: str | None = None):
    """Use pydicom package's embedded test DICOM files for testing or a custom folder of DICOM files."""

    # function's own util function
    def _is_supported_dicom_file(path: str) -> bool:
        try:
            dataset = dcmread(path, stop_before_pixels=True)  # ignore non-compliant DICOM files
            transfer_syntax = dataset.file_meta.TransferSyntaxUID
            return transfer_syntax in SUPPORTED_TRANSFER_SYNTAXES
        except Exception:
            return False

    dcm_paths = []
    if folder_path is not None:
        folder_path_p = Path(folder_path)
        if folder_path_p.exists():
            dcm_paths = sorted(folder_path_p.glob("*.dcm"))
        else:
            raise FileNotFoundError(f"Custom folder {folder_path} does not exist")
    else:
        # use pydicom package's embedded test DICOM files for testing
        dcm_paths = [Path(x) for x in get_testdata_files("*.dcm")]

    for dcm_path in dcm_paths:
        if not _is_supported_dicom_file(str(dcm_path)):
            continue
        yield str(dcm_path)


@pytest.mark.skipif(
    (not _is_nvimgcodec_available()),
    reason="NVIDIA nvimgcodec must be available",
)
@pytest.mark.parametrize("path", list(get_test_dicoms()))
def test_nvimgcodec_decoder_matches_default(path: str) -> None:
    """Ensure NVIDIA nvimgcodec decoder matches default decoding for supported syntaxes."""

    rtol = 0.01
    atol = 1.0

    baseline_pixels: np.ndarray = np.array([])
    nv_pixels: np.ndarray = np.array([])

    # Baseline (default pydicom) decode
    default_decoder_errored = False
    nvimgcodec_decoder_errored = False
    default_decoder_error_message = None
    nvimgcodec_decoder_error_message = None
    transfer_syntax = None
    try:
        ds_default = dcmread(path)
        transfer_syntax = ds_default.file_meta.TransferSyntaxUID
        baseline_pixels = ds_default.pixel_array
    except Exception as e:
        default_decoder_error_message = f"{e}"
        default_decoder_errored = True

    # Register the nvimgcodec decoder plugin and unregister it after each use.
    register_as_decoder_plugin()
    try:
        ds_custom = dcmread(path)
        nv_pixels = ds_custom.pixel_array
    except Exception as e:
        nvimgcodec_decoder_error_message = f"{e}"
        nvimgcodec_decoder_errored = True
    finally:
        unregister_as_decoder_plugin()

    if default_decoder_errored and nvimgcodec_decoder_errored:
        print(
            f"All decoders encountered errors for transfer syntax {transfer_syntax} in {Path(path).name}:\n"
            f"Default decoder error: {default_decoder_error_message}\n"
            f"nvimgcodec decoder error: {nvimgcodec_decoder_error_message}"
        )
        return
    elif nvimgcodec_decoder_errored and not default_decoder_errored:
        raise AssertionError(f"nvimgcodec decoder errored: {nvimgcodec_decoder_errored} but default decoder succeeded")

    assert baseline_pixels.shape == nv_pixels.shape, f"Shape mismatch for {Path(path).name}"
    assert baseline_pixels.dtype == nv_pixels.dtype, f"Dtype mismatch for {Path(path).name}"
    np.testing.assert_allclose(baseline_pixels, nv_pixels, rtol=rtol, atol=atol)


def performance_test_nvimgcodec_decoder_against_defaults(folder_path: str | None = None) -> None:
    """Test and compare the performance of the nvimgcodec decoder against the default decoders
    with all DICOM files of supported transfer syntaxes in a custom folder or pidicom dataset"""

    total_baseline_time = 0.0
    total_nvimgcodec_time = 0.0

    files_tested_with_perf: dict[str, dict[str, Any]] = {}  # key: path, value: performance_metrics
    files_with_errors = []

    try:
        unregister_as_decoder_plugin()  # Make sure nvimgcodec decoder plugin is not registered
    except Exception:
        pass

    for path in get_test_dicoms(folder_path):
        try:
            ds_default = dcmread(path)
            transfer_syntax = ds_default.file_meta.TransferSyntaxUID
            start = time.perf_counter()
            _ = ds_default.pixel_array
            baseline_execution_time = time.perf_counter() - start
            total_baseline_time += baseline_execution_time

            perf: dict[str, Any] = {}
            perf["transfer_syntax"] = transfer_syntax
            perf["baseline_execution_time"] = baseline_execution_time
            files_tested_with_perf[path] = perf
        except Exception:
            files_with_errors.append(Path(path).name)
            continue

    # Register the nvimgcodec decoder plugin and unregister it after each use.
    register_as_decoder_plugin()
    combined_perf = {}
    for path, perf in files_tested_with_perf.items():
        try:
            ds_custom = dcmread(path)
            start = time.perf_counter()
            _ = ds_custom.pixel_array
            perf["nvimgcodec_execution_time"] = time.perf_counter() - start
            total_nvimgcodec_time += perf["nvimgcodec_execution_time"]
            combined_perf[path] = perf
        except Exception:
            continue
    unregister_as_decoder_plugin()

    # Performance of the nvimgcodec decoder against the default decoders
    # with all DICOM files of supported transfer syntaxes
    print(
        "## nvimgcodec decoder performance against Pydicom default decoders for all supported transfer syntaxes in the test dataset"
        "\n"
        "**Note:** nvImgCodec is well suited for multiple-frame DICOM files, where decoder initialization time is less of a"
        " percentage of total execution time "
        "\n\n"
        "| Transfer Syntax | Default Decoder Execution Time | nvimgcodec Decoder Execution Time | File Name |"
        "\n"
        "| --- | --- | --- | --- |"
    )

    for path, perf in combined_perf.items():
        print(
            f"| {perf['transfer_syntax']} | {perf['baseline_execution_time']:.4f} |"
            f" {perf['nvimgcodec_execution_time']:.4f} | {Path(path).name}"
        )
    print(f"| **TOTAL** | {total_baseline_time} | {total_nvimgcodec_time} | - |")
    print(f"\n\n__Files not tested due to errors encountered by default decoders__: \n{files_with_errors}")


if __name__ == "__main__":

    # Use pytest to test the functionality with pydicom embedded DICOM files of supported transfer syntaxes individually
    # python -m pytest test_decoder_nvimgcodec.py
    #
    # The following compares the performance of the nvimgcodec decoder against the default decoders
    # with DICOM files in pidicom embedded dataset or an optional custom folder
    performance_test_nvimgcodec_decoder_against_defaults()  # e.g. "/tmp/multi-frame-dcm"
