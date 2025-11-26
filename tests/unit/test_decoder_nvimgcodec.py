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

# The JPEG 8-bit standard allows a maximum 1 bit of difference for each pixel component.
# It is normal for slight differences to exist due to varying internal precision in the
# decoders' inverse Discrete Cosine Transform (IDCT) implementations
# So, need to more closely inspect the pixel values for these transfer syntaxes.
TRANSFER_SYNTAXES_WITH_UNEQUAL_PIXEL_VALUES = [
    "1.2.840.10008.1.2.4.50",
    "1.2.840.10008.1.2.4.91",
    "1.2.840.10008.1.2.4.203",
]

# Files that cannot be decoded with the default decoders
SKIPPING_DEFAULT_ERRORED_FILES = {
    "UN_sequence.dcm": "1.2.840.10008.1.2.4.70",
    "JPEG-lossy.dcm": "1.2.840.10008.1.2.4.51",
    "JPEG2000-embedded-sequence-delimiter.dcm": "1.2.840.10008.1.2.4.91",
    "emri_small_jpeg_2k_lossless_too_short.dcm": "1.2.840.10008.1.2.4.90",
}

# Files that have unequal pixel values between default and nvimgcodec decoders
SKIPPING_UNEQUAL_PIXEL_FILES = {
    "SC_rgb_jpeg_lossy_gdcm.dcm": "1.2.840.10008.1.2.4.50",
    "SC_rgb_dcmtk_+eb+cr.dcm": "1.2.840.10008.1.2.4.50",
    "SC_rgb_jpeg_dcmtk.dcm": "1.2.840.10008.1.2.4.50",
    "SC_rgb_dcmtk_+eb+cy+n1.dcm": "1.2.840.10008.1.2.4.50",
    "SC_rgb_dcmtk_+eb+cy+s2.dcm": "1.2.840.10008.1.2.4.50",
    "examples_ybr_color.dcm": "1.2.840.10008.1.2.4.50",
    "SC_rgb_dcmtk_+eb+cy+n2.dcm": "1.2.840.10008.1.2.4.50",
    "SC_rgb_dcmtk_+eb+cy+s4.dcm": "1.2.840.10008.1.2.4.50",
    "SC_rgb_jpeg.dcm": "1.2.840.10008.1.2.4.50",
    "SC_rgb_jpeg_app14_dcmd.dcm": "1.2.840.10008.1.2.4.50",
    "SC_jpeg_no_color_transform.dcm": "1.2.840.10008.1.2.4.50",
    "SC_jpeg_no_color_transform_2.dcm": "1.2.840.10008.1.2.4.50",
    "SC_rgb_small_odd_jpeg.dcm": "1.2.840.10008.1.2.4.50",
    "SC_rgb_dcmtk_+eb+cy+np.dcm": "1.2.840.10008.1.2.4.50",
    "color3d_jpeg_baseline.dcm": "1.2.840.10008.1.2.4.50",
    "MR2_J2KI.dcm": "1.2.840.10008.1.2.4.91",
    "RG3_J2KI.dcm": "1.2.840.10008.1.2.4.91",
    "US1_J2KI.dcm": "1.2.840.10008.1.2.4.91",
}

CONFIRMED_EQUAL_PIXEL_FILES = {
    "JPGExtended.dcm": "1.2.840.10008.1.2.4.51",
    "examples_jpeg2k.dcm": "1.2.840.10008.1.2.4.90",
    "J2K_pixelrep_mismatch.dcm": "1.2.840.10008.1.2.4.90",
    "SC_rgb_gdcm_KY.dcm": "1.2.840.10008.1.2.4.91",
    "GDCMJ2K_TextGBR.dcm": "1.2.840.10008.1.2.4.90",
    "SC_rgb_jpeg_gdcm.dcm": "1.2.840.10008.1.2.4.70",
    "MR_small_jp2klossless.dcm": "1.2.840.10008.1.2.4.90",
    "JPEG2000.dcm": "1.2.840.10008.1.2.4.91",
    "693_J2KI.dcm": "1.2.840.10008.1.2.4.91",
    "693_J2KR.dcm": "1.2.840.10008.1.2.4.90",
    "bad_sequence.dcm": "1.2.840.10008.1.2.4.70",
    "emri_small_jpeg_2k_lossless.dcm": "1.2.840.10008.1.2.4.90",
    "explicit_VR-UN.dcm": "1.2.840.10008.1.2.4.90",
    "JPEG-LL.dcm": "1.2.840.10008.1.2.4.70",
    "JPGLosslessP14SV1_1s_1f_8b.dcm": "1.2.840.10008.1.2.4.70",
    "MR2_J2KR.dcm": "1.2.840.10008.1.2.4.90",
    "RG1_J2KI.dcm": "1.2.840.10008.1.2.4.91",
    "RG1_J2KR.dcm": "1.2.840.10008.1.2.4.90",
    "RG3_J2KR.dcm": "1.2.840.10008.1.2.4.90",
    "US1_J2KR.dcm": "1.2.840.10008.1.2.4.90",
}


@pytest.mark.skipif(not _is_nvimgcodec_available(), reason="nvimgcodec dependencies unavailable")
def test_nvimgcodec_decoder_matches_default():
    """Ensure nvimgcodec decoder matches default decoding for supported transfer syntaxes."""

    test_files = get_testdata_files("*.dcm")
    baseline_total = 0.0
    nvimgcodec_total = 0.0
    compared = 0
    _rtol = 0.01  # The relative tolerance parameter
    _atol = 1.00  # The absolute tolerance parameter

    default_errored_files = dict()
    nvimgcodec_errored_files = dict()
    unequal_pixel_files = dict()
    inspected_unequal_files = dict()
    confirmed_equal_pixel_files = dict()

    for path in test_files:
        default_errored = False
        nvimgcodec_errored = False

        try:
            dataset = dcmread(path, stop_before_pixels=True, force=True)
            transfer_syntax = dataset.file_meta.TransferSyntaxUID
        except Exception as e:
            print(f"Skipping: error reading DICOM file {path}: {e}")
            continue

        if transfer_syntax not in SUPPORTED_TRANSFER_SYNTAXES:
            print(f"Skipping: unsupported transfer syntax DICOM file {path}: {transfer_syntax}")
            continue

        try:
            ds_default = dcmread(path, force=True)
            start = time.perf_counter()
            baseline_pixels = ds_default.pixel_array
            baseline_total += time.perf_counter() - start
        except Exception as e:
            # Skip files that cannot be decoded with the default backend
            print(f"Skipping: default backends cannot decode DICOM file {path}: {e}")
            default_errored_files[Path(path).name] = transfer_syntax
            default_errored = True
            # Let's see if nvimgcodec can decode it.

        # Register the nvimgcodec decoder plugin and unregister it after each use.
        register_as_decoder_plugin()
        try:
            ds_custom = dcmread(path, force=True)
            start = time.perf_counter()
            nv_pixels = ds_custom.pixel_array
            nvimgcodec_total += time.perf_counter() - start
        except Exception as e:
            print(f"Skipping: nvimgcodec cannot decode DICOM file {path}: {e}")
            nvimgcodec_errored_files[Path(path).name] = transfer_syntax
            nvimgcodec_errored = True
        finally:
            unregister_as_decoder_plugin()

        if default_errored or nvimgcodec_errored:
            print(f"Skipping: either default or nvimgcodec decoder or both failed to decode DICOM file {path}")
            continue

        assert baseline_pixels.shape == nv_pixels.shape, f"Shape mismatch for {Path(path).name}"

        if baseline_pixels.dtype != nv_pixels.dtype:
            baseline_compare = baseline_pixels.astype(np.float32)
            nv_compare = nv_pixels.astype(np.float32)
        else:
            baseline_compare = baseline_pixels
            nv_compare = nv_pixels

        if not np.allclose(
            baseline_compare,
            nv_compare,
            rtol=_rtol,
            atol=_atol,
        ):
            if transfer_syntax in TRANSFER_SYNTAXES_WITH_UNEQUAL_PIXEL_VALUES:
                diff = baseline_compare.astype(np.float32) - nv_compare.astype(np.float32)
                peak_absolute_error = float(np.max(np.abs(diff)))
                mean_squared_error = float(np.mean(diff**2))
                inspected_unequal_files[Path(path).name] = {
                    "transfer_syntax": transfer_syntax,
                    "peak_absolute_error": peak_absolute_error,
                    "mean_squared_error": mean_squared_error,
                }
            else:
                unequal_pixel_files[Path(path).name] = {"transfer_syntax": transfer_syntax}
        else:
            confirmed_equal_pixel_files[Path(path).name] = transfer_syntax

        compared += 1

    print(f"Default decoder total time: {baseline_total:.4f}s")
    print(f"nvimgcodec decoder total time: {nvimgcodec_total:.4f}s")
    print(f"Total tested DICOM files: {compared}")
    print(f"Default errored files: {default_errored_files}")
    print(f"nvimgcodec errored files: {nvimgcodec_errored_files}")
    print(f"Unequal files (tolerance: {_rtol}, {_atol}): {unequal_pixel_files}")
    print(f"Inspected unequal files (tolerance: {_rtol}, {_atol}): {inspected_unequal_files}")
    print(f"Confirmed tested files: {confirmed_equal_pixel_files}")

    assert compared > 0, "No compatible DICOM files found for nvimgcodec decoder test."
    assert (
        x in default_errored_files.keys() for x in (nvimgcodec_errored_files.keys())
    ), "nvimgcodec decoder errored files found."
    assert len(unequal_pixel_files) == 0, "Unequal files found."
    assert len(confirmed_equal_pixel_files) > 0, "No files with equal pixel values after decoding with both decoders."


if __name__ == "__main__":
    test_nvimgcodec_decoder_matches_default()
