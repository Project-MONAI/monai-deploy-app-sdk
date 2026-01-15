import logging
import time
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
import pytest
from pydicom import dcmread
from pydicom.data import get_testdata_files

from monai.deploy.operators.decoder_nvimgcodec import (
    SUPPORTED_DECODER_CLASSES,
    SUPPORTED_TRANSFER_SYNTAXES,
    _is_nvimgcodec_available,
    register_as_decoder_plugin,
    unregister_as_decoder_plugin,
)

try:
    from PIL import Image as PILImage
except Exception:  # pragma: no cover - Pillow may be unavailable in some environments
    PILImage = None  # type: ignore[assignment]


_PNG_EXPORT_WARNING_EMITTED = False

_IGNORED_FILES_STEMS = ["GDCMJ2K_TextGBR".lower()]

_DEFAULT_PLUGIN_CACHE: dict[str, Any] = {}
_logger = logging.getLogger(__name__)


def _iter_frames(pixel_array: np.ndarray) -> Iterator[tuple[int, np.ndarray, bool]]:
    """Yield per-frame arrays and whether they represent color data."""
    arr = np.asarray(pixel_array)
    if arr.ndim == 2:
        yield 0, arr, False
        return

    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            yield 0, arr, True
        else:
            for index in range(arr.shape[0]):
                frame = arr[index]
                yield index, frame, False  # grayscale multi-frame images
        return

    if arr.ndim == 4:
        for index in range(arr.shape[0]):
            frame = arr[index]
            is_color = frame.shape[-1] in (3, 4)
            yield index, frame, is_color
        return

    raise ValueError(f"Unsupported pixel array shape {arr.shape!r} for PNG export")


def _prepare_frame_for_png(frame: np.ndarray, is_color: bool) -> np.ndarray:
    """Convert a decoded frame into a dtype supported by PNG writers."""
    arr = np.nan_to_num(np.asarray(frame), copy=False)

    # Remove singleton channel dimension for grayscale data.
    if not is_color and arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    arr_float = arr.astype(np.float64, copy=False)
    if np.issubdtype(arr.dtype, np.integer):
        arr_min = float(arr.min())
        arr_max = float(arr.max())
    else:
        arr_min = float(arr_float.min())
        arr_max = float(arr_float.max())

    if is_color:
        if arr.dtype == np.uint8:
            return arr
        if arr_max == arr_min:
            return np.zeros_like(arr, dtype=np.uint8)
        scaled = (arr_float - arr_min) / (arr_max - arr_min)
        return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)  # type: ignore[no-any-return]

    # Grayscale path
    if np.issubdtype(arr.dtype, np.integer):
        if arr_min >= 0 and arr_max <= 255:
            return arr.astype(np.uint8, copy=False)
        if arr_min >= 0 and arr_max <= 65535:
            return arr.astype(np.uint16, copy=False)

    if arr_max == arr_min:
        return np.zeros_like(arr_float, dtype=np.uint8)

    use_uint16 = arr_max - arr_min > 255.0
    scale = 65535.0 if use_uint16 else 255.0
    scaled = (arr_float - arr_min) / (arr_max - arr_min)
    scaled = np.clip(np.round(scaled * scale), 0, scale)
    target_dtype = np.uint16 if use_uint16 else np.uint8
    return scaled.astype(target_dtype)  # type: ignore[no-any-return]


def _save_frames_as_png(pixel_array: np.ndarray, output_dir: Path, file_stem: str) -> None:
    """Persist each frame as a PNG image in the specified directory."""
    global _PNG_EXPORT_WARNING_EMITTED

    if PILImage is None:
        if not _PNG_EXPORT_WARNING_EMITTED:
            _logger.info("Skipping PNG export because Pillow is not installed.")
            _PNG_EXPORT_WARNING_EMITTED = True
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    pil_image_cls = cast(Any, PILImage)

    for frame_index, frame, is_color in _iter_frames(pixel_array):
        frame_for_png = _prepare_frame_for_png(frame, is_color)
        image = pil_image_cls.fromarray(frame_for_png)
        filename = output_dir / f"{file_stem}_frame_{frame_index:04d}.png"
        image.save(filename)


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
        if dcm_path.stem.lower() in _IGNORED_FILES_STEMS:
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
    atol = 4.0

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

    # Remove and cache the other default decoder plugins first
    _remove_default_plugins()
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
        _restore_default_plugins()

    if default_decoder_errored and nvimgcodec_decoder_errored:
        _logger.info(
            f"All decoders encountered errors for transfer syntax: {transfer_syntax} in file: {Path(path).name}:\n"
            f"Default decoder error: {default_decoder_error_message}\n"
            f"nvimgcodec decoder error: {nvimgcodec_decoder_error_message}"
        )
        return
    elif nvimgcodec_decoder_errored and not default_decoder_errored:
        raise AssertionError(
            f"Only nvimgcodec encountered errors for transfer syntax: {transfer_syntax} in file: {Path(path).name}:\n"
            f"with error: {nvimgcodec_decoder_error_message}"
        )

    assert baseline_pixels.shape == nv_pixels.shape, f"Shape mismatch with transfer syntax {transfer_syntax}"
    assert baseline_pixels.dtype == nv_pixels.dtype, f"Dtype mismatch with transfer syntax {transfer_syntax}"
    try:
        np.testing.assert_allclose(baseline_pixels, nv_pixels, rtol=rtol, atol=atol)
        _logger.info(f"Pixels values matched for transfer syntax: {transfer_syntax} in file: {Path(path).name}")
    except AssertionError as e:
        raise AssertionError(
            f"Pixels values mismatch for transfer syntax: {transfer_syntax} in file: {Path(path).name}"
        ) from e


def performance_test_nvimgcodec_decoder_against_defaults(
    folder_path: str | None = None, png_output_dir: str | None = None
) -> None:
    """Test and compare the performance of the nvimgcodec decoder against the default decoders
    with all DICOM files of supported transfer syntaxes in a custom folder or pydicom embedded dataset.

    If `png_output_dir` is provided, decoded frames are saved as PNG files for both decoders."""

    total_baseline_time = 0.0
    total_nvimgcodec_time = 0.0

    files_tested_with_perf: dict[str, dict[str, Any]] = {}  # key: path, value: performance_metrics
    files_with_errors = []
    png_root = Path(png_output_dir).expanduser() if png_output_dir else None

    try:
        unregister_as_decoder_plugin()  # Make sure nvimgcodec decoder plugin is not registered
    except Exception:
        pass

    for path in get_test_dicoms(folder_path):
        try:
            ds_default = dcmread(path)
            transfer_syntax = ds_default.file_meta.TransferSyntaxUID
            start = time.perf_counter()
            baseline_pixels = ds_default.pixel_array
            baseline_execution_time = time.perf_counter() - start
            total_baseline_time += baseline_execution_time

            perf: dict[str, Any] = {}
            perf["transfer_syntax"] = transfer_syntax
            perf["baseline_execution_time"] = baseline_execution_time
            files_tested_with_perf[path] = perf

            if png_root is not None:
                baseline_dir = png_root / Path(path).stem / "default"
                _save_frames_as_png(baseline_pixels, baseline_dir, Path(path).stem)
        except Exception:
            files_with_errors.append(Path(path).name)
            continue

    _remove_default_plugins()
    # Register the nvimgcodec decoder plugin and unregister it after each use.
    register_as_decoder_plugin()

    combined_perf = {}
    for path, perf in files_tested_with_perf.items():
        try:
            ds_custom = dcmread(path)
            start = time.perf_counter()
            nv_pixels = ds_custom.pixel_array
            perf["nvimgcodec_execution_time"] = time.perf_counter() - start
            total_nvimgcodec_time += perf["nvimgcodec_execution_time"]
            combined_perf[path] = perf

            if png_root is not None:
                nv_dir = png_root / Path(path).stem / "nvimgcodec"
                _save_frames_as_png(nv_pixels, nv_dir, Path(path).stem)
        except Exception as e:
            _logger.info(f"Error decoding {path} with nvimgcodec decoder: {e}")
            continue
    unregister_as_decoder_plugin()
    _restore_default_plugins()

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


def _remove_default_plugins():
    """Remove the default plugins from the supported decoder classes."""

    global _DEFAULT_PLUGIN_CACHE
    _logger.debug("Removing default plugins from the supported decoder classes.")

    for decoder_class in SUPPORTED_DECODER_CLASSES:
        _DEFAULT_PLUGIN_CACHE[decoder_class.UID.name] = (
            decoder_class._available
        )  # white box, no API to get DecodeFunction
        decoder_class._available = {}  # remove all plugins, ref is still held by _DEFAULT_PLUGIN_CACHE
        _logger.info(f"Removed default plugins of {decoder_class.UID.name}: {decoder_class.available_plugins}.")


def _restore_default_plugins():
    """Restore the default plugins to the supported decoder classes."""

    global _DEFAULT_PLUGIN_CACHE
    _logger.debug("Restoring default plugins to the supported decoder classes.")

    for decoder_class in SUPPORTED_DECODER_CLASSES:
        decoder_class._available = _DEFAULT_PLUGIN_CACHE[decoder_class.UID.name]  # restore all plugins
        _logger.info(f"Restored default plugins of {decoder_class.UID.name}: {decoder_class.available_plugins}.")
    # Clear the cache
    _DEFAULT_PLUGIN_CACHE = {}


if __name__ == "__main__":

    # Use pytest to test the functionality with pydicom embedded DICOM files of supported transfer syntaxes individually
    # python -m pytest test_decoder_nvimgcodec.py
    #
    # The following compares the performance of the nvimgcodec decoder against the default decoders
    # with DICOM files in pydicom embedded dataset or an optional custom folder
    performance_test_nvimgcodec_decoder_against_defaults(
        png_output_dir="decoded_png"
    )  # or use (folder_path="/data/dcm", png_output_dir="decoded_png")
