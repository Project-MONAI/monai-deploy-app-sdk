# Copyright 2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This decoder plugin for nvimgcodec <https://github.com/NVIDIA/nvImageCodec> decompresses
# encoded Pixel Data for the following transfer syntaxes:
#     JPEGBaseline8Bit, # 1.2.840.10008.1.2.4.50, JPEG Baseline (Process 1)
#     JPEGExtended12Bit, # 1.2.840.10008.1.2.4.51, JPEG Extended (Process 2 & 4)
#     JPEGLossless, # 1.2.840.10008.1.2.4.57, JPEG Lossless, Non-Hierarchical (Process 14)
#     JPEGLosslessSV1, # 1.2.840.10008.1.2.4.70, JPEG Lossless, Non-Hierarchical, First-Order Prediction
#     JPEG2000Lossless, # 1.2.840.10008.1.2.4.90, JPEG 2000 Image Compression (Lossless Only)
#     JPEG2000, # 1.2.840.10008.1.2.4.91, JPEG 2000 Image Compression
#     HTJ2KLossless, # 1.2.840.10008.1.2.4.201, HTJ2K Image Compression (Lossless Only)
#     HTJ2KLosslessRPCL, # 1.2.840.10008.1.2.4.202, HTJ2K with RPCL Options Image Compression (Lossless Only)
#     HTJ2K, # 1.2.840.10008.1.2.4.203, HTJ2K Image Compression
#
# There are two ways to add a custom decoding plugin to pydicom:
# 1. Using the pixel_data_handlers backend, though pydicom.pixel_data_handlers module is deprecated
#    and will be removed in v4.0.
# 2. Using the pixels backend by adding a decoder plugin to existing decoders with the add_plugin method,
#    see https://pydicom.github.io/pydicom/stable/guides/decoding/decoder_plugins.html
#
# It is noted that pydicom.dataset.Dataset.pixel_array changed in version 3.0 where the backend used for
# pixel data decoding changed from the pixel_data_handlers module to the pixels module.
#
# So, this implementation uses the pixels backend.
#
# Plugin Requirements:
# A custom decoding plugin must implement three objects within the same module:
#  - A function named is_available with the following signature:
#     def is_available(uid: pydicom.uid.UID) -> bool:
#       Where uid is the Transfer Syntax UID for the corresponding decoder as a UID
#  - A dict named DECODER_DEPENDENCIES with the type dict[pydicom.uid.UID, tuple[str, ...], such as:
#     DECODER_DEPENDENCIES = {JPEG2000Lossless: ('numpy', 'pillow', 'imagecodecs'),}
#       This will be used to provide the user with a list of dependencies required by the plugin.
#  - A function that performs the decoding with the following function signature as in Github repo:
#     def _decode_frame(src: bytes, runner: DecodeRunner) -> bytearray | bytes
#       src is a single frame’s worth of raw compressed data to be decoded, and
#       runner is a DecodeRunner instance that manages the decoding process.
#
# Adding plugins to a Decoder:
# Additional plugins can be added to an existing decoder with the add_plugin() method
#  ```python
#   from pydicom.pixels.decoders import RLELosslessDecoder
#   RLELosslessDecoder.add_plugin(
#     'my_decoder',  # the plugin's label
#     ('my_package.decoders', 'my_decoder_func')  # the import paths
#   )
#  ```


import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from pydicom.pixels.decoders import (
    HTJ2KDecoder,
    HTJ2KLosslessDecoder,
    HTJ2KLosslessRPCLDecoder,
    JPEG2000Decoder,
    JPEG2000LosslessDecoder,
    JPEGBaseline8BitDecoder,
    JPEGExtended12BitDecoder,
    JPEGLosslessDecoder,
    JPEGLosslessSV1Decoder,
)
from pydicom.pixels.decoders.base import DecodeRunner
from pydicom.pixels.utils import _passes_version_check
from pydicom.uid import UID

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from nvidia import nvimgcodec as nvimgcodec

    nvimgcodec_version = tuple(int(x) for x in nvimgcodec.__version__.split("."))
except ImportError:
    nvimgcodec = None

# nvimgcodec pypi package name, minimum version required and the label for this decoder plugin.
NVIMGCODEC_MODULE_NAME = "nvidia.nvimgcodec"  # from nvidia-nvimgcodec-cu12 or other variants
NVIMGCODEC_MIN_VERSION = "0.6"
NVIMGCODEC_MIN_VERSION_TUPLE = tuple(int(x) for x in NVIMGCODEC_MIN_VERSION.split("."))
NVIMGCODEC_PLUGIN_LABEL = "0.6+nvimgcodec"  # to be sorted to first in ascending order of plugins
NVIMGCODEC_PLUGIN_FUNC_NAME = "_decode_frame"

# Supported decoder classes of the corresponding transfer syntaxes by this decoder plugin.
SUPPORTED_DECODER_CLASSES = [
    JPEGBaseline8BitDecoder,  # 1.2.840.10008.1.2.4.50, JPEG Baseline (Process 1)
    JPEGExtended12BitDecoder,  # 1.2.840.10008.1.2.4.51, JPEG Extended (Process 2 & 4)
    JPEGLosslessDecoder,  # 1.2.840.10008.1.2.4.57, JPEG Lossless, Non-Hierarchical (Process 14)
    JPEGLosslessSV1Decoder,  # 1.2.840.10008.1.2.4.70, JPEG Lossless, Non-Hierarchical, First-Order Prediction
    JPEG2000LosslessDecoder,  # 1.2.840.10008.1.2.4.90, JPEG 2000 Image Compression (Lossless Only)
    JPEG2000Decoder,  # 1.2.840.10008.1.2.4.91, JPEG 2000 Image Compression
    HTJ2KLosslessDecoder,  # 1.2.840.10008.1.2.4.201, HTJ2K Image Compression (Lossless Only)
    HTJ2KLosslessRPCLDecoder,  # 1.2.840.10008.1.2.4.202, HTJ2K with RPCL Options Image Compression (Lossless Only)
    HTJ2KDecoder,  # 1.2.840.10008.1.2.4.203, HTJ2K Image Compression
]

SUPPORTED_TRANSFER_SYNTAXES: Iterable[UID] = [x.UID for x in SUPPORTED_DECODER_CLASSES]

_logger = logging.getLogger(__name__)

# Required for decoder plugin
DECODER_DEPENDENCIES = {
    x: ("numpy", "cupy", f"{NVIMGCODEC_MODULE_NAME}>={NVIMGCODEC_MIN_VERSION}") for x in SUPPORTED_TRANSFER_SYNTAXES
}


# Required for decoder plugin
def is_available(uid: UID) -> bool:
    """Return ``True`` if a pixel data decoder for ``uid`` is available.

    Args:
        uid (UID): The transfer syntax UID to check.

    Returns:
        bool: ``True`` if a pixel data decoder for ``uid`` is available,
        ``False`` otherwise.
    """

    _logger.debug(f"Checking if CUDA and nvimgcodec available for transfer syntax: {uid}")

    if uid not in SUPPORTED_TRANSFER_SYNTAXES:
        _logger.debug(f"Transfer syntax {uid} not supported by nvimgcodec.")
        return False
    if not _is_nvimgcodec_available():
        _logger.debug(f"Module {NVIMGCODEC_MODULE_NAME} is not available.")
        return False

    return True


# Required function for decoder plugin (specific signature but flexible name to be registered to a decoder)
# see also https://github.com/pydicom/pydicom/blob/v3.0.1/src/pydicom/pixels/decoders/base.py#L334
def _decode_frame(src: bytes, runner: DecodeRunner) -> bytearray | bytes:
    """Return the decoded image data in `src` as a :class:`bytearray` or :class:`bytes`.

    This function is called by the pydicom.pixels.decoders.base.DecodeRunner.decode method.

    Args:
        src (bytes): An encoded frame of pixel data to be passed to the decoding plugins.
        runner (DecodeRunner): The runner instance that manages the decoding process.

    Returns:
        bytearray | bytes: The decoded frame as a :class:`bytearray` or :class:`bytes`.
    """

    # The frame data bytes object is passed in by the runner, which it gets via pydicom.encaps.get_frame
    # and other pydicom.encaps functions, e.g. pydicom.encaps.generate_frames, generate_fragmented_frames, etc.
    # So we can directly decode the frame using nvimgcodec.

    # Though a fragment may not contain encoded data from more than one frame, the encoded data from one frame
    # may span multiple fragments to support buffering during compression or to avoid exceeding the maximum size
    # of a fixed length fragment, see https://dicom.nema.org/dicom/2013/output/chtml/part05/sect_8.2.html.
    # In this case, pydicom.encaps.generate_fragmented_frames yields a tuple of bytes for each frame and
    # each tuple element is passed in as the src argument.

    # Double check if the transfer syntax is supported although the runner should be correct.
    tsyntax = runner.transfer_syntax
    if not is_available(tsyntax):
        raise ImportError(f"Transfer syntax {tsyntax} not supported; see details in the debug log.")

    nvimgcodec_decoder = nvimgcodec.Decoder()
    decode_params = nvimgcodec.DecodeParams(allow_any_depth=True, color_spec=nvimgcodec.ColorSpec.UNCHANGED)

    decoded_data = nvimgcodec_decoder.decode(src, params=decode_params)
    return bytearray(decoded_data.cpu())  # HWC layout, interleaved format, and contiguous array in C-style


# End of required function for decoder plugin


def register_as_decoder_plugin(module_path: str | None = None) -> bool:
    """Register as a preferred decoder plugin with supported decoder classes.

    The Decoder class does not support sorting the plugins and uses the order in which plugins were added.
    Further more, the properties of ``available_plugins`` returns sorted labels only but not the Callables or
    their module and function names, and the function ``remove_plugin`` only returns a boolean.
    So there is no way to remove the available plugins before adding them back after this plugin is added.

    For now, have to access the ``private`` property ``_available`` of the Decoder class to sort the available
    plugins and make sure this custom plugin is the first in the sorted list by its label. It is known that the
    first plugin in the default list is always ``gdcm`` for the supported decoder classes, so label name needs
    to be lexicographically greater than ``gdcm`` to be the first in the sorted list.

    Args:
        module_path (str | None): The importable module path for this plugin.
            When ``None`` or ``"__main__"``, search the loaded modules for an entry whose ``__file__`` resolves
            to the current file, e.g. module paths that start with ``monai.deploy.operators`` or ``monai.data``.

    Returns:
        bool: ``True`` if the decoder plugin is registered successfully, ``False`` otherwise.
    """

    if not _is_nvimgcodec_available():
        _logger.info(f"Module {NVIMGCODEC_MODULE_NAME} is not available.")
        return False

    func_name = NVIMGCODEC_PLUGIN_FUNC_NAME

    if module_path is None:
        module_path = _find_module_path(__name__)
    else:
        # Double check if the module path exists and if it is the same as the one for the callable origin.
        module_path_found, func_name_found = _get_callable_origin(_decode_frame)  # get the func's module path.
        if module_path_found:
            if module_path.casefold() != module_path_found.casefold():
                _logger.info(f"Module path {module_path} does not match {module_path_found} for decoder plugin.")
        else:
            _logger.info(f"Module path {module_path} not found for decoder plugin.")
            return False

        if func_name_found != NVIMGCODEC_PLUGIN_FUNC_NAME:
            _logger.warning(
                f"Function name {func_name_found} does not match {NVIMGCODEC_PLUGIN_FUNC_NAME} for decoder plugin."
            )

    for decoder_class in SUPPORTED_DECODER_CLASSES:
        _logger.info(
            f"Adding plugin {NVIMGCODEC_PLUGIN_LABEL} with module path {module_path} and func name {func_name} "
            f"for transfer syntax {decoder_class.UID}"
        )
        decoder_class.add_plugin(NVIMGCODEC_PLUGIN_LABEL, (module_path, func_name))

        # Need to sort the plugins to make sure the custom plugin is the first in items() of
        # the decoder class search for the plugin to be used.
        decoder_class._available = dict(sorted(decoder_class._available.items(), key=lambda item: item[0]))
        _logger.info(
            f"Registered decoder plugin {NVIMGCODEC_PLUGIN_LABEL} for transfer syntax {decoder_class.UID}: "
            f"{decoder_class._available}"
        )
    _logger.info(f"Registered nvimgcodec decoder plugin with {len(SUPPORTED_DECODER_CLASSES)} decoder classes.")

    return True


def _find_module_path(module_name: str | None) -> str:
    """Return the importable module path for this file.

    When *module_name* is ``None`` or ``"__main__"``, search the loaded modules
    for an entry whose ``__file__`` resolves to the current file.
    Likely to be in module paths that start with ``monai.deploy.operators`` or ``monai.data``.
    """

    current_file = Path(__file__).resolve()
    candidates: list[str] = []

    for name, module in sys.modules.items():
        if not name or name == "__main__":
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            if Path(module_file).resolve() == current_file:
                candidates.append(name)
        except (OSError, RuntimeError):
            continue

    preferred_prefixes = ("monai.deploy.operators", "monai.data")
    for prefix in preferred_prefixes:
        for name in candidates:
            if name.startswith(prefix):
                return name

    if candidates:
        # deterministic fallback
        return sorted(candidates)[0]

    return __name__


def _get_callable_origin(obj: Callable[..., Any]) -> tuple[str | None, str | None]:
    """Return the importable module path and attribute name for *obj*.

    Can be used to get the importable module path and func name of existing loaded functions.

    Args:
        obj: Callable retrieved via :func:`getattr` or similar.

    Returns:
        tuple[str | None, str | None]: ``(module_path, attr_name)``; each element
        is ``None`` if it cannot be determined. When both values are available,
        the same callable can be re-imported using
        :func:`importlib.import_module` followed by :func:`getattr`.
    """

    if not callable(obj):
        return None, None

    target = inspect.unwrap(obj)
    attr_name = getattr(target, "__name__", None)
    module = inspect.getmodule(target)
    module_path = getattr(module, "__name__", None)

    # If the callable is defined in a different module, find the attribute name in the module.
    if module_path and attr_name:
        module_obj = sys.modules.get(module_path)
        if module_obj and getattr(module_obj, attr_name, None) is not target:
            for name in dir(module_obj):
                try:
                    if getattr(module_obj, name) is target:
                        attr_name = name
                        break
                except AttributeError:
                    continue

    return module_path, attr_name


def _is_nvimgcodec_available() -> bool:
    """Return ``True`` if nvimgcodec is available, ``False`` otherwise."""

    if not nvimgcodec or not _passes_version_check(NVIMGCODEC_MODULE_NAME, NVIMGCODEC_MIN_VERSION_TUPLE) or not cp:
        _logger.debug(f"nvimgcodec (version >= {NVIMGCODEC_MIN_VERSION}) or CuPy missing.")
        return False
    if not cp.cuda.is_available():
        _logger.debug("CUDA device not found.")
        return False

    return True
