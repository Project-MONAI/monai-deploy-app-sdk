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

"""
This decoder plugin for nvimgcodec <https://github.com/NVIDIA/nvImageCodec> decompresses
encoded Pixel Data for the following transfer syntaxes:
    JPEGBaseline8Bit, 1.2.840.10008.1.2.4.50, JPEG Baseline (Process 1)
    JPEGExtended12Bit, 1.2.840.10008.1.2.4.51, JPEG Extended (Process 2 & 4)
    JPEGLossless, 1.2.840.10008.1.2.4.57, JPEG Lossless, Non-Hierarchical (Process 14)
    JPEGLosslessSV1, 1.2.840.10008.1.2.4.70, JPEG Lossless, Non-Hierarchical, First-Order Prediction
    JPEG2000Lossless, 1.2.840.10008.1.2.4.90, JPEG 2000 Image Compression (Lossless Only)
    JPEG2000, 1.2.840.10008.1.2.4.91, JPEG 2000 Image Compression
    HTJ2KLossless, 1.2.840.10008.1.2.4.201, HTJ2K Image Compression (Lossless Only)
    HTJ2KLosslessRPCL, 1.2.840.10008.1.2.4.202, HTJ2K with RPCL Options Image Compression (Lossless Only)
    HTJ2K, 1.2.840.10008.1.2.4.203, HTJ2K Image Compression

There are two ways to add a custom decoding plugin to pydicom:
1. Using the pixel_data_handlers backend, though pydicom.pixel_data_handlers module is deprecated
   and will be removed in v4.0.
2. Using the pixels backend by adding a decoder plugin to existing decoders with the add_plugin method,
   see https://pydicom.github.io/pydicom/stable/guides/decoding/decoder_plugins.html

It is noted that pydicom.dataset.Dataset.pixel_array changed in version 3.0 where the backend used for
pixel data decoding changed from the pixel_data_handlers module to the pixels module.

So, this implementation uses the pixels backend.

Plugin Requirements:
A custom decoding plugin must implement three objects within the same module:
 - A function named is_available with the following signature:
    def is_available(uid: pydicom.uid.UID) -> bool:
      Where uid is the Transfer Syntax UID for the corresponding decoder as a UID
 - A dict named DECODER_DEPENDENCIES with the type dict[pydicom.uid.UID, tuple[str, ...], such as:
    DECODER_DEPENDENCIES = {JPEG2000Lossless: ('numpy', 'pillow', 'imagecodecs'),}
      This will be used to provide the user with a list of dependencies required by the plugin.
 - A function that performs the decoding with the following function signature as in Github repo:
    def _decode_frame(src: bytes, runner: DecodeRunner) -> bytearray | bytes
      src is a single frame’s worth of raw compressed data to be decoded, and
      runner is a DecodeRunner instance that manages the decoding process.

Adding plugins to a Decoder:
Additional plugins can be added to an existing decoder with the add_plugin() method
 ```python
  from pydicom.pixels.decoders import RLELosslessDecoder
  RLELosslessDecoder.add_plugin(
    'my_decoder',  the plugin's label
    ('my_package.decoders', 'my_decoder_func')  the import paths
  )
 ```
"""

import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
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
    from nvidia import nvimgcodec

    # Parse version string, extracting only numeric components to handle suffixes like "0.6.0rc1"
    try:
        import re

        version_parts = []
        for part in nvimgcodec.__version__.split("."):
            # Extract leading digits from each version component
            match = re.match(r"^(\d+)", part)
            if match:
                version_parts.append(int(match.group(1)))
            else:
                break  # Stop at first non-numeric component
        nvimgcodec_version = tuple(version_parts) if version_parts else (0,)
    except (AttributeError, ValueError):
        nvimgcodec_version = (0,)
except ImportError:
    nvimgcodec = None

# nvimgcodec pypi package name, minimum version required and the label for this decoder plugin.
NVIMGCODEC_MODULE_NAME = "nvidia.nvimgcodec"  # from nvidia-nvimgcodec-cu12 or other variants
NVIMGCODEC_MIN_VERSION = "0.6"
NVIMGCODEC_MIN_VERSION_TUPLE = tuple(int(x) for x in NVIMGCODEC_MIN_VERSION.split("."))
NVIMGCODEC_PLUGIN_LABEL = "0.6+nvimgcodec"  # to be sorted to first in ascending order of plugins

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

# Lazy singletons for nvimgcodec resources; initialized on first use
if nvimgcodec:
    _NVIMGCODEC_DECODER: Any = None
    _NVIMGCODEC_DECODE_PARAMS: Any = None
else:  # pragma: no cover - nvimgcodec not installed
    _NVIMGCODEC_DECODER = None
    _NVIMGCODEC_DECODE_PARAMS = None

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


def _get_decoder_resources() -> tuple[Any, Any]:
    """Return cached nvimgcodec decoder and decode parameters."""

    if nvimgcodec is None:
        raise ImportError("nvimgcodec package is not available.")

    global _NVIMGCODEC_DECODER, _NVIMGCODEC_DECODE_PARAMS

    if _NVIMGCODEC_DECODER is None:
        _NVIMGCODEC_DECODER = nvimgcodec.Decoder()
    if _NVIMGCODEC_DECODE_PARAMS is None:
        _NVIMGCODEC_DECODE_PARAMS = nvimgcodec.DecodeParams(
            allow_any_depth=True,
            color_spec=nvimgcodec.ColorSpec.UNCHANGED,
        )

    return _NVIMGCODEC_DECODER, _NVIMGCODEC_DECODE_PARAMS


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
    _logger.debug(f"transfer_syntax: {tsyntax}")

    if not is_available(tsyntax):
        raise ValueError(f"Transfer syntax {tsyntax} not supported; see details in the debug log.")

    decoder, params = _get_decoder_resources()
    decoded_surface = decoder.decode(src, params=params).cpu()
    np_surface = np.ascontiguousarray(np.asarray(decoded_surface))
    return np_surface.tobytes()


def _is_nvimgcodec_available() -> bool:
    """Return ``True`` if nvimgcodec is available, ``False`` otherwise."""

    if not nvimgcodec or not _passes_version_check(NVIMGCODEC_MODULE_NAME, NVIMGCODEC_MIN_VERSION_TUPLE) or not cp:
        _logger.debug(f"nvimgcodec (version >= {NVIMGCODEC_MIN_VERSION}) or CuPy missing.")
        return False
    if not cp.cuda.is_available():
        _logger.debug("CUDA device not found.")
        return False

    return True


# Helper functions for an application to register this decoder plugin with Pydicom at application startup.


def register_as_decoder_plugin(module_path: str | None = None) -> bool:
    """Register as a preferred decoder plugin with supported decoder classes.

    The Decoder class does not support sorting the plugins and uses the order in which plugins were added.
    Furthermore, the properties of ``available_plugins`` returns sorted labels only but not the Callables or
    their module and function names, and the function ``remove_plugin`` only returns a boolean.
    So there is no way to remove the available plugins before adding them back after this plugin is added.

    For now, have to access the ``private`` property ``_available`` of the Decoder class to sort the available
    plugins and make sure this custom plugin is the first in the sorted list by its label. It is known that the
    first plugin in the default list is always ``gdcm`` for the supported decoder classes, so label name needs
    to be lexicographically less than ``gdcm`` to be the first in the sorted list.

    Args:
        module_path (str | None): The importable module path for this plugin.
            When ``None`` or ``"__main__"``, search the loaded modules for an entry whose ``__file__`` resolves
            to the current file, e.g. module paths that start with ``monai.deploy.operators`` or ``monai.data``.

    Returns:
        bool: ``True`` if the decoder plugin is registered successfully, ``False`` otherwise.
    """

    if not _is_nvimgcodec_available():
        _logger.warning(f"Module {NVIMGCODEC_MODULE_NAME} is not available.")
        return False

    try:
        func_name = getattr(_decode_frame, "__name__", None)
    except NameError:
        _logger.error("Decoder function `_decode_frame` not found.")
        return False

    if module_path is None:
        module_path = _find_module_path(__name__)
    else:
        # Double check if the module path exists and if it is the same as the one for the callable origin.
        module_path_found, func_name_found = _get_callable_origin(_decode_frame)  # get the func's module path.
        if module_path_found:
            if module_path.casefold() != module_path_found.casefold():
                _logger.warning(f"Module path {module_path} does not match {module_path_found} for decoder plugin.")
        else:
            _logger.error(f"Module path {module_path} not found for decoder plugin.")
            return False

        if func_name != func_name_found:
            _logger.warning(
                f"Function {func_name_found} in {module_path_found} instead of {func_name} used for decoder plugin."
            )

    for decoder_class in SUPPORTED_DECODER_CLASSES:
        if NVIMGCODEC_PLUGIN_LABEL in decoder_class.available_plugins:
            _logger.debug(f"{NVIMGCODEC_PLUGIN_LABEL} already registered for transfer syntax {decoder_class.UID}.")
            continue

        decoder_class.add_plugin(NVIMGCODEC_PLUGIN_LABEL, (module_path, str(func_name)))
        _logger.info(
            f"Added plugin for transfer syntax {decoder_class.UID}: "
            f"{NVIMGCODEC_PLUGIN_LABEL} with {func_name} in module path {module_path}."
        )

        # Need to sort the plugins to make sure the custom plugin is the first in items() of
        # the decoder class search for the plugin to be used.
        decoder_class._available = dict(sorted(decoder_class._available.items(), key=lambda item: item[0]))
        _logger.debug(f"Sorted plugins for transfer syntax {decoder_class.UID}: {decoder_class._available}")

    _logger.info(f"{NVIMGCODEC_MODULE_NAME} registered with {len(SUPPORTED_DECODER_CLASSES)} decoder classes.")

    return True


def unregister_as_decoder_plugin() -> bool:
    """Unregister the decoder plugin from the supported decoder classes."""

    for decoder_class in SUPPORTED_DECODER_CLASSES:
        if NVIMGCODEC_PLUGIN_LABEL in decoder_class.available_plugins:
            decoder_class.remove_plugin(NVIMGCODEC_PLUGIN_LABEL)
        _logger.info(f"Unregistered plugin for transfer syntax {decoder_class.UID}: {NVIMGCODEC_PLUGIN_LABEL}")

    return True


def _find_module_path(module_name: str | None) -> str:
    """Return the importable module path for *module_name* file.

    When *module_name* is ``None`` or ``"__main__"``, search the loaded modules
    for an entry whose ``__file__`` resolves to the current file.

    When *module_name* is provided and not ``"__main__"``, validate it exists in
    loaded modules and corresponds to the current file, returning it if valid.

    When used in MONAI, likely in module paths ``monai.deploy.operators`` or ``monai.data``.
    """

    current_file = Path(__file__).resolve()

    # If a specific module name is provided (not None or "__main__"), validate it
    if module_name and module_name != "__main__":
        module = sys.modules.get(module_name)
        if module:
            module_file = getattr(module, "__file__", None)
            if module_file:
                try:
                    if Path(module_file).resolve() == current_file:
                        return module_name
                    else:
                        _logger.warning(f"Module {module_name} found but its file path does not match current file.")
                except (OSError, RuntimeError):
                    _logger.warning(f"Could not resolve file path for module {module_name}.")
            else:
                _logger.warning(f"Module {module_name} has no __file__ attribute.")
        else:
            _logger.warning(f"Module {module_name} not found in loaded modules.")
        # Fall through to search for the correct module

    # Search for modules that correspond to the current file
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
    """Return the importable module path and attribute(function) name for *obj*.

    Can be used to get the importable module path and func name of existing callables.

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
