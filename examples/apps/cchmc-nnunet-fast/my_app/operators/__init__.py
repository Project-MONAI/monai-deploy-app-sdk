# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Package for custom operators used by the CCHMC nnU-Net Fast app.

from .gpu_util import (
    GpuTiming,
    StudyTimingCollector,
    assert_cuda_available,
    assert_on_gpu,
    get_study_id,
    nvtx_range,
    set_study_id,
)
from .dicom_sc_writer_operator import DICOMSCWriterOperator
from .dicom_series_selector_operator import DICOMSeriesSelectorOperator
from .ensemble_average_operator import EnsembleAverageOperator, argmax_to_segmentation, average_probabilities
from .sc_overlay import create_overlay, generate_contour, reference_label_to_contour, save_sc_dicom, write_sc_overlay
from .postprocess_operator import (
    PostprocessOperator,
    cc_label_gpu,
    calculate_volume_ml,
    keep_largest_component_gpu,
    load_postprocessing_rules,
    postprocess_gpu,
    remove_all_but_largest_component_gpu,
)
from .preprocess_operator import PreprocessOperator, preprocess_reference, to_holoscan_gpu_tensor
from .postresample_operator import PostResampleOperator, postresample_reference, revert_crop_and_transpose_gpu
from .slidewindow_operator import SlideWindowOperator

__all__ = [
    "PreprocessOperator",
    "preprocess_reference",
    "to_holoscan_gpu_tensor",
    "SlideWindowOperator",
    "PostResampleOperator",
    "postresample_reference",
    "revert_crop_and_transpose_gpu",
    "EnsembleAverageOperator",
    "average_probabilities",
    "argmax_to_segmentation",
    "PostprocessOperator",
    "load_postprocessing_rules",
    "postprocess_gpu",
    "cc_label_gpu",
    "keep_largest_component_gpu",
    "remove_all_but_largest_component_gpu",
    "calculate_volume_ml",
    "DICOMSeriesSelectorOperator",
    "DICOMSCWriterOperator",
    "generate_contour",
    "reference_label_to_contour",
    "create_overlay",
    "save_sc_dicom",
    "write_sc_overlay",
    "GpuTiming",
    "assert_cuda_available",
    "assert_on_gpu",
    "nvtx_range",
    "set_study_id",
    "get_study_id",
    "StudyTimingCollector",
]
