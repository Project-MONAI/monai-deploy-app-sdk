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

# Package for custom operators used by the TotalSegmentator Fast app.

from .cascade_ops import CascadePrepOp, CropMaskOp, CropOp, PasteOp
from .dicom_series_selector_operator import DICOMSeriesSelectorOperator
from .gpu_util import (
    GpuTiming,
    StudyTimingCollector,
    assert_cuda_available,
    assert_on_gpu,
    get_study_id,
    nvtx_range,
    set_study_id,
)
from .merge_5part_operator import MergeRemapOperator
from .merge_operator import MergeOperator, merge_single_part
from .merge_remap import back_resample_merged, flip_sar_to_dhw, merge_remap
from .part_postprocess import (
    RULE_KEEP_LARGEST,
    apply_part_postprocess,
    find_postprocessing_pkl,
)
from .postprocess_operator import (
    PostprocessOperator,
    calculate_volume_ml,
    cc_label_gpu,
    keep_largest_component_gpu,
    load_postprocessing_rules,
    postprocess_gpu,
    remove_all_but_largest_component_gpu,
)
from .postresample_operator import PostResampleOperator, postresample_reference, revert_crop_and_transpose_gpu
from .pre_resample_zoom import (
    TARGET_SPACING,
    gpu_preresample_enabled,
    preresample_channel_cpu,
    preresample_channel_gpu,
    preresample_should_skip,
    preresample_volume,
    preresample_zoom_factors,
)
from .preprocess_operator import PreprocessOperator, preprocess_reference, reorient_to_ras, to_holoscan_gpu_tensor
from .seg_emit_operator import SegEmitOperator
from .seg_volume_metrics_operator import SegVolumeMetricsOperator
from .slidewindow_operator import SlideWindowOperator
from .task_postprocess_operator import (
    TaskPostprocessOperator,
    apply_task_postprocess,
    model_voxel_volume_mm3,
)
from .volume_metrics import (
    VolumeMetricsResult,
    compute_slice_stats,
    compute_volume_metrics,
)
from .volume_sink_operator import VolumeSinkOperator

__all__ = [
    "CascadePrepOp",
    "CropMaskOp",
    "CropOp",
    "PasteOp",
    "PreprocessOperator",
    "preprocess_reference",
    "reorient_to_ras",
    "to_holoscan_gpu_tensor",
    "SlideWindowOperator",
    "VolumeSinkOperator",
    "PostResampleOperator",
    "postresample_reference",
    "revert_crop_and_transpose_gpu",
    "TARGET_SPACING",
    "gpu_preresample_enabled",
    "preresample_zoom_factors",
    "preresample_should_skip",
    "preresample_channel_cpu",
    "preresample_channel_gpu",
    "preresample_volume",
    "PostprocessOperator",
    "load_postprocessing_rules",
    "apply_part_postprocess",
    "find_postprocessing_pkl",
    "RULE_KEEP_LARGEST",
    "postprocess_gpu",
    "cc_label_gpu",
    "keep_largest_component_gpu",
    "remove_all_but_largest_component_gpu",
    "calculate_volume_ml",
    "MergeOperator",
    "merge_single_part",
    "TaskPostprocessOperator",
    "apply_task_postprocess",
    "model_voxel_volume_mm3",
    "MergeRemapOperator",
    "merge_remap",
    "back_resample_merged",
    "flip_sar_to_dhw",
    "SegEmitOperator",
    "SegVolumeMetricsOperator",
    "VolumeMetricsResult",
    "compute_volume_metrics",
    "compute_slice_stats",
    "DICOMSeriesSelectorOperator",
    "GpuTiming",
    "assert_cuda_available",
    "assert_on_gpu",
    "nvtx_range",
    "set_study_id",
    "get_study_id",
    "StudyTimingCollector",
]
