# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except as stated in the docstring below.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""P7 terminal — replaced by TotalSegSubgraph in Phases 8-9.

Minimal terminal operator so the ``DICOMSeriesToVolumeOperator`` output has a
receiver (GXF requires every emitted object to have a downstream port). It
declares ONE input port (``image``), receives the SDK volume, logs its shape
and dtype, and emits nothing.
"""

import logging

from monai.deploy.core import Image, Operator, OperatorSpec


class VolumeSinkOperator(Operator):
    """Terminal sink: receives the DICOM-to-volume ``image`` and discards it.

    P7 scaffold only — no inference. Phases 8-9 replace this operator with
    the 5-part TotalSegSubgraph (inference + ensemble + postprocess + writers).
    """

    INPUT_IMAGE = "image"

    def __init__(self, fragment, *args, **kwargs):
        # flags-before-super() discipline: instance state set before the
        # base Operator.__init__ (holoscan 4.2 pattern).
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.INPUT_IMAGE)

    def compute(self, op_input, op_output, context):
        image = op_input.receive(self.INPUT_IMAGE)
        if image is None:
            self._logger.warning("volume_sink: received no 'image' input")
            return
        # The SDK DICOMSeriesToVolumeOperator emits an ``Image`` wrapping a
        # 3D numpy array; ``asnumpy()`` is its array accessor.
        arr = image.asnumpy() if isinstance(image, Image) else image
        self._logger.info(
            "volume_sink: received shape=%s dtype=%s",
            tuple(arr.shape),
            arr.dtype,
        )
