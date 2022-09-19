# Copyright 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from random import randint
from typing import List

import highdicom as hd
import numpy as np
import logging
from app.inference import DetectionResult, DetectionResultList

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries


@md.input("original_dicom", List[StudySelectedSeries], IOType.IN_MEMORY)
@md.input("detection_predictions", DetectionResultList, IOType.IN_MEMORY)
@md.output("gsps_files", DataPath, IOType.DISK)
class GenerateGSPSOp(Operator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        selected_study = op_input.get("original_dicom")[0]  # assuming a single study
        selected_series = selected_study.selected_series[0]  # assuming a single series
        detection_result: DetectionResult = op_input.get("detection_predictions").detection_list[0]
        output_path = op_output.get("gsps_files").path

        # slice_coords = [inst.first_pixel_on_slice_normal[2] for inst in selected_series.series.get_sop_instances()]
        slice_coords = [inst for inst in range(len(selected_series.series.get_sop_instances()))]

        for inst_num, (box_data, box_score) in enumerate(zip(detection_result.box_data, detection_result.score_data)):

            polyline = hd.pr.GraphicObject(
                graphic_type=hd.pr.GraphicTypeValues.POLYLINE,
                graphic_data=np.array([
                    [box_data[0], box_data[1]],
                    [box_data[3], box_data[1]],
                    [box_data[3], box_data[4]],
                    [box_data[0], box_data[4]],
                    [box_data[0], box_data[1]],
                ]),  # coordinates of polyline vertices
                units=hd.pr.AnnotationUnitsValues.PIXEL,  # units for graphic data
                tracking_id='lung_nodule_MONAI',  # site-specific ID
                tracking_uid=hd.UID()  # highdicom will generate a unique ID
            )

            _left = box_data[0] if box_data[0] < box_data[2] else box_data[2]
            _right = box_data[2] if box_data[0] < box_data[2] else box_data[0]
            _top = box_data[1] if box_data[1] < box_data[3] else box_data[3]
            _bottom = box_data[3] if box_data[1] < box_data[3] else box_data[1]
            self.logger.info(f"Box: {[_left, _top, _right, _bottom]}")

            # text = hd.pr.TextObject(
            #     text_value=f"{box_score:.2f}",
            #     bounding_box=np.array(
            #         [_left, _top, _right, _bottom]  # left, top, right, bottom
            #     ),
            #     units=hd.pr.AnnotationUnitsValues.PIXEL,  # units for bounding box
            #     tracking_id='LungNoduleMONAI',  # site-specific ID
            #     tracking_uid=hd.UID()  # highdicom will generate a unique ID
            # )

            layer = hd.pr.GraphicLayer(
                layer_name='LUNG_NODULE',
                order=1,
                description='Lung Nodule Detection',
            )

            affected_slice_idx = [idx for idx, slice_coord in enumerate(slice_coords) if slice_coord >= box_data[2] and slice_coord <= box_data[5]]
            ref_images = [selected_series.series.get_sop_instances()[idx].get_native_sop_instance() for idx in affected_slice_idx]
            self.logger.info(f"Slice: {[box_data[2], box_data[5]]}, Instances: {affected_slice_idx}")

            if not ref_images:
                raise ValueError("Finding does not correspond to any series SOP instance")

            # A GraphicAnnotation may contain multiple text and/or graphic objects
            # and is rendered over all referenced images
            annotation = hd.pr.GraphicAnnotation(
                referenced_images=ref_images,
                graphic_layer=layer,
                # text_objects=[text],
                graphic_objects=[polyline],
            )

            # Assemble the components into a GSPS object
            gsps = hd.pr.GrayscaleSoftcopyPresentationState(
                referenced_images=ref_images,
                series_instance_uid=hd.UID(),
                series_number=randint(1, 100000),
                sop_instance_uid=hd.UID(),
                instance_number=inst_num + 1,
                manufacturer='MONAI',
                manufacturer_model_name='lung_nodule_ct_detection',
                software_versions='v1.1',
                device_serial_number='',
                content_label='ANNOTATIONS',
                graphic_layers=[layer],
                graphic_annotations=[annotation],
                institution_name='MONAI',
                institutional_department_name='Deploy',
            )

            gsps.save_as(os.path.join(output_path, f"gsps-{inst_num:04d}.dcm"))
