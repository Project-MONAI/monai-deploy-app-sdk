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

import logging
import os
from random import randint
from typing import Callable, List

import diagnostic_report as dr
import highdicom as hd
import numpy as np
from ai_service.utility import JSON_MIME_TYPE
from app.environment import PARTNER_NAME, SERVICE_NAME
from app.inference import DetectionResult, DetectionResultList

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries


@md.input("original_dicom", List[StudySelectedSeries], IOType.IN_MEMORY)
@md.input("detection_predictions", DetectionResultList, IOType.IN_MEMORY)
@md.output("gsps_files", DataPath, IOType.DISK)
class GenerateGSPSOp(Operator):
    def __init__(self, upload_gsps_fn: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.upload_gsps = upload_gsps_fn

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        selected_study = op_input.get("original_dicom")[0]  # assuming a single study
        selected_series = selected_study.selected_series[0]  # assuming a single series
        detection_result: DetectionResult = op_input.get("detection_predictions").detection_list[0]
        output_path = op_output.get("gsps_files").path

        slice_coords = list(range(len(selected_series.series.get_sop_instances())))

        series_uid = hd.UID()
        series_number = randint(1, 100000)

        for inst_num, (box_data, box_score) in enumerate(zip(detection_result.box_data, detection_result.score_data)):

            polyline = hd.pr.GraphicObject(
                graphic_type=hd.pr.GraphicTypeValues.POLYLINE,
                graphic_data=np.array(
                    [
                        [box_data[0], box_data[1]],
                        [box_data[3], box_data[1]],
                        [box_data[3], box_data[4]],
                        [box_data[0], box_data[4]],
                        [box_data[0], box_data[1]],
                    ]
                ),  # coordinates of polyline vertices
                units=hd.pr.AnnotationUnitsValues.PIXEL,  # units for graphic data
                tracking_id="lung_nodule_MONAI",  # site-specific ID
                tracking_uid=hd.UID(),  # highdicom will generate a unique ID
            )

            self.logger.info(f"Box: {[box_data[0], box_data[1], box_data[3], box_data[4]]}")

            text = hd.pr.TextObject(
                text_value=f"{box_score:.2f}",
                bounding_box=(box_data[0], box_data[1], box_data[3], box_data[4]),  # left, top, right, bottom
                units=hd.pr.AnnotationUnitsValues.PIXEL,  # units for bounding box
                tracking_id="LungNoduleMONAI",  # site-specific ID
                tracking_uid=hd.UID(),  # highdicom will generate a unique ID
            )

            layer = hd.pr.GraphicLayer(
                layer_name="LUNG_NODULE",
                order=1,
                description="Lung Nodule Detection",
            )

            affected_slice_idx = [
                idx
                for idx, slice_coord in enumerate(slice_coords)
                if slice_coord >= box_data[2] and slice_coord <= box_data[5]
            ]
            ref_images = [
                selected_series.series.get_sop_instances()[idx].get_native_sop_instance() for idx in affected_slice_idx
            ]
            self.logger.info(f"Slice: {[box_data[2], box_data[5]]}, Instances: {affected_slice_idx}")

            if not ref_images:
                self.logger.error("Finding does not correspond to any series SOP instance")
                continue

            # A GraphicAnnotation may contain multiple text and/or graphic objects
            # and is rendered over all referenced images
            annotation = hd.pr.GraphicAnnotation(
                referenced_images=ref_images,
                graphic_layer=layer,
                text_objects=[text],
                graphic_objects=[polyline],
            )

            # Assemble the components into a GSPS object
            gsps = hd.pr.GrayscaleSoftcopyPresentationState(
                referenced_images=ref_images,
                series_instance_uid=series_uid,
                series_number=series_number,
                sop_instance_uid=hd.UID(),
                instance_number=inst_num + 1,
                manufacturer="MONAI",
                manufacturer_model_name="lung_nodule_ct_detection",
                software_versions="v0.2.0",
                device_serial_number="",
                content_label="ANNOTATIONS",
                graphic_layers=[layer],
                graphic_annotations=[annotation],
                institution_name="MONAI",
                institutional_department_name="Deploy",
                voi_lut_transformations=[
                    hd.pr.SoftcopyVOILUTTransformation(
                        window_center=-550.0,
                        window_width=1350.0,
                    )
                ],
            )

            gsps.save_as(os.path.join(output_path, f"gsps-{inst_num:04d}.dcm"))

            self.upload_gsps(
                file=os.path.join(output_path, f"gsps-{inst_num:04d}.dcm"),
                document_detail="MONAI Lung Nodule Detection v0.2.0",
                series_uid=series_uid,
            )


@md.input("original_dicom", List[StudySelectedSeries], IOType.IN_MEMORY)
@md.input("detection_predictions", DetectionResultList, IOType.IN_MEMORY)
@md.output("pin_report", DataPath, IOType.DISK)
class CreatePINDiagnosticsReportOp(Operator):
    def __init__(self, upload_doc_fn: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.upload_doc = upload_doc_fn

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        selected_study = op_input.get("original_dicom")[0]  # assuming a single study
        selected_series = selected_study.selected_series[0]  # assuming a single series
        detection_result: DetectionResult = op_input.get("detection_predictions").detection_list[0]
        output_path = op_output.get("pin_report").path

        present_str = "not" if detection_result.box_data.shape[0] == 0 else ""

        report = dr.DiagnosticReport(
            divtext="""
            <div xmlns=\"http://www.w3.org/1999/xhtml\">
                <p>
                    <b>AI Lung Nodule Detection Report</b>
                </p>
            </div>
            """,
            conclusion=f"Lung nodules are {present_str} present.",
        )
        report.add_device("monai_deploy", "MONAI", "MONAI Model Zoo - Lung Nodule Detection", "v0.2.0")
        report.set_ai_technique("MONAI Model Zoo - Lung Nodule Detector v0.2.0")
        report.set_study(
            dcm=selected_series.series.get_sop_instances()[0].get_native_sop_instance(),
            procedure_code="RPID16",
            procedure_display="CT Chest wo",
            procedure_text="CT Chest wo IV Contrast",
        )

        slice_coords = list(range(len(selected_series.series.get_sop_instances())))

        for box_data, box_score in zip(detection_result.box_data, detection_result.score_data):

            affected_slice_idx = [
                idx
                for idx, slice_coord in enumerate(slice_coords)
                if slice_coord >= box_data[2] and slice_coord <= box_data[5]
            ]
            ref_images = [
                selected_series.series.get_sop_instances()[idx].get_native_sop_instance() for idx in affected_slice_idx
            ]
            self.logger.info(f"Slice: {[box_data[2], box_data[5]]}, Instances: {affected_slice_idx}")

            for dcm_img in ref_images:
                message = f"Lung nodule present with probability {box_score:.2f}"
                observation = report.add_observation(
                    body_part_code="RID1301",
                    body_part_text="lung",
                    dcm=dcm_img,
                    derived_from=report.study,
                    note=message,  # is box_score associated with a specific image?
                    observation_code="RID50149",
                    observation_system="http://nuancepowerscribe.com/saf",
                    observation_text="Pulmonary nodule",
                )
                observation.set_probability(round(box_score * 100, 2))  # probability's unit of measure is percent
                observation.set_summary(message)
                self.logger.info(message)
                if box_score > .8:
                    observation.set_present_qualifier()
                elif box_score > .15:
                    observation.set_indeterminate_qualifier()
                else:
                    observation.set_absent_qualifier()

        report.write_to_file(os.path.join(output_path, f"{PARTNER_NAME}-{SERVICE_NAME}-FHIR.json"))

        self.upload_doc(
            file=os.path.join(output_path, f"{PARTNER_NAME}-{SERVICE_NAME}-FHIR.json"),
            content_type=JSON_MIME_TYPE,
            series_uid=selected_series.series.SeriesInstanceUID,
        )
