# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import datetime
import json
import logging
import os
import pathlib
from typing import List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from dicomseg_writer import DICOMSegWriter
from metadata_parser import DICOMParser
from monai.inferers import SlidingWindowInferer
from monai.transforms import (Activations, AddChannel, Compose, Lambda, LoadNifti, ScaleIntensityRange, Spacing,
                              SqueezeDim, ToNumpy, ToTensor)
from series_converter import SeriesConverter
from series_selector import SeriesSelector
from series_instance_parser import SeriesInstanceParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoInputFoundError(ValueError):
    pass


NII_EXT = ('.nii.gz', '.nii')
MHD_EXT = ('.mhd')


def dicom_to_nifti(input_path: str, output_path: str):
    try:
        dicom_parser = DICOMParser()
        dicom_parser.parse(input_path)

        dicom_meta_path = "/tmp/dicom_meta.json"

        with open(dicom_meta_path, 'w') as json_file:
            json_file.write(dicom_parser.MetadataInJSON)

        # Check the metadata can be read back
        with open(dicom_meta_path, 'r') as json_file:
            json_data = json.load(json_file)
            logger.info('Verify loading output back into: {}'.format(type(json_data)))
            for p in json_data['studies']:
                logger.info('Loaded study, instance UID: {}'.format(p['StudyInstanceUID']))
    except Exception as ex:
        logger.exception('Failed to parse and save DICOM instances:\n{}'.format(ex))

    series_selector = SeriesSelector(json.loads(dicom_parser.MetadataInJSON))
    selected_series = series_selector.select(None)
    selected_series_uids = [x.casefold() for x in selected_series.values()]
    logger.info('{} Series matched:\n{}'.format(len(selected_series.keys()), selected_series.values()))

    # Convert the DICOM instances to volume image files
    series_converter = SeriesConverter()
    _ = series_converter.series_to_image(
        input_dir=input_path,
        output_dir=output_path,
        output_format="nii",
        metadata=dicom_parser.meta_data,
        selected_series_uids=selected_series_uids
    )


def list_files(input_path: str, file_exts: List[str]):

    file_names = []

    logging.info("Checking {} for {} files.".format(input_path, file_exts))

    for root, dirs, files in os.walk(input_path):
        for file in files:
            if any(file.endswith(file_ext) for file_ext in file_exts):
                file_names.append(os.path.join(root, file))

    return file_names


def execute():

    try:

        logger.info('Operator started: {:s}'.format(datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]))
        t0 = datetime.datetime.now()

        # get the clara_payload inputs we want to populate
        input_path = os.getenv('MONAI_INPUTPATH', 'input')
        output_path = os.getenv('MONAI_OUTPUTPATH', 'output')
        logger.info('Input path: {}'.format(input_path))
        logger.info('Output path: {}'.format(output_path))

        # create a temporary folder for the nifti outputs (from dicom )
        nifti_output_path = os.path.join(input_path, 'nifti')
        os.makedirs(nifti_output_path, exist_ok=True)
        dicom_dataset_list = dicom_to_nifti(input_path=input_path, output_path=nifti_output_path)
        file_names = list_files(input_path=nifti_output_path, file_exts=NII_EXT)
        if len(file_names) == 0:
            raise NoInputFoundError(f'No input files found in input path: {input_path}')

        logger.info('File read time: {:.0f} ms'.format((datetime.datetime.now() - t0).total_seconds() * 1000))
        t0 = datetime.datetime.now()

        inference_device = torch.device('cpu')
        if torch.cuda.is_available():
            inference_device = torch.device('cuda')
            cudnn.enabled = True
        else:
            logger.info('Did not detect CUDA-capable device. Using CPU.')

        # load model to be used for inference
        model = torch.jit.load('/opt/monai/models/spleen_model.ts', map_location=inference_device)

        logger.info('Model read and load time: {:.0f} ms'.format((datetime.datetime.now() - t0).total_seconds() * 1000))
        t0 = datetime.datetime.now()

        # setup MONAI scanning window inferer as the inference engine
        # set `device` and `sw_device` to be the same as the inference and input images should be in the same device for inference to succeed
        inferer = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=128, overlap=0.25,
                                       device=inference_device, sw_device=inference_device)

        # setup pre-inference image transformations
        pre_transforms = Compose([
            LoadNifti(image_only=True, dtype=np.float32),
            AddChannel(),                                       # add channel dimension (3d (H,W,D) -> 4d (C,H,W,D))
            AddChannel(),                                       # add batch dimension (4d -> 5d (N,C,H,W,D))
            Spacing(pixdim=(1., 1., 1.), mode='bilinear'),
            ScaleIntensityRange(a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            ToTensor(),                                         # convert to pytorch tensor before inference
        ])

        # setup post-inference transformations
        post_transforms = Compose([
            Activations(softmax=True),                   # convert net output to activations
            ToNumpy(),                                   # convert back to numpy
            SqueezeDim(dim=0),                           # remove the batch dimension
            Lambda(func=lambda x: x[1, ...]),            # use custom function to return only the foreground channel (1)
        ])

        logger.info('Inferer, Pre-, and post-transform configuration time: {:.0f} ms'.format(
            (datetime.datetime.now() - t0).total_seconds() * 1000))

        # perform actual inference on all input files
        for file_name in file_names:

            t0 = datetime.datetime.now()

            # perform pre-transformations
            pretransform_output = pre_transforms(file_name)

            logger.info('Pre-transform time: {:.0f} ms'.format((datetime.datetime.now() - t0).total_seconds() * 1000))
            t0 = datetime.datetime.now()

            # perform inference
            transformed_image = pretransform_output[0]  # primary output of `Spacing` pre-transform - used

            with torch.no_grad():  # prevent backpropagation during inference
                inference_output = inferer(transformed_image, model)

            logger.info('Inference time: {:.0f} ms'.format((datetime.datetime.now() - t0).total_seconds() * 1000))
            t0 = datetime.datetime.now()

            # perform post-transformations
            output_image = post_transforms(inference_output)

            series_inst_parser = SeriesInstanceParser()
            dicom_dataset_list = series_inst_parser.parse(input_path)
            dicom_seg_writer = DICOMSegWriter()
            dicom_seg_writer.write(
                seg_img=output_image,
                input_ds=dicom_dataset_list,
                outfile=os.path.join(output_path, f"{pathlib.Path(file_name).stem}-dcmseg.dcm"),
                seg_labels=['spleen'],
            )

            logger.info('Write output time: {:.0f} ms'.format((datetime.datetime.now() - t0).total_seconds() * 1000))

    except Exception as e:
        logger.error('Error: {}'.format(e))
        raise e

    logger.info('Operator ended: {:s}'.format(datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]))


if __name__ == "__main__":

    execute()
