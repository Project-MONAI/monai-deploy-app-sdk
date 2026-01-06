import boto3
import json
from strands import Agent, tool
import SimpleITK as sitk
from anatomy_llm import identify_anatomy_with_llm

@tool
def create_total_segmentator_command(image_modality: str, task_name: str = "coronaries", output_dir: str = "./output") -> str:
    """
    Use the image modality and anatomy to create the TotalSegmentator command that can be used to segment the image. 
    This function generates a command string for running TotalSegmentator on a given image.
    The command string includes the image file path, output directory, and other parameters for segmentation.
    Always use the GPU for running TotalSegmentator if available. 
    and use the verbose flag to get more output from the command.

    Args:
        image_modality (str): The modality of the image. This can be 'CT' or 'MRI'.
        anatomy (str, optional): The anatomy to be segmented. Defaults to "heart".

    Returns:
        str: The command string for running TotalSegmentator on the image.

    Example:
        >>> create_total_segmentator_command("CT", "heart")
        'total_segmentator -i /path/to/image.nii.gz -o /path/to/output/directory --ta task_name '
        >>> create_total_segmentator_command("MRI", "brain")
        'total_segmentator -i /path/to/image.nii.gz -o /path/to/output/directory --ml'
    """
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    prompt = f"""Based on the total segmentator help usage:
        usage: TotalSegmentator [-h] -i filepath -o directory [-ot OUTPUT_TYPE [OUTPUT_TYPE ...]] [-ml]
                        [-nr NR_THR_RESAMP] [-ns NR_THR_SAVING] [-f] [-ff] [-t NORA_TAG] [-p]
                        [-ta total,body,body_mr,vertebrae_mr,lung_vessels,cerebral_bleed,hip_implant,coronary_arteries,pleural_pericard_effusion,test,appendicular_bones,appendicular_bones_mr,tissue_types,heartchambers_highres,face,vertebrae_body,total_mr,tissue_types_mr,tissue_4_types,face_mr,head_glands_cavities,head_muscles,headneck_bones_vessels,headneck_muscles,brain_structures,liver_vessels,oculomotor_muscles,thigh_shoulder_muscles,thigh_shoulder_muscles_mr,lung_nodules,kidney_cysts,breasts,ventricle_parts,aortic_sinuses,liver_segments,liver_segments_mr,total_highres_test,craniofacial_structures,abdominal_muscles,teeth,trunk_cavities,brain_aneurysm]
                        [-rs ROI_SUBSET [ROI_SUBSET ...]] [-rsr ROI_SUBSET_ROBUST [ROI_SUBSET_ROBUST ...]]
                        [-rc] [-ho] [-s] [-r] [-sii] [-cp CROP_PATH] [-bs] [-fs] [-ss] [-ndm] [-v1o]
                        [-rmb] [-d DEVICE] [-q] [-sp SAVE_PROBABILITIES] [-v] [-l LICENSE_NUMBER]
                        [--test 0|1|3] [--version]

    Segment 104 anatomical structures in CT images.

    options:
    -h, --help            show this help message and exit
    -i filepath           CT nifti image or folder of dicom slices or zip file of dicom slices.
    -o directory          Output directory for segmentation masks. Or path of multilabel output nifti file
                            if --ml option is used.Or path of output dicom seg file if --output_type is set to
                            'dicom_seg' or 'dicom_rtstruct'
    -ot, --output_type OUTPUT_TYPE [OUTPUT_TYPE ...]
                            Select output type(s). Choices: nifti, dicom_rtstruct, dicom_seg. Multiple are
                            allowed e.g. -ot nifti dicom_seg OR -ot nifti,dicom_seg).
    -ml, --ml             Save one multilabel image for all classes
    -nr, --nr_thr_resamp NR_THR_RESAMP
                            Nr of threads for resampling
    -ns, --nr_thr_saving NR_THR_SAVING
                            Nr of threads for saving segmentations
    -f, --fast            Run faster lower resolution model (3mm)
    -ff, --fastest        Run even faster lower resolution model (6mm)
    -t, --nora_tag NORA_TAG
                            tag in nora as mask. Pass nora project id as argument.
    -p, --preview         Generate a png preview of segmentation
    -ta, --task total,body,body_mr,vertebrae_mr,lung_vessels,cerebral_bleed,hip_implant, \
            coronary_arteries,pleural_pericard_effusion,test,appendicular_bones,appendicular_bones_mr, \
            tissue_types,heartchambers_highres,face,vertebrae_body,total_mr,tissue_types_mr,tissue_4_types, \
            face_mr,head_glands_cavities,head_muscles,headneck_bones_vessels,headneck_muscles,brain_structures, \
            liver_vessels,oculomotor_muscles,thigh_shoulder_muscles,thigh_shoulder_muscles_mr,lung_nodules,kidney_cysts,\
            breasts,ventricle_parts,aortic_sinuses,liver_segments,liver_segments_mr, \
            total_highres_test,craniofacial_structures,abdominal_muscles,teeth,trunk_cavities,brain_aneurysm
                            Select which model to use. This determines what is predicted.
    -rs, --roi_subset ROI_SUBSET [ROI_SUBSET ...]
                            Define a subset of classes to save (space separated list of class names). If
                            running 1.5mm model, will only run the appropriate models for these rois.
    -rsr, --roi_subset_robust ROI_SUBSET_ROBUST [ROI_SUBSET_ROBUST ...]
                            Like roi_subset but uses a slower but more robust model to find the rois.
    -rc, --robust_crop    For cropping (which is required for several task) or roi_subset, use the more
                            robust 3mm model instead of the default and faster 6mm model.
    -ho, --higher_order_resampling
                            Use higher order resampling for segmentations. Results in smoother segmentations
                            on high resolution images but uses more runtime + memory.
    -s, --statistics      Calc volume (in mm3) and mean intensity. Results will be in statistics.json
    -r, --radiomics       Calc radiomics features. Requires pyradiomics. Results will be in
                            statistics_radiomics.json
    -sii, --stats_include_incomplete
                            Normally statistics are only calculated for ROIs which are not cut off by the
                            beginning or end of image. Use this option to calc anyways.
    -cp, --crop_path CROP_PATH
                            Custom path to masks used for cropping. If not set will use output directory.
    -bs, --body_seg       Do initial rough body segmentation and crop image to body region
    -fs, --force_split    Process image in 3 chunks for less memory consumption. (do not use on small
                            images)
    -ss, --skip_saving    Skip saving of segmentations for faster runtime if you are only interested in
                            statistics.
    -ndm, --no_derived_masks
                            Do not create derived masks (e.g. skin from body mask).
    -v1o, --v1_order      In multilabel file order classes as in v1. New v2 classes will be removed.
    -rmb, --remove_small_blobs
                            Remove small connected components (<0.2ml) from the final segmentations.
    -d, --device DEVICE   Device type: 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing
                            the GPU device ID.
    -q, --quiet           Print no intermediate outputs
    -sp, --save_probabilities SAVE_PROBABILITIES
                            Save probabilities to this path. Only for experienced users. Python skills
                            required.
    -v, --verbose         Show more intermediate output
    -l, --license_number LICENSE_NUMBER
                            Set license number. Needed for some tasks. Only needed once, then stored in config
                            file.
    --test 0|1|3          Only needed for unittesting.
    --version             show program's version number and exit
    Just give me the command without any explanation remove any bash marks or quotes Chose the task value based on the "anatomy" variable provided for the function
    Use your best judgement to create the command. Only choose a single value for ta. 
    """ 


    payload = {
        "schemaVersion": "messages-v1",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": prompt}
                ]
            }
        ],
        "inferenceConfig": {
            "max_new_tokens": 100
        }
    }

    response = client.invoke_model(
        body=json.dumps(payload),
        modelId="amazon.nova-lite-v1:0",
        accept="application/json",
        contentType="application/json",
    )

    result = json.loads(response['body'].read())
    print(result['output']['message']['content'][0]['text'])
    return result


@tool
def create_ts_command(task_name: str, image_modality: str, output_dir: str) -> str:
    """
    Create the TotalSegmentator command based on the task name and image modality.
    """
    command = f"TotalSegmentator -i /path/to/image.nii.gz -o {output_dir} --ta {task_name}"
    if image_modality == "CT":
        command += " --ml"
    # if gpu is available, add the flag to use it
    # check gpu availability and add the flag to use it
    import torch
    if torch.cuda.is_available():
        command += " --device gpu"
    return command

@tool
def calculate_segmentation_statistics(image_name = "", segmentation_name = ""):
    """
    Calculate statistics for a given image and its segmentation.
    Given the paths to a medical image and its corresponding segmentation,
    Calculate and return basic statistics such as volume and mean intensity for each segmented region.
    
    :param image_name: Path to the medical image file (e.g., NIfTI format).
    :param segmentation_name: Path to the segmentation file (e.g., NIfTI format).
    
    """
    
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_name)
    image = reader.Execute()
    
    reader.SetFileName(segmentation_name)
    mask = reader.Execute()
    print(mask.GetPixelIDTypeAsString())
    
    # Convert mask to binary for label statistics filter
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    
    stats_filter = sitk.LabelStatisticsImageFilter()
    stats_filter.Execute(image, mask)
    
    
    statistics = {}
    for label in stats_filter.GetLabels():
        if label == 0:
            continue  # Skip background
        volume = stats_filter.GetCount(label) * image.GetSpacing()[0] * image.GetSpacing()[1] * image.GetSpacing()[2]
        mean_intensity = stats_filter.GetMean(label)
        statistics[label] = {
            "volume_mm3": volume,
            "mean_intensity": mean_intensity
        }   
    return statistics
    
@tool
def get_image_metadata(image_file_name = ""):
    """
    Retrieve metadata from a medical image file.
    
    :param image_name: Path to the medical image file (e.g., NIfTI format).
    :return: Dictionary containing metadata such as dimensions, spacing, and origin.
    """
    
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_file_name)
    image = reader.Execute()
    
    metadata = {
        "dimensions": image.GetSize(),
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection()
    }
    
    return metadata

@tool
def get_image_characteristics(image_name=""):
    """
    Retrieve characteristics of a medical image file.
    
    :param image_name: Path to the medical image file (e.g., NIfTI format).
    :return: Dictionary containing characteristics such as data type and number of components.
    """
    
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_name)
    image = reader.Execute()
    
    image_dimension = image.GetDimension()
    # Get mean and stddev of pixel intensities
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(image)
    mean_intensity = stats_filter.GetMean()
    stddev_intensity = stats_filter.GetSigma()
    
    # Calculate gray-level co-occurrence matrix (GLCM) features depending on dimension
    if image_dimension == 2:
        glcm_filter = sitk.GrayLevelCooccurrenceMatrixImageFilter()
        glcm_filter.SetOffset([1, 0])  # Example offset
    elif image_dimension == 3:
        glcm_filter = sitk.GrayLevelCooccurrenceMatrixImageFilter()
        glcm_filter.SetOffset([1, 0, 0])  # Example offset
    glcm_image = glcm_filter.Execute(image)
    contrast = sitk.LabelStatisticsImageFilter()
    contrast.Execute(glcm_image, image)
    contrast_value = contrast.GetMean(1)  # Assuming label 1 for GLCM

    homogeneity = sitk.LabelStatisticsImageFilter()
    homogeneity.Execute(glcm_image, image)
    homogeneity_value = homogeneity.GetMean(1)  # Assuming label 1 for GLCM

    characteristics = {
        "data_type": image.GetPixelIDTypeAsString(),
        "number_of_components": image.GetNumberOfComponentsPerPixel(),
        "mean_intensity": mean_intensity,
        "stddev_intensity": stddev_intensity,
        "homogeneity": homogeneity_value,
        "contrast": contrast_value
    }

    
    return characteristics

@tool
def identify_modality(image_name: str, metadata: dict) -> str:
    modality = "unknown"
    image_name_lower = image_name.lower()
    
    # Check file name for clues
    if "ct" in image_name_lower:
        modality = "CT"
    elif "mri" in image_name_lower or "mr" in image_name_lower:
        modality = "MRI"
    elif "pet" in image_name_lower:
        modality = "PET"
    elif "us" in image_name_lower or "ultrasound" in image_name_lower:
        modality = "Ultrasound"
    
    # Check metadata for clues if modality is still unknown
    if modality == "unknown":
        if 'Modality' in metadata:
            modality_meta = metadata['Modality'].upper()
            if modality_meta == "CT":
                modality = "CT"
            elif modality_meta == "MR":
                modality = "MRI"
            elif modality_meta == "PT":
                modality = "PET"
            elif modality_meta == "US":
                modality = "Ultrasound"
    
    return modality

@tool
def run_total_segmentator(command: str) -> str:
    """
    It runs the command in the bash script. If no license is provided, it will use the default license.
    If it is not possible to run the command return an error message. 
    Do not default to run the command with "total" flag for -ta 
    if the command does not include it. Always use the GPU for running TotalSegmentator if available.
    :param command: Description
    :type command: str
    :param output_dir: Description
    :type output_dir: str
    :return: Description
    :rtype: str
    """
    print(f"Running command: {command}")
    import subprocess
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e.stderr.decode('utf-8')}"