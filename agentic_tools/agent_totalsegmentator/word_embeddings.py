import numpy as np
from anatomy_llm import identify_anatomy_with_llm, use_llm_find_closest_ta_list
from strands import Agent, tool


# Example usage
word_list = "total,body,body_mr,vertebrae_mr,lung_vessels,cerebral_bleed,hip_implant, \
        coronary_arteries,pleural_pericard_effusion,test,appendicular_bones,appendicular_bones_mr, \
        tissue_types,heartchambers_highres,face,vertebrae_body,total_mr,tissue_types_mr,tissue_4_types, \
        face_mr,head_glands_cavities,head_muscles,headneck_bones_vessels,headneck_muscles,brain_structures, \
        liver_vessels,oculomotor_muscles,thigh_shoulder_muscles,thigh_shoulder_muscles_mr,lung_nodules,kidney_cysts,\
        breasts,ventricle_parts,aortic_sinuses,liver_segments,liver_segments_mr, \
        total_highres_test,craniofacial_structures,abdominal_muscles,teeth,trunk_cavities,brain_aneurysm".split(",")

# Example usage
text = "The patient has pain in the chest and abdomen. The heart rate is elevated and liver function tests are abnormal."
anatomy_words = identify_anatomy_with_llm(text)
# print(f"Identified anatomy words: {anatomy_words}")
      
@tool
def find_ta_names(report: str=""):
    """
    Given a text report, it identifies the associated anatomy words and then finds the anatomy words
    that matches closely in the task list (ta_list) needed for the total segmentator.
    Only return the closest terms in the ta_list as a comma-separated list, nothing else. I do not want any other text in the output.
    :param report: Description
    return: A comma-separated list of task names that are closely related to the anatomy words identified in the report.
    """
    anatomy_words = identify_anatomy_with_llm(report)
    task_list = use_llm_find_closest_ta_list(anatomy_words, "CT")
    
    # Clean the response - extract only valid task names
    valid_tasks = []
    # TODO: Check why the response is not always a comma-separated list and sometimes contains other text. We need to extract only the task names from the response.
    
    for item in task_list:
        # Extract task names from verbose responses
        for task in word_list:
            if task.strip() in str(item):
                valid_tasks.append(task.strip())
    
    return list(set(valid_tasks))