import boto3
import json
import re
from strands import tool

def identify_anatomy_with_llm(text):
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    prompt = f"""Extract all human anatomy words and any anatomy words only closely related to any diseases mentioned in the following text: "{text}"

    Do not extract any generic anatomy words that are not specifically mentioned or implied in the text. If there is a negation context, do not extract the anatomy word.
    For example, if the text mentions "no fractures in the bones", do not extract "bones" as it is relevant to the context of fractures.
    Return only the anatomy words as a comma-separated list, nothing else.
    Examples: heart, brain, liver, lung, kidney, spine, abdomen"""
    
    payload = {
        "schemaVersion": "messages-v1",
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
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
    anatomy_text = result['output']['message']['content'][0]['text'].strip()
    
    # Parse comma-separated anatomy words
    anatomy_words = [word.strip() for word in anatomy_text.split(',') if word.strip()]
    return anatomy_words

def use_llm_find_closest_ta_list(anatomy_list: list=[], modality: str="CT") -> list:
    """
    Find the closest entry in the ta_list to an entry in the anatomy_list. The words may not match exactly. 
    But they should be semantically similar.
    Given a list of anatomy words, use an LLM to find the closest matching terms in the  ta_list.
    Do not return words that are not in the ta_list. Use the LLM to match the words semantically.
    For example if the anatomy word is "heart", and the ta_list contains "heartchambers", 
    the LLM should be able to return "heartchambers" as the closest match.
    In addition, if the modality is CT, do not suggest any TAs that have "mr" in their name, as they are likely to be for MRI images.
    :param anatomy_list: List of anatomy words
    """
    print(anatomy_list)
    all_closest = []
    ta_list = "total,body,body_mr,vertebrae_mr,lung_vessels,cerebral_bleed,hip_implant, \
        coronary_arteries,pleural_pericard_effusion,test,appendicular_bones,appendicular_bones_mr, \
        tissue_types,heartchambers_highres,face,vertebrae_body,total_mr,tissue_types_mr,tissue_4_types, \
        face_mr,head_glands_cavities,head_muscles,headneck_bones_vessels,headneck_muscles,brain_structures, \
        liver_vessels,oculomotor_muscles,thigh_shoulder_muscles,thigh_shoulder_muscles_mr,lung_nodules,kidney_cysts,\
        breasts,ventricle_parts,aortic_sinuses,liver_segments,liver_segments_mr, \
        total_highres_test,craniofacial_structures,abdominal_muscles,teeth,trunk_cavities,brain_aneurysm".split(",")
    
    
    for anatomy in anatomy_list:
        # Use LLM to find closest TAs
        prompt = f"""Find the closest entry in the ta_list to an entry in the anatomy_list. The words may not match exactly. 
        Given a list of anatomy words, use an LLM to find the closest matching terms in the  ta_list. 
        Do not return words which are not contained in ta_list.
        For example if the anatomy word is "heart", and the ta_list contains "heartchambers_highres", the LLM should be able to return "heartchambers_highres" as the closest match.
        Here is the anatomy word: "{anatomy}" and here is the modality: "{modality}". 
        TA List: {', '.join(ta_list)}
        Return only the closest terms in the ta_listas a comma-separated list, nothing else.
        Example: heart, brain, liver, lung, kidney, spine, abdomen"""
        client = boto3.client("bedrock-runtime", region_name="us-east-1")

        payload = {
        "schemaVersion": "messages-v1",
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
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
        closest_tas = result['output']['message']['content'][0]['text'].strip()
        
        # Create a list of closest TAs and filter based on modality
        closest_tas = [ta.strip() for ta in closest_tas.split(',') if ta.strip()]
        print(f"Closest TAs for {anatomy}: {closest_tas}")
        for ta in closest_tas:
            if "mr" in ta and modality.lower() == "ct":
                print(f"Skipping {ta} as it is likely for MRI images")
                continue
            else:
                print(f"Adding {ta} to the closest TAs list")
                all_closest.append(ta)
        
        
    print(" All Closest", all_closest)    
    return list(set(all_closest))

def main():
    # Example usage
    text = "The patient has pain in the chest and abdomen. The heart rate is elevated and liver function tests are abnormal."
    anatomy_words = identify_anatomy_with_llm(text)
    print(f"Identified anatomy words: {anatomy_words}")
    # task_list = use_llm_find_closest_ta_list(anatomy_words)
    # print(f"Task names identified: {task_list}")
    task_list = use_llm_find_closest_ta_list(anatomy_words, modality="CT")
    print(f"Task names identified for liver: {task_list}")
    
if __name__ == "__main__":
    main()
    
        
