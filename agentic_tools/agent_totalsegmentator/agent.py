from strands import Agent

from agent_tools import identify_modality, create_ts_command, \
        calculate_segmentation_statistics, get_image_metadata, get_image_characteristics, run_total_segmentator

from word_embeddings import find_ta_names
from anatomy_llm import identify_anatomy_with_llm, use_llm_find_closest_ta_list

def main():
    print("Starting agent...")
    monai_agent = Agent(
    name="MONAI Agent",
    system_prompt="An agent that identifies the modality of the \
        medical image and creates the command to segment the image using TotalSegmentator. \
            Then it will execute the command and produce the sementation",
    tools=[
        create_ts_command,
        calculate_segmentation_statistics, get_image_metadata,
        get_image_characteristics, identify_modality, find_ta_names, run_total_segmentator,
    ],
)
    image_filename = "./img_ct.nii.gz"
    sample_radiology_report = """The CT scan reveals a significant enlargement of the heart, indicating possible cardiomegaly. 
        There is also evidence of fluid accumulation in the lungs, suggestive of pulmonary edema. 
        Additionally, the liver appears to be enlarged, which may point to hepatomegaly. 
        No fractures or dislocations are observed in the visible bones."""

    # Run the test on tools
    # test_tools(monai_agent)

    monai_agent(f"Given the following radiology report: '{sample_radiology_report}' and the image: '{image_filename}' identify the modality of the medical image, \
        find the relevant task names for segmentation, and create the command to run TotalSegmentator. \
        Then execute the command and produce the segmentation.")

    
    
def test_tools(agent):
    
    sample_radiology_report = """The CT scan reveals a significant enlargement of the heart, indicating possible cardiomegaly. 
        There is also evidence of fluid accumulation in the lungs, suggestive of pulmonary edema. 
        Additionally, the liver appears to be enlarged, which may point to hepatomegaly. 
        No fractures or dislocations are observed in the visible bones."""
    
    # anatomy_words = identify_anatomy_with_llm(sample_radiology_report)
    # print("Anatomy words identified:", anatomy_words)

    # closest_ta_list = use_llm_find_closest_ta_list(anatomy_list=anatomy_words, modality="CT")
    # print("Closest TA list:", closest_ta_list)
    
    # image_metadata = get_image_metadata("./img_ct.nii.gz")
    # print("Image metadata:", image_metadata)
    
    # segmentation_statistics = calculate_segmentation_statistics(image_name = "./img_ct.nii.gz", segmentation_name="./label.nii.gz")
    # print("Segmentation statistics:", segmentation_statistics)
    
    task_names = find_ta_names(sample_radiology_report)
    print("Task names:", task_names)
    for _ in task_names:
        print(_)
        a = create_ts_command(task_name=_, image_modality="CT", output_dir="./output")
        print(a)

if __name__ == "__main__":
    main()