# MONAI Agent

An AI agent that automatically identifies medical image modalities, extracts anatomy information from radiology reports, and generates TotalSegmentator commands for medical image segmentation.

## Features

- **Modality Detection**: Automatically identifies CT, MRI, PET, and other medical imaging modalities
- **Anatomy Extraction**: Uses LLM to extract anatomical structures from radiology reports
- **Task Matching**: Maps anatomy words to TotalSegmentator task names using semantic similarity
- **Command Generation**: Creates optimized TotalSegmentator commands
- **Segmentation Execution**: Runs segmentation and calculates statistics

## Installation

```bash
pip install -r requirements_minimal.txt
python -m spacy download en_core_web_sm
```

## AWS Setup

Configure AWS credentials for Bedrock access:
```bash
aws configure
```

## Usage

```python
from strands import Agent
from agent_tools import *
from word_embeddings import find_ta_names

# Create agent
monai_agent = Agent(
    name="MONAI Agent",
    tools=[create_ts_command, identify_modality, find_ta_names, run_total_segmentator]
)

# Run segmentation
monai_agent("Analyze CT image ./img_ct.nii.gz with report: 'Heart enlargement detected'")
```

## Files

- `agent.py` - Main agent script
- `agent_tools.py` - Medical image processing tools
- `anatomy_llm.py` - LLM-based anatomy extraction
- `word_embeddings.py` - Semantic matching for task names

## Example

```bash
python agent.py
```

The agent will:
1. Identify image modality (CT/MRI/PET)
2. Extract anatomy from radiology report
3. Find matching TotalSegmentator tasks
4. Generate and execute segmentation command
5. Calculate segmentation statistics