#! /bin/bash

# List all available pipelines
uv run pg list

rm -r results* test_*
uv run pg gen MONAI/breast_density_classification --output test_breast_den_cls
uv run pg run test_breast_den_cls --input test_breast_den_cls/model/sample_data/A/ --output ./results3
uv run pg run test_breast_den_cls --input test_breast_den_cls/model/sample_data/B/ --output ./results3
uv run pg run test_breast_den_cls --input test_breast_den_cls/model/sample_data/C/ --output ./results3

rm -r results* test_*
uv run pg gen MONAI/multi_organ_segmentation --output test_multiorgan_seg
uv run pg run test_multiorgan_seg/ --input /home/vicchang/Downloads/Task09_Spleen/Task09_Spleen/imagesTs --output ./results2

rm -r results* test_*
uv run pg gen MONAI/spleen_ct_segmentation --output test_spleen_ct_seg
uv run pg run test_spleen_ct_seg/ --input /home/vicchang/Downloads/Task09_Spleen/Task09_Spleen/imagesTs --output ./results

rm -r results* test_*
uv run pg gen MONAI/endoscopic_tool_segmentation --output test_endo_tool_seg
uv run pg run test_endo_tool_seg/ --input /home/vicchang/Downloads/instrument_5_8_testing/instrument_dataset_5/left_frames --output ./results

rm -r results* test_*
uv run pg gen MONAI/wholeBrainSeg_Large_UNEST_segmentation --output test_whole_brainseg_large
uv run pg run test_whole_brainseg_large/ --input /home/vicchang/Downloads/Task01_BrainTumour/imagesTs --output ./results

rm -r results* test_*
uv run pg gen MONAI/wholeBody_ct_segmentation --output test_wholeBody_ct_segmentation
uv run pg run test_wholeBody_ct_segmentation/ --input /home/vicchang/Downloads/Task09_Spleen/Task09_Spleen/imagesTs --output ./results

rm -r results* test_*
uv run pg gen MONAI/swin_unetr_btcv_segmentation --output test_swin_unetr_btcv_segmentation
uv run pg run test_swin_unetr_btcv_segmentation --input /home/vicchang/Downloads/Task09_Spleen/Task09_Spleen/imagesTs --output ./results

rm -r results* test_*
uv run pg gen MONAI/Llama3-VILA-M3-3B --output test_llama3
uv run pg run test_llama3 --input /home/vicchang/sc/github/monai/monai-deploy-app-sdk/tools/test_inputs --output ./results

rm -r results* test_*
uv run pg gen MONAI/Llama3-VILA-M3-8B --output test_llama3_8b
uv run pg run test_llama3_8b --input /home/vicchang/sc/github/monai/monai-deploy-app-sdk/tools/test_inputs --output ./results

rm -r results* test_*
uv run pg gen MONAI/Llama3-VILA-M3-13B --output test_llama3_13b
uv run pg run test_llama3_13b --input /home/vicchang/sc/github/monai/monai-deploy-app-sdk/tools/test_inputs --output ./results


rm -r results* test_*
uv run pg gen MONAI/retinalOCT_RPD_segmentation --output test_retinal_oct_seg
uv run pg run test_retinal_oct_seg --input /home/vicchang/sc/github/monai/monai-deploy-app-sdk/tools/pipeline-generator/test_retinal_oct_seg/model/sample_data --output ./results


