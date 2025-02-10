# execute model bundle locally (pythonically)

# load in environment variables
source .env

# remove the output directory
rm -rf "$HOLOSCAN_OUTPUT_PATH"

# execute model bundle locally (pythonically)
python cchmc_ped_abd_ct_seg_app -i "$HOLOSCAN_INPUT_PATH" -o "$HOLOSCAN_OUTPUT_PATH" -m "$HOLOSCAN_MODEL_PATH"
