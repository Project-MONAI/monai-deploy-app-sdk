#!/bin/bash

# This script tests the RESTful endpoints of the wrapper application.

# Base URL of the web application
BASE_URL="http://127.0.0.1:5000"

# Get the absolute path to the input and output directories
INPUT_DIR="$HOLOSCAN_INPUT_PATH" #"$(pwd)/inputs/spleen_ct_tcia"
OUTPUT_DIR="$HOLOSCAN_OUTPUT_PATH" #"$(pwd)/output_spleen"
CALLBACK_PORT=9005
CALLBACK_URL="http://127.0.0.1:$CALLBACK_PORT/callback"

# Function to print test headers
print_header() {
    echo ""
    echo "======================================================"
    echo "$1"
    echo "======================================================"
}

# 1. Start a simple netcat listener to act as our callback server
print_header "Starting callback listener on port $CALLBACK_PORT"
# Listen for one request, then exit immediately
nc -l $CALLBACK_PORT -q 1 > callback_output.txt &
NC_PID=$!
echo "Netcat listener started with PID $NC_PID"
sleep 2 # Give it a moment to start up

# 2. Check the initial status (should be IDLE)
print_header "Checking initial status (should be IDLE)"
curl -X GET "$BASE_URL/status"
echo ""

# 3. Send a request to process data
print_header "Sending request to process data"
curl -X POST "$BASE_URL/process" \
    -H "Content-Type: application/json" \
    -d '{
        "input_folder": "'"$INPUT_DIR"'",
        "output_folder": "'"$OUTPUT_DIR"'",
        "callback_url": "'"$CALLBACK_URL"'"
    }'
echo ""

# 4. Check the status immediately after (should be BUSY)
print_header "Checking status immediately after request (should be BUSY)"
sleep 1 # Give the server a moment to switch state
curl -X GET "$BASE_URL/status"
echo ""

# 5. Wait for processing to complete
print_header "Waiting for processing to complete..."
wait $NC_PID
echo "Netcat listener has received the callback and exited."


# 6. Display the callback data received
print_header "Callback data received"
if [ -f "callback_output.txt" ]; then
    cat callback_output.txt
    # The actual HTTP headers and body are captured. We'll just show the JSON part.
    print_header "Callback message content in JSON:"
    grep -o '{.*}' callback_output.txt
    rm callback_output.txt
else
    echo "No callback output file found."
fi
echo ""

# 7. Check the final status (should be IDLE again)
print_header "Checking final status (should be IDLE)"
# Give a second for the status to be updated post-callback
sleep 1
curl -X GET "$BASE_URL/status"
echo ""

echo "Test script finished."
