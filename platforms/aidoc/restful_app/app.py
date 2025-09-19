# Copyright 2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import importlib
import json
import logging
import os
import sys
import threading
from http import HTTPStatus

import requests
from flask import Flask, jsonify, request
from flask_wtf.csrf import CSRFProtect

# The MONAI Deploy application to be wrapped.
# This can be changed to any other application in the repository.
# Provide the module path and the class name of the application.

APP_MODULE_NAME = "ai_spleen_seg_app"
APP_CLASS_NAME = "AISpleenSegApp"

# Flask application setup
app = Flask(__name__)
# It is recommended to use a securely generated random string for the secret key,
# and store it in an environment variable or a secure configuration file.
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "a-secure-default-secret-key-for-dev")
csrf = CSRFProtect(app)


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Global state to track processing status. A lock is used for thread safety.
PROCESSING_STATUS = "IDLE"
PROCESSING_LOCK = threading.Lock()


def set_processing_status(status):
    """Sets the processing status in a thread-safe manner."""
    global PROCESSING_STATUS
    with PROCESSING_LOCK:
        PROCESSING_STATUS = status


def get_processing_status():
    """Gets the processing status in a thread-safe manner."""
    with PROCESSING_LOCK:
        return PROCESSING_STATUS


def run_processing(input_folder, output_folder, callback_url):
    """
    This function runs in a separate thread to execute the MONAI Deploy application.
    """

    # Define the callback function that the MONAI Deploy app will call.
    def app_status_callback(summary: str):
        """Callback function to handle the final status from the application."""
        logging.info(f"Received status from application: {summary}")
        if callback_url:
            try:
                logging.info(f"Sending final status callback to {callback_url}")
                # Here you could map the summary to the expected format of the callback.
                # For now, we'll just forward the summary.
                response = requests.post(callback_url, data=summary, timeout=5)
                response.raise_for_status()  # for bad status codes (4xx or 5xx)
                logging.info("Sent final status callback.")

            except requests.exceptions.Timeout:
                logging.error("The request timed out.")
            except requests.exceptions.ConnectionError:
                logging.error("A connection error occurred.")
            except requests.exceptions.RequestException as e:
                logging.error(f"An unexpected error occurred: {e}")
            except Exception as e:
                logging.error(f"Failed to send callback to {callback_url}: {e}")

    try:
        logging.info("Starting processing in a background thread.")
        set_processing_status("BUSY")

        # Set environment variables for the MONAI Deploy application.
        # The application context will pick these up.
        os.environ["MONAI_INPUTPATH"] = input_folder
        os.environ["MONAI_OUTPUTPATH"] = output_folder
        os.environ["HOLOSCAN_INPUT_PATH"] = input_folder  # For Holoscan-based apps
        os.environ["HOLOSCAN_OUTPUT_PATH"] = output_folder  # For Holoscan-based apps

        # Dynamically import the application class from the specified module.
        logging.info(f"Loading application: {APP_MODULE_NAME}.{APP_CLASS_NAME}")
        module = importlib.import_module(APP_MODULE_NAME)
        app_class = getattr(module, APP_CLASS_NAME)
        monai_app = app_class(status_callback=app_status_callback)

        # Run the MONAI Deploy application which calls the callback if successful.
        logging.info("Running the MONAI Deploy application.")
        monai_app.run()
        logging.info("Processing completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        # If the app fails, we need to handle it here and report a failure.
        callback_msg = {
            "run_success": False,
            "error_message": f"Error during processing: {e}",
            "error_code": 500,
        }
        app_status_callback(json.dumps(callback_msg))

    finally:
        set_processing_status("IDLE")
        logging.info("Processor is now IDLE.")


@app.route("/status", methods=["GET"])
def status():
    """Endpoint to check the current processing status."""
    return jsonify({"status": get_processing_status()})


@app.route("/process", methods=["POST"])
@csrf.exempt
def process():
    """Endpoint to start a new processing job."""
    if get_processing_status() == "BUSY":
        return jsonify({"error": "Processor is busy."}), HTTPStatus.CONFLICT

    data = request.get_json()
    if not data or "input_folder" not in data or "output_folder" not in data:
        return jsonify({"error": "Missing required fields."}), HTTPStatus.BAD_REQUEST

    input_folder = data["input_folder"]
    output_folder = data["output_folder"]
    callback_url = data.get("callback_url")  # Callback URL is optional

    # Start the processing in a background thread.
    thread = threading.Thread(target=run_processing, args=(input_folder, output_folder, callback_url))
    thread.start()

    return jsonify({"message": "Processing started."}), HTTPStatus.ACCEPTED


if __name__ == "__main__":
    # Note: For production, use a proper WSGI server like Gunicorn or uWSGI.
    parser = argparse.ArgumentParser(description="Run the MONAI Deploy RESTful wrapper application.")
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("FLASK_HOST", "0.0.0.0"),
        help="Host address to bind the Flask server to. Defaults to env var FLASK_HOST or 0.0.0.0.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FLASK_PORT", 5000)),
        help="Port to listen on. Defaults to env var FLASK_PORT or 5000.",
    )
    args = parser.parse_args()
    host = args.host or os.environ.get("FLASK_HOST", "0.0.0.0")
    port = args.port or int(os.environ.get("FLASK_PORT", 5000))
    app.run(host=host, port=port)
