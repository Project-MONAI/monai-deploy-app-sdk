# RESTful Wrapper Application for MONAI Deploy

This application provides a RESTful web interface to run MONAI Deploy applications.

It allows you to start a processing job, check the status, and receive a callback when the job is complete.

As it stands now, the callback message content is stubbed/generated in the wrapper app, and this will change to the design
where the wrapper app will pass a static callback function to the MONAI Deploy app which will have a reporter operator
that gathers the operations and domain specific info in the app's pipeline and then reports back the content via
this callback. The wrapper app will then have a mapping function to transform the reported data to that expected by
the external callback endpoint.

Also, the whole Restful application can be packaged into a container image using MONAI Deploy app packager, but not doner here.

## How to Run

Change working directory to the same level as this README.

1.  **Install Dependencies**

    Create and activate a Python virtual environment.

    ```bash
    pip install -r restful_app/requirements.txt
    ```
2.  **Download Test Data and Set Env Vars**
    The model and test DICOM series are shared on Google Drive requiring first gaining access permission, and
    the zip file is [here](https://drive.google.com/uc?id=1IwWMpbo2fd38fKIqeIdL8SKTGvkn31tK).

    Please make a request so that it can be shared to specific Gmail account.

    `gdown` may also work.
    ```
    pip install gdown
    gdown https://drive.google.com/uc?id=1IwWMpbo2fd38fKIqeIdL8SKTGvkn31tK
    ```

    Unzip the file to local folders. If deviating from the path noted below, please adjuest the env var values

    ```
    unzip -o "ai_spleen_seg_bundle_data.zip"
    rm -rf models && mkdir -p models/model && mv model.ts models/model && ls models/model
    ```

    Set the environment vars so that the model can be found by the Spleen Seg app. Also,
    the settings are consolidated in the `env_settings.sh`.

    ```
    export HOLOSCAN_MODEL_PATH=models
    ```

3.  **Run the Web Application**

    ```bash
    python restful_app/app.py
    ```

    The application will start on `http://127.0.0.1:5000`.

## Test API Endpoints

A simplest test client is provided, which makes call to the endpoint, as well as providing
a callback endpoint to receives message content at the specidied port.

Open another console window and change directory to the same as this file.

Set the environment vars so that the test script can get the input DCM and write the callback contents.
Also, once the Restful app completes each processing, the Spleen Seg app's output will also be saved in
the output folder speficied below (the script passes the output folder via the Rest API).

```
export HOLOSCAN_INPUT_PATH=dcm
export HOLOSCAN_OUTPUT_PATH=output
```

Run the test script, and examine its console output.

```
source test_endpoints.sh
```

Once the script completes, examine the `output` folder, which should conatain the following (dcm file
name will be different)

```
output
├── 1.2.826.0.1.3680043.10.511.3.22611096892439837402906545708809852.dcm
└── stl
    └── spleen.stl
```

The script can run multiple times, or modified to loop with different output folder setting.

### Check Status

-   **URL**: `/status`
-   **Method**: `GET`
-   **Description**: Checks the current status of the processor.
-   **Success Response**:
    -   **Code**: 200 OK
    -   **Content**: `{ "status": "IDLE" }` or `{ "status": "BUSY" }`

### Process Data

-   **URL**: `/process`
-   **Method**: `POST`
-   **Description**: Starts a new processing job.
-   **Body**:

    ```json
    {
        "input_folder": "/path/to/your/input/data",
        "output_folder": "/path/to/your/output/folder",
        "callback_url": "http://your-service.com/callback"
    }
    ```

-   **Success Response**:
    -   **Code**: 202 ACCEPTED
    -   **Content**: `{ "message": "Processing started." }`
-   **Error Response**:
    -   **Code**: 409 CONFLICT
    -   **Content**: `{ "error": "Processor is busy." }`
    -   **Code**: 400 BAD REQUEST
    -   **Content**: `{ "error": "Missing required fields." }`

### Callback

When processing is complete, the application will send a `POST` request to the `callback_url` provided in the process request. The body of the callback will be:

```json
{
    "run_success": true,
    "result": "Processing completed successfully.",
    "output_files": ["test.json", "seg.com"],
    "error_message": null,
    "error_code": null
}
```

Or in case of an error:

```json
{
    "run_success": False,
    "error_message": "E.g., Model network is not load and model file not found.",
    "error_code": 500
}
```
