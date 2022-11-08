# Running MONAI Apps in Nuance PIN

MONAI Deploy Apps can be deployed as Nuance PIN applications with minimal effort and near-zero coding.

This folder includes an example MONAI app, AI-based Lung Nodule Segmentation, which is wrapped by the Nuance PIN API.
The Nuance PIN wrapper code allows MONAI app developer to deploy their existing MONAI apps in Nuance
with minimal code changes.

## Prerequisites

Before setting up and running the example MONAI lung nodule detection app to run as a Nuance PIN App, the user will need to install/download the following libraries.
It is optional to use a GPU for the example app, however, it is recommended that a GPU is used for inference as it is very computationally intensive.

Minimum software requirements:
- [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- [NVIDIA Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#pre-requisites)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Nuance PIN SDK](https://www.nuance.com/healthcare/diagnostics-solutions/precision-imaging-network.html)

> **Note**: Nuance PIN SDK does not require host installation to make the example app work. We will explore options in the [Quickstart](#quickstart) section 

## Quickstart

If you are reading this guide on the MONAI Github repo, you will need to clone the MONAI repo and change the directory to the Nuance PIN integration path.
```bash
git clone https://github.com/Project-MONAI/monai-deploy-app-sdk.git
cd platforms/nuance_pin
```

In this folder you will see the following directory structure
```bash
nuance_pin
    ├── app/                    # directory with MONAI app code ├── lib/                    # you should create this directory where we will place Nuance PIN wheels
    ├── model/                  # directory where we will place the model used by our MONAI app
    ├── app_wrapper.py          # Nuance PIN wrapper code
    ├── docker-compose.yml      # docker compose runtime script
    ├── Dockerfile              # container image build script
    ├── README.md               # this README
    └── requirements.txt        # libraries required for the example integration to work
```

When building the base container image the [Lung Nodule Detection model](https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/lung_nodule_ct_detection_v0.2.0.zip), available from the [MONAI model zoo](https://github.com/Project-MONAI/model-zoo/releases/tag/hosting_storage_v1) will automatically download in the `model` folder. Should the developer choose a different location of their own within the `nuance_pin` subtree for their model, then an update to `MODEL_PATH` variable in `docker-compose.yml` is required.

### Downloading Data for Spleen Segmentation

To download the test data you may follow the instructions in the [Lund Nodule Detection Documentation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/lung_nodule_ct_detection#data).

### Download Nuance PIN SDK

Place the Nuance PIN `ai_service` wheel in the `nuance_pin/lib` folder. This can be obtained in the link provided in step 4 of of the [prerequisites](#prerequisites).

### Running the Example App in the Container

Now we are ready to build and start the container that runs our MONAI app as a Nuance service.
```bash
docker-compose up --build
```

If the build is successful the service will start on `localhost:5000`. We can verify the service is running
by issuing a "live" request such as
```bash
curl -v http://localhost:5000/aiservice/2/live && echo ""
```
The issued command should return the developer, app, and version of the deployed example app.

Now we can run the example app with the example spleen data as the payload using Nuance PIN AI Service Test
(`AiSvcTest`) utility obtained with the Nuance PIN SDK.
```bash
# create a virtual environment and activate it
python3 -m venv /opt/venv
. /opt/venv/bin/activate

# install AiSvcTest
pip install AiSvcTest-<version>-py3-none-any.whl

# create an output directory for the inference results
mkdir -p ~/Downloads/dcm/out

# run AiSvcTest with spleen dicom payload
python -m AiSvcTest -i ~/Downloads/dcm -o ~/Downloads/dcm/out -s http://localhost:5000 -V 2 -k
```

### Running the Example App on the Host

Alternatively the user may choose to run the Nuance PIn service directly on the host. For this we must install the following:
- Nuance PIN AI Service libraries
- Libraries in the `requirements.txt`

```bash
# create a virtual environment and activate it
python3 -m venv /opt/venv
. /opt/venv/bin/activate

# install Nuance Ai Service
pip install ai_service-<version>-py3-none-any.whl

# install requirements
pip install -r requirements.txt

# download the lung nodule detection model
wget -q https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/lung_nodule_ct_detection_v0.2.0.zip && \
unzip lung_nodule_ct_detection_v0.2.0.zip -d /tmp/ && \
cp /tmp/lung_nodule_ct_detection/models/model.ts model/. && \
rm -rf /tmp/lung_nodule_ct_detection && \
rm lung_nodule_ct_detection_v0.2.0.zip

# run the service
export AI_PARTNER_NAME=NVIDIA
export AI_SVC_NAME=ai_spleen_seg_app
export AI_SVC_VERSION=0.1.0
export AI_MODEL_PATH=model/model.ts
export MONAI_APP_CLASSPATH=app.spleen_seg.AISpleenSegApp
export PYTHONPATH=$PYTHONPATH:.
python app_wrapper.py
```

Now we can issue a "live" request to check whether the service is running
```bash
curl -v http://localhost:5000/aiservice/2/live && echo ""
```
As we did in the last section, we can now run the example app with the example spleen data as the payload using Nuance PIN AI Service Test
(`AiSvcTest`) utility obtained with the Nuance PIN SDK.
```bash
. /opt/venv/bin/activate

# install AiSvcTest
pip install AiSvcTest-<version>-py3-none-any.whl

# create an output directory for the inference results
mkdir -p ~/Downloads/dcm/out

# run AiSvcTest with spleen dicom payload
python -m AiSvcTest -i ~/Downloads/dcm -o ~/Downloads/dcm/out -s http://localhost:5000 -V 2 -k
```

### Bring Your Own MONAI App

This example integration may be modified to fit any existing MONAI app by tailoring the files within the `app/` directory, however, there may be caveats.

Nuance PIN requires all artifacts present in the output folder to be also added into the `resultManifest.json` output file
to consider the run successful. To see what this means in practical terms, check the `resultManifest.json` output from the
example app we ran the in previous sections. You will notice an entry in `resultManifest.json` that corresponds to the DICOM
GSPS output generated by the underlying MONAI app
```json
  "study": {
    "uid": "1.2.826.0.1.3680043.2.1125.1.67295333199898911264201812221946213",
    "artifacts": [],
    "series": [
      {
        "uid": "1.2.826.0.1.3680043.2.1125.1.67295333199898911264201812221946213",
        "artifacts": [
          {
            "documentType": "application/dicom",
            "groupCode": "default",
            "name": "gsps.dcm",
            "trackingUids": []
          }
        ]
      }
    ]
  },
```
This entry is generated automatically by Nuance's `ai_service` library as a result of uploading the DICOM GSPS object in `app/post_inference_ops.py`.
In general, however, the developer may need to tailor some of the code in `app_wrapper.py` to provide more insight to Nuance's network, such as adding findings, conclusions, etc. and generating more insight
using SNOMED codes. All of this is handled within the Nuance PIN SDK libraries - for more information please consult Nuance PIN [documentation](https://www.nuance.com/healthcare/diagnostics-solutions/precision-imaging-network.html).

In simpler cases, the developer will need to place their code and model under `nuance_pin`. Placing the model under `model` is optional as the model may be placed
anywhere where the code under `app` can access it, however, considerations must be taken when needing to deploy the model inside a container image. The MONAI app code
is placed in `app` and structured as a small Python project.
