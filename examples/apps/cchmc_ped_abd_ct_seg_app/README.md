# MONAI Application Package (MAP) for CCHMC Pediatric Abdominal CT Segmentation MONAI Bundle

This MAP is based on the [CCHMC Pediatric Abdominal CT Segmentation MONAI Bundle](https://github.com/cchmc-dll/pediatric_abdominal_segmentation_bundle/tree/original). This model was developed at Cincinnati Children's Hospital Medical Center by the Department of Radiology.

The PyTorch and TorchScript DynUNet models can be downloaded from the [MONAI Bundle Repository](https://github.com/cchmc-dll/pediatric_abdominal_segmentation_bundle/tree/original/models).

For questions, please feel free to contact Elan Somasundaram (Elanchezhian.Somasundaram@cchmc.org) and Bryan Luna (Bryan.Luna@cchmc.org).

## Unique Features

Some unique features of this MAP pipeline include:
- **Custom Inference Operator:** custom `AbdomenSegOperator` enables either PyTorch or TorchScript model loading
- **DICOM Secondary Capture Output:** custom `DICOMSecondaryCaptureWriterOperator` writes a DICOM SC with organ contours
- **Output Filtering:** model produces Liver-Spleen-Pancreas segmentations, but seg visibility in the outputs (DICOM SEG, SC, SR) can be controlled in `app.py`
- **MONAI Deploy Express MongoDB Write:** custom operators (`MongoDBEntryCreatorOperator` and `MongoDBWriterOperator`) allow for writing to the MongoDB database associated with MONAI Deploy Express

## Scripts
Several scripts have been compiled that quickly execute useful actions (such as running the app code locally with Python interpreter, MAP packaging, MAP execution, etc.). Some scripts require the input of command line arguments; review the `scripts` folder for more details.

## Notes
The DICOM Series selection criteria has been customized based on the model's training and CCHMC use cases. By default, Axial CT series with Slice Thickness between 3.0 - 5.0 mm (inclusive) will be selected for. 

If MongoDB writing is not desired, please comment out the relevant sections in `app.py` and the `AbdomenSegOperator`. 

To execute the pipeline with MongoDB writing enabled, it is best to create a `.env` file that the `MongoDBWriterOperator` can load in. Below is an example `.env` file that follows the format outlined in this operator; note that these values are the default variable values as defined in the [.env](https://github.com/Project-MONAI/monai-deploy/blob/main/deploy/monai-deploy-express/.env) and [docker-compose.yaml](https://github.com/Project-MONAI/monai-deploy/blob/main/deploy/monai-deploy-express/docker-compose.yml) files of v0.6.0 of MONAI Deploy Express:

```dotenv
MONGODB_USERNAME=root
MONGODB_PASSWORD=rootpassword
MONGODB_PORT=27017
MONGODB_IP_DOCKER=172.17.0.1 # default Docker bridge network IP
```

Prior to packaging into a MAP, the MongoDB credentials should be hardcoded into the `MongoDBWriterOperator`.

The MONAI Deploy Express MongoDB Docker container (`mdl-mongodb`) needs to be connected to the Docker bridge network in order for the MongoDB write to be successful. Executing the following command in a MONAI Deploy Express terminal will establish this connection:

```bash
docker network connect bridge mdl-mongodb
```
