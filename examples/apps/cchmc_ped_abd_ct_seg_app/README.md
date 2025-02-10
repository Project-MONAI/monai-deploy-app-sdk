# MONAI Application Package (MAP) for CCHMC Pediatric Abdominal CT Segmentation MONAI Bundle

This MAP is based on the [CCHMC Pediatric Abdominal CT Segmentation MONAI Bundle](https://github.com/cchmc-dll/pediatric_abdominal_segmentation_bundle/tree/original). This model was developed at Cincinnati Children's Hospital Medical Center by the Department of Radiology.

The PyTorch and TorchScript DynUNet models can be downloaded from the [MONAI Bundle Repository](https://github.com/cchmc-dll/pediatric_abdominal_segmentation_bundle/tree/original/models).

For questions, please feel free to contact Elan Somasundaram (Elanchezhian.Somasundaram@cchmc.org) and Bryan Luna (Bryan.Luna@cchmc.org).

## Unique Features

Some unique features of this MAP pipeline include:
- **Custom Inference Operator:** custom `AbdomenSegOperator` enables either PyTorch or TorchScript model loading as desired
- **DICOM Secondary Capture Output:** custom `DICOMSecondaryCaptureWriterOperator` writes a DICOM SC with organ contours
- **Output Filtering:** model produces Liver-Spleen-Pancreas segmentations, but seg visibility in the outputs (DICOM SEG, SC, SR) can be controlled in `app.py`
- **MONAI Deploy Express MongoDB Write:** custom operators (`MongoDBEntryCreatorOperator` and `MongoDBWriterOperator`) allow for writing to the MongoDB database associated with MONAI Deploy Express

## Scripts
Several scripts have been compiled that quickly execute useful actions (such as running the app code locally with Python interpreter, MAP packaging, MAP execution, etc.). Some scripts require the input of command line arguments; review the `scripts` folder for more details.

## Notes
The DICOM Series selection criteria has been customized based on the model's training and CCHMC use cases. By default, Axial CT series with Slice Thickness between 3.0 - 5.0 mm (inclusive) will be selected for. 

If PyTorch model loading is desired, please uncomment the "PyTorch Model Loading" section in the `AbdomenSegOperator`.

If MongoDB writing is desired, please uncomment the relevant sections in `app.py` and the `AbdomenSegOperator`. Please note that MongoDB connection values (username, password, and port) are the default values pulled from the v0.6.0 MONAI Deploy Express [.env](https://github.com/Project-MONAI/monai-deploy/blob/main/deploy/monai-deploy-express/.env) and [docker-compose.yaml](https://github.com/Project-MONAI/monai-deploy/blob/main/deploy/monai-deploy-express/docker-compose.yml) files; these default values are harcoded into the `MongoDBWriterOperator`. If your instance of MONAI Deploy Express has modified values for these fields, the `MongoDBWriterOperator` will need to be udpated accordingly.

The MONAI Deploy Express MongoDB Docker container (`mdl-mongodb`) needs to be connected to the Docker bridge network in order for the MongoDB write to be successful. Executing the following command in a MONAI Deploy Express terminal will establish this connection:

```bash
docker network connect bridge mdl-mongodb
```
