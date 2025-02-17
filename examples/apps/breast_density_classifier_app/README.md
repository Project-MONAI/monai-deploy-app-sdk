## A MONAI Application Package to deploy breast density classification algorithm
This MAP is based on the Breast Density Model in MONAI [Model-Zoo](https://github.com/Project-MONAI/model-zoo). This model is developed at the Center for Augmented Intelligence in Imaging at the Mayo Clinic, Florida.
For any questions, feel free to contact Vikash Gupta (gupta.vikash@mayo.edu)
Sample data and a torchscript model can be downloaded from https://drive.google.com/drive/folders/1Dryozl2MwNunpsGaFPVoaKBLkNbVM3Hu?usp=sharing


## Run the application code with Python interpreter
```
python app.py -i <input_dir> -o <out_dir> -m <breast_density_model>
```

## Package the application as a MONAI Application Package (container image)
In order to build the MONAI App Package, go a level up and execute the following command.
```
monai-deploy package breast_density_classification_app -m <breast_density_model> -c breast_density_classifer_app/app.yaml --tag breast_density:0.1.0 --platform x64-workstation -l DEBUG
```

## Run the MONAI Application Package using MONAI Deploy CLI
```
monai-deploy run breast_density-x64-workstation-dgpu-linux-amd64:0.1.0 -i <input_dir> -o <output_dir>
```

Once the container exits successfully, check the results in the output directory. There should be a newly created DICOM instance file and a `output.json` file containing the classification results.