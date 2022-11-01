## A MONAI Application Package to deploy breast density classification algorithm 
This MAP is based on the Breast Density Model in MONAI [Model-Zoo](https://github.com/Project-MONAI/model-zoo). This model is developed at the Center for Augmented Intelligence in Imaging at the Mayo Clinic, Florida. 
For any questions, feel free to contact Vikash Gupta (gupta.vikash@mayo.edu)
Sample data and a torchscript model can be downloaded from https://drive.google.com/drive/folders/1Dryozl2MwNunpsGaFPVoaKBLkNbVM3Hu?usp=sharing


## Run the application package
### Python CLI 
```
python app.py -i <input_dir> -o <out_dir> -m <breast_density_model> 
```

### MONAI Deploy CLI
```
monai-deploy exec app.py -i <input_dir> -o <out_dir> -m <breast_density_model> 
```
Alternatively, you can go a level higher and execute 
```
monai-deploy exec breast_density_classification_app -i <input_dir> -o <out_dir> -m <breast_density_model>
```


### Packaging the monai app
In order to build the monai app, Go a level up and execute the following command.
```
monai-deploy package -b nvcr.io/nvidia/pytorch:21.12-py3 breast_density_classification_app --tag breast_density:0.1.0 -m $breast_density_model
```


