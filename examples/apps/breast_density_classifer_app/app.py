from breast_density_classifier_operator import ClassifierOperator

from monai.deploy.core import Application, env
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.dicom_text_sr_writer_operator import DICOMTextSRWriterOperator, EquipmentInfo, ModelInfo


@env(pip_packages=["highdicom>=0.18.2"])
class BreastClassificationApp(Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compose(self):
        model_info = ModelInfo(
            "MONAI Model for Breast Density",
            "BreastDensity",
            "0.1",
            "Center for Augmented Intelligence in Imaging, Mayo Clinic, Florida",
        )
        my_equipment = EquipmentInfo(manufacturer="MONAI Deploy App SD", manufacturer_model="DICOM SR Writer")
        my_special_tags = {"SeriesDescription": "Not for clinical use"}
        study_loader_op = DICOMDataLoaderOperator()
        series_selector_op = DICOMSeriesSelectorOperator(rules="")
        series_to_vol_op = DICOMSeriesToVolumeOperator()
        classifier_op = ClassifierOperator()
        sr_writer_op = DICOMTextSRWriterOperator(
            copy_tags=True, model_info=model_info, equipment_info=my_equipment, custom_tags=my_special_tags
        )  # copy_tags=True to use Study and Patient modules of the original input

        self.add_flow(study_loader_op, series_selector_op, {"dicom_study_list": "dicom_study_list"})
        self.add_flow(
            series_selector_op, series_to_vol_op, {"study_selected_series_list": "study_selected_series_list"}
        )
        self.add_flow(series_to_vol_op, classifier_op, {"image": "image"})
        self.add_flow(classifier_op, sr_writer_op, {"result_text": "classification_result"})
        # Pass the Study series to the SR writer for copying tags
        self.add_flow(series_selector_op, sr_writer_op, {"study_selected_series_list": "study_selected_series_list"})


def test():
    app = BreastClassificationApp()
    image_dir = "./sampleDICOMs/1/BI_BREAST_SCREENING_BILATERAL_WITH_TOMOSYNTHESIS-2019-07-08/1/L_CC_C-View"

    model_path = "./model/traced_ts_model.pt"
    app.run(input=image_dir, output="./output", model=model_path)


if __name__ == "__main__":
    app = BreastClassificationApp(do_run=True)
