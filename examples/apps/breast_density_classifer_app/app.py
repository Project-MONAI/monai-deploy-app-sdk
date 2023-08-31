import logging
from pathlib import Path

from breast_density_classifier_operator import ClassifierOperator

from monai.deploy.conditions import CountCondition
from monai.deploy.core import AppContext, Application
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.dicom_text_sr_writer_operator import DICOMTextSRWriterOperator, EquipmentInfo, ModelInfo


# @env(pip_packages=["monai~=1.1.0", "highdicom>=0.18.2", "pydicom >= 2.3.0"])
class BreastClassificationApp(Application):
    """This is an AI breast density classification application.

    The DL model was trained by Center for Augmented Intelligence in Imaging, Mayo Clinic, Florida,
    and published on MONAI Model Zoo at
        https://github.com/Project-MONAI/model-zoo/tree/dev/models/breast_density_classification
    """

    def __init__(self, *args, **kwargs):
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__(*args, **kwargs)

    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""
        logging.info(f"Begin {self.compose.__name__}")

        # Use command line options over environment variables to init context.
        app_context: AppContext = Application.init_app_context(self.argv)
        app_input_path = Path(app_context.input_path)
        app_output_path = Path(app_context.output_path)
        model_path = Path(app_context.model_path)

        model_info = ModelInfo(
            "MONAI Model for Breast Density",
            "BreastDensity",
            "0.1",
            "Center for Augmented Intelligence in Imaging, Mayo Clinic, Florida",
        )

        my_equipment = EquipmentInfo(manufacturer="MONAI Deploy App SDK", manufacturer_model="DICOM SR Writer")
        my_special_tags = {"SeriesDescription": "Not for clinical use"}
        study_loader_op = DICOMDataLoaderOperator(
            self, CountCondition(self, 1), input_folder=app_input_path, name="study_loader_op"
        )
        series_selector_op = DICOMSeriesSelectorOperator(self, rules=Sample_Rules_Text, name="series_selector_op")
        series_to_vol_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_op")
        classifier_op = ClassifierOperator(
            self, app_context=app_context, output_folder=app_output_path, model_path=model_path, name="classifier_op"
        )
        sr_writer_op = DICOMTextSRWriterOperator(
            self,
            copy_tags=True,
            model_info=model_info,
            equipment_info=my_equipment,
            custom_tags=my_special_tags,
            output_folder=app_output_path,
            name="sr_writer_op",
        )  # copy_tags=True to use Study and Patient modules of the original input

        self.add_flow(study_loader_op, series_selector_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(
            series_selector_op, series_to_vol_op, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(series_to_vol_op, classifier_op, {("image", "image")})
        self.add_flow(classifier_op, sr_writer_op, {("result_text", "text")})
        # Pass the Study series to the SR writer for copying tags
        self.add_flow(series_selector_op, sr_writer_op, {("study_selected_series_list", "study_selected_series_list")})

        logging.info(f"End {self.compose.__name__}")


# This is a sample series selection rule in JSON, simply selecting a MG series.
# If the study has more than 1 MG series, then all of them will be selected.
# Please see more detail in DICOMSeriesSelectorOperator.
# For list of string values, e.g. "ImageType": ["PRIMARY", "ORIGINAL"], it is a match if all elements
# are all in the multi-value attribute of the DICOM series.

Sample_Rules_Text = """
{
    "selections": [
        {
            "name": "MG Series",
            "conditions": {
                "Modality": "(?i)MG",
                "ImageType": ["PRIMARY"]
            }
        }
    ]
}
"""


def test():
    app = BreastClassificationApp()
    image_dir = "./sampleDICOMs/1/BI_BREAST_SCREENING_BILATERAL_WITH_TOMOSYNTHESIS-2019-07-08/1/L_CC_C-View"

    model_path = "./model/traced_ts_model.pt"
    app.run(input=image_dir, output="./output", model=model_path)


if __name__ == "__main__":
    logging.info(f"Begin {__name__}")
    BreastClassificationApp().run()
    logging.info(f"End {__name__}")
