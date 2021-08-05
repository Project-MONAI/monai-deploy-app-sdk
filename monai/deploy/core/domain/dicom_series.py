
import math
from .dicom_sop_instance import DICOMSOPInstance
from .domain import Domain


class DICOMSeries(Domain):
    def __init__(self, series_instance_uid):
        super().__init__(None)
        self._series_instance_uid = series_instance_uid
        self._sop_instances = []

    
    def get_series_instance_uid(self):
        return self._series_instance_uid

    
    def add_sop_instance(self, sop_instance):
        dicom_sop_instance = DICOMSOPInstance(sop_instance)
        self._sop_instances.append(dicom_sop_instance)

    
    def get_sop_instances(self):
        return self._sop_instances

    
    @property
    def series_date(self):
        return self.__series_date

    @series_date.setter
    def series_date(self, val):
        self.__series_date = val


    @property
    def series_time(self):
        return self.__series_time

    @series_time.setter
    def series_time(self, val):
        self.__series_time = val

    
    @property
    def modality(self):
        return self.__modality

    @modality.setter
    def modality(self, val):
        self.__modality = val


    @property
    def series_description(self):
        return self.__series_description

    @series_description.setter
    def series_description(self, val):
        self.__series_description = val

    @property
    def body_part_examined(self):
        return self.__body_part_examined

    @body_part_examined.setter
    def body_part_examined(self, val):
        self.__body_part_examined = val

    @property
    def patient_position(self):
        return self.__patient_position

    @patient_position.setter
    def patient_position(self, val):
        self.__patient_position = val

    @property
    def series_number(self):
        return self.__series_number

    @series_number.setter
    def series_number(self, val):
        self.__series_number = val

    @property
    def laterality(self):
        return self.__laterality

    @laterality.setter
    def laterality(self, val):
        self.__laterality = val


    @property
    def row_pixel_spacing(self):
        return self.__row_pixel_spacing

    @row_pixel_spacing.setter
    def row_pixel_spacing(self, val):
        self.__row_pixel_spacing = val


    @property
    def col_pixel_spacing(self):
        return self.__col_pixel_spacing

    @col_pixel_spacing.setter
    def col_pixel_spacing(self, val):
        self.__col_pixel_spacing = val

    @property
    def depth_pixel_spacing(self):
        return self.__depth_pixel_spacing

    @depth_pixel_spacing.setter
    def depth_pixel_spacing(self, val):
        self.__depth_pixel_spacing = val


    def __str__(self):
        result = "---------------" +"\n"
        
        series_instance_uid_attr = "Series Instance UID: " + self._series_instance_uid + "\n"
        result += series_instance_uid_attr
        
        try:
            num_sop_instances =  "Num SOP Instances: " + str(len(self._sop_instances))  + "\n"
            result += num_sop_instances
        except AttributeError:
            pass
        

        try:
            series_date_attr =  "Series Date: " + self.series_date + "\n"
            result += series_date_attr
        except AttributeError:
            pass
        
        try:
            series_time_attr =  "Series Time: " + self.series_time + "\n"
            result += series_time_attr
        except AttributeError:
            pass 
        
        try:
            modality_attr =  "Modality: " + self.modality + "\n"
            result += modality_attr
        except AttributeError:
            pass

        
        try:
            depth_pixel_spacing_attr =  "Depth Pixel Spacing: " + str(self.depth_pixel_spacing) + "\n"
            result += depth_pixel_spacing_attr
        except AttributeError:
            pass
        
        result += "---------------" +"\n"
        

        return result

