import sys
sys.path.insert(0,'../../../')
import numpy as np


class Image():
    
    
    def __init__(self):
        super().__init__()
        
    
    @property
    def ndim(self):
        return self.__ndim

    @ndim.setter
    def ndim(self, val):
        self.__ndim = val


    @property
    def shape(self):
        return self.__shape


    @shape.setter
    def shape(self, val):
        self.__shape = val

    
    @property
    def spacing(self):
        return self.__spacing


    @spacing.setter
    def spacing(self, val):
        self.__spacing = val


    @property
    def direction_cosines(self):
        return self.__direction_cosines


    @direction_cosines.setter
    def direction_cosines(self, val):
        self.__direction_cosines = val


    @property
    def modality(self):
        return self.__modality

    @modality.setter
    def modality(self, val):
        self.__modality = val


    @property
    def pixel_data(self):
        return self.__pixel_data

    @pixel_data.setter
    def pixel_data(self, val):
        self.__pixel_data = val


    def __str__(self):
        
        result = ""
        modality = "Modality: " + self.modality
        result = result + modality +"\n"
        num_dim = "Number of Dimensions: " + str(self.ndim)
        result = result + num_dim +"\n"
        spacing = "Pixel Spacing: " + str(self.spacing)
        result = result + spacing +"\n"
        dir_cosines = "Direction Cosines: " + str(self.direction_cosines)
        result = result + dir_cosines +"\n"

        return result
       


    