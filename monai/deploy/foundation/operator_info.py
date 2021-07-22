
from monai.deploy.foundation.io_type import IOType


class OperatorInfo():

    """ This is the Operator Info
    """

    def __init__(self):
        ## initializing the attribute
        self.num_input_ports = 0
        self.num_output_ports = 0
        self.input_storage_type = []
        self.output_storage_type = []
        self.input_data_type = []
        self.output_data_type = []



    @property
    def num_input_ports(self):
        return self.__num_input_ports


    ## the attribute name and the method name must be same which is used to set the value for the attribute
    @num_input_ports.setter
    def num_input_ports(self, val):
        self.__num_input_ports = val


    @property
    def num_output_ports(self):
        return self.__num_output_ports


    ## the attribute name and the method name must be same which is used to set the value for the attribute
    @num_output_ports.setter
    def num_output_ports(self, val):
        self.__num_output_ports = val


    
    def set_input_storage_type(self, port_num, storage_type):
        self.input_storage_type.insert(port_num, storage_type)


    def set_input_data_type(self, port_num, data_type):
        self.input_data_type.insert(port_num, data_type)


    def set_output_storage_type(self, port_num, storage_type):
        self.output_storage_type.insert(port_num, storage_type)


    def set_output_data_type(self, port_num, storage_type):
        self.output_data_type.insert(port_num, data_type)


    def get_input_storage_type(self, port_num):
        return self.input_storage_type[port_num]
    
    def get_output_storage_type(self, port_num):
        return self.output_storage_type[port_num]

    def get_input_data_type(self, port_num):
        return self.input_data_type[port_num]
    
    def get_output_data_type(self, port_num):
        return self.output_data_type[port_num]




