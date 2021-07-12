import uuid
import os


class ExecutionContext():

    def __init__(self):
        super().__init__()
        self._storage = {}


    def set_operator_input(self, op_uid, input_port_num, val):
        key = (op_uid, 'input', input_port_num)
        self._storage[key] = val


    def set_operator_output(self, op_uid, output_port_num, val):
        key = (op_uid, 'output', output_port_num)
        self._storage[key] = val


    def get_operator_input(self, op_uid, input_port_num):
        key = (op_uid, 'input', input_port_num)
        return self._storage[key]

    def get_operator_output(self, op_uid, output_port_num):
        key = (op_uid, 'output', output_port_num)
        return self._storage[key]
    

