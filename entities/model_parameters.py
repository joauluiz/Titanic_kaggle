from typing import Any

from enums.model_function import Model_Function
from enums.molder_solver import Model_Solver

class Model_Parameters:
    def __init__(self, number_neurons, function, solver, train_input_norm, train_output_norm, test_input_norm,test_output_norm):

        self.number_neurons= number_neurons
        self.function= function
        self.solver= solver
        self.train_input_norm= train_input_norm  #investigar depois
        self.train_output_norm= train_output_norm
        self.test_input_norm= test_input_norm
        self.test_output_norm= test_output_norm




