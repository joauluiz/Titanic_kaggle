from typing import Any

from enums.model_function import Model_Function
from enums.molder_solver import Model_Solver

class Model_Parameters():
    number_neurons : int
    function : Model_Function
    solver : Model_Solver
    train_input_norm : Any #investigar depois
    train_output_norm : Any
    test_input_norm : Any
    test_output_norm : Any