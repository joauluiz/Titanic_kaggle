from enum import Enum

# class syntax
class Model_Solver (Enum):
    LBFGS = 'lbfgs'
    SGD = 'sgd'
    ADAM = 'adam'