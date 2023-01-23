from enum import Enum

# class syntax
class Message_Error (Enum):
    ACTIVE_FUNCTION = 'Wrong active function, please try again: '
    POSITIVE_INTEGER = 'The number must be positive and integer'
    POSITIVE_FLOAT = 'The number must be positive and float'
    INTEGER = "The value must be an integer. Try again."
    LETTER = "The value must be a letter. Try again."
    SOLVER = 'Wrong solver, must be one of the three.'
    DECIMAL_NUMBER = "The value must be a number, with the decimal separator being the point '.'. Try again."
    ZERO_OR_ONE = "The value must be 0 or 1. Try again."
    ONE_TWO_THREE = "The value must be 1, 2 or 3. Try again."
    C_Q_S = "The value must be C, Q or S. Try again."
    WRONG_VALUE = 'Wrong value, please try again: '
    NUMBER = 'The value must be 0, 1 or 2'
