from enum import Enum

from enums.message_error import Message_Error
from enums.message_inputs import Message_Inputs


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def input_float_values(message):
    while True:
        input_value = input(message)

        # Checking if the value is an integer number by the isdigit() function
        if is_float(input_value):
            break

        else:
            # If the value is not an integer number, inform the error and request the input again
            print(Message_Error.DECIMAL_NUMBER.value)

    return float(input_value)


def input_sex():
    while True:
        input_value = input(Message_Inputs.SEX.value)
        # Checking if the value is an integer number by the isdigit() function
        if input_value.isdigit():
            if input_value in ['0', '1']:
                break
            else:
                print(Message_Error.ZERO_OR_ONE.value)
        else:
            # If the value is not an integer, inform the error and request the input again
            print(Message_Error.INTEGER.value)

    return int(input_value)


def input_socieconomic_class():
    while True:
        input_value = input(Message_Inputs.SOCIECONOMIC_CLASS.value)
        # Checking if the value is an integer number by the isdigit() function
        if input_value.isdigit():

            if input_value in ['1', '2', '3']:
                break

            else:
                print(Message_Error.ONE_TWO_OR_THREE.value)
        else:
            # If the value is not an integer, inform the error and request the input again
            print(Message_Error.INTEGER.value)

    return int(input_value)


def input_port_embarkation():
    while True:
        input_value = input(Message_Inputs.PORT_EMBARKATION.value)

        # Checking if the value is a letter by the method isalpha()
        if input_value.isalpha():

            if input_value == "C" or input_value == "c":
                input_value = 1
                break

            elif input_value == "Q" or input_value == "q":
                input_value = 2
                break

            elif input_value == "S" or input_value == "s":
                input_value = 3
                break

            else:
                print(Message_Error.C_Q_S.value)
        else:
            # If the value is not an integer, inform the error and request the input again
            print(Message_Error.LETTER.value)

    return input_value


class Model_Inputs:

    def __init__(self):
        self.age = input_float_values(Message_Inputs.AGE.value)

        self.siblings_spouses = input_float_values(Message_Inputs.SIBLINGS_SPOUSES.value)

        self.parents_children = input_float_values(Message_Inputs.PARENTS_CHILDREN.value)

        self.fare_paid = input_float_values(Message_Inputs.FARE_PAID.value)

        self.sex = input_sex()

        self.socieconomic_class = input_socieconomic_class()

        self.port_embarkation = input_port_embarkation()
