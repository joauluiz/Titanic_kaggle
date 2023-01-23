import pandas as pd
import numpy as np
import mlp
import warnings
import time
from fastapi import FastAPI
from pydantic import BaseModel
from entities.model_inputs import Model_Inputs
from entities.model_parameters import Model_Parameters
from enums.message_error import Message_Error
from enums.model_function import Model_Function
from enums.molder_solver import Model_Solver
from enums.train_choice import Train_Choice
from enums.message_inputs import Message_Inputs

warnings.filterwarnings("ignore", category=Warning)

app = FastAPI()


# @app.get('/')
# def main_output():
#     output = main()
#     return output

class questions(BaseModel):
    age: int
    simblings: int
    parents_children: int
    fare_paid: float
    sex: int
    socioeconomic_class: int  # take a look on what restrict values
    port_embarkation: str


@app.post('/inputs/')
def inputs(teste: questions):
    return teste.age, teste.simblings, teste.parents_children, teste.fare_paid, teste.sex, \
        teste.socioeconomic_class, teste.port_embarkation


# print(teste.age)

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def is_int(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def training_choice():
    print('\nChoose one of the options:')
    print(Message_Inputs.BEST_PARAMETERS.value)
    print(Message_Inputs.DEFINE_PARAMETERS.value)
    print(Message_Inputs.PREVIOUS_BEST_PARAMETERS.value)
    train_choice_input = input('Type:')
    return train_choice_input


def number_int_positive(message):
    while True:
        question = input(message)
        if is_int(question) and int(question) > 0:
            break
        else:
            print(Message_Error.POSITIVE_INTEGER.value)
            time.sleep(5)
    return question


def number_float_positive(message):
    while True:
        question = input(message)
        if is_float(question) and float(question) > 0:
            break
        else:
            print(Message_Error.POSITIVE_FLOAT.value)
            time.sleep(5)
    return question


def model_function():
    while True:
        question = input(Message_Inputs.ACTIVE_FUNCTION.value)
        if question == 'relu' or question == 'logistic' or question == 'tanh':
            break
        else:
            print(Message_Error.ACTIVE_FUNCTION.value)
            time.sleep(2)
    return question


def model_solver():
    while True:
        question = input(Message_Inputs.SOLVER.value)
        if question == 'lbfgs' or question == 'sgd' or question == 'adam':
            break
        else:
            print(Message_Error.SOLVER.value)
            time.sleep(2)
    return question


def model_neurons():
    while True:
        question = input(Message_Inputs.NUMBER_NEURONS.value)
        if is_int(question) and int(question) > 0:
            break
        else:
            print(Message_Error.POSITIVE_INTEGER.value)
            time.sleep(2)
    return int(question) - 1


def data_norm():
    train_data, test_data, aux_test_data = mlp.load_data()

    train_output_norm, train_input_norm = mlp.treat_data(train_data)

    aux_test_data = aux_test_data[["Survived"]]

    test_data = pd.concat([test_data, aux_test_data], axis=1)

    test_output_norm, test_input_norm = mlp.treat_data(test_data)

    return train_output_norm, train_input_norm, test_output_norm, test_input_norm


def model_parameters():
    print('\nChoose the parameters:')

    train_data, test_data, aux_test_data = mlp.load_data()

    train_output_norm, train_input_norm = mlp.treat_data(train_data)

    aux_test_data = aux_test_data[["Survived"]]

    test_data = pd.concat([test_data, aux_test_data], axis=1)

    test_output_norm, test_input_norm = mlp.treat_data(test_data)

    number_neurons = model_neurons()

    func = model_function()

    solver = model_solver()

    return Model_Parameters(number_neurons, func, solver, train_input_norm, train_output_norm, test_input_norm,
                            test_output_norm)


def continue_or_not():
    while True:

        print(Message_Inputs.CONTINUE_PREDICT.value)

        print(Message_Inputs.TEST_OTHER_PARAMETERS.value)

        print(Message_Inputs.PREVIOUS_OPTIONS.value)

        question = input('Type:')

        if int(question) in [0, 1, 2]:
            break

        else:
            print(Message_Error.WRONG_VALUE.value)
            time.sleep(5)

    return int(question)


def user_inputs():
    inputs = Model_Inputs()

    # Creating a list that stores the values entered by the user

    input_values = [inputs.age, inputs.sex, inputs.siblings_spouses, inputs.parents_children,
                    inputs.fare_paid, inputs.socieconomic_class, inputs.port_embarkation]

    input_values = np.array(input_values).reshape(1, -1)

    return input_values


def user_model_choose():
    while True:

        train_choice_input = training_choice()

        # trazer para lower case tudo
        if train_choice_input == Train_Choice.BEST_PARAMETERS.value:

            acc, model = mlp.retrieve_best_parameters()

            time.sleep(2)

            break

        elif train_choice_input == Train_Choice.PREVIOUS_PARAMETERS.value:

            train_output_norm, train_input_norm, test_output_norm, test_input_norm = data_norm()

            acc, model = mlp.train_model(Model_Parameters(1, Model_Function.RELU.value, Model_Solver.SGD.value,
                                                          train_input_norm, train_output_norm,
                                                          test_input_norm, test_output_norm))
            time.sleep(2)

            break

        while train_choice_input == Train_Choice.DEFINE_PARAMETERS.value:

            parameters = model_parameters()

            acc, model = mlp.train_model(Model_Parameters(parameters.number_neurons, parameters.function,
                                                          parameters.solver, parameters.train_input_norm,
                                                          parameters.train_output_norm, parameters.test_input_norm,
                                                          parameters.test_output_norm))

            print("The accuracy for the model is: ", round(acc * 100, 2), "%\n")

            time.sleep(2)

            question = continue_or_not()

            if question == 0:
                break

            if question == 2:
                train_choice_input = Train_Choice.CONTINUE.value

        if train_choice_input == Train_Choice.DEFINE_PARAMETERS.value:
            break

        if train_choice_input == Train_Choice.CONTINUE.value:
            # time.sleep(0)
            pass

        else:
            print(Message_Error.NUMBER.value)

            time.sleep(1)

    return acc, model
