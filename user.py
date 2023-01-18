import pandas as pd
import numpy as np
import mlp
import warnings
import time
from fastapi import FastAPI
from pydantic import BaseModel

from enums.train_choice import Train_Choice

warnings.filterwarnings("ignore", category=Warning)

app = FastAPI()

# @app.get('/')
# def main_output():
#     output = main()
#     return output

class questions(BaseModel):
    age : int
    simblings : int
    parents_children : int
    fare_paid: float
    sex: int
    socioeconomic_class: int #take a look on what restrict values
    port_embarkation: str



@app.post('/inputs/')
def inputs(teste: questions):
    return teste.age, teste.simblings, teste.parents_children, teste.fare_paid, teste.sex,\
        teste.socioeconomic_class, teste.port_embarkation

#print(teste.age)

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

def number_int_positive (message):

    while True:
        question = input(message)
        if is_int(question) and int(question)>0:
            break
        else:
            print('The number must be positive and integer')
            time.sleep(5)
    return question

def number_float_positive(message):

    while True:
        question = input(message)
        if is_float(question) and float(question)>0:
            break
        else:
            print('The number must be positive and float')
            time.sleep(5)
    return question

def model_function():

    while True:
        question = input('Choose an active function: relu, logistic or tanh: ')
        if question == 'relu' or question == 'logistic' or  question == 'tanh':
            break
        else:
            print('Wrong active function, must be one of the three.')
            time.sleep(2)
    return question

def model_solver():

    while True:
        question = input('Choose one of the solvers: lbfgs, sgd or adam: ')
        if question == 'lbfgs' or question == 'sgd' or  question == 'adam':
            break
        else:
            print('Wrong solver, must be one of the three.')
            time.sleep(2)
    return question

def model_neurons():

    while True:
        question = input('Please, type the number of neurons in the hidden layer:')
        if is_int(question) and int(question) > 0:
            break
        else:
            print('The number must be positive and integer')
            time.sleep(2)
    return int(question)

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
    return number_neurons, func, solver,train_input_norm, train_output_norm, test_input_norm, test_output_norm

def continue_or_not():

    while True:
        print('Do you want go ahead and predict with your own values? (type 0)')
        print('Do you want to test other parameters? (type 1)')
        print('Do you want go back to the previous options? (Type 2)')

        question = input ('Type:')

        if int(question) in [0,1,2]:
            break

        else:
            print('Wrong value, please try again: ')
            time.sleep(5)

    return int(question)
def user_inputs():

    message = [None] * 7
    # Fazendo um loop para garantir que as informações colocadas são número inteiros. Neste caso não estou limitando os valores
    for i in range (4):
        if (i==0):
            message[i] = input("\nEnter values for Age: ")
            msg = "Enter values for Age: "
        elif (i == 1):
            message[i] = input("Enter values for the number of siblings/spouses of the passenger on board: ")
            msg = "Enter values for the number of siblings/spouses of the passenger on board: "
        elif i == 2:
            message[i] = input("Enter values for the number of parents/children of the passenger on board: ")
            msg = "Enter values for the number of parents/children of the passenger on board: "
        elif (i == 3):
            message[i] = input("Enter values for the fare paid by the passenger: ")
            msg = "Enter values for the fare paid by the passenger: "

        #This loop will always enter to confirm that the entered value makes sense, for this loop will always analyze if the value is only an INT
        while True:

            # Checking if the value is an integer number by the isdigit() function
            if is_float(message[i]):
                break

            else:
                # If the value is not an integer number, inform the error and request the input again
                print("The value must be a number, with the decimal separator being the point '.'. Try again.")
                # Requesting the value again
                message[i] = input(msg)

    while True:
        message[4] = input("Enter the values for the Sex (0 male and 1 female): ")
        # Checking if the value is an integer number by the isdigit() function
        if message[4].isdigit():
            if message[4] in ['0', '1']:
                break
            else:
                print("The value must be 0 or 1. Try again.")
        else:
            # If the value is not an integer number, inform the error and request the input again
            print("The value must be an integer number. Try again.")

    while True:
        message[5] = input("Enter the value for the passenger's socioeconomic class (1 = first class, 2 = second class, 3 = third class): ")
        # Checking if the value is an integer number by the isdigit() function
        if message[5].isdigit():
            if message[5] in ['1', '2', '3']:
                break
            else:
                print("The value must be 1, 2 or 3. Try again.")
        else:
            # If the value is not an integer, inform the error and request the input again
            print("The value must be an integer. Try again.")

    while True:
        message[6] = input("Enter the values for the passenger's port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton): ")

        # Checking if the value is a letter by the method isalpha()
        if message[6].isalpha():

            if message[6] == "C" or message[6] == "c":
                message[6] = 1
                break

            elif message[6] == "Q" or message[6] == "q":
                message[6] = 2
                break

            elif message[6] == "S" or message[6] == "s":
                message[6] = 3
                break

            else:
                print("The value must be C, Q or S. Try again.")
        else:
            # If the value is not an integer, inform the error and request the input again
            print("The value must be a letter. Try again.")

    # Creating a list that stores the values entered by the user
    input_values = [float(message[5]), float(message[4]), float(message[0]), float(message[1]), float(message[2]), float(message[3]), float(message[6])]
    input_values = np.array(input_values).reshape(1, -1)

    return input_values

def user_model_choose():
    while True:
        print('\nChoose one of the options:')
        print('Would you like to train the machine learning model to find the best parameters? (Type: 0)')
        print('Would you like to define the parameters to analyze the accuracy? (Type: 1)')
        print('Would you like to use the previous best parameters of the model (96 % accuracy)? (Type: 2)')
        train_choice_input = input('Type:')

        # trazer para lower case tudo
        if train_choice_input == Train_Choice.BEST_PARAMETERS:
            acc, model = mlp.retrieve_best_parameters()
            time.sleep(2)
            break

        elif train_choice_input == Train_Choice.PREVIOUS_PARAMETERS:
            train_data, test_data, aux_test_data = mlp.load_data()
            train_output_norm, train_input_norm = mlp.treat_data(train_data)
            aux_test_data = aux_test_data[["Survived"]]
            test_data = pd.concat([test_data, aux_test_data], axis=1)
            test_output_norm, test_input_norm = mlp.treat_data(test_data)
            acc, model = mlp.train_model(number_neurons=1, function='relu', solver='sgd',
                                                    train_input_norm=train_input_norm, train_output_norm=train_output_norm,
                                                    test_input_norm=test_input_norm,
                                                    test_output_norm=test_output_norm)
            time.sleep(2)
            break

        while train_choice_input == Train_Choice.DEFINE_PARAMETERS:
            numb_neur, func, solver, train_input_norm, train_output_norm, test_input_norm, test_output_norm = model_parameters()
            acc, model = mlp.train_model(numb_neur, func, solver, train_input_norm,
                                                    train_output_norm, test_input_norm, test_output_norm)

            print("The accuracy for the model is: ", round(acc * 100, 2), "%\n")

            time.sleep(2)

            question = continue_or_not()

            if question == 0:
                break

            if question == 2:
                train_choice_input = Train_Choice.CONTINUE

        if train_choice_input == Train_Choice.DEFINE_PARAMETERS:
            break

        if train_choice_input == Train_Choice.CONTINUE:
            #time.sleep(0)
            pass

        else:
            print('The value typed must be 0 or 1, try again:')
            time.sleep(5)

    return acc, model







