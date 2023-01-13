import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
import mlp_titanic
import warnings
import time
warnings.filterwarnings("ignore", category=Warning)

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

def ml_function():
    while True:
        question = input('Choose an active function: relu, logistic or tanh: ')
        if question == 'relu' or question == 'logistic' or  question == 'tanh':
            break
        else:
            print('Wrong active function, must be one of the three.')
            time.sleep(2)
    return question

def ml_solver():
    while True:
        question = input('Choose one of the solvers: lbfgs, sgd or adam: ')
        if question == 'lbfgs' or question == 'sgd' or  question == 'adam':
            break
        else:
            print('Wrong solver, must be one of the three.')
            time.sleep(2)
    return question

def ml_neurons():
    while True:
        question = input('Please, type the number of neurons in the hidden layer:')
        if is_int(question) and int(question) > 0:
            break
        else:
            print('The number must be positive and integer')
            time.sleep(2)
    return int(question)

def ml_parameters():
    print('Choose the parameters:')
    train_data, test_data, aux_test_data = mlp_titanic.load_data()
    train_output_norm, train_input_norm = mlp_titanic.treat_data(train_data)
    aux_test_data = aux_test_data[["Survived"]]
    test_data = pd.concat([test_data, aux_test_data], axis=1)
    test_output_norm, test_input_norm = mlp_titanic.treat_data(test_data)
    number_neurons = ml_neurons()
    func = ml_function()
    solver = ml_solver()
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
    message = []
    # Fazendo um loop para garantir que as informações colocadas são número inteiros. Neste caso não estou limitando os valores
    for i in range (4):
        if (i==0):
            message[i] = input("Digite os valores para a Idade: ")
            msg = "Digite os valores para a Idade: "
        elif (i==1):
            message[i] = input("Digite os valores para o número de irmãos/cônjuges do passageiro a bordo do navio: ")
            msg = "Digite os valores para o número de irmãos/cônjuges do passageiro a bordo do navio: "
        elif (i==2):
            message[i]= input("Digite os valores para o número de pais/filhos do passageiro a bordo do navio: ")
            msg = "Digite os valores para o número de pais/filhos do passageiro a bordo do navio: "
        elif (i == 3):
            message[i] = input("Digite os valores para o valor pago pelo bilhete do passageiro: ")
            msg = "Digite os valores para o valor pago pelo bilhete do passageiro: "

        #Sempre entrará neste loop para confirmar que o valor digitado faz sentido, para esse loop sempre irá naalisar se apenas o valor é um INT
        while True:

          #Verificando se o valor é um número inteiro pela função isdigit()
          if is_float(message[i]):
            break

          else:
            # Caso o valor não for um número inteiro, informa o erro e solicita novamente o input
            print("O valor deve ser um número, sendo o separador decimal o ponto '.'. Tente novamente.")
                    #Solicitando novamento o valor
            message[i] = input(msg)

    while True:
      message[4] = input("Digite os valores para o Sexo (0 masculino e 1 feminino): ")
      #Verificando se o valor é um número inteiro pela função isdigit()
      if message[4].isdigit():
        if message[4] in ['0', '1']:
            break
        else:
          print("O valor deve ser 0 ou 1. Tente novamente.")
      else:
        #Caso o valor não for um número inteiro, informa o erro e solicita novamente o input
        print("O valor deve ser um número inteiro. Tente novamente.")

    while True:
      message[5] = input("Digite o valor para a classe socioeconômica do passageiro (1 = primeira classe, 2 = segunda classe, 3 = terceira classe): ")
      #Verificando se o valor é um número inteiro pela função isdigit()
      if message[5].isdigit():
        if message[5] in ['1', '2', '3']:
            break
        else:
          print("O valor deve ser 1, 2 ou 3. Tente novamente.")
      else:
        #Caso o valor não for um número inteiro, informa o erro e solicita novamente o input
        print("O valor deve ser um número inteiro. Tente novamente.")

    while True:
      message[6] = input("Digite os valores para o o porto de embarque do passageiro (C = Cherbourg, Q = Queenstown, S = Southampton): ")
      #Verificando se o valor é uma letra pelo meétodo isalpha()
      if message[6].isalpha():
        if message[6] == "C" or message[6] == "c":
            message[6]=1
            break
        elif message[6] == "Q" or message[6] == "q":
            message[6] = 2
            break
        elif message[6] == "S" or message[6] == "s":
            message[6] = 3
            break
        else:
            print("O valor deve ser C, Q ou S. Tente novamente.")
      else:
        #Caso o valor não for um número inteiro, informa o erro e solicita novamente o input
        print("O valor deve ser uma Letra. Tente novamente.")

    #Criando uma lista na qual armazena os valores digitados pelo usuário
    input_values = [float(message[5]), float(message[4]), float(message[0]), float(message[1]), float(message[2]), float(message[3]), float(message[6])]
    input_values = np.array(input_values).reshape(1, -1)
    return input_values



def main ():
    while True:
        print('Choose one of the options:')
        print('Would you like to train the machine learning model to find the best parameters? (Type: 0)')
        print('Would you like to define the parameters to analyze the accuracy? (Type: 1)')
        print('Would you like to use the previous best parameters of the model? (Type: 2)')
        train_choose = input('type:')
        #trazer para lower case tudo
        if train_choose == '0':
            acc,ml_model = mlp_titanic.main()
            time.sleep(2)
            break

        while train_choose == '1':
            numb_neur, func, solver, train_input_norm, train_output_norm, test_input_norm, test_output_norm = ml_parameters()
            acc,ml_model = mlp_titanic.train_model(numb_neur, func, solver,train_input_norm,
                                   train_output_norm, test_input_norm, test_output_norm)
            time.sleep(2)
            question = continue_or_not()
            if question == 0:
                break
            if question == 2:
                train_choose = 10

        elif train_choose == '2':
            train_data, test_data, aux_test_data = mlp_titanic.load_data()
            train_output_norm, train_input_norm = mlp_titanic.treat_data(train_data)
            aux_test_data = aux_test_data[["Survived"]]
            test_data = pd.concat([test_data, aux_test_data], axis=1)
            test_output_norm, test_input_norm = mlp_titanic.treat_data(test_data)
            acc,ml_model = mlp_titanic.train_model(numb_neur =1, func = 'relu', solver = 'sgd',
                                                   train_input_norm=train_input_norm, train_output_norm=train_output_norm, test_input_norm=test_input_norm,
                                                   test_output_norm=test_output_norm)
            time.sleep(2)
            break

        elif question == 2:
            print()
        else:
            print('The value typed must be 0 or 1, try again:' )
            time.sleep(5)


    inputs = user_inputs()

    print(inputs)

    V_Rede = ml_model.predict(inputs)

    if V_Rede[0]==1:
        print("O resultado da rede é: Sobreviveu")
    else:
        print("O resultado da rede é: Morreu")

if __name__ == '__main__':
    main()