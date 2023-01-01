import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
import itertools
import warnings
warnings.filterwarnings("ignore", category=Warning)


#Leitura dos dados de treinamento
Data_train = pd.read_csv("C:/Users/joao_/Desktop/train.csv")
Data_test = pd.read_csv("C:/Users/joao_/Desktop/test.csv")


#Tratamentos dos dados
#Excluindo linhas vazias do dataset quando estiver faltando dados em pelo menos 1 coluna
##
Data_train = Data_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']].dropna()

#Atribuindo às varias de treinamento os inputs e outputs
Output_train = Data_train[['Survived']] #PRECISA CONCATENAR COM A OUTRA PLANILHA QUE FALA SE VIVEU OU MORREU
#Utilizei essas variáveis pos acreditei que elas tinham maior relação no resultado de sobreviver/morrer. Talvez seja interessante testar mais váriáveis depois
Input_train = Data_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]


#Prints para analisar o dataset
#Informações relevantes
#PassengerId: um ID único para cada passageiro.
# Survived: indica se o passageiro sobreviveu (1) ou não (0) ao acidente. Este é o rótulo que deve ser previsto para os passageiros do arquivo de teste.
# Pclass: a classe socioeconômica do passageiro (1 = primeira classe, 2 = segunda classe, 3 = terceira classe).
# Name: o nome do passageiro.
# Sex: o gênero do passageiro (masculino ou feminino).
# Age: a idade do passageiro (em anos).
# SibSp: o número de irmãos/cônjuges do passageiro a bordo do navio.
# Parch: o número de pais/filhos do passageiro a bordo do navio.
# Ticket: o número do bilhete do passageiro.
# Fare: o valor pago pelo bilhete do passageiro.
# Cabin: o número da cabine do passageiro.
# Embarked: o porto de embarque do passageiro (C = Cherbourg, Q = Queenstown, S = Southampton).

#print(Input_train[["Sex"]])
#print(Input_train.columns)

#Se por acaso há alguma valor sem ser male e female excluir as linhas (Isso serve para um hipotético caso em que informações sejam adicionadas indevidamente)
Mask = Input_train.loc[(Input_train["Sex"] != "male") & (Input_train["Sex"] != "female")]

Input_train = Input_train.drop(Mask.index)

#Alterando os valores de masculino para 0, para que possam entrar no modelo da rede neural
Mask = Input_train["Sex"] == "male"
Input_train.loc[Mask,"Sex"] = 0

#Alterando os valores de  feminino para 1, para que possam entrar no modelo da rede neural
Mask = Input_train["Sex"] == "female"
Input_train.loc[Mask,"Sex"] = 1

#Se por acaso há alguma valor sem ser S, Q e C, excluir as linhas (Isso serve para um hipotético caso em que informações sejam adicionadas indevidamente)
Mask = Input_train.loc[(Input_train["Embarked"] != "C") & (Input_train["Embarked"] != "Q") & (Input_train["Embarked"] != "S")]

Input_train = Input_train.drop(Mask.index)

#Alterando os valores dos pontos de embarque, C=1, Q=2 e S=3 para que possam entrar no modelo da rede neural
Mask = Input_train["Embarked"] == "C"
Input_train.loc[Mask,"Embarked"] = 1

Mask = Input_train["Embarked"] == "Q"
Input_train.loc[Mask,"Embarked"] = 2

Mask = Input_train["Embarked"] == "S"
Input_train.loc[Mask,"Embarked"] = 3

#Realizando a normalização dos dados
Input_train_Norm = MinMaxScaler(feature_range=(-1, 1)).fit(Input_train).transform(Input_train)
Output_train_Norm = MinMaxScaler(feature_range=(-1, 1)).fit(Output_train).transform(Output_train)

#Input_test_Norm = MinMaxScaler(feature_range=(-1, 1)).fit(Input_test).transform(Input_test)
min_train = np.min(Input_train_Norm)
max_train = np.max(Input_train_Norm)

print(min_train)
print(max_train)

# A função de ativação que obteve melhor resultado foi:  relu
# O otimizador de peso que obteve melhor resultado foi:  sgd
# A quantidade de neuronios na camada escondida que obteve melhor resultado foi:  1

#Criando a rede neural
mlpc = MLPClassifier(hidden_layer_sizes=(1),
                    max_iter=10000,
                    learning_rate_init=0.01,
                    validation_fraction=0.15,
                    activation='relu',
                    solver='sgd',
                    tol=1e-4,
                    random_state=1)

#Realizando o Cross-Validation k fold = 5 para avaliar qual modelo possui melhor resultado e que será usado para a fase de teste posteriormente
scores = cross_validate(mlpc, Input_train_Norm, Output_train_Norm.ravel(), cv=5,
                        scoring=('accuracy'),
                        return_train_score=True,
                        return_estimator=True)

#Obtendos os scores dos modelos criados pelo Cross-Validation k fold=5
teste = (scores['train_score'][:])

#Obtendo os modelos criados pelo Cross-Validation
modelo = scores['estimator'][:]

#Pegando o índíce que obteve melhor score
max_index = np.argmax(teste)

#Criando um array que permite receber vaariáveis de qualquer tipo
message = np.zeros(7, dtype=object)

def is_float(string):
  try:
    float(string)
    return True
  except ValueError:
    return False

#Fazendo um loop para garantir que as informações colocadas são número inteiros. Neste caso não estou limitando os valores
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


#Sempre entrará neste loop para confirmar que o valor digitado faz sentido, para esse loop sempre irá naalisar se apenas o valor é um INT e se é 0 ou 1
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

#Data_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]


Data_user = np.zeros(7, dtype='float')

#Criando uma lista na qual armazena os valores digitados pelo usuário
input_values = [float(message[5]), float(message[4]), float(message[0]), float(message[1]), float(message[2]), float(message[3]), float(message[6])]

input_values = np.array(input_values).reshape(1, -1)

print(input_values)

V_Rede = modelo[max_index].predict(input_values)

if V_Rede[0]==1:
    print("O resultado da rede é: Sobreviveu")
else:
    print("O resultado da rede é: Morreu")