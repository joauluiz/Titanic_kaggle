import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
import itertools
import warnings
warnings.filterwarnings("ignore", category=Warning)


#Leitura dos dados de treinamento
Data_train = pd.read_csv("C:/Users/joao_/Desktop/train.csv")
Data_test = pd.read_csv("C:/Users/joao_/Desktop/test.csv")
Data_test_aux = pd.read_csv("C:/Users/joao_/Desktop/gender_submission.csv")

Data_test_aux = Data_test_aux[["Survived"]]

#Como as informações de teste sobre o Survived estavam em um outro csv, foi necessário fazer a concatenação das linhas desses dois documentos
Data_test = pd.concat([Data_test, Data_test_aux], axis=1)


#Tratamentos dos dados
#Excluindo linhas vazias do dataset quando estiver faltando dados em pelo menos 1 coluna
##
Data_train = Data_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']].dropna()
Data_test = Data_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']].dropna()

#Atribuindo às varias de treinamento os inputs e outputs
Output_train = Data_train[['Survived']]

#Utilizei essas variáveis pos acreditei que elas tinham maior relação no resultado de sobreviver/morrer. Talvez seja interessante testar mais váriáveis depois
Input_train = Data_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

#Atribuindo as varias de teste os inputs e outputs
Output_test = Data_test[['Survived']]

#Utilizei essas variáveis pos acreditei que elas tinham maior relação no resultado de sobreviver/morrer. Talvez seja interessante testar mais váriáveis depois
Input_test = Data_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

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

#Se por acaso há alguma valor sem ser male e female excluir as linhas (Isso serve para um hipotético caso em que informações sejam adicionadas indevidamente)
Mask = Input_train.loc[(Input_train["Sex"] != "male") & (Input_train["Sex"] != "female")]
Input_train = Input_train.drop(Mask.index)

#Alterando os valores de masculino para 0, para que possam entrar no modelo da rede neural
Mask = Input_train["Sex"] == "male"
Input_train.loc[Mask,"Sex"] = 0
Mask = Input_test["Sex"] == "male"
Input_test.loc[Mask,"Sex"] = 0

#Alterando os valores de  feminino para 1, para que possam entrar no modelo da rede neural
Mask = Input_train["Sex"] == "female"
Input_train.loc[Mask,"Sex"] = 1
Mask = Input_test["Sex"] == "female"
Input_test.loc[Mask,"Sex"] = 1

#Se por acaso há alguma valor sem ser S, Q e C, excluir as linhas (Isso serve para um hipotético caso em que informações sejam adicionadas indevidamente)
Mask = Input_train.loc[(Input_train["Embarked"] != "C") & (Input_train["Embarked"] != "Q") & (Input_train["Embarked"] != "S")]
Input_train = Input_train.drop(Mask.index)

#Alterando os valores dos pontos de embarque, C=1, Q=2 e S=3 para que possam entrar no modelo da rede neural
Mask = Input_train["Embarked"] == "C"
Input_train.loc[Mask,"Embarked"] = 1
Mask = Input_test["Embarked"] == "C"
Input_test.loc[Mask,"Embarked"] = 1

Mask = Input_train["Embarked"] == "Q"
Input_train.loc[Mask,"Embarked"] = 2
Mask = Input_test["Embarked"] == "Q"
Input_test.loc[Mask,"Embarked"] = 2

Mask = Input_train["Embarked"] == "S"
Input_train.loc[Mask,"Embarked"] = 3
Mask = Input_test["Embarked"] == "S"
Input_test.loc[Mask,"Embarked"] = 3

#Realizando a normalização dos dados
Input_train_Norm = MinMaxScaler(feature_range=(-1, 1)).fit(Input_train).transform(Input_train)
Output_train_Norm = MinMaxScaler(feature_range=(-1, 1)).fit(Output_train).transform(Output_train)
Input_test_Norm = MinMaxScaler(feature_range=(-1, 1)).fit(Input_test).transform(Input_test)




acc_best=0
#Criando um looping para fazer a variação dos parametros da rede para que seja possível avaliar qual mode possui melhor acurácia

#Loop variação das funções de ativação dos neurônios
for k in range (3):
    if (k == 0):
        func = "tanh"
    elif (k == 1):
        func = "logistic"
    elif (k == 2):
        func = "relu"

    #Loop para alteração dos otimizadores de pesos
    for j in range (3):
        if(j==0):
            solver="lbfgs"
        elif (j==1):
            solver = "sgd"
        elif (j==2):
            solver = "adam"
        #Resetando o número de neurônios, pois estão sendo trocados as funções de ativação e os otimizadores
        numb_neur = 0
        for i in range (5):
            #Número de neurônios aumentando em cada iteração
            numb_neur = numb_neur + 1
            #Criando o modelo da rede neural
            mlpc = MLPClassifier(hidden_layer_sizes=(numb_neur),
                                max_iter=10000,
                                learning_rate_init=0.005,
                                validation_fraction=0.15,
                                activation=func,
                                solver = solver,
                                tol=1e-4,
                                random_state=1)

            #Realizando o Cross-Validation k fold = 5 para avaliar qual modelo possui melhor resultado e que será usado para a fase de teste posteriormente
            scores = cross_validate(mlpc, Input_train_Norm, Output_train_Norm.ravel(), cv=5,
                                    scoring=('accuracy'),
                                    return_train_score=True,
                                    return_estimator=True)

            #Obtendos os scores dos modelos criados pelo Cross-Validation k fold=5
            pontos = (scores['train_score'][:])

            #Obtendo os modelos criados pelo Cross-Validation
            modelo = scores['estimator'][:]

            #Pegando o índíce que obteve melhor score
            max_index = np.argmax(pontos)

            #Calculando o as respostas da rede
            V_Rede = modelo[max_index].predict(Input_test_Norm).reshape(-1,1)

            #Realizando a desnomalização dos dados, para que seja possível os valores voltarem a ser 0 e 1 e assim fazer a comparação com os valores esperados
            V_Rede = MinMaxScaler(feature_range = (Output_test.min().values[0], Output_test.max().values[0])).fit(V_Rede).transform(V_Rede)


            #Calculo da acurácia, comparando os valores desejados com os valores reais
            acc = accuracy_score(Output_test, V_Rede)

            print(acc)

            #BEST PARAMETERS
            if (acc>acc_best):
                best_func = func
                best_solver = solver
                best_numb_neur = numb_neur
                best_model = scores['estimator'][max_index]
                acc_best = acc

#Print sobre as informações do melhor modelo
print("A melhor acurácia foi: ",acc_best)
print("A função de ativação que obteve melhor resultado foi: ",best_func)
print("O otimizador de peso que obteve melhor resultado foi: ",best_solver)
print("A quantidade de neuronios na camada escondida que obteve melhor resultado foi: ",best_numb_neur)


