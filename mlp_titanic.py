import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=Warning)
from IPython.display import clear_output


def load_data():
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    aux_test_data = pd.read_csv("data/gender_submission.csv")
    return train_data, test_data, aux_test_data





#Tratamentos dos dados
#Excluindo linhas vazias do dataset quando estiver faltando dados em pelo menos 1 coluna
def treat_data(data):
    data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']].dropna()
    # Atribuindo às varias de treinamento os inputs e outputs
    output_columns = data[['Survived']]
    # Utilizei essas variáveis pos acreditei que elas tinham maior relação no resultado de sobreviver/morrer. Talvez seja interessante testar mais váriáveis depois
    input_columns = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    # Informações relevantes
    # PassengerId: um ID único para cada passageiro.
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
    # Se por acaso há alguma valor sem ser male e female excluir as linhas (Isso serve para um hipotético caso em que informações sejam adicionadas indevidamente)
    mask = input_columns.loc[(input_columns["Sex"] != "male") & (input_columns["Sex"] != "female")]
    input_columns = input_columns.drop(mask.index)
    # Alterando os valores de masculino para 0, para que possam entrar no modelo da rede neural
    mask = input_columns["Sex"] == "male"
    input_columns.loc[mask, "Sex"] = 0
    # Alterando os valores de  feminino para 1, para que possam entrar no modelo da rede neural
    mask = input_columns["Sex"] == "female"
    input_columns.loc[mask, "Sex"] = 1
    # Se por acaso há alguma valor sem ser S, Q e C, excluir as linhas (Isso serve para um hipotético caso em que informações sejam adicionadas indevidamente)
    mask = input_columns.loc[
    (input_columns["Embarked"] != "C") & (input_columns["Embarked"] != "Q") & (input_columns["Embarked"] != "S")]
    input_columns = input_columns.drop(mask.index)
    # Alterando os valores dos pontos de embarque, C=1, Q=2 e S=3 para que possam entrar no modelo da rede neural
    mask = input_columns["Embarked"] == "C"
    input_columns.loc[mask, "Embarked"] = 1
    mask = input_columns["Embarked"] == "Q"
    input_columns.loc[mask, "Embarked"] = 2
    mask = input_columns["Embarked"] == "S"
    input_columns.loc[mask, "Embarked"] = 3
    # Realizando a normalização dos dados
    input_norm = MinMaxScaler(feature_range=(-1, 1)).fit(input_columns).transform(input_columns)
    output_norm = MinMaxScaler(feature_range=(-1, 1)).fit(output_columns).transform(output_columns)
    return output_norm, input_norm


#Criando um looping para fazer a variação dos parametros da rede para que seja possível avaliar qual mode possui melhor acurácia
def train_model(numb_neur, func, solver,train_input_norm, train_output_norm, test_input_norm, test_output_norm):
    # Criando o modelo da rede neural
    mlpc = MLPClassifier(hidden_layer_sizes=(numb_neur + 1),
                         max_iter=10000,
                         learning_rate_init=0.005,
                         validation_fraction=0.15,
                         activation=func,
                         solver=solver,
                         tol=1e-4,
                         random_state=1)
    # Realizando o Cross-Validation k fold = 5 para avaliar qual modelo possui melhor resultado e que será usado para a fase de teste posteriormente
    scores = cross_validate(mlpc, train_input_norm, train_output_norm.ravel(), cv=5,
                            scoring=('accuracy'),
                            return_train_score=True,
                            return_estimator=True)
    # Obtendos os scores dos modelos criados pelo Cross-Validation k fold=5
    pontos = (scores['train_score'][:])
    # Obtendo os modelos criados pelo Cross-Validation
    modelo = scores['estimator'][:]
    # Pegando o índíce que obteve melhor score
    max_index = np.argmax(pontos)
    #melhor modelo
    best_model = modelo[max_index]
    # Calculando o as respostas da rede
    rede = modelo[max_index].predict(test_input_norm).reshape(-1, 1)
    # Realizando a desnomalização dos dados, para que seja possível os valores voltarem a ser 0 e 1 e assim fazer a comparação com os valores esperados
    rede = MinMaxScaler(feature_range=(test_output_norm.min(), test_output_norm.max())).fit(rede).transform(rede)
    # Calculo da acurácia, comparando os valores desejados com os valores reais
    acc = accuracy_score(test_output_norm, rede)
    return acc, best_model


def main():
    # Leitura dos dados de treinamento
    train_data, test_data, aux_test_data = load_data()

    aux_test_data = aux_test_data[["Survived"]]

    # Como as informações de teste sobre o Survived estavam em um outro csv, foi necessário fazer a concatenação das linhas desses dois documentos
    test_data = pd.concat([test_data, aux_test_data], axis=1)

    train_output_norm, train_input_norm = treat_data(train_data)
    test_output_norm, test_input_norm = treat_data(test_data)

    acc_best = 0

    k=0
    for i in range(3):
        if (i == 0):
            func = "tanh"
        elif (i == 1):
            func = "logistic"
        elif (i == 2):
            func = "relu"

        # Loop para alteração dos otimizadores de pesos
        for j in range(3):
            if (j == 0):
                solver = "lbfgs"
            elif (j == 1):
                solver = "sgd"
            elif (j == 2):
                solver = "adam"
            # Resetando o número de neurônios, pois estão sendo trocados as funções de ativação e os otimizadores
            for numb_neur in range(5):
                k=k+1
                acc, model = train_model(numb_neur, func, solver, train_input_norm, train_output_norm,
                                  test_input_norm, test_output_norm)
                # BEST PARAMETERS

                if (acc > acc_best):
                    print("\nThe best accuracy found so far is: ",round(acc * 100, 2), "%")
                    print("The models are still being tested")
                    print("Number of iterations so far:", k)
                    best_func = func
                    best_solver = solver
                    best_numb_neur = numb_neur
                    acc_best = acc
                    best_model = model

    # Print sobre as informações do melhor modelo
    print("A melhor acurácia foi: ", round(acc_best*100,2), "%")
    print("A função de ativação que obteve melhor resultado foi: ", best_func)
    print("O otimizador de peso que obteve melhor resultado foi: ", best_solver)
    print("A quantidade de neuronios na camada escondida que obteve melhor resultado foi: ", best_numb_neur + 1)
    return best_model

if __name__ == '__main__':
    main()