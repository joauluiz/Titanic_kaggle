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






#Data cleaning
#Excluding empty rows of the dataset when there is missing data in at least 1 column
def treat_data(data):

    data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']].dropna()

    # Assigning the training variables the inputs and outputs
    output_columns = data[['Survived']]

    # I used these variables because I believed they had a higher relationship in the result of surviving/dying. Maybe it would be interesting to test more variables later
    input_columns = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

    # Relevant information
    # PassengerId: a unique ID for each passenger.
    # Survived: indicates whether the passenger survived (1) or not (0) the accident. This is the label that should be predicted for the passengers in the test file.
    # Pclass: the passenger's socioeconomic class (1 = first class, 2 = second class, 3 = third class).
    # Name: the passenger's name.
    # Sex: the passenger's gender (male or female).
    # Age: the passenger's age (in years).
    # SibSp: the number of siblings/spouses of the passenger on board the ship.
    # Parch: the number of parents/children of the passenger on board the ship.
    # Ticket: the passenger's ticket number.
    # Fare: the value paid for the passenger's ticket.
    # Cabin: the passenger's cabin number.
    # Embarked: the passenger's port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

    # If by chance there is any value other than male and female, delete the rows (This serves for a hypothetical case in which information is added improperly)
    mask = input_columns.loc[(input_columns["Sex"] != "male") & (input_columns["Sex"] != "female")]
    input_columns = input_columns.drop(mask.index)

    # Changing the values of male to 0, so that they can enter the neural network model
    mask = input_columns["Sex"] == "male"
    input_columns.loc[mask, "Sex"] = 0

    # Changing the values of female to 1, so that they can enter the neural network model
    mask = input_columns["Sex"] == "female"
    input_columns.loc[mask, "Sex"] = 1

    # If by chance there is any value other than S, Q, and C, delete the rows (This serves for a hypothetical case in which information is added improperly)
    mask = input_columns.loc[(input_columns["Embarked"] != "C") & (input_columns["Embarked"] != "Q") & (input_columns["Embarked"] != "S")]
    input_columns = input_columns.drop(mask.index)

    # Changing the values from embarkation points, C=1, Q=2 e S=3 so then they can be used by the neural network model
    mask = input_columns["Embarked"] == "C"
    input_columns.loc[mask, "Embarked"] = 1
    mask = input_columns["Embarked"] == "Q"
    input_columns.loc[mask, "Embarked"] = 2
    mask = input_columns["Embarked"] == "S"
    input_columns.loc[mask, "Embarked"] = 3

    # Doing the data normalization
    input_norm = MinMaxScaler(feature_range=(-1, 1)).fit(input_columns).transform(input_columns)
    output_norm = MinMaxScaler(feature_range=(-1, 1)).fit(output_columns).transform(output_columns)

    return output_norm, input_norm


#Criando um looping para fazer a variação dos parametros da rede para que seja possível avaliar qual mode possui melhor acurácia
def train_model(number_neurons, function, solver, train_input_norm, train_output_norm, test_input_norm, test_output_norm):
    # Criando o modelo da rede neural
    multilayer_perceptron_classifier = MLPClassifier(hidden_layer_sizes=(number_neurons + 1),
                                                     max_iter=10000,
                                                     learning_rate_init=0.005,
                                                     validation_fraction=0.15,
                                                     activation=function,
                                                     solver=solver,
                                                     tol=1e-4,
                                                     random_state=1)
    # Realizando o Cross-Validation k fold = 5 para avaliar qual modelo possui melhor resultado e que será usado para a fase de teste posteriormente
    scores = cross_validate(multilayer_perceptron_classifier, train_input_norm, train_output_norm.ravel(), cv=5,
                            scoring=('accuracy'),
                            return_train_score=True,
                            return_estimator=True)
    # Obtendos os scores dos modelos criados pelo Cross-Validation k fold=5
    scores = (scores['train_score'][:])
    # Obtendo os modelos criados pelo Cross-Validation
    model = scores['estimator'][:]
    # Pegando o índíce que obteve melhor score
    max_index = np.argmax(scores)
    #melhor modelo
    best_model = model[max_index]
    # Calculando o as respostas da rede
    output_model = model[max_index].predict(test_input_norm).reshape(-1, 1)
    # Realizando a desnomalização dos dados, para que seja possível os valores voltarem a ser 0 e 1 e assim fazer a comparação com os valores esperados
    output_model = MinMaxScaler(feature_range=(test_output_norm.min(), test_output_norm.max())).fit(output_model).transform(output_model)
    # Calculo da acurácia, comparando os valores desejados com os valores reais
    acc = accuracy_score(test_output_norm, output_model)
    return acc, best_model


def main():
    # Leitura dos dados de treinamento
    train_data, test_data, aux_test_data = load_data()

    aux_test_data = aux_test_data[["Survived"]]

    # Como as informações de teste sobre o Survived estavam em um outro csv, foi necessário fazer a concatenação das linhas desses dois documentos
    test_data = pd.concat([test_data, aux_test_data], axis=1)

    train_output_norm, train_input_norm = treat_data(train_data)
    test_output_norm, test_input_norm = treat_data(test_data)

    best_accuracy = 0

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
                accuracy, model = train_model(numb_neur, func, solver, train_input_norm, train_output_norm,
                                  test_input_norm, test_output_norm)
                # BEST PARAMETERS

                if (accuracy > best_accuracy):
                    print("\nThe best accuracy found so far is: ",round(accuracy * 100, 2), "%")
                    print("The models are still being tested")
                    print("Number of iterations so far:", k)
                    best_function = func
                    best_solver = solver
                    best_number_neurons = numb_neur
                    best_accuracy = accuracy
                    best_model = model

    # Print sobre as informações do melhor modelo
    print("A melhor acurácia foi: ", round(best_accuracy*100,2), "%")
    print("A função de ativação que obteve melhor resultado foi: ", best_function)
    print("O otimizador de peso que obteve melhor resultado foi: ", best_solver)
    print("A quantidade de neuronios na camada escondida que obteve melhor resultado foi: ", best_number_neurons + 1)
    return best_model

if __name__ == '__main__':
    main()