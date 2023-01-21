import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
import warnings

from entities.model_parameters import Model_Parameters
from enums.message_answer import Message_Answer, Message_Best, Messages
from enums.model_function import Model_Function
from enums.molder_solver import Model_Solver

warnings.filterwarnings("ignore", category=Warning)
from IPython.display import clear_output


def load_data():
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    aux_test_data = pd.read_csv("data/gender_submission.csv")
    return train_data, test_data, aux_test_data

def treat_data(data):

    # Data cleaning
    # Excluding empty rows of the dataset when there is missing data in at least 1 column
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

def train_model(parameters: Model_Parameters):

    # Creating the neural network mode
    multilayer_perceptron_classifier = MLPClassifier(hidden_layer_sizes=(parameters.number_neurons + 1),
                                                     max_iter=10000,
                                                     learning_rate_init=0.005,
                                                     validation_fraction=0.15,
                                                     activation=parameters.function,
                                                     solver=parameters.solver,
                                                     tol=1e-4,
                                                     random_state=1)

    # Performing k-fold Cross-Validation with k=5 to evaluate which model has the best result and will be used for the subsequent test phase
    scores = cross_validate(multilayer_perceptron_classifier, parameters.train_input_norm, parameters.train_output_norm.ravel(), cv=5,
                            scoring=('accuracy'),
                            return_train_score=True,
                            return_estimator=True)

    # Obtaining the scores of the models created by k-fold Cross-Validation
    score = (scores['train_score'][:])

    # Obtaining the models created by k-fold Cross-Validation
    model = scores['estimator'][:]

    # Getting the index that obtained the best score
    max_index = np.argmax(score)

    # Best model
    best_model = model[max_index]

    # Calculating the network's responses
    output_model = model[max_index].predict(parameters.test_input_norm).reshape(-1, 1)

    # Denormalizing the data so that the values can return to 0 and 1 and thus make the comparison with the expected values
    output_model = MinMaxScaler(feature_range=(parameters.test_output_norm.min(), parameters.test_output_norm.max())).fit(
        output_model).transform(output_model)

    # Accuracy calculation, comparing the desired values with the actual values
    acc = accuracy_score(parameters.test_output_norm, output_model)

    return acc, best_model


def retrieve_best_parameters():

    # Reading the training data
    train_data, test_data, aux_test_data = load_data()

    aux_test_data = aux_test_data[["Survived"]]

    # As the test information about Survived were in another csv, it was necessary to concatenate the rows of these two documents.
    test_data = pd.concat([test_data, aux_test_data], axis=1)

    # Getting the train and test data normalized
    train_output_norm, train_input_norm = treat_data(train_data)
    test_output_norm, test_input_norm = treat_data(test_data)

    best_accuracy = 0

    #Creating a loop so I can verify what parameters have the accuracy
    k=0
    for i in range(3):
        if (i == 0):
            func = Model_Function.TANH
        elif (i == 1):
            func = Model_Function.LOGISTIC
        elif (i == 2):
            func = Model_Function.RELU

        # Loop for changing the weights optimizers
        for j in range(3):
            if (j == 0):
                solver = Model_Solver.LBFGS
            elif (j == 1):
                solver = Model_Solver.SGD
            elif (j == 2):
                solver = Model_Solver.ADAM

            # Resetting the number of neurons, as the activation functions and optimizers are being swapped.
            for numb_neur in range(5):
                k=k+1

                parameters = Model_Parameters()
                parameters.number_neurons = numb_neur
                parameters.function = func #todo investigar
                parameters.solver = solver #todo investigar
                parameters.train_input_norm = train_input_norm
                parameters.train_output_norm = train_output_norm
                parameters.test_input_norm = test_input_norm
                parameters.test_output_norm = test_output_norm


                accuracy, model = train_model(parameters)

                #Enter in this if only if the current accuracy is higher than the best accuracy
                if (accuracy > best_accuracy):
                    print(Message_Best.ACCURACY_SO_FAR,round(accuracy * 100, 2), "%")
                    print(Messages.RUNNING_MODELS)
                    print(Messages.ITERATIONS, k)
                    best_function = func
                    best_solver = solver
                    best_number_neurons = numb_neur
                    best_accuracy = accuracy
                    best_model = model

    # Prints about the informations of the best parameters
    print(Message_Best.ACCURACY, round(best_accuracy * 100, 2), "%")
    print(Message_Best.FUNCTION, best_function)
    print(Message_Best.ANSWER, best_solver)
    print(Message_Best.NUMBER_NEURONS, best_number_neurons + 1)
    return best_accuracy, best_model


