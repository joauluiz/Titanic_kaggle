from enum import Enum

# class syntax
class Message_Inputs (Enum):
    AGE = "\nEnter values for Age: "
    SIBLINGS_SPOUSES = "Enter values for the number of siblings/spouses of the passenger on board: "
    PARENTS_CHILDREN = "Enter values for the number of parents/children of the passenger on board: "
    FARE_PAID = "Enter values for the fare paid by the passenger: "
    SOCIECONOMIC_CLASS = "Enter the value for the passenger's socioeconomic class (1 = first class, 2 = second class, 3 = third class): "
    PORT_EMBARKATION = "Enter the values for the passenger's port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton): "
    SEX = "Enter the values for the Sex (0 male and 1 female): "
    BEST_PARAMETERS = 'Would you like to train the machine learning model to find the best parameters? (Type: 0)'
    DEFINE_PARAMETERS = 'Would you like to define the parameters to analyze the accuracy? (Type: 1)'
    PREVIOUS_BEST_PARAMETERS = 'Would you like to use the previous best parameters of the model (96 % accuracy)? (Type: 2)'
    NUMBER_NEURONS = 'Please, type the number of neurons in the hidden layer:'
    SOLVER = 'Choose one of the solvers: lbfgs, sgd or adam: '
    ACTIVE_FUNCTION = 'Choose an active function: relu, logistic or tanh: '
    CONTINUE_PREDICT = 'Do you want go ahead and predict with your own values? (type 0)'
    TEST_OTHER_PARAMETERS = 'Do you want to test other parameters? (type 1)'
    PREVIOUS_OPTIONS = 'Do you want go back to the previous options? (Type 2)'



