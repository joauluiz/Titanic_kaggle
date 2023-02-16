from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from user import data_norm

train_output_norm, train_input_norm, test_output_norm, test_input_norm = data_norm()


def mlp_train():
    mlp = MLPClassifier(max_iter=1000)

    # Setup the parameters to be tested
    param_grid = {
        'hidden_layer_sizes': [(5,), (10,), (15,), (20,), (25,), (30,), (35,), (40,)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }

    model = GridSearchCV(mlp, param_grid, refit=True, cv=5)

    model.fit(train_input_norm, train_output_norm)

    output_model = model.predict(test_input_norm).reshape(-1, 1)

    acc = accuracy_score(test_output_norm, output_model)

    print('A acurácia do modelo MLP foi de: ', acc * 100, '%')

    return model


def svc_train():
    svc = SVC()

    # Setup the parameters to be tested
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

    model = GridSearchCV(svc, param_grid, refit=True, cv=5)

    model.fit(train_input_norm, train_output_norm)

    output_model = model.predict(test_input_norm).reshape(-1, 1)

    acc = accuracy_score(test_output_norm, output_model)

    print('A acurácia do modelo SVC foi de: ', acc * 100, '%')

    return model


def decision_tree():
    model = RandomForestClassifier()

    param_grid = {
        'n_estimators': [100, 300, 500, 800, 1000],
        'max_depth': [5, 10, 15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    grid_search = GridSearchCV(model, param_grid, refit=True, cv=5)

    grid_search.fit(train_input_norm, train_output_norm)

    output_model = model.predict(test_input_norm).reshape(-1, 1)

    acc = accuracy_score(test_output_norm, output_model)

    print('A acurácia do modelo RandomForest foi de: ', acc * 100, '%')

    return model


def ensemble():

    ensemble_model = VotingClassifier(estimators=[("mlp", mlp_train()),
                                                  ("gnb", svc_train()),
                                                  ("rf", decision_tree())],
                                      voting="hard")

    ensemble_model.fit(train_input_norm, train_output_norm)

    output_model = ensemble_model.predict(test_input_norm).reshape(-1, 1)

    acc = ensemble_model.score(test_output_norm, output_model)

    print('A acurácia do ensemble foi de: ', acc * 100, '%')

    return ensemble_model
