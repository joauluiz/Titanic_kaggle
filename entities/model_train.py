from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from user import data_norm

train_output_norm, train_input_norm, test_output_norm, test_input_norm = data_norm()


class Model_Train:

    def svc_train(self):
        svc = SVC()

        # Setup the parameters to be tested
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

        model = GridSearchCV(svc, param_grid, refit=True, cv=5)

        model.fit(train_input_norm, train_output_norm)

        output_model = model.predict(test_input_norm).reshape(-1, 1)

        acc = accuracy_score(test_output_norm, output_model)

        return acc, model

    def mlp_train(self):

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

        return acc, model
