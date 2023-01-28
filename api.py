from fastapi import FastAPI, Depends
import mlp
from entities.model_parameters import Model_Parameters
from entities.model_parameters_api import Model_Inputs_Api
from entities.user_credentials import User_Credentials
from user import data_norm
from fastapi_users import FastAPIUsers
from fastapi_users.authentication.basic import BasicAuth

app = FastAPI()


def validate_credentials(credentials: User_Credentials):
    if credentials.user_name == "admin" and credentials.password == "123":
        return True
    return False


basic_auth = BasicAuth(validate_user=validate_credentials(User_Credentials()))
fastapi_users = FastAPIUsers(app, auth_backend=basic_auth)


def output_message(number_output):
    if number_output == 1:
        message = "result : Survived"
    else:
        message = "result : Died"

    return message


@app.post("/")
def out_model(parameters: Model_Inputs_Api):
    train_output_norm, train_input_norm, \
        test_output_norm, test_input_norm = data_norm()

    acc, model = mlp.train_model(Model_Parameters(1, 'relu',
                                                  'sgd', train_input_norm,
                                                  train_output_norm, test_input_norm,
                                                  test_output_norm))

    input_values = [[parameters.age, parameters.siblings_spouses, parameters.parents_children,
                     parameters.fare_paid, parameters.gender,
                     parameters.socieconomic_class, parameters.port_embarkation]]

    output_model = model.predict(input_values).reshape(-1, 1)

    number_output = output_model[0]

    return output_message(number_output)
