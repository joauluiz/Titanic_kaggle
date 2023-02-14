import asyncio
import subprocess
from fastapi import FastAPI, Depends, HTTPException, status
from starlette import status
import mlp
from entities.model_parameters import Model_Parameters
from entities.model_parameters_api import Model_Inputs_Api
from entities.user_credentials import User_Credentials
from user import data_norm
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

app = FastAPI()

security = HTTPBasic()


@app.get("/users/me")
# def read_current_user(credentials: HTTPBasicCredentials = Depends(security)):
#     return {"username": credentials.username, "password": credentials.password}
#
#
# def validate_credentials(credentials: User_Credentials):
#     if credentials.user_name == "admin" and credentials.password == "123":
#         return True
#     return False
async def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = b"admin"
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = b"123"
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return "Released access"


def output_message(number_output):
    if number_output == 1:
        message = "result : Survived"
    else:
        message = "result : Died"

    return message


@app.post("/predict")
async def out_model(parameters: Model_Inputs_Api,  user = Depends(get_current_username)):

    train_output_norm, train_input_norm, \
        test_output_norm, test_input_norm = data_norm()

    #chamada Banco de Dados

    acc, model = mlp.train_model(Model_Parameters(1, 'relu',
                                                  'sgd', train_input_norm,
                                                  train_output_norm, test_input_norm,
                                                  test_output_norm))

    input_values = [[parameters.age, parameters.siblings_spouses, parameters.parents_children,
                     parameters.fare_paid, parameters.gender,
                     parameters.socieconomic_class, parameters.port_embarkation]]

    output_model = model.predict(input_values).reshape(-1, 1)

    number_output = output_model[0]

    return output_message(number_output) # Todo refazer

if __name__ == "__main__":

    app.run(host="0.0.0.0:$PORT")
