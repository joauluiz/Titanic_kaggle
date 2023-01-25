import time

import numpy as np
from fastapi import FastAPI

import mlp
from entities.model_parameters import Model_Parameters
from entities.model_parameters_api import Model_Parameters_Api, Model_Inputs_Api
from main import main
from mlp import train_model
from user import user_inputs, inputs, data_norm, model_parameters, continue_or_not

app = FastAPI()


@app.get(f"/teste/{Model_Parameters_Api.number_neurons}/{Model_Parameters_Api.function}/{Model_Parameters_Api.solver}"
         f"/{Model_Inputs_Api.age}/{Model_Inputs_Api.siblings_spouses}/{Model_Inputs_Api.parents_children}/{Model_Inputs_Api.fare_paid}/{Model_Inputs_Api.sex}"
         f"/{Model_Inputs_Api.socieconomic_class}"/{Model_Inputs_Api.port_embarkation})
def out_model(parameters: Model_Parameters_Api, input_values: Model_Inputs_Api):

    parameters.train_output_norm, parameters.train_input_norm, \
        parameters.test_output_norm, parameters.test_input_norm = data_norm()

    acc, model = mlp.train_model(Model_Parameters(parameters.number_neurons, parameters.function,
                                                  parameters.solver, parameters.train_input_norm,
                                                  parameters.train_output_norm, parameters.test_input_norm,
                                                  parameters.test_output_norm))

    print("The accuracy for the model is: ", round(acc * 100, 2), "%\n")

    input_value = [[input_values.age, input_values.siblings_spouses, input_values.parents_children,
                    input_values.fare_paid, input_values.sex,
                    input_values.socieconomic_class, input_values.port_embarkation]]

    output_model = model.predict(input_value).reshape(-1, 1)

    if output_model[0] == 1:
        teste = "result : Survived"
    else:
        teste = "result : Died"

    return teste


@app.get("/train")
def out_model():
    parameters = Model_Parameters_Api()

    parameters.train_output_norm, parameters.train_input_norm, \
        parameters.test_output_norm, parameters.test_input_norm = data_norm()

    acc, model = mlp.train_model(Model_Parameters(parameters.number_neurons, parameters.function,
                                                  parameters.solver, parameters.train_input_norm,
                                                  parameters.train_output_norm, parameters.test_input_norm,
                                                  parameters.test_output_norm))

    print("The accuracy for the model is: ", round(acc * 100, 2), "%\n")

    input_values = Model_Inputs_Api()

    input_value = [[input_values.age, input_values.siblings_spouses, input_values.parents_children,
                    input_values.fare_paid, input_values.sex,
                    input_values.socieconomic_class, input_values.port_embarkation]]

    output_model = model.predict(input_value).reshape(-1, 1)

    if output_model[0] == 1:
        teste = "result : Survived"
    else:
        teste = "result : Died"

    return teste


@app.get('/main')
def main_output():
    output = main()
    return output
