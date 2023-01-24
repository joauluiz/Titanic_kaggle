from fastapi import FastAPI

from entities.model_parameters_api import Model_Parameters_Api, Model_Inputs_Api
from main import main
from mlp import train_model
from user import user_inputs, inputs, data_norm

app = FastAPI()


# @app.get("/train")
# def out_model():
#     parameters = Model_Parameters_Api()
#
#     parameters.train_output_norm, parameters.train_input_norm, parameters.test_output_norm, \
#         parameters.test_input_norm = data_norm()
#
#     acc, best_model = train_model(parameters)
#
#     input_values = Model_Inputs_Api()
#
#     output_model = best_model.predict(input_values)
#
#     if output_model[0] == 1:
#         return {"result": "Survived"}
#     else:
#         return {"result": "Died"}


@app.get('/main')
def main_output():
    output = main()
    return output
