from fastapi import FastAPI
import mlp
from entities.model_parameters import Model_Parameters
from entities.model_parameters_api import  Model_Inputs_Api
from user import  data_norm
app = FastAPI()


@app.post("/")
def out_model(parameters: Model_Inputs_Api, response_model= message):

    train_output_norm, train_input_norm, \
        test_output_norm, test_input_norm = data_norm()

    acc, model = mlp.train_model(Model_Parameters(1, 'relu',
                                                  'sgd', train_input_norm,
                                                  train_output_norm, test_input_norm,
                                                  test_output_norm))

    print("The accuracy for the model is: ", round(acc * 100, 2), "%\n")

    input_values = [[parameters.age, parameters.siblings_spouses, parameters.parents_children,
                     parameters.fare_paid, parameters.sex,
                     parameters.socieconomic_class, parameters.port_embarkation]]

    output_model = model.predict(input_values).reshape(-1, 1)

    if output_model[0] == 1:
        message = "result : Survived"
    else:
        message = "result : Died"

    return message


# @app.get("/train")
# def out_model():
#     parameters = Model_Parameters_Api()
#
#     parameters.train_output_norm, parameters.train_input_norm, \
#         parameters.test_output_norm, parameters.test_input_norm = data_norm()
#
#     acc, model = mlp.train_model(Model_Parameters(parameters.number_neurons, parameters.function,
#                                                   parameters.solver, parameters.train_input_norm,
#                                                   parameters.train_output_norm, parameters.test_input_norm,
#                                                   parameters.test_output_norm))
#
#     print("The accuracy for the model is: ", round(acc * 100, 2), "%\n")
#
#     input_values = Model_Inputs_Api()
#
#     input_value = [[input_values.age, input_values.siblings_spouses, input_values.parents_children,
#                     input_values.fare_paid, input_values.sex,
#                     input_values.socieconomic_class, input_values.port_embarkation]]
#
#     output_model = model.predict(input_value).reshape(-1, 1)
#
#     if output_model[0] == 1:
#         teste = "result : Survived"
#     else:
#         teste = "result : Died"
#
#     return teste


# @app.get('/main')
# def main_output():
#     output = main()
#     return output
