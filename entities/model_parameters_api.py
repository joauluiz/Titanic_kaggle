from entities.model_inputs import input_float_values, input_sex, input_socieconomic_class, input_port_embarkation
from enums.message_inputs import Message_Inputs
from user import model_function, data_norm, model_solver, model_neurons
from pydantic import BaseModel


class Model_Parameters_Api(BaseModel):
    arbitrary_types_allowed = True
    number_neurons = model_neurons()
    function = model_function()
    solver = model_solver()
    train_output_norm = 0
    train_input_norm = 0
    test_output_norm = 0
    test_input_norm = 0


class Model_Inputs_Api(BaseModel):
    age = input_float_values(Message_Inputs.AGE.value)

    siblings_spouses = input_float_values(Message_Inputs.SIBLINGS_SPOUSES.value)

    parents_children = input_float_values(Message_Inputs.PARENTS_CHILDREN.value)

    fare_paid = input_float_values(Message_Inputs.FARE_PAID.value)

    sex = input_sex()

    socieconomic_class = input_socieconomic_class()

    port_embarkation = input_port_embarkation()
