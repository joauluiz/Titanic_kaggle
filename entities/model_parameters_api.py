from enums.gender import Gender
from enums.port_embarkation import Port_Embarkation
from pydantic import BaseModel, validator


class Model_Inputs_Api(BaseModel):
    age = float

    siblings_spouses = float

    parents_children = float

    fare_paid = float

    sex = Gender

    socieconomic_class = int

    port_embarkation = Port_Embarkation
    #
    # @validator('age', allow_reuse=True, check_fields=False)
    # def must_be_greater_than_zero(cls, age):
    #     if age <= 0:
    #         raise ValueError('Age must be greater than zero')
    #     return age
    #
    # @validator('siblings_spouses', allow_reuse=True, check_fields=False)
    # def must_be_greater_than_zero(cls, siblings_spouses):
    #     if siblings_spouses <= 0:
    #         raise ValueError('The siblings or spouses values must be greater than zero')
    #     return siblings_spouses
    #
    # @validator('parents_children', allow_reuse=True, check_fields=False)
    # def must_be_greater_than_zero(cls, parents_children):
    #     if parents_children <= 0:
    #         raise ValueError('The parents or children values must be greater than zero')
    #     return parents_children
    #
    # @validator('fare_paid', allow_reuse=True, check_fields=False)
    # def must_be_greater_than_zero(cls, fare_paid):
    #     if fare_paid <= 0:
    #         raise ValueError('The fare paid values must be greater than zero')
    #     return fare_paid
    #
    # @validator('socieconomic_class', allow_reuse=True, check_fields=False)
    # def must_be_greater_than_zero(cls, socieconomic_class):
    #     if socieconomic_class in [1, 2, 3]:
    #         raise ValueError('The socieconomic class values must be greater than zero')
    #     return socieconomic_class
