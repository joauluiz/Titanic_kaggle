from enums.gender import Gender
from enums.port_embarkation import Port_Embarkation
from pydantic import BaseModel, validator


class Model_Inputs_Api(BaseModel):

    age: float

    siblings_spouses: float

    parents_children: float

    fare_paid: float

    gender: Gender

    socieconomic_class: int

    port_embarkation: Port_Embarkation

    @validator('age')
    def must_be_greater_than_zero_age(cls, age):
        if age <= 0:
            raise ValueError('Age must be greater than zero')
        return age

    @validator('siblings_spouses')
    def must_be_greater_than_zero_simblings(cls, siblings_spouses):
        if siblings_spouses <= 0:
            raise ValueError('The siblings or spouses values must be greater than zero')
        return siblings_spouses

    @validator('parents_children')
    def must_be_greater_than_zero_parents(cls, parents_children):
        if parents_children <= 0:
            raise ValueError('The parents or children values must be greater than zero')
        return parents_children

    @validator('fare_paid')
    def must_be_greater_than_zero_fare(cls, fare_paid):
        if fare_paid <= 0:
            raise ValueError('The fare paid values must be greater than zero')
        return fare_paid

    @validator('socieconomic_class')
    def must_be_1_3_3 (cls, socieconomic_class):
        if socieconomic_class in [1, 2, 3]:
            return socieconomic_class
        else:
            raise ValueError('The socieconomic class values must be 1, 2 or 3.')


    @validator('gender')
    def must_be_male_female(cls, gender):

       if gender not in Gender:

            raise ValueError('Gender must be male or female')

       else:
            if gender == "male":
                gender = 0

            else:
                gender = 1

       return gender

    @validator('port_embarkation')
    def must_be_SQC(cls, port_embarkation):

       if port_embarkation not in Port_Embarkation:

            raise ValueError('Port embarkation must be S, Q or C')

       else:

            if port_embarkation == "c":
                port_embarkation = 1

            elif port_embarkation == "q":
                port_embarkation = 2

            else:
                port_embarkation = 3

       return port_embarkation
