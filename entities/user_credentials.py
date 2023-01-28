from pydantic import BaseModel


class User_Credentials(BaseModel):
    user_name: str
    password: str
