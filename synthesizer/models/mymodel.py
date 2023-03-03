"Template to define a CustomModel"

# load the BaseModel class
from gridder.custom_model import BaseModel

# Define the CustomModel class
class CustomModel(BaseModel):

    @property
    def dens(self,):
        "define your own expression"
        r = np.sqrt(x*x + y*y + z*z)
        return 1/r