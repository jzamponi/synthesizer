import numpy as np

from synthesizer.gridder.models import BaseModel
from synthesizer.gridder.vector_field import VectorField

class CustomModel(BaseModel):
    """ 
        This object is meant to create user-defined analytical models for 
        density and optionally temperature and vector field.

        Use this file to create your own density distributions in cartesian 
        coordinates. 

        Coordinates x, y and z are already provided for you to use them.
        
        All you need to create it's an expression for self.dens and optionally
        for self.temp. 

        To create a vector field, make sure to create 
        self.vx, self.vy and self.vz.
    """

    @property
    def dens(self):
        # Example: create a sphere
        r = np.sqrt(x*x + y*y + z*z)
        return 1 / r

