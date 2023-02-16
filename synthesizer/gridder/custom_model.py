import numpy as np

from synthesizer.gridder.vector_field import VectorField

class CustomModel:
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
    def __init__(self, x, y, z, morphology='custom'):
        self.dens = np.zeros((x.shape[0], y.shape[0], z.shape[0]))
        self.temp = np.zeros((x.shape[0], y.shape[0], z.shape[0]))
        self.vfield = VectorField(x, y, z, morphology)

        # Example: create a sphere
        r = np.sqrt(x*x + y*y + z*z)
        self.dens = 1 / r

