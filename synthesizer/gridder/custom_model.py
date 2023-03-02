import numpy as np
from synthesizer.gridder.vector_field import VectorField
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """ 
        This object is meant to create user-defined analytical models for 
        density and optionally temperature and vector field.

        Use this file to create your own density distributions in cartesian 
        coordinates. 

        Coordinates x, y and z are already provided for you to use them.
        
        All you need to create is an expression for dens and optionally
        for self.temp. 

        To create a vector field, make sure to create 
        self.vx, self.vy and self.vz.
    """
    def __init__(self, x, y, z, morphology='custom'):
        self.morphology=morphology
        
    @property
    @abstractmethod # this will raise an error if dens is not provided by the user
    def dens(self):
        pass

    @property
    def temp(self):
        return self._temp

    @temp.setter
    def temp(self, temp):
        if type(temp) != np.ndarray:
            raise ValueError(f"temp must be a numpy.array, current input is of type {type(temp)}")
        self._temp=temp

    @property
    def vfield(self):
        return self._vfield

    @vfield.setter
    def vfield(self, vfield):
        if type(vfield) != VectorField:
            raise ValueError(f"vfield must be a VectorField, current input is of type {type(vfield)}")
        self._vfield=vfield